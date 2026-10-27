"""
Model 01: LightGBM urban-heat model -> per-pixel heat prediction and hotspot maps.

    python models/train_lgbm.py                       # clean 14-city table, leave-one-city-out CV
    python models/train_lgbm.py --cv-sample 0.15      # faster CV (fewer training rows per fold)
    python models/train_lgbm.py --data PATH --cv block   # spatial-block CV (used automatically with < 3 cities)

Target   LST_anom = LST - mean LST of its city (deg C). Hotspots are defined city-relative (top 10 % of urban
         pixels), and a city's baseline temperature is not something 14 cities can teach a model, so we predict
         the within-city pattern. Water pixels (LANDCOVER 80) are dropped unless --keep-water.
Features satellite indices, neighbourhood means, GHSL urban form, terrain, distances, weather, and a physics term
         ABSORBED_SW = (1 - albedo) * shortwave. Target-derived columns (LST_z, SUHI, HOTSPOT, LST_std, n_obs),
         coordinates and city ids are NOT used.
Physics  monotonic constraints: more vegetation (NDVI, and its neighbourhood means) can never raise predicted LST.
Validate leave-one-city-out: every city is predicted by a model that never saw it. Ridge is the baseline.
Outputs  models/artifacts/   lgbm model, feature list, LOCO metrics, importances
         models/outputs/     out-of-fold predictions table + observed-vs-predicted hotspot maps
"""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from hotspots import hotspot_flag  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
ROOT = Path(__file__).resolve().parents[1]
ART, OUT = Path(__file__).parent / "artifacts", Path(__file__).parent / "outputs"

MET = ["MET_T2M", "MET_RH", "MET_WIND", "MET_SW", "MET_RAIN7", "MET_RAIN30", "MET_T2M_SEASON_MAX"]
BASE = ["NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "SLOPE_deg", "LANDCOVER",
        "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP",
        "NDVI_nbr150", "NDVI_nbr450", "BUILT_nbr150", "BUILT_nbr450", "HEIGHT_nbr150", "HEIGHT_nbr450",
        "DIST_WATER_m", "DIST_GREEN_m"]
NORM = ["NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "NDVI_nbr150", "NDVI_nbr450",
        "BUILT_nbr150", "HEIGHT_nbr150", "DIST_WATER_m", "DIST_GREEN_m"]
MONO_DOWN = {"NDVI", "NDVI_nbr150", "NDVI_nbr450", "NDVI_cn", "NDVI_nbr150_cn", "NDVI_nbr450_cn"}      # more green -> LST does not go up


def read_table(path):
    """Read a training table fast: .parquet directly, or a CSV with the multi-threaded pyarrow reader."""
    path = str(path)
    pq = path.replace(".csv.gz", ".parquet")
    if path.endswith(".parquet") or (path.endswith(".csv.gz") and os.path.exists(pq)):
        return pd.read_parquet(pq if not path.endswith(".parquet") else path)
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:  # noqa: BLE001  (older pandas / pyarrow missing)
        return pd.read_csv(path)


def load(path, keep_water, use_met, target, city_norm, context=False):
    df = read_table(path)
    n0 = len(df)
    if not keep_water:
        df = df[df["LANDCOVER"] != 80]
    df = df.dropna(subset=BASE).reset_index(drop=True)
    print(f"loaded {n0:,} rows -> {len(df):,} after dropping water / NaN | {df['city'].nunique()} cities")
    df["LST_anom"] = df["LST"] - df.groupby("city")["LST"].transform("mean")
    df["ABSORBED_SW"] = (1 - df["ALBEDO"]) * df["MET_SW"]
    sd = df.groupby("city")["LST"].transform("std")
    df["TARGET"] = df["LST_anom"] / sd if target == "z" else df["LST_anom"]
    feats = BASE + (MET + ["ABSORBED_SW"] if use_met else [])
    if city_norm:        # feature minus its city mean: lets the model learn "greener than the rest of THIS city"
        for f in NORM:
            df[f + "_cn"] = df[f] - df.groupby("city")[f].transform("mean")
        feats += [f + "_cn" for f in NORM]
    if context:          # multi-scale surroundings from extraction/add_context_features.py
        ctx = [c for c in df.columns if re.search(r"_(c\d+|sd\d+)$", c)]
        if not ctx:
            raise SystemExit("no context columns in the data: run extraction/add_context_features.py and pass --data <..._ctx.csv.gz>")
        feats += ctx
        print(f"context features: {len(ctx)}")
    lc = sorted(df["LANDCOVER"].astype(int).unique())
    df["LANDCOVER"] = pd.Categorical(df["LANDCOVER"].astype(int), categories=lc)
    return df, feats


def lgb_params(args, feats):
    simple = getattr(args, "simple", False)
    return dict(objective="regression", n_estimators=args.trees, learning_rate=0.03 if simple else 0.05,
                num_leaves=15 if simple else 63, max_depth=5 if simple else -1,
                min_child_samples=3000 if simple else 200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.6 if simple else 0.8, reg_lambda=50.0 if simple else 5.0,
                monotone_constraints=[-1 if (f in MONO_DOWN or re.fullmatch(r"NDVI_c\d+", f)) else 0 for f in feats] if not args.no_monotone else None,
                n_jobs=-1, verbose=-1, random_state=0)


def folds(df, cv):
    """Yield (fold name, test mask, unit codes). Units are integer codes (sorted order), not strings: sorting / comparing
    9 million text values for each of 50 folds took minutes of single-core time."""
    if cv == "loco":
        codes, names = pd.factorize(df["city"], sort=True)
        for k, c in enumerate(names):
            yield c, codes == k, codes
    else:
        codes = pd.factorize(df["block_id"], sort=True)[0]
        for i, (_, te) in enumerate(GroupKFold(5).split(df, groups=df["block_id"])):
            m = np.zeros(len(df), bool)
            m[te] = True
            yield f"fold{i}", m, codes


def hotspot_scores(g, pred_col):
    """Hotspots as defined in build_dataset: top 10 % of URBAN pixels in a city. Returns metrics for one city."""
    u = g[g["IS_URBAN"] == 1]
    out = {"urban_px": len(u)}
    if u["HOTSPOT"].nunique() < 2:
        return out
    thr = u[pred_col].quantile(0.9)
    flag = u[pred_col] >= thr
    tp = int((flag & (u["HOTSPOT"] == 1)).sum())
    prec, rec = tp / max(flag.sum(), 1), tp / max((u["HOTSPOT"] == 1).sum(), 1)
    out.update(hot_auc=roc_auc_score(u["HOTSPOT"], u[pred_col]), hot_ap=average_precision_score(u["HOTSPOT"], u[pred_col]),
               hot_prec=prec, hot_rec=rec, hot_f1=2 * prec * rec / max(prec + rec, 1e-9))
    # hotspot AREAS: smooth the prediction over neighbours and keep connected clusters only
    fs = hotspot_flag(g, pred_col)[u.index]
    tps = int((fs & (u["HOTSPOT"] == 1)).sum())
    ps, rs = tps / max(fs.sum(), 1), tps / max((u["HOTSPOT"] == 1).sum(), 1)
    out.update(hot_f1_area=2 * ps * rs / max(ps + rs, 1e-9))
    return out


def city_metrics(df, pred_col):
    rows = []
    for c, g in df.groupby("city"):
        rows.append({"city": c, "rows": len(g), "rmse": float(np.sqrt(((g["TARGET"] - g[pred_col]) ** 2).mean())),
                     "r2": r2_score(g["TARGET"], g[pred_col]), "spearman": spearmanr(g["TARGET"], g[pred_col])[0],
                     # R2 after the best per-city linear rescale of the prediction: ranking quality without scale / offset errors
                     "r2_rescaled": float(np.corrcoef(g["TARGET"], g[pred_col])[0, 1] ** 2),
                     **hotspot_scores(g, pred_col)})
    return pd.DataFrame(rows).set_index("city")


def maps(df, out_dir, label="held-out city"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, g in df.groupby("city"):
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)
        lim = float(np.nanpercentile(np.abs(g["TARGET"]), 99))
        for a, col, title in ((ax[0], "TARGET", "observed heat anomaly"), (ax[1], "pred_lgbm", f"predicted ({label})")):
            s = a.scatter(g["x"], g["y"], c=g[col], s=1.2, marker="s", cmap="RdYlBu_r", vmin=-lim, vmax=lim, linewidths=0)
            a.set_title(f"{c}: {title}")
            plt.colorbar(s, ax=a, shrink=0.8)
        u = g[g["IS_URBAN"] == 1]
        thr = u["pred_lgbm"].quantile(0.9)
        ax[2].scatter(g["x"], g["y"], c="#dddddd", s=1.2, marker="s", linewidths=0)
        hit = u[(u["pred_lgbm"] >= thr) & (u["HOTSPOT"] == 1)]
        miss = u[(u["pred_lgbm"] >= thr) & (u["HOTSPOT"] != 1)]
        lost = u[(u["pred_lgbm"] < thr) & (u["HOTSPOT"] == 1)]
        for d, col, lab in ((hit, "#d7191c", f"hotspot found ({len(hit):,})"), (miss, "#fdae61", f"false alarm ({len(miss):,})"),
                            (lost, "#2c7bb6", f"hotspot missed ({len(lost):,})")):
            ax[2].scatter(d["x"], d["y"], c=col, s=2, marker="s", linewidths=0, label=lab)
        ax[2].legend(loc="lower left", fontsize=8, markerscale=4)
        ax[2].set_title(f"{c}: predicted top-10 % urban vs true hotspots")
        for a in ax:
            a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
        fig.savefig(out_dir / f"{c}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("maps ->", out_dir)


LEAKY = {"LST", "LST_z", "SUHI", "HOTSPOT", "LST_std", "n_obs", "IS_URBAN", "LST_anom", "TARGET", "pred_lgbm", "pred_ridge",
         "x", "y", "lon", "lat", "city", "climate", "block_id"}


def avail_gb():
    """Free RAM in GB (Linux /proc/meminfo, or psutil); None if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    except Exception:  # noqa: BLE001
        pass
    return None


def data_report(df, feats, args):
    """Print what the model will train on, so it can be verified once (--check-only) before a long run."""
    num_f = [f for f in feats if f != "LANDCOVER"]
    X = df[num_f]
    bar = "=" * 78
    print(bar)
    print("DATA CHECK: what the model will be trained on")
    print(bar)
    tgt = "(LST - city mean) / city std" if args.target == "z" else "LST - city mean (C)"
    print(f"rows {len(df):,} | cities {df['city'].nunique()} | features {len(feats)} | target = {tgt}")
    print("features: " + ", ".join(feats))
    leak = sorted(set(feats) & LEAKY)
    print("leakage guard (no target-derived / location columns among the features): " + ("FAIL " + str(leak) if leak else "ok"))
    if leak:
        raise SystemExit("remove the leaking features above")
    n_nan, n_inf = int(X.isna().sum().sum()), int(np.isinf(X.to_numpy(dtype="float64", na_value=0)).sum())
    const = [f for f in num_f if X[f].nunique() <= 1]
    print(f"NaN in features: {n_nan:,} | inf: {n_inf:,} | constant features: {const or 'none'}")
    print(f"target: NaN {int(df['TARGET'].isna().sum()):,} | overall mean {df['TARGET'].mean():+.3f} | std {df['TARGET'].std():.3f}")
    g = df.groupby("city")
    tbl = pd.DataFrame({"rows": g.size(), "target_mean": g["TARGET"].mean().round(3), "target_std": g["TARGET"].std().round(3),
                        "urban_%": (g["IS_URBAN"].mean() * 100).round(1), "hotspot_px": g["HOTSPOT"].sum().astype(int)})
    print("per city (target_mean should be ~0 everywhere" + (", target_std ~1 for --target z" if args.target == "z" else "") + "):")
    print(tbl.to_string())
    small = tbl.index[tbl["rows"] < 20_000].tolist()
    print("cities with fewer than 20k rows: " + (str(small) if small else "none"))
    if args.city_norm:
        cn = [f for f in feats if f.endswith("_cn")]
        worst = max(abs(g[f].mean()).max() for f in cn)
        print(f"city-relative features ({len(cn)}): largest per-city mean = {worst:.2e} (should be ~0)")
    est = len(df) * (len(feats) + 1) * 4 / 1e9
    print(f"memory: data about {est:.1f} GB as float32 per process; free RAM now {avail_gb() or float('nan'):.1f} GB")
    print(bar)


_W = {}      # per-process copy of the data used by the fold workers


def _init_worker(path, feats, num, params):
    _W.update(df=pd.read_parquet(path), feats=feats, num=num, params=params)


def _fit_fold(task):
    """Fit LightGBM + Ridge for one held-out fold. Runs in a worker process (or in the main one if workers == 1)."""
    name, test, fit_idx, val_idx, threads = task
    d, feats, num = _W["df"], _W["feats"], _W["num"]
    params = {**_W["params"], "n_jobs": threads}
    try:
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(limits=threads) if threads > 0 else contextlib.nullcontext()
    except Exception:  # noqa: BLE001
        limit = contextlib.nullcontext()
    with limit:
        m = lgb.LGBMRegressor(**params)
        m.fit(d.loc[fit_idx, feats], d.loc[fit_idx, "TARGET"], eval_set=[(d.loc[val_idx, feats], d.loc[val_idx, "TARGET"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p_l = m.predict(d.loc[test, feats])
        rdg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(d.loc[fit_idx, num], d.loc[fit_idx, "TARGET"])
        p_r = rdg.predict(d.loc[test, num])
    return name, test, p_l, p_r, m.best_iteration_, r2_score(d.loc[test, "TARGET"], p_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/multi_city/all_cities_clean.csv.gz"))
    ap.add_argument("--cv", choices=["loco", "block"], default="loco")
    ap.add_argument("--cv-sample", type=float, default=0.3, help="fraction of training rows used in each CV fold")
    ap.add_argument("--trees", type=int, default=800, help="max trees (early stopping picks fewer)")
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--no-met", action="store_true", help="drop the city-level weather features")
    ap.add_argument("--drop-feats", nargs="*", default=[], help="features to leave out, e.g. ELEV LANDCOVER (city-specific offsets do not transfer)")
    ap.add_argument("--check-only", action="store_true", help="prepare the data, print the DATA CHECK report and stop (about a minute)")
    ap.add_argument("--simple", action="store_true", help="shallower, more heavily regularised trees (better transfer to unseen cities)")
    ap.add_argument("--keep-water", action="store_true")
    ap.add_argument("--target", choices=["anom", "z"], default="anom",
                    help="anom: LST - city mean (C). z: that divided by the city's LST std (same scale for every city)")
    ap.add_argument("--city-norm", action="store_true", help="add features minus their city mean (*_cn)")
    ap.add_argument("--context", action="store_true", help="use multi-scale surroundings features (_c90.._c2010, _sd*) from the *_ctx table")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--workers", type=int, default=0,
                    help="CV folds trained in parallel (default 0 = auto: min(#folds, #cores) for big tables). Each gets cores/workers threads")
    ap.add_argument("--tag", default="lgbm_lst_anom")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)      # show progress live under nohup / redirected logs
    ART.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    df, feats = load(args.data, args.keep_water, not args.no_met, args.target, args.city_norm, args.context)
    feats = [f for f in feats if f not in args.drop_feats]
    for c in feats + ["TARGET"]:                       # float32 halves memory (matters with 50 cities x parallel folds)
        if c != "LANDCOVER" and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    data_report(df, feats, args)
    if args.check_only:
        print("--check-only: stopping before training.")
        return
    if args.cv == "loco" and df["city"].nunique() < 3:
        print("fewer than 3 cities -> using spatial-block CV")
        args.cv = "block"
    params = lgb_params(args, feats)
    rng = np.random.default_rng(0)
    df["pred_lgbm"] = np.nan
    df["pred_ridge"] = np.nan
    best_iters = []
    num = [f for f in feats if f != "LANDCOVER"]
    t0 = time.time()

    cores = os.cpu_count() or 1
    fold_list = list(folds(df, args.cv))
    workers = args.workers or (min(len(fold_list), cores) if len(df) > 200_000 else 1)
    workers = max(1, min(workers, len(fold_list), cores))
    free = avail_gb()
    per_worker = len(df) * (len(feats) + 1) * 4 * 2.5 / 1e9          # data + training copies, rough
    if free and workers > 1:
        by_ram = max(1, int(free * 0.75 / per_worker))
        if by_ram < workers:
            print(f"RAM guard: free {free:.0f} GB, about {per_worker:.1f} GB per worker -> {by_ram} fold(s) at a time instead of {workers}")
            workers = by_ram
    threads = max(1, cores // workers) if workers > 1 else -1
    print(f"{cores} cores -> {workers} fold(s) in parallel x {'all' if threads < 0 else threads} thread(s) each")
    print(f"training {len(fold_list)} folds; the first result appears after one fold finishes (a few minutes). Heartbeat every 60 s.")

    tasks = []                                         # sampling is done up front, in one fixed order -> same result for any #workers
    for name, test, unit in fold_list:
        tr_idx = np.flatnonzero(~test)
        units = np.unique(unit[tr_idx])
        val_units = rng.choice(units, max(1, round(0.1 * len(units))), replace=False)
        is_val = np.isin(unit[tr_idx], val_units)
        fit_idx, val_idx = tr_idx[~is_val], tr_idx[is_val]
        fit_idx = rng.choice(fit_idx, int(len(fit_idx) * args.cv_sample), replace=False)
        val_idx = rng.choice(val_idx, min(len(val_idx), 200_000), replace=False)
        tasks.append((name, np.flatnonzero(test), fit_idx, val_idx, threads))

    def collect(res):
        name, test, p_l, p_r, best_it, r2 = res
        df.loc[test, "pred_lgbm"] = p_l
        df.loc[test, "pred_ridge"] = p_r
        best_iters.append(best_it)
        k, el = len(best_iters), time.time() - t0
        eta = el / k * (len(tasks) - k)
        print(f"  held out {name:12s} trees {best_it:4d}  R2 {r2:5.2f}  [{k}/{len(tasks)} done, {el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left]")

    if workers == 1:
        _W.update(df=df, feats=feats, num=num, params=params)
        for t in tasks:
            collect(_fit_fold(t))
    else:
        tmp = ART / f".tmp_{args.tag}.parquet"
        df[feats + ["TARGET"]].to_parquet(tmp)         # workers read this once instead of receiving a pickled copy per task
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                     initargs=(str(tmp), feats, num, params)) as ex:
                pending = {ex.submit(_fit_fold, t) for t in tasks}
                while pending:
                    done, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                    for f in done:
                        collect(f.result())
                    if not done:
                        print(f"  ... {len(best_iters)}/{len(tasks)} folds done, {len(pending)} running or queued, "
                              f"{(time.time() - t0) / 60:.1f} min elapsed")
        finally:
            tmp.unlink(missing_ok=True)
    best_iters.sort()                                  # completion order varies; the mean does not

    df["pred_blend"] = (df["pred_lgbm"] + df["pred_ridge"]) / 2          # simple average of the two models
    lg, rd, bl = city_metrics(df, "pred_lgbm"), city_metrics(df, "pred_ridge"), city_metrics(df, "pred_blend")
    summary = lg.join(rd[["rmse", "r2", "hot_auc", "hot_f1"]].add_prefix("ridge_")).join(bl[["r2", "hot_auc", "hot_f1"]].add_prefix("blend_"))
    pd.set_option("display.width", 220)
    print("\nPer held-out city (LightGBM vs Ridge baseline):")
    print(summary.round(3).to_string())
    print("\nMEAN over cities:  " + "  ".join(f"{c}={summary[c].mean():.3f}" for c in
                                            ["rmse", "r2", "r2_rescaled", "spearman", "hot_auc", "hot_f1", "hot_f1_area", "ridge_r2", "ridge_hot_auc", "ridge_hot_f1", "blend_r2", "blend_hot_auc", "blend_hot_f1"]))
    summary.round(4).to_csv(ART / f"{args.tag}_cv_metrics.csv")

    # ---- final model on everything
    n_trees = max(50, int(np.mean(best_iters) * 1.1))
    final = lgb.LGBMRegressor(**{**params, "n_estimators": n_trees}).fit(df[feats], df["TARGET"])
    final.booster_.save_model(str(ART / f"{args.tag}.txt"))
    import joblib      # the Ridge half of the LightGBM + Ridge blend, saved so predict_city.py can use both
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(df[num], df["TARGET"])
    joblib.dump(ridge, ART / f"{args.tag}_ridge.joblib")
    rng_q = df[num].quantile([0.01, 0.99])             # what the model has seen, for the new-city trust check
    (ART / f"{args.tag}_ranges.json").write_text(json.dumps(
        {"ridge_features": num, "lo": rng_q.loc[0.01].to_dict(), "hi": rng_q.loc[0.99].to_dict()}, indent=2))
    imp = pd.Series(final.booster_.feature_importance("gain"), feats).sort_values(ascending=False)
    (imp / imp.sum()).round(4).to_csv(ART / f"{args.tag}_importance.csv", header=["gain_share"])
    (ART / f"{args.tag}_config.json").write_text(json.dumps({
        "features": feats, "target": "LST - city mean LST (C)" if args.target == "anom" else "(LST - city mean) / city std", "city_norm": args.city_norm, "context": args.context, "n_trees": n_trees, "cv": args.cv,
        "monotone_down": sorted(MONO_DOWN) if not args.no_monotone else [], "dropped_water": not args.keep_water,
        "landcover_classes": list(df["LANDCOVER"].cat.categories), "params": {k: v for k, v in params.items() if k != "monotone_constraints"}},
        indent=2, default=str))
    print(f"\nfinal model: {n_trees} trees -> {ART / (args.tag + '.txt')}")
    print("top drivers (share of gain):\n" + (imp / imp.sum()).round(3).head(10).to_string())

    # ---- out-of-fold predictions (honest: every city predicted by a model that never saw it) + maps
    cols = ["city", "x", "y", "lon", "lat", "LST", "LST_anom", "TARGET", "pred_lgbm", "IS_URBAN", "HOTSPOT", "block_id"]
    df[cols].round(4).to_csv(OUT / f"{args.tag}_oof_predictions.csv.gz", index=False)
    if not args.no_maps:
        maps(df, OUT / f"{args.tag}_maps")


if __name__ == "__main__":
    main()
