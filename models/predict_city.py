"""
Predict hotspots for ANY city with the saved global model (average of LightGBM and Ridge). No training.

    python models/predict_city.py --city Pune --tag global_simple
    python models/predict_city.py --table data/multi_city/all_cities_clean_ctx.csv.gz --city Delhi --tag global_simple

The table must hold that city's pixels with the same input columns as the training table (built by
extraction/build_dataset.py, finalize_dataset.py and add_context_features.py). If it also has LST / HOTSPOT, the
prediction is checked against them; for a truly new city those columns are not needed.

Output (models/outputs/predictions/<City>_<tag>...)
    .csv.gz   x, y, lon, lat, heat_score (relative heat, higher = hotter), hotspot, hotspot_area
    .png      predicted heat map + hotspot map
Prints a trust report: share of pixels whose inputs fall outside what the model saw in training.

heat_score is a RELATIVE ranking inside the city (in units of the city's own temperature spread), not degrees C.
hotspot = top 10 % of urban pixels by heat_score; hotspot_area = the same after smoothing and keeping clusters.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import train_lgbm as tl                      # noqa: E402  (BASE / NORM feature lists)
from hotspots import area_scores, cluster_table, hotspot_areas  # noqa: E402


def prepare(df, cfg):
    """Rebuild exactly the features the model was trained on."""
    n0 = len(df)
    if cfg.get("dropped_water", True):
        df = df[df["LANDCOVER"] != 80]
    df = df.dropna(subset=tl.BASE).copy()
    print(f"{n0:,} pixels -> {len(df):,} usable (water / missing inputs dropped, as in training)")
    feats = cfg["features"]
    if any(f.startswith("MET_") or f == "ABSORBED_SW" for f in feats):
        df["ABSORBED_SW"] = (1 - df["ALBEDO"]) * df["MET_SW"]
    for f in tl.NORM:                                  # city-relative features use THIS city's own means
        df[f + "_cn"] = df[f] - df[f].mean()
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"table lacks model inputs {missing}; run add_context_features.py for this city")
    df["LANDCOVER"] = pd.Categorical(df["LANDCOVER"].astype(int), categories=cfg["landcover_classes"])
    return df, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--tag", default="global_simple")
    ap.add_argument("--table", default=str(tl.ROOT / "data/multi_city/all_cities_clean_ctx.csv.gz"))
    ap.add_argument("--out", default=str(tl.OUT / "predictions"))
    ap.add_argument("--top", type=float, default=0.10, help="share of urban pixels flagged as hotspots")
    ap.add_argument("--min-ha", type=float, default=5.0, help="smallest connected patch (hectares) that counts as a hotspot area")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((tl.ART / f"{a.tag}_config.json").read_text())
    rng = json.loads((tl.ART / f"{a.tag}_ranges.json").read_text())
    booster = lgb.Booster(model_file=str(tl.ART / f"{a.tag}.txt"))
    ridge = joblib.load(tl.ART / f"{a.tag}_ridge.joblib")

    df = tl.read_table(a.table)
    df = df[df["city"] == a.city]
    if df.empty:
        raise SystemExit(f"no rows for city {a.city} in {a.table}")
    df, feats = prepare(df, cfg)
    df["heat_lgbm"] = booster.predict(df[feats])
    df["heat_ridge"] = ridge.predict(df[rng["ridge_features"]])
    df["heat_score"] = (df["heat_lgbm"] + df["heat_ridge"]) / 2
    urban = df["IS_URBAN"] == 1
    df["hotspot"] = (df["heat_score"] >= df.loc[urban, "heat_score"].quantile(1 - a.top)) & urban
    df["cluster_id"] = hotspot_areas(df, "heat_score", q=1 - a.top, min_ha=a.min_ha)      # connected patches
    df["hotspot_area"] = df["cluster_id"] > 0
    clusters = cluster_table(df, df["cluster_id"], "heat_score")

    # ---------------- trust report
    print(f"\nTRUST REPORT for {a.city}  (model: {a.tag})")
    outside = {f: float(((df[f] < rng["lo"][f]) | (df[f] > rng["hi"][f])).mean()) for f in rng["ridge_features"]}
    worst = sorted(outside.items(), key=lambda kv: -kv[1])[:5]
    print("  share of pixels outside the 1st-99th percentile range seen in training (expected about 2 %):")
    for f, v in worst:
        print(f"    {f:22s} {v:6.1%}")
    flag = [f for f, v in outside.items() if v > 0.25]
    print("  WARNING: inputs far from the training cities for " + ", ".join(flag) + ". Treat the ranking with caution."
          if flag else "  inputs look similar to the training cities.")
    agree = spearmanr(df["heat_lgbm"], df["heat_ridge"])[0]
    print(f"  LightGBM vs Ridge ranking agreement (Spearman): {agree:.2f}  (higher = more reliable; below 0.6 = be careful)")
    if "HOTSPOT" in df and df.loc[urban, "HOTSPOT"].nunique() == 2:
        u = df[urban]
        tp = int((u["hotspot"] & (u["HOTSPOT"] == 1)).sum())
        p, r = tp / max(u["hotspot"].sum(), 1), tp / max((u["HOTSPOT"] == 1).sum(), 1)
        print(f"  check against observed satellite hotspots: AUC {roc_auc_score(u['HOTSPOT'], u['heat_score']):.2f}, "
              f"F1 {2 * p * r / max(p + r, 1e-9):.2f}  (for a city the model was trained on this is optimistic)")

    if "LST" in df:
        s = area_scores(df, "heat_score", "LST", q=1 - a.top, min_ha=a.min_ha)
        print(f"  hotspot AREAS vs observed: {s['pred_areas']} predicted / {s['true_areas']} observed patches; "
              f"pixel overlap F1 {s['area_f1']:.2f}, patches found {s['cluster_recall']:.0%}, predicted patches real {s['cluster_precision']:.0%}")

    # ---------------- outputs
    print(f"\nHOTSPOT AREAS (patches of >= {a.min_ha:g} ha): {len(clusters)}, total {clusters['area_ha'].sum():,.0f} ha")
    print(clusters.head(10).round(3).to_string(index=False))
    cols = ["x", "y", "lon", "lat", "heat_score", "hotspot", "hotspot_area", "cluster_id"]
    stem = out / f"{a.city}_{a.tag}"
    df[cols].round(4).to_csv(str(stem) + ".csv.gz", index=False)
    clusters.round(4).to_csv(str(stem) + "_hotspot_areas.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    lim = float(np.nanpercentile(np.abs(df["heat_score"]), 99))
    s = ax[0].scatter(df["x"], df["y"], c=df["heat_score"], s=1.5, marker="s", cmap="RdYlBu_r", vmin=-lim, vmax=lim, linewidths=0)
    plt.colorbar(s, ax=ax[0], shrink=0.7, label="relative heat (city's own spread)")
    ax[0].set_title(f"{a.city}: predicted relative heat")
    ax[1].scatter(df["x"], df["y"], c="#dddddd", s=1.5, marker="s", linewidths=0)
    h = df[df["hotspot_area"]]
    ax[1].scatter(h["x"], h["y"], c="#d7191c", s=2.5, marker="s", linewidths=0,
                  label=f"{len(clusters)} hotspot areas, {clusters['area_ha'].sum():,.0f} ha")
    ax[1].legend(loc="lower left", markerscale=4, fontsize=9)
    ax[1].set_title(f"{a.city}: hotspot areas (top {a.top:.0%} of urban land, patches >= {a.min_ha:g} ha)")
    for x_ in ax:
        x_.set_aspect("equal")
        x_.set_xticks([])
        x_.set_yticks([])
    fig.savefig(str(stem) + ".png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {stem}.csv.gz, {stem}_hotspot_areas.csv and {stem}.png")


if __name__ == "__main__":
    main()
