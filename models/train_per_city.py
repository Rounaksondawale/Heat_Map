"""
Model 02: one LightGBM per city, validated with spatial blocks inside that city.

    python models/train_per_city.py                        # all cities
    python models/train_per_city.py --cities Pune Delhi    # a subset

Why: the goal is to find hotspots in a city and test cooling there, not to predict a city the model never saw.
A model trained on a city's own pixels learns that city's relationship between greenery, built-up area, albedo
and heat. It is validated on 3 km spatial blocks (GroupKFold on block_id) that the model did not train on, so the
score is honest about unseen *places*, not unseen cities. Compare with models/train_lgbm.py (leave-one-city-out).

Outputs  models/artifacts/per_city/<City>.txt        final model per city (use these for scenarios)
         models/artifacts/per_city_cv_metrics.csv    block-CV metrics per city
         models/outputs/per_city_maps/<City>.png     observed vs predicted vs hotspots (out-of-block predictions)
         models/outputs/per_city_oof_predictions.csv.gz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import train_lgbm as tl            # noqa: E402  (shares loading, features, metrics and maps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(tl.ROOT / "data/multi_city/all_cities_clean.csv.gz"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--trees", type=int, default=1500)
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--keep-water", action="store_true")
    ap.add_argument("--context", action="store_true", help="use multi-scale surroundings features from the *_ctx table")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    (tl.ART / "per_city").mkdir(parents=True, exist_ok=True)
    tl.OUT.mkdir(exist_ok=True)

    # weather is a single value per city -> useless inside one city, so it is left out
    df, feats = tl.load(args.data, args.keep_water, use_met=False, target="anom", city_norm=False, context=args.context)
    args.trees_cap = args.trees
    params = tl.lgb_params(argparse.Namespace(trees=args.trees, no_monotone=args.no_monotone), feats)
    cities = args.cities or sorted(df["city"].unique())
    df["pred_lgbm"] = np.nan
    t0 = time.time()
    config = {}

    for c in cities:
        idx = np.flatnonzero(df["city"] == c)
        g = df.iloc[idx]
        n_blocks = g["block_id"].nunique()
        iters = []
        for tr, te in GroupKFold(min(5, n_blocks)).split(g, groups=g["block_id"]):
            # early-stopping set = a few whole training blocks, never the test blocks
            tr_blocks = g["block_id"].iloc[tr].unique()
            rng = np.random.default_rng(0)
            vb = rng.choice(tr_blocks, max(1, round(0.15 * len(tr_blocks))), replace=False)
            is_val = g["block_id"].iloc[tr].isin(vb).to_numpy()
            fit, val = tr[~is_val], tr[is_val]
            m = lgb.LGBMRegressor(**params)
            m.fit(g.iloc[fit][feats], g.iloc[fit]["TARGET"], eval_set=[(g.iloc[val][feats], g.iloc[val]["TARGET"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            df.iloc[idx[te], df.columns.get_loc("pred_lgbm")] = m.predict(g.iloc[te][feats])
            iters.append(m.best_iteration_)
        n_trees = max(50, int(np.mean(iters) * 1.1))
        final = lgb.LGBMRegressor(**{**params, "n_estimators": n_trees}).fit(g[feats], g["TARGET"])
        final.booster_.save_model(str(tl.ART / "per_city" / f"{c}.txt"))
        config[c] = {"n_trees": n_trees, "city_mean_LST": float(g["LST"].mean())}
        r2 = 1 - ((g["TARGET"] - df.iloc[idx]["pred_lgbm"]) ** 2).sum() / ((g["TARGET"] - g["TARGET"].mean()) ** 2).sum()
        print(f"  {c:12s} blocks {n_blocks:3d}  trees {n_trees:4d}  block-CV R2 {r2:5.2f}  ({time.time() - t0:4.0f}s)")

    d = df[df["city"].isin(cities)]
    met = tl.city_metrics(d, "pred_lgbm")
    pd.set_option("display.width", 200)
    print("\nPer-city models, scored on held-out spatial blocks:")
    print(met.round(3).to_string())
    print("\nMEAN over cities:  " + "  ".join(f"{k}={met[k].mean():.3f}" for k in ["rmse", "r2", "r2_rescaled", "spearman", "hot_auc", "hot_f1", "hot_f1_area"]))
    print("(leave-one-city-out baseline was:  rmse=2.002  r2=0.396  spearman=0.769  hot_auc=0.821  hot_f1=0.400)")
    met.round(4).to_csv(tl.ART / "per_city_cv_metrics.csv")
    (tl.ART / "per_city" / "config.json").write_text(json.dumps({"features": feats, "cities": config}, indent=2))
    d[["city", "x", "y", "lon", "lat", "LST", "LST_anom", "pred_lgbm", "IS_URBAN", "HOTSPOT", "block_id"]].round(4) \
        .to_csv(tl.OUT / "per_city_oof_predictions.csv.gz", index=False)
    if not args.no_maps:
        tl.maps(d, tl.OUT / "per_city_maps", label="held-out 3 km blocks")


if __name__ == "__main__":
    main()
