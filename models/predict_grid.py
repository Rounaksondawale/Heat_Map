"""
Predict heat and find hotspots on the COMPLETE 30 m grid of a city (every pixel, not a sampled table).

    python models/predict_grid.py --city Pune --tag combined_v2
    python models/predict_grid.py --city Delhi --tag combined_v2 --z 1.96 --min-excess 0.7 --min-ha 3

1. Builds the model's input features for every usable pixel of cities/<City>/aligned.nc (same features as training,
   including the surroundings windows, computed on the full grid).
2. Predicts relative heat with the saved combined model (average of LightGBM and Ridge).
3. Finds hotspots with the Getis-Ord Gi* statistic on the whole map: patches where heat is concentrated compared
   with the rest of the area. A pixel is hot if Gi* >= --z (2.58 = 99 % confidence) AND its neighbourhood is at least
   --min-excess standard deviations above the area mean; patches smaller than --min-ha hectares are dropped. The
   share of hot land is NOT fixed: an even area has none.
4. If the grid also holds the satellite temperature, finds the observed hotspots the same way and scores the match.

Outputs (models/outputs/grid/<City>_<tag>.*):  .nc (heat, Gi* z, hotspot ids), .png maps, _hotspot_areas.csv
(one row per predicted hotspot: hectares, mean heat, centre lon/lat).
For a city the model was trained on, the comparison is optimistic; use it to inspect, not as the accuracy figure.
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
import xarray as xr
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extraction"))
import train_lgbm as tl                                   # noqa: E402
from add_context_features import context_grids           # noqa: E402
from hotspots import grid_area_scores, hotspot_grid      # noqa: E402


def build_features(ds, cfg, usable):
    ctx = context_grids(ds) if any("_c" in f or "_sd" in f for f in cfg["features"]) else {}
    cols = {}
    for f in cfg["features"]:
        if f.endswith("_cn"):                              # relative to THIS area's mean over usable pixels
            a = ds[f[:-3]].values.astype("float64")
            arr = a - np.nanmean(a[usable])
        elif f in ctx:
            arr = ctx[f].values
        elif f in ds:
            arr = ds[f].values
        else:
            raise SystemExit(f"cannot build feature {f} from the grid (weather features are not supported here)")
        cols[f] = arr[usable].astype("float32")
    X = pd.DataFrame(cols)
    if "LANDCOVER" in X:
        X["LANDCOVER"] = pd.Categorical(X["LANDCOVER"].astype(int), categories=cfg["landcover_classes"])
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--tag", default="combined_v2")
    ap.add_argument("--root", default=str(HERE.parent / "data/multi_city"))
    ap.add_argument("--out", default=str(tl.OUT / "grid"))
    ap.add_argument("--win", type=int, default=15, help="Gi* window in pixels (15 = 450 m)")
    ap.add_argument("--z", type=float, default=2.58, help="Gi* threshold (1.96 = 95 %, 2.58 = 99 %)")
    ap.add_argument("--min-excess", type=float, default=1.0, help="neighbourhood must be this many std above the area mean")
    ap.add_argument("--min-ha", type=float, default=5.0)
    ap.add_argument("--urban-only", action="store_true", help="only count hotspots on urban pixels")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((tl.ART / f"{a.tag}_config.json").read_text())
    rng = json.loads((tl.ART / f"{a.tag}_ranges.json").read_text())
    booster = lgb.Booster(model_file=str(tl.ART / f"{a.tag}.txt"))
    ridge = joblib.load(tl.ART / f"{a.tag}_ridge.joblib")

    ds = xr.open_dataset(Path(a.root) / "cities" / a.city / "aligned.nc")
    res = abs(float(ds.x[1] - ds.x[0]))
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    print(f"{a.city}: grid {ny} x {nx} = {ny * nx:,} pixels at {res:g} m")

    # pixels the model can score: all base inputs present and (as in training) not water
    usable = np.ones((ny, nx), bool)
    for b in tl.BASE:
        usable &= np.isfinite(ds[b].values)
    if cfg.get("dropped_water", True):
        usable &= ds["LANDCOVER"].values != 80
    print(f"usable (inputs present, not water): {usable.sum():,} pixels ({usable.mean():.0%})")

    X = build_features(ds, cfg, usable)
    heat = np.full((ny, nx), np.nan)
    heat[usable] = (booster.predict(X) + ridge.predict(X[rng["ridge_features"]])) / 2

    urban = ds["IS_URBAN"].values == 1 if "IS_URBAN" in ds else None
    allow = urban if (a.urban_only and urban is not None) else None
    pred_ids, gi_z, excess = hotspot_grid(heat, usable, res, a.win, a.z, a.min_excess, a.min_ha, allow)
    hot = pred_ids > 0
    print(f"\nPREDICTED hotspot areas: {pred_ids.max()} patches, {hot.sum() * res ** 2 / 1e4:,.0f} ha "
          f"= {hot.sum() / usable.sum():.1%} of usable land")
    if urban is not None and hot.any():
        print(f"  share of hotspot land that is urban: {urban[hot].mean():.0%} (urban share of all land: {urban[usable].mean():.0%})")

    # ---- observed hotspots from the satellite temperature, same method
    lst = ds["LST"].values if "LST" in ds else None
    obs_ids = None
    if lst is not None:
        valid_obs = usable & np.isfinite(lst)
        obs_ids, _, _ = hotspot_grid(lst, valid_obs, res, a.win, a.z, a.min_excess, a.min_ha, allow)
        idx = np.flatnonzero(valid_obs.ravel())
        idx = np.random.default_rng(0).choice(idx, min(len(idx), 500_000), replace=False)
        rho = spearmanr(heat.ravel()[idx], lst.ravel()[idx])[0]
        sc = grid_area_scores(np.where(valid_obs, obs_ids, 0), np.where(valid_obs, pred_ids, 0))
        print(f"\nOBSERVED hotspot areas (satellite LST): {obs_ids.max()} patches, {(obs_ids > 0).sum() * res ** 2 / 1e4:,.0f} ha")
        print(f"pixel ranking, predicted heat vs observed LST (Spearman, all {valid_obs.sum():,} pixels): {rho:.3f}")
        print(f"predicted vs observed hotspot areas: overlap F1 {sc['area_f1']:.2f}, IoU {sc['area_iou']:.2f}, "
              f"observed patches found {sc['cluster_recall']:.0%}, predicted patches real {sc['cluster_precision']:.0%}")

    # ---- patch table
    yy, xx = np.nonzero(hot)
    t = pd.DataFrame({"cluster_id": pred_ids[yy, xx], "x": ds.x.values[xx], "y": ds.y.values[yy],
                      "heat": heat[yy, xx], "gi_z": gi_z[yy, xx]})
    if len(t):
        from pyproj import Transformer
        tr = Transformer.from_crs(f"EPSG:{int(ds.attrs['epsg'])}", "EPSG:4326", always_xy=True)
        g = t.groupby("cluster_id").agg(pixels=("x", "size"), mean_heat=("heat", "mean"), max_heat=("heat", "max"),
                                        mean_gi_z=("gi_z", "mean"), x=("x", "mean"), y=("y", "mean")).reset_index()
        g["area_ha"] = g["pixels"] * res ** 2 / 1e4
        g["lon"], g["lat"] = tr.transform(g["x"].values, g["y"].values)
        g = g.sort_values("mean_heat", ascending=False)[["cluster_id", "pixels", "area_ha", "mean_heat", "max_heat", "mean_gi_z", "lon", "lat"]]
    else:
        g = pd.DataFrame(columns=["cluster_id", "pixels", "area_ha", "mean_heat", "max_heat", "mean_gi_z", "lon", "lat"])
    print("\nTOP HOTSPOT AREAS")
    print(g.head(10).round(3).to_string(index=False))

    stem = out / f"{a.city}_{a.tag}"
    g.round(4).to_csv(str(stem) + "_hotspot_areas.csv", index=False)
    d = xr.Dataset({"heat": (("y", "x"), heat.astype("float32")), "gi_z": (("y", "x"), gi_z.astype("float32")),
                    "hotspot_id": (("y", "x"), pred_ids.astype("int32"))}, coords={"y": ds.y, "x": ds.x})
    if obs_ids is not None:
        d["observed_hotspot_id"] = (("y", "x"), obs_ids.astype("int32"))
    d.to_netcdf(str(stem) + ".nc")

    # ---- maps
    ext = [float(ds.x.min()) - res / 2, float(ds.x.max()) + res / 2, float(ds.y.min()) - res / 2, float(ds.y.max()) + res / 2]
    origin = "upper" if float(ds.y[0]) > float(ds.y[-1]) else "lower"
    panels = 3 if obs_ids is not None else 2
    fig, ax = plt.subplots(1, panels, figsize=(6 * panels, 6), sharex=True, sharey=True)
    k = 0
    if obs_ids is not None:
        anom = lst - np.nanmean(lst[valid_obs])
        lim = np.nanpercentile(np.abs(anom), 99)
        im = ax[k].imshow(anom, cmap="RdYlBu_r", vmin=-lim, vmax=lim, extent=ext, origin=origin)
        ax[k].contour(np.where(obs_ids > 0, 1, 0), levels=[0.5], colors="k", linewidths=0.8, extent=ext, origin=origin)
        ax[k].set_title(f"{a.city}: observed temperature anomaly (C)\nblack = observed hotspot areas")
        plt.colorbar(im, ax=ax[k], shrink=0.7)
        k += 1
    lim = np.nanpercentile(np.abs(heat), 99)
    im = ax[k].imshow(heat, cmap="RdYlBu_r", vmin=-lim, vmax=lim, extent=ext, origin=origin)
    ax[k].contour(np.where(hot, 1, 0), levels=[0.5], colors="k", linewidths=0.8, extent=ext, origin=origin)
    ax[k].set_title(f"{a.city}: predicted relative heat\nblack = predicted hotspot areas ({pred_ids.max()})")
    plt.colorbar(im, ax=ax[k], shrink=0.7)
    k += 1
    if obs_ids is not None:
        agree = np.zeros((ny, nx))
        agree[(obs_ids > 0) & hot] = 3
        agree[(obs_ids == 0) & hot] = 2
        agree[(obs_ids > 0) & ~hot] = 1
        cm = matplotlib.colors.ListedColormap(["#eeeeee", "#2c7bb6", "#fdae61", "#d7191c"])
        ax[k].imshow(agree, cmap=cm, vmin=0, vmax=3, extent=ext, origin=origin, interpolation="nearest")
        ax[k].set_title("agreement: red = both, orange = predicted only, blue = missed")
    for x_ in ax:
        x_.set_xticks([])
        x_.set_yticks([])
    fig.savefig(str(stem) + ".png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {stem}.nc, {stem}_hotspot_areas.csv, {stem}.png")


if __name__ == "__main__":
    main()
