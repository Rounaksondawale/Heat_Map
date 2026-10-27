"""
Prepare U-Net training data: one normalised image stack + hotspot mask per city, from the full 30 m grids.

    python models/unet/prepare_grids.py --root data/multi_city --out data/unet
    python models/unet/prepare_grids.py --root data/multi_city --out data/unet --val Pune Bhopal --test Delhi Kochi Jaipur

For every city (cities/<City>/aligned.nc) this writes data/unet/<City>/
    X.npy         float16 [C, H, W]  input channels (see channels.json), z-scored WITHIN the city, missing = 0
    y.npy         uint8   [H, W]     hotspot mask = Getis-Ord Gi* patches of the observed satellite temperature
    aux.npy       float16 [H, W]     observed temperature anomaly in the city's own std units (optional second target)
    loss_mask.npy bool    [H, W]     pixels the loss may use: inputs present, not water, temperature observed
    meta.json     city, shape, pixel size, hotspot share, ...
and data/unet/channels.json, data/unet/splits.json (train / val / test cities).

Inputs are the RAW layers; the network learns the surroundings itself, so no hand-made window averages are used.
Land cover becomes one channel per class. Continuous channels are clipped at the city's 0.5 / 99.5 percentiles and
z-scored per city, so a model can work across climates. Patches are cut at training time (random crops + flips).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hotspots import hotspot_grid          # noqa: E402

CONT = ["NDVI", "NDBI", "MNDWI", "ALBEDO", "SLOPE_deg", "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP",
        "DIST_WATER_m", "DIST_GREEN_m", "ELEV"]
LOG = {"GHSL_POP", "DIST_WATER_m", "DIST_GREEN_m"}             # long-tailed: log1p first
LC_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90, 95]               # ESA WorldCover: tree, shrub, grass, crop, built, bare, water, wetland, mangrove
CHANNELS = CONT + [f"LC_{c}" for c in LC_CLASSES] + ["VALID"]
BASE_CHECK = CONT + ["LANDCOVER"]


def zscore(a, valid):
    v = a[valid]
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    mu, sd = v.mean(), v.std()
    out = np.zeros_like(a, dtype="float32")
    out[valid] = (np.clip(a[valid], lo, hi) - mu) / max(sd, 1e-6)
    return out


def prepare_city(root, city, out, hot):
    ds = xr.open_dataset(Path(root) / "cities" / city / "aligned.nc")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    res = abs(float(ds.x[1] - ds.x[0]))
    lc = ds["LANDCOVER"].values
    inputs_ok = np.ones((ny, nx), bool)
    for b in BASE_CHECK:
        inputs_ok &= np.isfinite(ds[b].values)
    usable = inputs_ok & (lc != 80)                              # water is not scored (as in the tabular model)
    X = np.zeros((len(CHANNELS), ny, nx), "float32")
    for i, b in enumerate(CONT):
        a = ds[b].values.astype("float64")
        if b in LOG:
            a = np.log1p(np.clip(a, 0, None))
        ok = np.isfinite(a) & usable
        if ok.sum() > 100:
            X[i] = zscore(a, ok)
    for j, c in enumerate(LC_CLASSES):
        X[len(CONT) + j] = (lc == c).astype("float32")
    X[-1] = usable.astype("float32")
    X[:, ~usable] = 0
    X[-1] = usable.astype("float32")

    lst = ds["LST"].values
    loss_mask = usable & np.isfinite(lst)
    ids, _, _ = hotspot_grid(lst, loss_mask, res, hot["win"], hot["z"], hot["min_excess"], hot["min_ha"])
    y = (ids > 0).astype("uint8")
    aux = np.zeros((ny, nx), "float32")
    mu, sd = lst[loss_mask].mean(), lst[loss_mask].std()
    aux[loss_mask] = (lst[loss_mask] - mu) / max(sd, 1e-6)

    d = Path(out) / city
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", X.astype("float16"))
    np.save(d / "y.npy", y)
    np.save(d / "aux.npy", aux.astype("float16"))
    np.save(d / "loss_mask.npy", loss_mask)
    meta = {"city": city, "shape": [ny, nx], "res_m": res, "usable_share": float(usable.mean()),
            "hotspot_patches": int(ids.max()), "hotspot_share_of_valid": float(y[loss_mask].mean()),
            "lst_mean": float(mu), "lst_std": float(sd)}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    ds.close()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multi_city")
    ap.add_argument("--out", default="data/unet")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--val", nargs="*", default=["Pune", "Bhopal", "Kolkata", "Udaipur"], help="validation cities (never trained on)")
    ap.add_argument("--test", nargs="*", default=["Delhi", "Jaipur", "Kochi", "Lucknow", "Nashik", "Guwahati"],
                    help="test cities: touched only for the final score")
    ap.add_argument("--win", type=int, default=15)
    ap.add_argument("--z", type=float, default=2.58)
    ap.add_argument("--min-excess", type=float, default=1.0)
    ap.add_argument("--min-ha", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    avail = sorted(p.parent.name for p in (root / "cities").glob("*/aligned.nc"))
    cities = a.cities or avail
    missing = [c for c in cities if c not in avail]
    if missing:
        raise SystemExit(f"no aligned.nc for {missing}")
    hot = {"win": a.win, "z": a.z, "min_excess": a.min_excess, "min_ha": a.min_ha}
    # about 1.8 M pixels x 22 channels x 4 B = 160 MB per city while it is processed
    workers = max(1, min(a.workers or (os.cpu_count() or 1), len(cities)))
    print(f"{len(cities)} cities on {workers} process(es); channels: {len(CHANNELS)}")
    metas = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(prepare_city, str(root), c, str(out), hot): c for c in cities}
        for f in as_completed(futs):
            m = f.result()
            metas[m["city"]] = m
            print(f"  {m['city']:20s} {m['shape'][0]} x {m['shape'][1]}  usable {m['usable_share']:.0%}  "
                  f"hotspot patches {m['hotspot_patches']:3d}  hotspot share {m['hotspot_share_of_valid']:.1%}")
    (out / "channels.json").write_text(json.dumps(CHANNELS, indent=2))
    val = [c for c in a.val if c in cities]
    test = [c for c in a.test if c in cities and c not in val]
    train = [c for c in cities if c not in val and c not in test]
    (out / "splits.json").write_text(json.dumps({"train": train, "val": val, "test": test}, indent=2))
    sh = np.array([m["hotspot_share_of_valid"] for m in metas.values()])
    print(f"\nsplit: {len(train)} train, {len(val)} val {val}, {len(test)} test {test}")
    print(f"hotspot share of valid land per city: median {np.median(sh):.1%}, min {sh.min():.1%}, max {sh.max():.1%}")
    print(f"cities with no hotspot at all: {[c for c, m in metas.items() if m['hotspot_patches'] == 0] or 'none'}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
