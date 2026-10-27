"""
Cut the city grids into 64 x 64 patches for a U-Net.

    python pre_processing/make_patches.py                                   # grids in pre_processing/grids
    python pre_processing/make_patches.py --val-cities Jaipur Chennai --workers 4

Reads <grids>/<City>_features.nc + <City>_temp.nc (export_city_grids.py) and writes, per city,
<out>/<split>/<City>.npz with
    X     (N, F, 64, 64) float16  numeric features, standardised with TRAINING-city mean/std; missing -> 0
    lc    (N, 64, 64)    uint8    LANDCOVER class index (categorical; codes in meta.json)
    y     (N, 64, 64)    float32  LST in degrees C (not normalised)
    mask  (N, 64, 64)    uint8    1 = usable pixel: LST and every feature present, not water. Compute the loss on mask == 1 only.
    pos   (N, 2)         int32    (row, col) of each patch's top-left pixel in the city grid
plus <out>/meta.json (feature names, mean/std, split, patch size) and <out>/patch_summary.csv.
Splits are by whole city. Training patches overlap by --overlap (default 15 %); validation patches do not overlap,
so validation scores are not inflated. Patches with less than --min-valid usable pixels are dropped.
"""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LC_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]      # ESA WorldCover classes; 80 = water
WATER = 80
VAL_DEFAULT = ["Jaipur", "Lucknow", "Nagpur", "Chennai", "Bengaluru", "Kolkata", "Bhubaneswar", "Dehradun",
               "Amritsar", "Coimbatore"]                        # spread over climates; the hardest cities stay in training


def load(grids, city):
    f = xr.open_dataset(grids / f"{city}_features.nc")
    names = [v for v in f.data_vars if v != "LANDCOVER"]
    X = np.stack([f[v].values.astype("float32") for v in names])
    lc = f["LANDCOVER"].values
    y = xr.open_dataset(grids / f"{city}_temp.nc")["LST"].values.astype("float32")
    f.close()
    return names, X, lc, y


def stats(args):
    grids, city = args
    names, X, lc, y = load(Path(grids), city)
    ok = np.isfinite(y) & (lc != WATER) & np.isfinite(X).all(0)
    S = X[:, ok][:, ::7].astype("float64")                      # every 7th usable pixel is plenty
    return names, S.sum(1), (S ** 2).sum(1), S.shape[1]


def starts(n, size, stride):
    s = list(range(0, n - size + 1, stride))
    if s[-1] != n - size:
        s.append(n - size)                                      # cover the far edge
    return s


def cut(args):
    grids, city, split, out, size, stride, min_valid, mean, std = args
    names, X, lc, y = load(Path(grids), city)
    mean, std = np.array(mean, "float32")[:, None, None], np.array(std, "float32")[:, None, None]
    ok = np.isfinite(y) & (lc != WATER) & np.isfinite(X).all(0)
    Z = np.where(np.isfinite(X), (X - mean) / std, 0).astype("float16")
    lci = np.full(lc.shape, 255, "uint8")
    for i, c in enumerate(LC_CODES):
        lci[lc == c] = i
    xs, ls, ys, ms, ps = [], [], [], [], []
    for r in starts(y.shape[0], size, stride):
        for c in starts(y.shape[1], size, stride):
            m = ok[r:r + size, c:c + size]
            if m.mean() < min_valid:
                continue
            xs.append(Z[:, r:r + size, c:c + size])
            ls.append(lci[r:r + size, c:c + size])
            ys.append(y[r:r + size, c:c + size])
            ms.append(m.astype("uint8"))
            ps.append((r, c))
    if not xs:
        return {"city": city, "split": split, "patches": 0}
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{city}.npz", X=np.stack(xs), lc=np.stack(ls), y=np.stack(ys), mask=np.stack(ms),
                        pos=np.array(ps, "int32"))
    return {"city": city, "split": split, "patches": len(xs), "usable_%": round(float(np.mean([m.mean() for m in ms]) * 100), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="pre_processing/grids")
    ap.add_argument("--out", default="pre_processing/patches")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--overlap", type=float, default=0.15, help="training-patch overlap (validation uses none)")
    ap.add_argument("--min-valid", type=float, default=0.5, help="drop patches with a smaller usable share")
    ap.add_argument("--val-cities", nargs="*", default=VAL_DEFAULT)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    grids, out = Path(a.grids), Path(a.out)
    cities = sorted(p.name[:-len("_features.nc")] for p in grids.glob("*_features.nc") if (grids / p.name.replace("_features", "_temp")).exists())
    if not cities:
        raise SystemExit(f"no <City>_features.nc / <City>_temp.nc pairs in {grids}")
    val = [c for c in cities if c in a.val_cities]
    train = [c for c in cities if c not in val]
    if not train:
        raise SystemExit("no training cities left")
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(cities)))
    stride = max(1, round(a.size * (1 - a.overlap)))
    print(f"{len(train)} training cities, {len(val)} validation cities | {a.size}x{a.size} patches, train stride {stride} "
          f"({1 - stride / a.size:.0%} overlap), validation stride {a.size} | {workers} process(es)")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        parts = list(ex.map(stats, [(str(grids), c) for c in train]))
        names = parts[0][0]
        n = sum(p[3] for p in parts)
        mean = sum(p[1] for p in parts) / n
        std = np.sqrt(np.maximum(sum(p[2] for p in parts) / n - mean ** 2, 0)) + 1e-6
        jobs = [(str(grids), c, "train", str(out), a.size, stride, a.min_valid, mean.tolist(), std.tolist()) for c in train] + \
               [(str(grids), c, "val", str(out), a.size, a.size, a.min_valid, mean.tolist(), std.tolist()) for c in val]
        rows = []
        for r in ex.map(cut, jobs):
            rows.append(r)
            print(f"  {r['city']} ({r['split']}): {r['patches']} patches", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "patch_summary.csv", index=False)
    json.dump({"features": names, "mean": mean.tolist(), "std": std.tolist(), "landcover_codes": LC_CODES,
               "size": a.size, "train_stride": stride, "train_cities": train, "val_cities": val},
              open(out / "meta.json", "w"), indent=1)
    s = pd.DataFrame(rows).groupby("split")["patches"].sum()
    print(f"\npatches: {s.to_dict()} | {len(names)} numeric features + landcover class | summary -> {out / 'patch_summary.csv'}")


if __name__ == "__main__":
    main()
