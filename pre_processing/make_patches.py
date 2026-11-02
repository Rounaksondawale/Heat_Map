"""
Cut the city grids into 64 x 64 patches for a U-Net.

    python pre_processing/make_patches.py                                   # grids in pre_processing/grids
    python pre_processing/make_patches.py --val-cities Jaipur Chennai --workers 4

Reads <grids>/<City>_features.nc + <City>_temp.nc (export_city_grids.py) and writes, per city,
<out>/<split>/<City>.npz with
    X     (N, F, 64, 64) float16  numeric features, standardised with TRAINING-city mean/std; missing -> 0. The last 6 channels
                                  (NDVI_rel, NDBI_rel, ...) are the same features relative to the pixel's own city (mean 0, sd 1 per city)
    lc    (N, 64, 64)    uint8    LANDCOVER class index (categorical; codes in meta.json)
    y     (N, 64, 64)    float32  LST in degrees C (not normalised)
    mask  (N, 64, 64)    uint8    1 = usable pixel: LST and every feature present, not water, and at least --min-pass x the city's
                                  median number of satellite passes (few-pass pixels are often cloud / snow leftovers).
                                  Compute the loss on mask == 1 only.
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
REL_VARS = ["NDVI", "NDBI", "ALBEDO", "GHSL_BUILT_FRAC", "GHSL_HEIGHT", "ELEV"]   # also given relative to the city (_rel channels)
REL_FLOOR = 0.25     # a city's spread of a feature counts as at least this share of the spread over all training cities
VAL_DEFAULT = ["Jaipur", "Lucknow", "Nagpur", "Chennai", "Bengaluru", "Kolkata", "Bhubaneswar", "Dehradun",
               "Amritsar", "Coimbatore"]                        # spread over climates; the hardest cities stay in training


def city_rel(values, usable, floor):
    """(mean, std) of one feature over a city's usable pixels; the std never falls below `floor`."""
    v = values[usable]
    return float(v.mean()), max(float(v.std()), floor)


def load(grids, city):
    f = xr.open_dataset(grids / f"{city}_features.nc")
    names = [v for v in f.data_vars if v != "LANDCOVER"]
    X = np.stack([f[v].values.astype("float32") for v in names])
    lc = f["LANDCOVER"].values
    tf = xr.open_dataset(grids / f"{city}_temp.nc")
    y = tf["LST"].values.astype("float32")
    n = tf["n_obs"].values.astype("float32") if "n_obs" in tf else None
    tf.close()
    f.close()
    return names, X, lc, y, n


def stats(args):
    grids, city = args
    names, X, lc, y, _ = load(Path(grids), city)
    ok = np.isfinite(y) & (lc != WATER) & np.isfinite(X).all(0)
    S = X[:, ok][:, ::7].astype("float64")                      # every 7th usable pixel is plenty
    return names, S.sum(1), (S ** 2).sum(1), S.shape[1]


def starts(n, size, stride):
    s = list(range(0, n - size + 1, stride))
    if s[-1] != n - size:
        s.append(n - size)                                      # cover the far edge
    return s


def cut(args):
    grids, city, split, out, size, stride, min_valid, mean, std, min_pass = args
    names, X, lc, y, n = load(Path(grids), city)
    mean, std = np.array(mean, "float32"), np.array(std, "float32")
    base_ok = (lc != WATER) & np.isfinite(X).all(0)
    ok = np.isfinite(y) & base_ok
    few = float("nan")                                          # share of otherwise usable pixels dropped for too few passes
    if n is not None:
        keep = np.isfinite(n) & (n >= min_pass * float(np.median(n[ok & np.isfinite(n)])))
        few = float((ok & ~keep).sum() / max(ok.sum(), 1) * 100)
        ok &= keep
    Z = np.where(np.isfinite(X), (X - mean[:, None, None]) / std[:, None, None], 0).astype("float16")
    rel = []
    for v in REL_VARS:
        i = names.index(v)
        m, s = city_rel(X[i], base_ok, REL_FLOOR * float(std[i]))
        rel.append(np.where(np.isfinite(X[i]), (X[i] - m) / s, 0).astype("float16"))
    Z = np.concatenate([Z, np.stack(rel)])
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
        return {"city": city, "split": split, "patches": 0, "few_pass_%": few}
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{city}.npz", X=np.stack(xs), lc=np.stack(ls), y=np.stack(ys), mask=np.stack(ms),
                        pos=np.array(ps, "int32"))
    return {"city": city, "split": split, "patches": len(xs), "usable_%": round(float(np.mean([m.mean() for m in ms]) * 100), 1),
            "few_pass_%": round(few, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="pre_processing/grids")
    ap.add_argument("--out", default="pre_processing/patches")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--overlap", type=float, default=0.15, help="training-patch overlap (validation uses none)")
    ap.add_argument("--min-valid", type=float, default=0.5, help="drop patches with a smaller usable share")
    ap.add_argument("--min-pass", type=float, default=0.6, help="mask pixels with fewer satellite passes than this x the city median (0 = off)")
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
        jobs = [(str(grids), c, "train", str(out), a.size, stride, a.min_valid, mean.tolist(), std.tolist(), a.min_pass) for c in train] + \
               [(str(grids), c, "val", str(out), a.size, a.size, a.min_valid, mean.tolist(), std.tolist(), a.min_pass) for c in val]
        rows = []
        for r in ex.map(cut, jobs):
            rows.append(r)
            print(f"  {r['city']} ({r['split']}): {r['patches']} patches, {r['few_pass_%']}% of usable pixels masked for few passes", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "patch_summary.csv", index=False)
    rel_names = [f"{v}_rel" for v in REL_VARS]
    if pd.DataFrame(rows)["few_pass_%"].isna().any():
        print("WARNING: some grids have no n_obs, so no few-pass masking was done; re-run export_city_grids.py --force")
    json.dump({"features": names + rel_names, "mean": mean.tolist() + [0.0] * len(rel_names), "std": std.tolist() + [1.0] * len(rel_names),
               "rel": {"vars": REL_VARS, "floor": REL_FLOOR}, "min_pass": a.min_pass, "landcover_codes": LC_CODES,
               "size": a.size, "train_stride": stride, "train_cities": train, "val_cities": val},
              open(out / "meta.json", "w"), indent=1)
    s = pd.DataFrame(rows).groupby("split")["patches"].sum()
    print(f"\npatches: {s.to_dict()} | {len(names) + len(rel_names)} numeric features ({len(rel_names)} relative to the city) + landcover class | summary -> {out / 'patch_summary.csv'}")


if __name__ == "__main__":
    main()
