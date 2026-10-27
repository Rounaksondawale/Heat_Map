"""
Add multi-scale "surroundings" features to the training table, computed from each city's aligned.nc grid.

    python add_context_features.py                       # ../data/multi_city  ->  all_cities_clean_ctx.csv.gz
    python add_context_features.py --root ../data/smoke_test --src ../data/smoke_test/all_cities_training.csv.gz

For every table pixel and for NDVI, NDBI, ALBEDO, MNDWI, GHSL_BUILT_FRAC, GHSL_HEIGHT:
    <VAR>_c90, _c330, _c930, _c2010     mean of the layer in a 3 / 11 / 31 / 67 pixel window (30 m pixels)
    NDVI_sd330/930, GHSL_BUILT_FRAC_sd330/930   local variation (how mixed the surroundings are)
Means ignore NaN pixels (water holes, cloud gaps, nodata) and are computed on the full 30 m grid, not on the
strided table, so the windows see every neighbouring pixel. The source file is not modified.
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

VARS = ["NDVI", "NDBI", "ALBEDO", "MNDWI", "GHSL_BUILT_FRAC", "GHSL_HEIGHT"]
WINDOWS = {3: "c90", 11: "c330", 31: "c930", 67: "c2010"}          # window in 30 m pixels -> name
TEXTURE = {"NDVI": (11, 31), "GHSL_BUILT_FRAC": (11, 31)}


def masked_mean(a, w):
    ok = np.isfinite(a)
    num = uniform_filter(np.where(ok, a, 0.0), w, mode="constant")
    den = uniform_filter(ok.astype("float64"), w, mode="constant")
    return np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)


def context_grids(ds):
    """Surroundings features on the full 30 m grid: {name: DataArray(y, x)}."""
    out = {}
    for v in VARS:
        a = ds[v].values.astype("float64")
        means = {}
        for w, tag in WINDOWS.items():
            means[w] = masked_mean(a, w)
            out[f"{v}_{tag}"] = ds[v].copy(data=means[w])
        for w in TEXTURE.get(v, ()):
            var = masked_mean(a ** 2, w) - means.get(w, masked_mean(a, w)) ** 2
            out[f"{v}_sd{w * 30}"] = ds[v].copy(data=np.sqrt(np.clip(var, 0, None)))
    return out


def city_features(nc_path, xs, ys):
    ds = xr.open_dataset(nc_path)
    ix = xr.DataArray(xs, dims="p")
    iy = xr.DataArray(ys, dims="p")
    out = context_grids(ds)
    cols = {k: g.sel(x=ix, y=iy).values.astype("float32") for k, g in out.items()}
    ds.close()
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/multi_city", help="folder holding cities/<City>/aligned.nc")
    ap.add_argument("--src", default=None, help="default: <root>/all_cities_clean.csv.gz")
    ap.add_argument("--dst", default=None, help="default: <root>/all_cities_clean_ctx.csv.gz")
    ap.add_argument("--workers", type=int, default=0, help="cities processed in parallel (default 0 = all CPU cores)")
    ap.add_argument("--no-csv", action="store_true", help="write only the (much faster) parquet file, not the CSV")
    a = ap.parse_args()
    root = Path(a.root)
    src = Path(a.src) if a.src else root / "all_cities_clean.csv.gz"
    src_pq = src.with_name(src.name.replace(".csv.gz", ".parquet"))
    dst = Path(a.dst) if a.dst else root / "all_cities_clean_ctx.csv.gz"

    if src_pq.exists():
        df = pd.read_parquet(src_pq)
    else:
        try:
            df = pd.read_csv(src, engine="pyarrow")
        except Exception:  # noqa: BLE001
            df = pd.read_csv(src)
    jobs = {}
    for city, g in df.groupby("city", sort=False):
        nc = root / "cities" / city / "aligned.nc"
        if not nc.exists():
            raise SystemExit(f"missing {nc}")
        jobs[city] = (nc, g["x"].to_numpy(), g["y"].to_numpy(), g.index)
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(jobs)))
    print(f"{len(jobs)} cities on {workers} worker process(es)")
    parts = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(city_features, nc, xs, ys): (city, idx) for city, (nc, xs, ys, idx) in jobs.items()}
        for f in as_completed(futs):
            city, idx = futs[f]
            cols = f.result()
            parts.append(pd.DataFrame(cols, index=idx))
            print(f"  {city:12s} {len(idx):7,} rows  +{len(cols)} context features")
    ctx = pd.concat(parts).loc[df.index]
    out = pd.concat([df, ctx], axis=1)
    out.to_parquet(dst.with_name(dst.name.replace(".csv.gz", ".parquet")), index=False)
    if not a.no_csv:
        out.to_csv(dst, index=False)
    print(f"\n{out.shape[0]:,} rows x {out.shape[1]} cols -> {dst}  (NaN in context: {int(ctx.isna().sum().sum()):,})")


if __name__ == "__main__":
    main()
