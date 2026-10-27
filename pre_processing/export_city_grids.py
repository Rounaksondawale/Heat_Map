"""
One full-resolution feature grid per city: every 30 m pixel of the 40 km box with ALL features.

    python pre_processing/export_city_grids.py --root data/multi_city
    python pre_processing/export_city_grids.py --root data/multi_city --cities Delhi Pune --workers 2

For each cities/<City>/aligned.nc this writes <out>/<City>.nc (compressed float32, dims y, x) holding
  - every layer of aligned.nc (LST, NDVI, NDBI, MNDWI, ALBEDO, ELEV, SLOPE_deg, LANDCOVER, GHSL_*, weather, n_obs, ...)
  - the surroundings features from extraction/add_context_features.py (3x3 ... 67x67 pixel means, NDVI/built texture)
and <out>/grid_summary.csv (grid size, number of features, share valid LST, file size).
The pixel (y, x) coordinates are the same as in the training table, so the grids and the table line up.
"""
import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))
from add_context_features import context_grids  # noqa: E402


def export(nc_path, out_path):
    ds = xr.open_dataset(nc_path).load()
    ds = ds.assign({k: v for k, v in context_grids(ds).items()})
    enc = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind == "f":
            ds[v] = ds[v].astype("float32")
        enc[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(out_path, encoding=enc)
    r = {"city": Path(nc_path).parent.name, "grid": f"{ds.sizes['y']} x {ds.sizes['x']}", "features": len(ds.data_vars),
         "valid_LST_%": round(float(np.isfinite(ds["LST"].values).mean() * 100), 1),
         "size_MB": round(os.path.getsize(out_path) / 1e6, 1)}
    ds.close()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multi_city", help="folder holding cities/<City>/aligned.nc")
    ap.add_argument("--out", default="pre_processing/grids")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=0, help="cities in parallel (default 0 = all cores, capped by city count)")
    ap.add_argument("--force", action="store_true", help="rewrite grids that already exist")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {p.parent.name: p for p in sorted((root / "cities").glob("*/aligned.nc"))}
    if a.cities:
        files = {c: p for c, p in files.items() if c in a.cities}
    if not files:
        raise SystemExit(f"no aligned.nc found under {root}/cities")
    todo = {c: p for c, p in files.items() if a.force or not (out / f"{c}.nc").exists()}
    print(f"{len(files)} cities, {len(todo)} to export -> {out}")
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(todo) or 1))
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(export, str(p), str(out / f"{c}.nc")): c for c, p in todo.items()}
        for f in as_completed(futs):
            try:
                r = f.result()
                rows.append(r)
                print(f"  {r['city']}: {r['grid']}, {r['features']} features, {r['size_MB']} MB", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {futs[f]} failed: {type(e).__name__}: {e}", flush=True)
    if rows:
        s = pd.DataFrame(rows).set_index("city").sort_index()
        old = out / "grid_summary.csv"
        if old.exists():
            s = pd.concat([pd.read_csv(old, index_col="city").drop(index=s.index, errors="ignore"), s]).sort_index()
        s.to_csv(old)
        print(f"\nsummary -> {old}")


if __name__ == "__main__":
    main()
