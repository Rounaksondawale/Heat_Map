"""
Per city, two aligned full-resolution grids (every 30 m pixel of the 40 km box):

    python pre_processing/export_city_grids.py --root data/multi_city
    python pre_processing/export_city_grids.py --root data/multi_city --cities Delhi Pune --workers 2

For each cities/<City>/aligned.nc this writes
  <out>/<City>_features.nc   the model INPUT features only (FEATURES below, incl. the surroundings features)
  <out>/<City>_temp.nc       one grid: LST (land surface temperature, the target)
Both have the same (y, x) coordinates as aligned.nc and the training table, so they line up pixel for pixel.
No leakage columns (LST_z, SUHI, HOTSPOT, ...) and no weather. <out>/grid_summary.csv lists size and feature count per city.
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


BASE = ["NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "SLOPE_deg", "LANDCOVER", "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP",
        "DIST_WATER_m", "DIST_GREEN_m", "NDVI_nbr150", "NDVI_nbr450", "BUILT_nbr150", "BUILT_nbr450",
        "HEIGHT_nbr150", "HEIGHT_nbr450"]
CONTEXT = [f"{v}_{t}" for v in ("NDVI", "NDBI", "ALBEDO", "MNDWI", "GHSL_BUILT_FRAC", "GHSL_HEIGHT")
           for t in ("c90", "c330", "c930", "c2010")] + ["NDVI_sd330", "NDVI_sd930", "GHSL_BUILT_FRAC_sd330", "GHSL_BUILT_FRAC_sd930"]
FEATURES = BASE + CONTEXT


def _write(ds, path):
    enc = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind == "f":
            ds[v] = ds[v].astype("float32")
        enc[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, encoding=enc)


def export(nc_path, out_dir):
    city = Path(nc_path).parent.name
    ds = xr.open_dataset(nc_path).load()
    ds = ds.assign({k: v for k, v in context_grids(ds).items()})
    feats, temp = Path(out_dir) / f"{city}_features.nc", Path(out_dir) / f"{city}_temp.nc"
    _write(ds[FEATURES], feats)
    _write(ds[["LST"]], temp)
    r = {"city": city, "grid": f"{ds.sizes['y']} x {ds.sizes['x']}", "features": len(FEATURES),
         "valid_LST_%": round(float(np.isfinite(ds["LST"].values).mean() * 100), 1),
         "size_MB": round((os.path.getsize(feats) + os.path.getsize(temp)) / 1e6, 1)}
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
    todo = {c: p for c, p in files.items() if a.force or not ((out / f"{c}_features.nc").exists() and (out / f"{c}_temp.nc").exists())}
    print(f"{len(files)} cities, {len(todo)} to export -> {out}")
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(todo) or 1))
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(export, str(p), str(out)): c for c, p in todo.items()}
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
