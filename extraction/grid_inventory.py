"""
How much data do we have, grid-wise? Read-only inventory of every city's full 30 m grid (cities/<City>/aligned.nc).

    python extraction/grid_inventory.py --root data/multi_city
    python extraction/grid_inventory.py --root data/multi_city --maps Delhi Pune     # also draw observation-count maps

Per city: grid size and pixel count, share of pixels with a valid land surface temperature, how many satellite
observations each pixel has (mean / minimum / share with fewer than 5), share of water, share of pixels each input
layer is missing for, share usable by the model (all inputs present and not water), table rows (every 3rd pixel).
Writes <root>/grid_inventory.csv.
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = ["NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "SLOPE_deg", "LANDCOVER", "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP",
        "NDVI_nbr150", "NDVI_nbr450", "BUILT_nbr150", "BUILT_nbr450", "HEIGHT_nbr150", "HEIGHT_nbr450",
        "DIST_WATER_m", "DIST_GREEN_m"]


def inventory(path, city):
    ds = xr.open_dataset(path)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    n = ny * nx
    r = {"city": city, "grid": f"{ny} x {nx}", "pixels_M": round(n / 1e6, 3)}
    lst = ds["LST"].values
    r["valid_LST_%"] = round(float(np.isfinite(lst).mean() * 100), 1)
    nobs = ds["n_obs"].values.astype("float64")
    ok = np.isfinite(lst)
    r["obs_mean"] = round(float(np.nanmean(nobs[ok])), 1)
    r["obs_min"] = int(np.nanmin(nobs[ok]))
    r["few_obs_%"] = round(float((nobs[ok] < 5).mean() * 100), 2)        # pixels averaged from fewer than 5 passes
    lc = ds["LANDCOVER"].values
    r["water_%"] = round(float((lc == 80).mean() * 100), 1)
    usable = np.ones((ny, nx), bool)
    miss = {}
    for b in BASE:
        f = np.isfinite(ds[b].values)
        usable &= f
        miss[b] = float((~f).mean() * 100)
    worst = max(miss, key=miss.get)
    r["missing_worst_layer"] = f"{worst} {miss[worst]:.1f}%"
    usable &= lc != 80
    r["usable_%"] = round(float(usable.mean() * 100), 1)
    r["usable_pixels_M"] = round(float(usable.sum() / 1e6), 3)
    r["urban_%"] = round(float((ds["IS_URBAN"].values == 1).mean() * 100), 1) if "IS_URBAN" in ds else np.nan
    ds.close()
    return r


def draw(path, city, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = xr.open_dataset(path)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    im = ax[0].imshow(ds["n_obs"].values, cmap="viridis")
    ax[0].set_title(f"{city}: satellite observations per pixel")
    plt.colorbar(im, ax=ax[0], shrink=0.7)
    miss = np.zeros(ds["LST"].shape)
    for b in BASE:
        miss += ~np.isfinite(ds[b].values)
    im = ax[1].imshow(miss, cmap="Reds")
    ax[1].set_title("number of missing input layers per pixel")
    plt.colorbar(im, ax=ax[1], shrink=0.7)
    ax[2].imshow(np.where(ds["LANDCOVER"].values == 80, 2, np.where(np.isfinite(ds["LST"].values), 0, 1)),
                 cmap=matplotlib.colors.ListedColormap(["#d9f0d3", "#fdae61", "#4575b4"]), interpolation="nearest")
    ax[2].set_title("green = valid LST, orange = no LST, blue = water")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    ds.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multi_city")
    ap.add_argument("--maps", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    files = {p.parent.name: p for p in sorted((root / "cities").glob("*/aligned.nc"))}
    if not files:
        raise SystemExit(f"no aligned.nc files under {root}/cities")
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(files)))
    print(f"{len(files)} cities on {workers} process(es)\n")
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(inventory, str(p), c): c for c, p in files.items()}
        for f in as_completed(futs):
            rows.append(f.result())
    t = pd.DataFrame(rows).set_index("city").sort_index()
    for c in t.index:                                  # rows kept in the training table (every 3rd pixel, valid LST+NDVI)
        tab = root / "cities" / c / "table.csv.gz"
        t.loc[c, "table_rows"] = len(pd.read_csv(tab, usecols=["x"])) if tab.exists() else np.nan
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(t.to_string())
    t.to_csv(root / "grid_inventory.csv")
    print(f"\nTOTAL: {t['pixels_M'].sum():,.1f} M grid pixels in {len(t)} cities | usable by the model {t['usable_pixels_M'].sum():,.1f} M "
          f"({t['usable_pixels_M'].sum() / t['pixels_M'].sum():.0%}) | training-table rows {t['table_rows'].sum() / 1e6:,.2f} M")
    print(f"valid LST: mean {t['valid_LST_%'].mean():.1f}% (lowest {t['valid_LST_%'].idxmin()} {t['valid_LST_%'].min():.1f}%) | "
          f"observations per pixel: mean {t['obs_mean'].mean():.1f}")
    weak = t[(t["valid_LST_%"] < 90) | (t["few_obs_%"] > 5)]
    print("cities with gaps worth a look: " + (", ".join(weak.index) if len(weak) else "none"))
    for c in a.maps:
        if c in files:
            draw(files[c], c, root / f"grid_inventory_{c}.png")
            print(f"map -> {root / f'grid_inventory_{c}.png'}")


if __name__ == "__main__":
    main()
