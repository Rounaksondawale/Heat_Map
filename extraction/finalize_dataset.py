"""
Final clean-up + validation before modeling.

    python finalize_dataset.py                                  # reads ../data/multi_city/all_cities_training.csv.gz
    python finalize_dataset.py --src PATH --dst PATH --drop-nan-city Mumbai

1. Drops rows with any NaN, but ONLY for the cities named in --drop-nan-city (default Mumbai: its GHSL layers are
   NaN over ~44 % of the box). Every other city must already be NaN-free or validation fails.
2. Runs the checks below on the result and prints PASS / WARN / FAIL for each.
3. Writes the cleaned table to --dst (default: all_cities_clean.csv.gz next to the source) only if nothing FAILed.
The source file is never modified. Exit code 1 if any check fails.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset as bd

MET = ["MET_T2M", "MET_RH", "MET_WIND", "MET_SW", "MET_RAIN7", "MET_RAIN30", "MET_T2M_SEASON_MAX"]
META_COLS = ["city", "climate", "block_id"]
REQUIRED = ["LST", "NDVI", "NDBI", "MNDWI", "ALBEDO", "LST_std", "n_obs", "ELEV", "LANDCOVER", "GHSL_HEIGHT",
            "GHSL_BUILT_FRAC", "GHSL_POP", "NDVI_nbr150", "NDVI_nbr450", "BUILT_nbr150", "BUILT_nbr450",
            "HEIGHT_nbr150", "HEIGHT_nbr450", "DIST_WATER_m", "DIST_GREEN_m", "SLOPE_deg", "LST_z", "SUHI",
            "IS_URBAN", "HOTSPOT", "lon", "lat", "x", "y"] + MET + META_COLS
# column: (min, max) physically plausible
RANGES = {"LST": (10, 75), "NDVI": (-1, 1), "NDBI": (-1, 1), "MNDWI": (-1, 1), "ALBEDO": (-0.2, 1.0),
          "GHSL_BUILT_FRAC": (0, 1), "GHSL_HEIGHT": (0, 600), "GHSL_POP": (0, 1e4), "SLOPE_deg": (0, 90),
          "DIST_WATER_m": (0, 1e5), "DIST_GREEN_m": (0, 1e5), "IS_URBAN": (0, 1), "HOTSPOT": (0, 1),
          "ELEV": (-50, 6000), "n_obs": (1, 400), "lat": (6, 37), "lon": (68, 98),
          "MET_RH": (0, 100), "MET_T2M": (0, 55), "MET_WIND": (0, 60), "MET_SW": (0, 1400),
          "MET_RAIN7": (0, 1500), "MET_RAIN30": (0, 3000)}
OUT_OF_RANGE_FAIL = 0.01       # >1 % of a city's rows outside the range = FAIL, any = WARN

results = []


def check(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status:4s}] {name}" + (f": {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/multi_city/all_cities_training.csv.gz")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--drop-nan-city", nargs="*", default=["all"],
                    help="cities whose NaN rows are dropped; default all (a city losing more than half its rows FAILS validation)")
    ap.add_argument("--keep-bad-indices", action="store_true",
                    help="do not drop rows whose NDVI / NDBI / MNDWI / ELEV are outside the physical range")
    ap.add_argument("--allow-missing-cities", action="store_true", help="for partial / test runs")
    ap.add_argument("--no-csv", action="store_true", help="write only the (much faster) parquet file, not the CSV")
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_name("all_cities_clean.csv.gz")

    try:
        df = pd.read_csv(src, engine="pyarrow")          # multi-threaded reader
    except Exception:  # noqa: BLE001
        df = pd.read_csv(src)
    print(f"loaded {len(df):,} rows x {df.shape[1]} cols from {src}\n")

    # ------------------------------------------------ 1. drop NaN rows in the named cities
    feat = [c for c in df.columns if c not in META_COLS]
    n_before = df["city"].value_counts()
    clean_cities = sorted(df["city"].unique()) if "all" in a.drop_nan_city else a.drop_nan_city
    for city in clean_cities:
        m = df["city"] == city
        if not m.any():
            continue
        bad = m & df[feat].isna().any(axis=1)
        if bad.any():
            lc = df.loc[bad, "LANDCOVER"]
            print(f"{city}: dropping {bad.sum():,} of {m.sum():,} rows ({bad.sum() / m.sum():.0%}) with NaN; "
                  f"LANDCOVER of dropped rows: {lc.value_counts(normalize=True).round(2).head(4).to_dict()} "
                  f"(80 = water)")
            nan_cols = df.loc[bad, feat].isna().any()
            print(f"   NaN columns: {nan_cols[nan_cols].index.tolist()}\n")
            df = df[~bad].reset_index(drop=True)

    # ------------------------------------------------ 1b. drop pixels with impossible spectral indices / elevation
    # Per-scene indices are in [-1, 1], but the scene means blow up where nir+red (or green+swir...) is ~0 or
    # negative after the surface-reflectance offset: dark water pixels (Chennai, Mumbai coast). Those values are noise.
    if not a.keep_bad_indices:
        bad = pd.Series(False, index=df.index)
        for col in ("NDVI", "NDBI", "MNDWI", "ELEV"):
            lo, hi = RANGES[col]
            b = (df[col] < lo) | (df[col] > hi)
            if b.any():
                print(f"{col}: {int(b.sum()):,} rows outside [{lo}, {hi}] " +
                      str(df.loc[b, "city"].value_counts().to_dict()))
            bad |= b
        if bad.any():
            wl = df.loc[bad, "LANDCOVER"].value_counts(normalize=True).round(2).head(4).to_dict()
            print(f"dropping {int(bad.sum()):,} rows ({bad.mean():.2%} of all); LANDCOVER of dropped rows: {wl}\n")
            df = df[~bad].reset_index(drop=True)

    # ------------------------------------------------ 2. validation
    lost = 1 - df["city"].value_counts().reindex(n_before.index).fillna(0) / n_before
    heavy = lost[lost > 0.5]
    check("PASS" if heavy.empty else "FAIL", "no city lost more than half its rows in cleaning", str(heavy.round(2).to_dict()) if len(heavy) else f"max loss {lost.max():.0%} ({lost.idxmax()})")
    # cities
    present = set(df["city"].unique())
    missing = [c for c in bd.CITIES if c not in present]
    extra = sorted(present - set(bd.CITIES))
    check("PASS" if not missing else ("WARN" if a.allow_missing_cities else "FAIL"), f"all {len(bd.CITIES)} cities present",
          f"missing {missing}" if missing else f"{len(present)} cities")
    if extra:
        check("FAIL", "unexpected cities", str(extra))

    # columns
    absent = [c for c in REQUIRED if c not in df.columns]
    check("PASS" if not absent else "FAIL", "required columns present", f"absent {absent}" if absent else f"{len(REQUIRED)} ok")
    optional = [c for c in ("BLDG_FRAC", "WATER_FRAC", "GREEN_FRAC") if c not in df.columns]
    if optional:
        check("WARN", "OSM columns not built (fine unless the model uses them)", str(optional))

    # NaN / inf
    num = df.select_dtypes("number")
    n_nan = int(num.isna().sum().sum())
    n_inf = int(np.isinf(num.to_numpy(dtype="float64", na_value=0)).sum())
    check("PASS" if n_nan == 0 else "FAIL", "no NaN values", f"{n_nan:,} NaN" + (
        f" in {num.columns[num.isna().any()].tolist()}; cities {df.loc[num.isna().any(axis=1), 'city'].unique().tolist()}"
        if n_nan else ""))
    check("PASS" if n_inf == 0 else "FAIL", "no inf values", f"{n_inf:,} inf")

    # duplicates
    dup = int(df.duplicated(["city", "x", "y"]).sum())
    check("PASS" if dup == 0 else "FAIL", "no duplicate pixels (city, x, y)", f"{dup:,} duplicates")

    # per-city sanity
    g = df.groupby("city")
    rows = g.size()
    small = rows[rows < 100_000]
    check("PASS" if small.empty else "WARN", "each city has >= 100k rows", f"small: {small.to_dict()}" if len(small) else
          f"min {rows.min():,} max {rows.max():,}")
    blocks = g["block_id"].nunique()
    check("PASS" if (blocks >= 20).all() else "WARN", "each city has >= 20 spatial blocks (for block CV)",
          f"min {blocks.min()} ({blocks.idxmin()})")
    clim = g["climate"].nunique()
    check("PASS" if (clim == 1).all() else "FAIL", "one climate class per city")
    hot = g["HOTSPOT"].mean()
    oddh = hot[(hot < 0.005) | (hot > 0.3)]
    check("PASS" if oddh.empty else "WARN", "hotspot share per city between 0.5 % and 30 %",
          f"{ (hot*100).round(1).to_dict() if len(oddh) else 'min %.1f%% max %.1f%%' % (hot.min()*100, hot.max()*100)}")
    urb = g["IS_URBAN"].mean()
    check("PASS" if (urb > 0.02).all() else "WARN", "every city has urban pixels (>2 %)",
          f"min {urb.min():.1%} ({urb.idxmin()})")
    sd = g["LST"].std()
    check("PASS" if (sd > 0.5).all() else "FAIL", "LST varies inside every city (std > 0.5 C)", f"min {sd.min():.2f}")

    # weather: present, constant inside a city, different between cities
    wvar = g[MET].nunique().max()
    check("PASS" if (wvar <= 1).all() else "WARN", "weather constant within each city (city-level values)")
    wcity = g[MET[0]].first()
    check("PASS" if wcity.nunique() == len(wcity) else "WARN", "weather differs between cities (no copy-paste fill)",
          f"{wcity.nunique()} distinct values for {len(wcity)} cities")

    # constant columns across the whole table
    const = [c for c in num.columns if num[c].nunique() <= 1]
    check("PASS" if not const else "FAIL", "no constant columns", str(const) if const else "")
    dead = [(c, k) for c in ("GHSL_HEIGHT", "GHSL_BUILT_FRAC", "ELEV", "LANDCOVER", "DIST_WATER_m", "DIST_GREEN_m")
            if c in df for k, v in g[c].nunique().items() if v <= 1]
    check("PASS" if not dead else "WARN", "no layer is constant inside a city", str(dead) if dead else "")

    # value ranges
    bad_ranges = []
    for col, (lo, hi) in RANGES.items():
        if col not in df:
            continue
        out = ((df[col] < lo) | (df[col] > hi)).groupby(df["city"]).mean()
        out = out[out > 0]
        if len(out):
            bad_ranges.append((col, lo, hi, out.round(4).to_dict(), bool((out > OUT_OF_RANGE_FAIL).any())))
    if not bad_ranges:
        check("PASS", "all values inside physical ranges")
    for col, lo, hi, out, fail in bad_ranges:
        check("FAIL" if fail else "WARN", f"{col} outside [{lo}, {hi}]", f"share of rows by city {out}")

    # ------------------------------------------------ summary + write
    counts = {s: sum(r[0] == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    print("\nrows per city:\n" + rows.reindex([c for c in bd.CITIES if c in rows.index]).to_string())
    print(f"\nTOTAL {len(df):,} rows x {df.shape[1]} columns, {df['city'].nunique()} cities")
    if counts["FAIL"]:
        print("\nNOT READY FOR MODELING: fix the FAIL items above (nothing was written).")
        sys.exit(1)
    df.to_parquet(dst.with_name(dst.name.replace(".csv.gz", ".parquet")), index=False)
    if not a.no_csv:
        df.to_csv(dst, index=False)
    pq = dst.with_name(dst.name.replace(".csv.gz", ".parquet"))
    print(f"\nREADY FOR MODELING -> {pq}" + ("" if a.no_csv else f" and {dst}"))


if __name__ == "__main__":
    main()
