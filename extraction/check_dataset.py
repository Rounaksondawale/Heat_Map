"""
Read-only check of a build_dataset.py output folder: did every city download, and is the data any good?

    python extraction/check_dataset.py                                   # data/multi_city
    python extraction/check_dataset.py --root data/multi_city --logs pipeline.log build.log

Nothing is modified. For every city in CITIES it reports
    status      OK / WARN / BAD / NOT BUILT, with the reasons
    files       table.csv.gz, aligned.nc (opens?), meta.json, size on disk
    coverage    rows, Landsat scenes, share of the 40 km box with a valid LST, observations per pixel
    quality     LST range, NaN share per column, constant columns, GHSL all-zero, weather present + plausible,
                land cover mix (water / urban), elevation range, duplicate pixels
and the stacked all_cities_training.csv.gz (row count must equal the sum over cities).
With --logs it also lists the build failures it finds ("!! City failed: reason").
Writes <root>/quality_report.csv.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

MET = ["MET_T2M", "MET_RH", "MET_WIND", "MET_SW", "MET_RAIN7", "MET_RAIN30", "MET_T2M_SEASON_MAX"]
# column: (low, high) physically plausible; share of rows outside is reported
RANGES = {"LST": (0, 75), "NDVI": (-1, 1), "NDBI": (-1, 1), "MNDWI": (-1, 1), "ALBEDO": (-0.2, 1.0),
          "GHSL_BUILT_FRAC": (0, 1), "ELEV": (-50, 6000), "SLOPE_deg": (0, 90)}
CORE = ["LST", "NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "LANDCOVER", "GHSL_HEIGHT", "GHSL_BUILT_FRAC",
        "GHSL_POP", "SLOPE_deg", "DIST_WATER_m", "DIST_GREEN_m"]


def check_city(root, name):
    """Return one dict of facts + a list of (level, message) problems for a city."""
    cdir = Path(root) / "cities" / name
    r = {"city": name, "problems": []}
    P = r["problems"]
    table, nc, meta_p = cdir / "table.csv.gz", cdir / "aligned.nc", cdir / "meta.json"
    if not table.exists():
        r["status"] = "NOT BUILT"
        return r
    r["size_MB"] = round(sum(f.stat().st_size for f in cdir.glob("*") if f.is_file()) / 1e6, 1)

    # ---- table
    try:
        df = pd.read_csv(table)
    except Exception as e:  # noqa: BLE001
        r["status"] = "BAD"
        P.append(("BAD", f"table unreadable: {type(e).__name__}"))
        return r
    r["rows"] = len(df)
    if len(df) < 100_000:
        P.append(("WARN", f"only {len(df):,} rows"))
    dup = int(df.duplicated(["x", "y"]).sum())
    if dup:
        P.append(("BAD", f"{dup:,} duplicate pixels"))

    # ---- meta
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        r["scenes"] = meta.get("n_scenes")
        r["hotspot_C"] = round(meta.get("hotspot_threshold_C", np.nan), 1)
        if (r["scenes"] or 0) < 8:
            P.append(("WARN", f"only {r['scenes']} Landsat scenes"))
    else:
        P.append(("WARN", "meta.json missing"))

    # ---- aligned grid (does it open? how much of the box has a valid LST?)
    if nc.exists():
        try:
            import xarray as xr
            with xr.open_dataset(nc) as ds:
                lst = ds["LST"].values
                r["grid_valid_%"] = round(float(np.isfinite(lst).mean() * 100), 1)
                r["n_obs_mean"] = round(float(np.nanmean(ds["n_obs"].values)), 1)
            if r["grid_valid_%"] < 50:
                P.append(("WARN", f"only {r['grid_valid_%']}% of the box has a valid LST (sea / clouds / nodata)"))
        except Exception as e:  # noqa: BLE001
            P.append(("BAD", f"aligned.nc does not open: {type(e).__name__}"))
    else:
        P.append(("BAD", "aligned.nc missing"))

    # ---- columns
    for c in CORE:
        if c not in df:
            P.append(("BAD", f"column {c} absent"))
    nan = df.drop(columns=[c for c in MET if c in df]).isna().mean()
    for c, v in nan[nan > 0.02].items():
        P.append(("WARN" if v < 0.3 else "BAD", f"{c}: {v:.0%} NaN"))
    for c in CORE:
        if c in df and df[c].nunique(dropna=True) <= 1:
            P.append(("BAD", f"{c} is constant"))
    for c in ("GHSL_BUILT_FRAC", "GHSL_HEIGHT"):
        if c in df and df[c].eq(0).all():
            P.append(("BAD", f"{c} all zero (GHSL tile missing?)"))
    if "GHSL_POP" in df and df["GHSL_POP"].eq(0).mean() > 0.9:
        P.append(("WARN", "GHSL_POP is zero for over 90% of pixels"))
    water = float((df["LANDCOVER"] == 80).mean()) if "LANDCOVER" in df else 0.0
    for c, (lo, hi) in RANGES.items():
        if c in df:
            out = float(((df[c] < lo) | (df[c] > hi)).mean())
            if out > 0.01:
                if c in ("NDVI", "NDBI", "MNDWI", "ELEV") and out <= water:
                    # index blow-ups happen over dark water pixels; finalize_dataset.py drops them
                    P.append(("WARN", f"{c}: {out:.1%} outside [{lo}, {hi}] (fewer than the {water:.0%} water pixels: removed in cleaning)"))
                else:
                    P.append(("WARN" if out < 0.1 else "BAD", f"{c}: {out:.1%} outside [{lo}, {hi}]"))

    # ---- LST / land cover / weather facts
    if "LST" in df:
        r["LST_p1"], r["LST_p99"] = round(df["LST"].quantile(0.01), 1), round(df["LST"].quantile(0.99), 1)
        r["LST_std"] = round(df["LST"].std(), 2)
        if r["LST_std"] < 0.5:
            P.append(("BAD", f"LST barely varies (std {r['LST_std']})"))
    if "LANDCOVER" in df:
        r["water_%"] = round(float((df["LANDCOVER"] == 80).mean() * 100), 1)
        r["urban_%"] = round(float((df["LANDCOVER"] == 50).mean() * 100), 1)
        if r["urban_%"] < 2:
            P.append(("WARN", f"only {r['urban_%']}% built-up land cover"))
    if all(c in df for c in MET):
        w = df[MET].iloc[0]
        if w.isna().any():
            P.append(("BAD", "weather missing (Open-Meteo failed): run repair_dataset.py"))
        else:
            r["T2M"], r["RH"] = round(w["MET_T2M"], 1), round(w["MET_RH"], 1)
            if not (5 <= w["MET_T2M"] <= 50 and 0 <= w["MET_RH"] <= 100):
                P.append(("WARN", f"weather implausible (T2M {w['MET_T2M']:.1f}, RH {w['MET_RH']:.0f})"))
    else:
        P.append(("BAD", "weather columns absent"))

    levels = [lvl for lvl, _ in P]
    r["status"] = "BAD" if "BAD" in levels else ("WARN" if "WARN" in levels else "OK")
    return r


def scan_logs(paths):
    """Find '!! City failed: reason' lines (and 'too few clear scenes') in build logs."""
    fails = {}
    pat = re.compile(r"!! (\w+) failed: (.*)")
    for p in paths:
        if not Path(p).exists():
            print(f"(log {p} not found)")
            continue
        for line in Path(p).read_text(errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                fails[m.group(1)] = m.group(2)[:150]
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multi_city")
    ap.add_argument("--logs", nargs="*", default=[], help="build logs to scan for failures, e.g. pipeline.log build.log")
    ap.add_argument("--workers", type=int, default=0, help="default 0 = all cores")
    a = ap.parse_args()
    root = Path(a.root)
    if not (root / "cities").exists():
        raise SystemExit(f"{root.resolve()} has no cities/ folder")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import build_dataset as bd
    names = list(bd.CITIES)
    workers = max(1, min(a.workers or os.cpu_count() or 1, len(names)))
    print(f"checking {len(names)} cities in {root} on {workers} process(es)\n")
    res = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(check_city, str(root), n): n for n in names}
        for f in as_completed(futs):
            res[futs[f]] = f.result()

    rows = [res[n] for n in names]
    cols = ["city", "status", "rows", "scenes", "grid_valid_%", "n_obs_mean", "LST_p1", "LST_p99", "LST_std", "hotspot_C",
            "water_%", "urban_%", "T2M", "RH", "size_MB"]
    tbl = pd.DataFrame(rows).reindex(columns=cols).set_index("city")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(tbl[tbl["status"] != "NOT BUILT"].to_string())          # not-built cities are listed separately below
    tbl.to_csv(root / "quality_report.csv")

    # ---------------- problems
    print("\nPROBLEMS")
    any_p = False
    for n in names:
        for lvl, msg in res[n]["problems"]:
            any_p = True
            print(f"  [{lvl:4s}] {n}: {msg}")
    if not any_p:
        print("  none")

    # ---------------- stacked file
    stack = root / "all_cities_training.csv.gz"
    built = [n for n in names if "rows" in res[n]]
    total = sum(res[n]["rows"] for n in built)
    if stack.exists():
        n_stack = len(pd.read_csv(stack, usecols=["city"]))
        print(f"\nSTACKED FILE all_cities_training.csv.gz: {n_stack:,} rows; sum of city tables {total:,} "
              f"-> {'match' if n_stack == total else 'MISMATCH: restack with build_dataset.py or repair_dataset.py'}")
    else:
        print("\nSTACKED FILE all_cities_training.csv.gz not found (the build has not finished stacking yet)")

    # ---------------- failures from logs
    not_built = [n for n in names if res[n]["status"] == "NOT BUILT"]
    fails = scan_logs(a.logs) if a.logs else {}
    if not_built:
        print(f"\nNOT BUILT ({len(not_built)}):")
        for n in not_built:
            print(f"  {n:20s} {fails.get(n, '(no failure message found; pass --logs, or the build has not reached it yet)')}")

    # ---------------- verdict
    s = tbl["status"].value_counts().to_dict()
    print(f"\nSUMMARY: {len(built)}/{len(names)} cities built | " + ", ".join(f"{k}: {v}" for k, v in s.items()) +
          f" | {total:,} rows | quality table -> {root / 'quality_report.csv'}")
    bad = [n for n in names if res[n]["status"] == "BAD"]
    if bad:
        print(f"Cities to fix or rebuild: {bad}  (python extraction/repair_dataset.py --out {root} --cities ...)")


if __name__ == "__main__":
    main()
