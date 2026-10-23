"""
Audit a build_dataset.py output folder and re-fetch ONLY the weather that failed (no Landsat re-download).

    python fix_weather.py --out data_multi              # audit, then patch every city missing weather
    python fix_weather.py --out data_multi --audit      # audit only
    python fix_weather.py --out data_multi --cities Delhi Jaipur --pause 30

For each city whose table has no MET_* values it re-runs the (cheap) Planetary Computer STAC search with the
same settings as build_dataset.py to recover the scene dates, calls Open-Meteo with retries, then patches
cities/<City>/table.csv.gz + meta.json and re-stacks all_cities_training.csv.gz.

Cities that are missing entirely (or whose Landsat read failed, e.g. Hyderabad) need a real rebuild:
    python build_dataset.py --cities Hyderabad --force --workers 1 --out data_multi
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import build_dataset as bd

MET = ["MET_T2M", "MET_RH", "MET_WIND", "MET_SW", "MET_RAIN7", "MET_RAIN30", "MET_T2M_SEASON_MAX"]


def audit(out_root):
    rows = []
    for name in bd.CITIES:
        t = out_root / "cities" / name / "table.csv.gz"
        if not t.exists():
            rows.append((name, "MISSING", 0, None, None))
            continue
        df = pd.read_csv(t, usecols=["LST"] + MET)
        meta = json.loads((out_root / "cities" / name / "meta.json").read_text())
        no_met = bool(df[MET].isna().all(axis=1).mean() > 0.5)
        rows.append((name, "no weather" if no_met else "ok", len(df), meta.get("n_scenes"), meta.get("hotspot_threshold_C")))
    rep = pd.DataFrame(rows, columns=["city", "status", "rows", "scenes", "hotspot_C"]).set_index("city")
    print(rep.to_string())
    return rep


def scene_dates(cat, name, years, months, max_scenes):
    """Same search + selection as get_landsat(), without reading any pixels."""
    lon, lat, _, cloud_max = bd.CITIES[name]
    bbox = bd.city_bbox(lon, lat, args.half)
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=f"{years[0]}-01-01/{years[1]}-12-31",
                            query={"eo:cloud_cover": {"lt": cloud_max},
                                   "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    items = [i for i in items if i.datetime.month in months]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    items = items[:max_scenes]
    return pd.DatetimeIndex(sorted({pd.Timestamp(i.datetime).tz_localize(None).normalize() for i in items}))


def restack(out_root):
    tabs = sorted((out_root / "cities").glob("*/table.csv.gz"))
    alldf = pd.concat((pd.read_csv(t) for t in tabs), ignore_index=True)
    alldf.to_csv(out_root / "all_cities_training.csv.gz", index=False)
    print(f"Stacked {len(tabs)} cities -> {len(alldf):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_multi")
    ap.add_argument("--cities", nargs="*", default=None, help="default: every city without weather")
    ap.add_argument("--audit", action="store_true", help="only print the audit")
    ap.add_argument("--pause", type=int, default=20, help="seconds between Open-Meteo calls (rate limit)")
    ap.add_argument("--years", nargs=2, type=int, default=[2022, 2025])
    ap.add_argument("--months", nargs="*", type=int, default=[3, 4, 5])
    ap.add_argument("--max-scenes", type=int, default=40)
    ap.add_argument("--half", type=float, default=20)
    args = ap.parse_args()

    out_root = Path(args.out)
    rep = audit(out_root)
    todo = args.cities or rep.index[rep.status == "no weather"].tolist()
    miss = rep.index[rep.status == "MISSING"].tolist()
    if miss:
        print("\nNot built at all (run build_dataset.py --cities ...):", miss)
    if args.audit or not todo:
        raise SystemExit(0)

    cat = bd.catalog()
    failed = []
    for n, name in enumerate(todo):
        if name not in bd.CITIES or not (out_root / "cities" / name / "table.csv.gz").exists():
            print(f"skip {name}: no table")
            continue
        lon, lat = bd.CITIES[name][:2]
        print(f"\n[{name}] fetching weather")
        try:
            dates = scene_dates(cat, name, args.years, args.months, args.max_scenes)
            met = bd.get_weather(lon, lat, dates)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {name} still failing: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        tp = out_root / "cities" / name / "table.csv.gz"
        df = pd.read_csv(tp)
        for k, v in met.items():
            df[k] = v
        df.to_csv(tp, index=False)
        mp = out_root / "cities" / name / "meta.json"
        meta = json.loads(mp.read_text())
        meta.update({k: float(v) for k, v in met.items()})
        mp.write_text(json.dumps(meta, indent=2))
        print(f"  patched {len(df):,} rows from {len(dates)} dates: " + ", ".join(f"{k}={v:.1f}" for k, v in met.items()))
        if n < len(todo) - 1:
            time.sleep(args.pause)

    restack(out_root)
    if failed:
        print("Still failing (re-run later):", failed)
