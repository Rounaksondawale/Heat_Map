"""
Audit EVERYTHING in a build_dataset.py output folder and repair what is missing.

    python repair_dataset.py --out ../data/multi_city --audit    # report only
    python repair_dataset.py --out ../data/multi_city            # report, repair, report again
    python repair_dataset.py --out ../data/multi_city --add-osm  # also add OSM layers (slow)

Checks per city:  table missing | weather (MET_*) empty | any other column with NaNs | empty / all-zero GHSL or
                  DEM / land-cover layers | too few Landsat scenes | OSM columns absent (only with --add-osm).

Repairs, cheapest first:
  1. city missing / layer problem  -> rebuild that city with build_dataset.build_city (force)
  2. weather still empty           -> re-fetch only the weather (Open-Meteo) and patch table + meta
  3. restack all_cities_training.csv.gz, then audit again and say what could NOT be fixed

Some gaps are upstream (a Landsat file that is corrupt on Planetary Computer, a city with no water) and cannot
be repaired by re-downloading; those are listed under "still not fixed".
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import build_dataset as bd

MET = ["MET_T2M", "MET_RH", "MET_WIND", "MET_SW", "MET_RAIN7", "MET_RAIN30", "MET_T2M_SEASON_MAX"]
OSM = ["BLDG_FRAC", "WATER_FRAC", "GREEN_FRAC", "BLDG_nbr150", "BLDG_nbr450"]
CORE = ["LST", "NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "LANDCOVER", "GHSL_HEIGHT", "GHSL_BUILT_FRAC",
        "GHSL_POP", "SLOPE_deg", "DIST_WATER_m", "DIST_GREEN_m", "SUHI", "HOTSPOT"]
NAN_TOL = 0.02          # >2 % NaN in a non-weather column is a problem
MIN_SCENES = 8


def audit_city(out_root, name, want_osm):
    """Return a list of (code, detail) issues for one city."""
    cdir = out_root / "cities" / name
    if not (cdir / "table.csv.gz").exists():
        return [("table", "not built")]
    df = pd.read_csv(cdir / "table.csv.gz")
    meta = json.loads((cdir / "meta.json").read_text()) if (cdir / "meta.json").exists() else {}
    issues = []
    if df[MET].isna().all(axis=1).mean() > 0.5 if all(c in df for c in MET) else True:
        issues.append(("weather", "MET_* empty"))
    for c in df.columns.difference(MET):
        if df[c].dtype != object and df[c].isna().mean() > NAN_TOL:
            issues.append(("layer", f"{c}: {df[c].isna().mean():.0%} NaN"))
    for c in CORE:
        if c not in df:
            issues.append(("layer", f"{c}: column absent"))
        elif df[c].nunique(dropna=True) <= 1:
            issues.append(("layer", f"{c}: constant ({df[c].dropna().iloc[0] if df[c].notna().any() else 'all NaN'})"))
    for c in ("GHSL_BUILT_FRAC", "GHSL_HEIGHT"):
        if c in df and df[c].eq(0).all():
            issues.append(("layer", f"{c}: all zero (GHSL tile missing?)"))
    if meta.get("n_scenes", 99) < MIN_SCENES:
        issues.append(("layer", f"only {meta.get('n_scenes')} Landsat scenes"))
    if want_osm and not all(c in df for c in OSM):
        issues.append(("osm", "OSM columns absent"))
    return issues


def audit(out_root, want_osm, label):
    print(f"\n===== {label} =====")
    res = {n: audit_city(out_root, n, want_osm) for n in bd.CITIES}
    for n, iss in res.items():
        print(f"{n:12s} " + ("OK" if not iss else ""))
        for code, d in iss:
            print(f"    [{code}] {d}")
    return res


def scene_dates(cat, name, a):
    """Same search + selection as get_landsat(), without reading any pixels."""
    lon, lat, _, cloud_max = bd.CITIES[name]
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=bd.city_bbox(lon, lat, a.half),
                            datetime=f"{a.years[0]}-01-01/{a.years[1]}-12-31",
                            query={"eo:cloud_cover": {"lt": cloud_max},
                                   "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    items = [i for i in items if i.datetime.month in a.months]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return pd.DatetimeIndex(sorted({pd.Timestamp(i.datetime).tz_localize(None).normalize()
                                    for i in items[:a.max_scenes]}))


def patch_weather(out_root, name, cat, a):
    lon, lat = bd.CITIES[name][:2]
    dates = scene_dates(cat, name, a)
    met = bd.get_weather(lon, lat, dates)
    tp = out_root / "cities" / name / "table.csv.gz"
    df = pd.read_csv(tp)
    for k, v in met.items():
        df[k] = v
    df.to_csv(tp, index=False)
    mp = out_root / "cities" / name / "meta.json"
    meta = json.loads(mp.read_text())
    meta.update({k: float(v) for k, v in met.items()})
    mp.write_text(json.dumps(meta, indent=2))
    print(f"  [{name}] weather patched from {len(dates)} dates")


def restack(out_root):
    tabs = sorted((out_root / "cities").glob("*/table.csv.gz"))
    alldf = pd.concat((pd.read_csv(t) for t in tabs), ignore_index=True)
    alldf.to_csv(out_root / "all_cities_training.csv.gz", index=False)
    print(f"\nStacked {len(tabs)} cities -> {len(alldf):,} rows -> {out_root / 'all_cities_training.csv.gz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/multi_city")
    ap.add_argument("--cities", nargs="*", default=None, help="limit repairs to these cities")
    ap.add_argument("--audit", action="store_true", help="report only, change nothing")
    ap.add_argument("--add-osm", action="store_true", help="treat missing OSM layers as a gap and fetch them (slow)")
    ap.add_argument("--pause", type=int, default=20, help="seconds between Open-Meteo calls")
    ap.add_argument("--years", nargs=2, type=int, default=[2022, 2025])
    ap.add_argument("--months", nargs="*", type=int, default=[3, 4, 5])
    ap.add_argument("--max-scenes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--half", type=float, default=20)
    a = ap.parse_args()

    out_root = Path(a.out)
    if not (out_root / "cities").exists():
        raise SystemExit(f"{out_root.resolve()} has no cities/ folder; pass the real --out (e.g. ../data/multi_city)")

    res = audit(out_root, a.add_osm, "AUDIT")
    if a.audit:
        raise SystemExit(0)
    scope = set(a.cities or bd.CITIES)

    # 1. rebuild cities that are missing or have layer / OSM problems (rebuild also re-fetches weather)
    rebuild = [n for n, iss in res.items() if n in scope and any(c in ("table", "layer", "osm") for c, _ in iss)]
    if rebuild:
        print(f"\nRebuilding {rebuild}")
        cat = bd.catalog()
        ns = SimpleNamespace(half=a.half, years=a.years, months=a.months, max_scenes=a.max_scenes, stride=a.stride,
                             osm=a.add_osm, force=True, workers=1)
        import dask
        dask.config.set(scheduler="threads")
        for n in rebuild:
            try:
                bd.build_city(n, ns, cat, out_root)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {n} rebuild failed: {type(e).__name__}: {e}")

    # 2. weather-only patch for whatever still has none
    cat = None
    todo = [n for n in bd.CITIES if n in scope and any(c == "weather" for c, _ in audit_city(out_root, n, a.add_osm))]
    for i, n in enumerate(todo):
        cat = cat or bd.catalog()
        try:
            patch_weather(out_root, n, cat, a)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {n} weather still failing: {type(e).__name__}: {e}")
        if i < len(todo) - 1:
            time.sleep(a.pause)

    restack(out_root)
    final = audit(out_root, a.add_osm, "STILL NOT FIXED (after repair)")
    left = {n: iss for n, iss in final.items() if iss and n in scope}
    print("\nAll clean." if not left else f"\n{len(left)} city(ies) still have gaps: {list(left)}  (see details above)")
