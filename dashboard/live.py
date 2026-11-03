"""
Live data pipeline: from a rectangle drawn anywhere in India to the grid files the model reads for the 50 prepared cities.

    rect = check_rect(west, south, east, north)            # validates the rectangle (inside India, 1 to 10 km a side)
    info = build_area_grids(rect, out_dir, ghsl_cache)     # downloads, derives the 46 features, writes <name>_features.nc / _temp.nc

It reuses extraction/build_dataset.py (Landsat 8/9 land surface temperature and indices from Microsoft Planetary Computer,
Copernicus DEM, ESA WorldCover, JRC GHSL) and pre_processing/export_city_grids.py (the context features), so a live area gets exactly
the same inputs as a prepared city. The area is processed inside a window around the rectangle (at least 16 km wide, 3 km of margin
on every side) because the model needs the surroundings: its neighbourhood features reach 2 km and its city-relative inputs are
computed over the window.

Differences from the prepared cities, so the results are a little less reliable:
  * fewer Landsat scenes (default 20 instead of up to 40), which makes the averaged inputs slightly noisier;
  * the "city-relative" inputs and the hotspot statistics are relative to this window, not to a 40 km city box;
  * the first request for a region downloads the GHSL tiles (about 85 MB per layer and tile) and caches them.
"""
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("extraction", "pre_processing"):
    sys.path.insert(0, str(ROOT / sub))

INDIA = (68.0, 6.5, 97.5, 37.5)               # west, south, east, north (bounding box of India)
MIN_SIDE_KM, MAX_SIDE_KM = 1.0, 10.0          # side of the rectangle the user draws
MIN_HALF_KM, PAD_KM = 8.0, 3.0                # window around it: half-size at least 8 km, margin 3 km
CLOUD_STEPS = (20, 40, 70)                    # tried in turn until enough clear Landsat scenes are found
VERSION = 1
_LOCK = threading.Lock()


class Busy(RuntimeError):
    """Another live build is already running (one at a time keeps memory and downloads in check)."""


def side_km(west, south, east, north):
    lat = (south + north) / 2
    return (north - south) * 110.54, (east - west) * 111.32 * np.cos(np.radians(lat))


def check_rect(west, south, east, north):
    """Return the rectangle as floats (west, south, east, north) or raise ValueError with a message for the user."""
    w, s, e, n = (float(v) for v in (west, south, east, north))
    if not (w < e and s < n):
        raise ValueError("the rectangle is empty")
    cx, cy = (w + e) / 2, (s + n) / 2
    if not (INDIA[0] <= cx <= INDIA[2] and INDIA[1] <= cy <= INDIA[3]):
        raise ValueError("the rectangle is outside India: the model was trained on Indian cities")
    h_km, w_km = side_km(w, s, e, n)
    if min(h_km, w_km) < MIN_SIDE_KM:
        raise ValueError(f"the rectangle is too small ({h_km:.1f} x {w_km:.1f} km); draw at least {MIN_SIDE_KM:g} km on each side")
    if max(h_km, w_km) > MAX_SIDE_KM:
        raise ValueError(f"the rectangle is too large ({h_km:.0f} x {w_km:.0f} km); keep each side under {MAX_SIDE_KM:g} km")
    return w, s, e, n


def window_for(rect):
    """(lon, lat, half_km) of the square window that is downloaded and processed around the rectangle."""
    w, s, e, n = rect
    h_km, w_km = side_km(w, s, e, n)
    return (w + e) / 2, (s + n) / 2, max(MIN_HALF_KM, max(h_km, w_km) / 2 + PAD_KM)


def area_key(rect, max_scenes=20, years=(2022, 2025), months=(3, 4, 5)):
    """Stable short name for a rectangle and its settings, used as the 'city' name of the live grids."""
    txt = json.dumps([round(v, 4) for v in rect] + [max_scenes, list(years), list(months), VERSION])
    return "live_" + hashlib.sha1(txt.encode()).hexdigest()[:10]


def is_built(out_dir, name):
    return (Path(out_dir) / f"{name}_features.nc").exists() and (Path(out_dir) / f"{name}_temp.nc").exists()


def build_area_grids(rect, out_dir, ghsl_cache, max_scenes=20, years=(2022, 2025), months=(3, 4, 5), progress=None, cloud_steps=CLOUD_STEPS):
    """Download and process the window around `rect` and write <out_dir>/<name>_features.nc and <name>_temp.nc. Returns an info dict."""
    import build_dataset as B
    from export_city_grids import export_dataset
    say = progress or (lambda msg: None)
    name = area_key(rect, max_scenes, years, months)
    if not _LOCK.acquire(blocking=False):
        raise Busy("another area is being analysed right now; try again in a minute")
    t0 = time.time()
    try:
        lon, lat, half = window_for(rect)
        bbox = B.city_bbox(lon, lat, half)
        crs = B.utm_epsg(lon, lat)
        Path(ghsl_cache).mkdir(parents=True, exist_ok=True)
        say(f"Searching clear Landsat 8/9 scenes (March to May {years[0]}-{years[1]}) for a {2 * half:.0f} km window")
        cat = B.catalog()
        stack = dates = None
        for cm in cloud_steps:
            try:
                stack, dates = B.get_landsat(cat, bbox, crs, years, months, cm, max_scenes)
                break
            except RuntimeError:
                continue
        if stack is None:
            raise RuntimeError("not enough clear Landsat scenes for this place in March to May 2022-2025; try another area")
        say(f"Averaged {len(dates)} Landsat scenes into temperature, vegetation, built-up, water and albedo layers")
        ds = stack.drop_vars("spatial_ref", errors="ignore").rio.write_crs(f"EPSG:{crs}")
        tmpl = ds["LST"]
        layers = {}
        say("Elevation (Copernicus DEM) and land cover (ESA WorldCover)")
        layers.update(B.get_dem_and_landcover(cat, bbox, stack))
        say("Buildings and population (JRC GHSL; the first request for a region downloads the tiles)")
        layers.update(B.get_ghsl(bbox, tmpl, Path(ghsl_cache)))
        for k, v in layers.items():
            ds[k] = v.reindex_like(tmpl, method="nearest", tolerance=B.RES) if v.shape != tmpl.shape else v.assign_coords(y=tmpl.y, x=tmpl.x)
        say("Deriving the neighbourhood, distance and surroundings features")
        ds = B.add_derived(ds).drop_vars("spatial_ref", errors="ignore")
        for v in ds.data_vars:
            ds[v] = ds[v].astype("float32")
        ds.attrs.update({"epsg": crs, "n_scenes": len(dates)})
        for need in ("GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP"):
            if need not in ds:
                raise RuntimeError(f"the building layer {need} could not be downloaded (the JRC GHSL server did not answer); try again later")
        info = export_dataset(ds, name, out_dir)
        info.update({"name": name, "rect": list(rect), "window": [lon, lat, half], "n_scenes": len(dates),
                     "dates": [str(dates.min().date()), str(dates.max().date())], "seconds": round(time.time() - t0),
                     "years": list(years), "months": list(months)})
        (Path(out_dir) / f"{name}_info.json").write_text(json.dumps(info, indent=1))
        say(f"Done in {info['seconds']} s")
        return info
    finally:
        _LOCK.release()
