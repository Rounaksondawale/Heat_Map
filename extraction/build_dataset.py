"""
Multi-city training-data builder for the urban-heat-hotspot model.

For every city it builds ONE aligned 30 m grid (local UTM) with:
    target      LST (Landsat 8/9, hot-season mean, cloud masked) + hotspot labels
    surface     NDVI, NDBI, MNDWI, albedo, ESA WorldCover land cover
    terrain     Copernicus DEM elevation, slope
    urban form  GHSL building height, built-up fraction, population   (+ OSM layers with --osm)
    context     150 m / 450 m neighbourhood means, distance to water / green
    weather     city-level conditions at Landsat overpass time (Open-Meteo archive = ERA5-based, no key)

and then stacks all cities into one table with city / climate / spatial-block columns,
ready for leave-one-city-out and spatial-block cross-validation.

    python build_dataset.py                       # all cities in CITIES
    python build_dataset.py --cities Pune Delhi   # a subset
    python build_dataset.py --half 8 --cities Pune --out data_test   # quick smoke test
    python build_dataset.py --osm                 # add OSM buildings / water / green / roads (slow)
    python build_dataset.py --workers 3           # build 3 cities in parallel (use cores-1)

Output (default ./data_multi)
    cities/<City>/aligned.nc         all layers, (y, x)
    cities/<City>/table.csv.gz       sub-sampled pixel table for that city
    all_cities_training.csv.gz       every city stacked
Runs are resumable: finished cities are skipped unless --force.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt, uniform_filter

# ---------------------------------------------------------------- config
# name: (lon, lat, Koppen-ish climate, max cloud % for Landsat scenes)
CITIES = {
    "Pune":        (73.856, 18.520, "semi-arid",          10),
    "Delhi":       (77.209, 28.614, "semi-arid/subtrop",  15),
    "Jaipur":      (75.787, 26.912, "semi-arid",          10),
    "Ahmedabad":   (72.571, 23.023, "semi-arid",          10),
    "Nagpur":      (79.088, 21.146, "tropical savanna",   10),
    "Hyderabad":   (78.487, 17.385, "semi-arid",          10),
    "Bengaluru":   (77.594, 12.972, "tropical savanna",   20),
    "Chennai":     (80.270, 13.083, "tropical coastal",   15),
    "Mumbai":      (72.878, 19.076, "tropical coastal",   15),
    "Kolkata":     (88.364, 22.572, "tropical humid",     20),
    "Lucknow":     (80.946, 26.847, "humid subtropical",  10),
    "Bhubaneswar": (85.824, 20.296, "tropical savanna",   15),
    "Guwahati":    (91.736, 26.145, "humid subtropical",  25),
    "Chandigarh":  (76.779, 30.733, "humid subtropical",  10),
    # ---- added to widen climate / terrain coverage for the combined model
    "Shimla":            (77.171, 31.105, "humid subtropical (hill)", 20),
    "Dehradun":          (78.032, 30.316, "humid subtropical",        15),
    "Ranchi":            (85.309, 23.344, "humid subtropical",        15),
    "Shillong":          (91.883, 25.578, "humid subtropical (hill)", 30),
    "Visakhapatnam":     (83.218, 17.687, "tropical coastal",         15),
    "Surat":             (72.831, 21.170, "tropical coastal",         15),
    "Kochi":             (76.267,  9.931, "tropical coastal",         25),
    "Thiruvananthapuram": (76.953, 8.524, "tropical coastal",         25),
    "Mangaluru":         (74.856, 12.914, "tropical coastal",         20),
    "Jodhpur":           (73.024, 26.238, "arid",                     10),
    "Indore":            (75.857, 22.719, "semi-arid",                10),
    "Bhopal":            (77.413, 23.259, "semi-arid",                10),
    "Amritsar":          (74.873, 31.634, "semi-arid/subtrop",        10),
    "Ludhiana":          (75.857, 30.901, "semi-arid/subtrop",        10),
    "Patna":             (85.144, 25.594, "humid subtropical",        15),
    "Raipur":            (81.630, 21.251, "tropical savanna",         15),
    "Coimbatore":        (76.956, 11.017, "tropical savanna",         20),
    "Vadodara":          (73.181, 22.307, "semi-arid",                10),
    "Kanpur":            (80.332, 26.449, "humid subtropical",        10),
    # ---- batch 3: reaches 50 cities
    "Srinagar":          (74.797, 34.083, "temperate (hill)",         20),
    "Jammu":             (74.857, 32.726, "humid subtropical",        10),
    "Agartala":          (91.280, 23.831, "humid subtropical",        25),
    "Imphal":            (93.937, 24.817, "humid subtropical (hill)", 25),
    "Puducherry":        (79.808, 11.934, "tropical coastal",         20),
    "Panaji":            (73.827, 15.499, "tropical coastal",         25),
    "Madurai":           (78.119,  9.925, "tropical savanna",         15),
    "Bikaner":           (73.311, 28.022, "arid",                     10),
    "Ajmer":             (74.640, 26.450, "semi-arid",                10),
    "Udaipur":           (73.712, 24.585, "semi-arid",                10),
    "Rajkot":            (70.802, 22.304, "semi-arid",                10),
    "Nashik":            (73.789, 19.998, "semi-arid",                10),
    "Aurangabad":        (75.343, 19.877, "semi-arid",                10),
    "Agra":              (78.008, 27.177, "semi-arid/subtrop",        10),
    "Varanasi":          (82.973, 25.317, "humid subtropical",        10),
    "Jabalpur":          (79.986, 23.181, "humid subtropical",        10),
    "Gwalior":           (78.182, 26.218, "semi-arid/subtrop",        10),
}

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
GHSL_LAYERS = {   # key: (product folder, file stem)
    "built_h": ("GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss",
                "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss_V1_0"),
    "built_s": ("GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss",
                "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0"),
    "pop":     ("GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_3ss",
                "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0"),
}
RES = 30
BLOCK_M = 3000          # spatial-CV block size


# ---------------------------------------------------------------- geometry helpers
def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def city_bbox(lon, lat, half_km):
    dlat = half_km / 110.54
    dlon = half_km / (111.32 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]       # W, S, E, N


def ghsl_tiles(bbox):
    W, S, E, N = bbox
    rows = {int((90 - lat) // 10) + 1 for lat in (S, N)}
    cols = {int((lon + 180) // 10) + 1 for lon in (W, E)}
    return [f"R{r}_C{c}" for r in sorted(rows) for c in sorted(cols)]


def catalog():
    import planetary_computer as pc
    from pystac_client import Client
    return Client.open(PC_STAC, modifier=pc.sign_inplace)


# ---------------------------------------------------------------- 1. Landsat LST + indices
def get_landsat(cat, bbox, crs, years, months, cloud_max, max_scenes):
    from odc.stac import configure_rio, load
    configure_rio(cloud_defaults=True)

    items = list(cat.search(
        collections=["landsat-c2-l2"], bbox=bbox,
        datetime=f"{years[0]}-01-01/{years[1]}-12-31",
        query={"eo:cloud_cover": {"lt": cloud_max},
               "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    items = [i for i in items if i.datetime.month in months]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    items = items[:max_scenes]
    print(f"  Landsat scenes: {len(items)}")
    if len(items) < 4:
        raise RuntimeError("too few clear Landsat scenes; widen --months/--years or raise cloud limit")

    # Load in small batches and re-sign the URLs right before each one: Planetary Computer's SAS tokens
    # expire after ~1 h, and a slow city used to lose its remaining scenes ("not recognized as being
    # in a supported file format") with fail_on_error=False hiding it. Means are rebuilt from sums/counts.
    import planetary_computer as pc
    BATCH = 8
    names = ["LST", "NDVI", "NDBI", "MNDWI", "ALBEDO"]
    tot = {k: 0 for k in names}
    cnt = {k: 0 for k in names}
    sumsq, all_dates = 0, []
    for i in range(0, len(items), BATCH):
        batch = []
        for it in items[i:i + BATCH]:
            it = it.clone()
            for a in it.assets.values():
                a.href = a.href.split("?")[0]                 # drop the old (possibly expired) token
            pc.sign_inplace(it)
            batch.append(it)
        raw = load(batch, bands=["lwir11", "qa_pixel", "blue", "green", "red", "nir08", "swir16", "swir22"],
                   bbox=bbox, crs=f"EPSG:{crs}", resolution=RES, groupby="solar_day",
                   chunks={"x": 512, "y": 512}, fail_on_error=False)
        bad = (raw["qa_pixel"].astype("uint16") & 0b11111) != 0      # fill, dilated cloud, cirrus, cloud, shadow
        sr = lambda b: (raw[b].where(raw[b] > 0) * 0.0000275 - 0.2).where(~bad)   # noqa: E731
        blue, green, red, nir, sw1, sw2 = (sr(b) for b in ["blue", "green", "red", "nir08", "swir16", "swir22"])
        lst = (raw["lwir11"].where(raw["lwir11"] > 0) * 0.00341802 + 149.0 - 273.15).where(~bad)
        layers = {
            "LST": lst,
            "NDVI": (nir - red) / (nir + red),
            "NDBI": (sw1 - nir) / (sw1 + nir),
            "MNDWI": (green - sw1) / (green + sw1),
            "ALBEDO": 0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * sw1 + 0.072 * sw2 - 0.0018,
        }
        part = xr.Dataset({**{f"s_{k}": v.astype("float64").sum("time", skipna=True) for k, v in layers.items()},
                           **{f"n_{k}": v.count("time") for k, v in layers.items()},
                           "q_LST": (lst.astype("float64") ** 2).sum("time", skipna=True)}).compute()
        for k in names:
            tot[k] = tot[k] + part[f"s_{k}"]
            cnt[k] = cnt[k] + part[f"n_{k}"]
        sumsq = sumsq + part["q_LST"]
        all_dates.extend(raw.time.values)

    mean = {k: (tot[k] / cnt[k].where(cnt[k] > 0)).astype("float32") for k in names}
    n = cnt["LST"].where(cnt["LST"] > 0)
    lst_std = ((sumsq / n) - (tot["LST"] / n) ** 2).clip(min=0) ** 0.5
    stack = xr.Dataset({**mean, "LST_std": lst_std.astype("float32"), "n_obs": cnt["LST"]})
    dates = pd.to_datetime(pd.Series(all_dates)).dt.normalize().drop_duplicates().sort_values()
    dates = pd.DatetimeIndex(dates)
    stack = stack.where(stack["n_obs"] >= min(5, len(items)))
    stack = stack.rio.write_crs(f"EPSG:{crs}")
    return stack, dates


# ---------------------------------------------------------------- 2. DEM + land cover
def get_dem_and_landcover(cat, bbox, stack):
    from odc.stac import load
    out = {}
    dem_items = list(cat.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    dem = load(dem_items, bands=["data"], like=stack, resampling="bilinear", chunks={}).compute()
    out["ELEV"] = dem["data"].where(dem["data"] > -500).mean("time")

    wc_items = list(cat.search(collections=["esa-worldcover"], bbox=bbox, datetime="2021-01-01/2021-12-31").items())
    wc = load(wc_items, bands=["map"], like=stack, resampling="mode", chunks={}).compute()
    out["LANDCOVER"] = wc["map"].max("time").astype("float32")
    return out


# ---------------------------------------------------------------- 3. GHSL
def _download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def get_ghsl(bbox, tmpl, cache):
    """GHSL 100 m layers -> master grid. Height is intensive; built-up m2 and population are extensive."""
    import rioxarray as rxr
    from rioxarray.merge import merge_arrays
    W, S, E, N = bbox
    out = {}
    for key, (folder, stem) in GHSL_LAYERS.items():
        parts = []
        for tile in ghsl_tiles(bbox):
            tif = cache / f"{stem}_{tile}.tif"
            if not tif.exists():
                url = f"{GHSL_BASE}/{folder}/V1-0/tiles/{stem}_{tile}.zip"
                zpath = cache / f"{stem}_{tile}.{os.getpid()}.zip"      # per-process temp names: parallel-safe
                tmpdir = cache / f"_tmp_{os.getpid()}"
                try:
                    print(f"  GHSL {key} {tile}: downloading")
                    _download(url, zpath)
                    with zipfile.ZipFile(zpath) as z:
                        z.extract(f"{stem}_{tile}.tif", tmpdir)
                    os.replace(tmpdir / f"{stem}_{tile}.tif", tif)          # atomic: tif is never half-written
                    zpath.unlink()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! GHSL {key} {tile} failed: {e}")
                    continue
            da = rxr.open_rasterio(tif, masked=True).squeeze(drop=True).rio.clip_box(W - .01, S - .01, E + .01, N + .01)
            parts.append(da)
        if not parts:
            continue
        da = merge_arrays(parts) if len(parts) > 1 else parts[0]
        if key == "built_h":
            out["GHSL_HEIGHT"] = da.rio.reproject_match(tmpl, resampling=Resampling.bilinear)
            continue
        lat = float(da.y.mean())
        cell_m2 = (abs(float(da.x[1] - da.x[0])) * 111_320 * np.cos(np.radians(lat))
                   * abs(float(da.y[1] - da.y[0])) * 110_540)
        dens = (da / cell_m2).rio.reproject_match(tmpl, resampling=Resampling.bilinear)
        if key == "built_s":
            out["GHSL_BUILT_FRAC"] = dens.clip(0, 1)
        else:
            out["GHSL_POP"] = dens * RES * RES                    # people per 30 m cell
    return out


# ---------------------------------------------------------------- 4. OSM (optional)
def get_osm(bbox, stack):
    import geopandas as gpd
    import osmnx as ox
    from rasterio import features
    ox.settings.use_cache = True
    ox.settings.timeout = 300
    SUB = 3
    t = stack.rio.transform()
    ft = t * t.scale(1 / SUB, 1 / SUB)
    shape = (stack.sizes["y"] * SUB, stack.sizes["x"] * SUB)
    crs = stack.rio.crs

    def tiles(step=0.1):
        W, S, E, N = bbox
        xs = np.arange(W, E, step).tolist() + [E]
        ys = np.arange(S, N, step).tolist() + [N]
        return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]

    def fetch(tags):
        parts = []
        for bb in tiles():
            try:
                g = ox.features_from_bbox(bb, tags=tags)
                g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])]
                if len(g):
                    parts.append(g[["geometry"] + [c for c in ("height", "building:levels") if c in g]].reset_index(drop=True))
            except Exception:  # noqa: BLE001  (empty tile)
                pass
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(crs) if parts else None

    def frac(gdf, vals=1.0):
        v = np.broadcast_to(vals, len(gdf))
        a = features.rasterize(zip(gdf.geometry, v), out_shape=shape, transform=ft, fill=0, dtype="float32")
        return xr.DataArray(a.reshape(stack.sizes["y"], SUB, stack.sizes["x"], SUB).mean(axis=(1, 3)),
                            coords={"y": stack.y, "x": stack.x}, dims=("y", "x"))

    out = {}
    b = fetch({"building": True})
    if b is not None:
        out["BLDG_FRAC"] = frac(b)
        num = lambda x: float(m.group()) if (m := re.search(r"[\d.]+", str(x))) else np.nan   # noqa: E731
        h = b["height"].map(num) if "height" in b else pd.Series(np.nan, index=b.index)
        lv = b["building:levels"].map(num) if "building:levels" in b else pd.Series(np.nan, index=b.index)
        b["h"] = np.where(h.notna(), h, lv * 3.0)
        bh = b[b["h"].notna()]
        if len(bh):
            cover = frac(bh)
            out["OSM_HEIGHT"] = (frac(bh, bh["h"].values) / cover.where(cover > 0))
    for name, tags in (("WATER_FRAC", {"natural": "water", "waterway": ["river", "canal"]}),
                       ("GREEN_FRAC", {"leisure": ["park", "garden"], "landuse": ["grass", "forest", "meadow"],
                                       "natural": ["wood", "scrub", "grassland"]})):
        g = fetch(tags)
        if g is not None:
            out[name] = frac(g)
    return out


# ---------------------------------------------------------------- 5. weather at overpass time
def get_weather(lon, lat, dates):
    """City-level conditions on the Landsat dates at ~10-11 h IST (overpass), from Open-Meteo's ERA5 archive."""
    d0, d1 = (dates.min() - pd.Timedelta(days=30)).date(), dates.max().date()
    params = {"latitude": lat, "longitude": lon, "start_date": str(d0), "end_date": str(d1),
              "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,precipitation",
              "timezone": "Asia/Kolkata"}
    for attempt in range(5):                       # the free API times out / rate-limits now and then
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=params)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 4:
                raise
            wait = 10 * (attempt + 1)
            print(f"  weather retry {attempt + 1}/4 in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")
    daily_p = h["precipitation"].resample("D").sum()
    pre7 = daily_p.rolling(7, min_periods=1).sum().shift(1)       # rain in the 7 days before each day
    pre30 = daily_p.rolling(30, min_periods=1).sum().shift(1)
    sel = h[(h.index.normalize().isin(dates)) & (h.index.hour.isin([10, 11]))]
    return {
        "MET_T2M": sel["temperature_2m"].mean(),
        "MET_RH": sel["relative_humidity_2m"].mean(),
        "MET_WIND": sel["wind_speed_10m"].mean(),
        "MET_SW": sel["shortwave_radiation"].mean(),
        "MET_RAIN7": pre7.reindex(dates).mean(),
        "MET_RAIN30": pre30.reindex(dates).mean(),
        "MET_T2M_SEASON_MAX": h["temperature_2m"].resample("D").max().mean(),
    }


# ---------------------------------------------------------------- 6. derived + labels
def add_derived(ds):
    def nbr(a, px):
        v = np.nan_to_num(a.values, nan=0.0)
        return xr.DataArray(uniform_filter(v, size=px, mode="nearest").astype("float32"), coords=a.coords, dims=a.dims)

    for src, tag in (("NDVI", "NDVI"), ("GHSL_BUILT_FRAC", "BUILT"), ("BLDG_FRAC", "BLDG"), ("GHSL_HEIGHT", "HEIGHT")):
        if src in ds:
            ds[f"{tag}_nbr150"] = nbr(ds[src], 5)
            ds[f"{tag}_nbr450"] = nbr(ds[src], 15)

    water = (ds["MNDWI"].values > 0.1) | (ds["LANDCOVER"].values == 80)
    if "WATER_FRAC" in ds:
        water |= ds["WATER_FRAC"].values > 0.3
    water &= ~np.isnan(ds["LST"].values) | True
    ds["DIST_WATER_m"] = xr.DataArray((distance_transform_edt(~water) * RES).astype("float32") if water.any()
                                      else np.full(water.shape, 5000, "float32"), coords=ds["LST"].coords, dims=("y", "x"))
    green = np.nan_to_num(ds["NDVI"].values, nan=0) > 0.4
    ds["DIST_GREEN_m"] = xr.DataArray((distance_transform_edt(~green) * RES).astype("float32") if green.any()
                                      else np.full(green.shape, 5000, "float32"), coords=ds["LST"].coords, dims=("y", "x"))
    gy, gx = np.gradient(ds["ELEV"].values.astype("float64"), RES)
    ds["SLOPE_deg"] = xr.DataArray(np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32"),
                                   coords=ds["LST"].coords, dims=("y", "x"))
    return ds


def add_labels(ds):
    """City-relative targets, so cities with different baseline temperatures are comparable."""
    lst = ds["LST"].values
    lc = ds["LANDCOVER"].values
    built = ds["GHSL_BUILT_FRAC"].values if "GHSL_BUILT_FRAC" in ds else np.zeros_like(lst)
    urban = (lc == 50) | (np.nan_to_num(built) >= 0.15)
    rural = np.isin(lc, [30, 40])                                  # grass, cropland reference
    ok = ~np.isnan(lst)
    mu, sd = np.nanmean(lst), np.nanstd(lst)
    ds["LST_z"] = xr.DataArray(((lst - mu) / sd).astype("float32"), coords=ds["LST"].coords, dims=("y", "x"))
    ref = np.nanmean(lst[rural & ok]) if (rural & ok).sum() > 100 else mu
    ds["SUHI"] = ds["LST"] - ref                                   # surface UHI vs rural reference
    ds["IS_URBAN"] = xr.DataArray(urban.astype("float32"), coords=ds["LST"].coords, dims=("y", "x"))
    thr = np.nanpercentile(lst[urban & ok], 90) if (urban & ok).sum() > 100 else np.nanpercentile(lst, 90)
    ds["HOTSPOT"] = xr.DataArray(((lst >= thr) & urban & ok).astype("float32"), coords=ds["LST"].coords, dims=("y", "x"))
    ds.attrs["hotspot_threshold_C"] = float(thr)
    return ds


# ---------------------------------------------------------------- per-city driver
def build_city(name, args, cat, out_root):
    lon, lat, climate, cloud_max = CITIES[name]
    cdir = out_root / "cities" / name
    if (cdir / "table.csv.gz").exists() and not args.force:
        print(f"[{name}] done, skipping (use --force)")
        return
    cdir.mkdir(parents=True, exist_ok=True)
    bbox = city_bbox(lon, lat, args.half)
    crs = utm_epsg(lon, lat)
    print(f"\n=== {name} | {climate} | EPSG:{crs} | bbox {np.round(bbox, 3).tolist()}")

    stack, dates = get_landsat(cat, bbox, crs, args.years, args.months, cloud_max, args.max_scenes)
    ds = stack.drop_vars("spatial_ref", errors="ignore").rio.write_crs(f"EPSG:{crs}")
    tmpl = ds["LST"]

    layers = {}
    layers.update(get_dem_and_landcover(cat, bbox, stack))
    layers.update(get_ghsl(bbox, tmpl, out_root / "_cache" / "ghsl"))
    if args.osm:
        try:
            layers.update(get_osm(bbox, ds))
        except Exception as e:  # noqa: BLE001
            print(f"  !! OSM failed: {e}")
    for k, v in layers.items():
        ds[k] = v.reindex_like(tmpl, method="nearest", tolerance=RES) if v.shape != tmpl.shape else v.assign_coords(
            y=tmpl.y, x=tmpl.x)

    ds = add_derived(ds)
    ds = add_labels(ds)
    ds = ds.drop_vars("spatial_ref", errors="ignore")
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")

    try:
        met = get_weather(lon, lat, dates)
    except Exception as e:  # noqa: BLE001
        print(f"  !! weather failed: {e}")
        met = {}
    ds.attrs.update({"city": name, "climate": climate, "epsg": crs, "n_scenes": len(dates),
                     **{k: float(v) for k, v in met.items()}})
    ds.to_netcdf(cdir / "aligned.nc")

    # ---- pixel table (strided to cut spatial autocorrelation + size)
    sub = ds.isel(y=slice(0, None, args.stride), x=slice(0, None, args.stride))
    df = sub.to_dataframe().reset_index()
    df = df[df["LST"].notna() & df["NDVI"].notna()].copy()
    from pyproj import Transformer
    tr = Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True)
    df["lon"], df["lat"] = tr.transform(df["x"].values, df["y"].values)
    df.insert(0, "city", name)
    df.insert(1, "climate", climate)
    df["block_id"] = (name + "_" + (df["x"] // BLOCK_M).astype(int).astype(str) + "_"
                      + (df["y"] // BLOCK_M).astype(int).astype(str))
    for k, v in met.items():
        df[k] = v
    df.to_csv(cdir / "table.csv.gz", index=False)
    (cdir / "meta.json").write_text(json.dumps(ds.attrs, indent=2))
    print(f"  {name}: {len(df):,} rows | hotspot threshold {ds.attrs['hotspot_threshold_C']:.1f} C | "
          f"scenes {len(dates)}")


class _Prefix:
    """Tag every output line with the city so parallel logs stay readable."""
    def __init__(self, tag, stream):
        self.tag, self.stream, self.buf = tag, stream, ""

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.stream.write(f"[{self.tag}] {line}\n")
        self.stream.flush()

    def flush(self):
        self.stream.flush()


def _run_city(name, args, out_root, threads):
    """Entry point of one worker process: builds a single city."""
    import dask
    dask.config.set(scheduler="threads", num_workers=threads)     # share the cores between workers
    if args.workers > 1:
        sys.stdout = _Prefix(name, sys.__stdout__)
    try:
        build_city(name, args, catalog(), out_root)
    finally:
        sys.stdout = sys.__stdout__
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["all"])
    ap.add_argument("--half", type=float, default=20, help="half box size in km (box = 2*half)")
    ap.add_argument("--years", nargs=2, type=int, default=[2022, 2025])
    ap.add_argument("--months", nargs="*", type=int, default=[3, 4, 5], help="hot, mostly cloud-free season")
    ap.add_argument("--max-scenes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth pixel in the stacked table")
    ap.add_argument("--osm", action="store_true", help="add OSM buildings/water/green (slow for big cities)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="data_multi")
    ap.add_argument("--dask-threads", type=int, default=0,
                    help="threads per city (default 0 = cores // workers). Reads are network-bound, so 2-3 keeps cores busy")
    ap.add_argument("--workers", type=int, default=1,
                    help="cities built in parallel (one process each). Try cores-1; each needs a few GB of RAM")
    args = ap.parse_args()

    names = list(CITIES) if args.cities == ["all"] else args.cities
    bad = [n for n in names if n not in CITIES]
    if bad:
        raise SystemExit(f"unknown cities {bad}; choose from {list(CITIES)}")
    out_root = Path(args.out)
    workers = max(1, min(args.workers, len(names)))
    threads = args.dask_threads or max(1, (os.cpu_count() or 1) // workers)
    print(f"{len(names)} cities | {workers} parallel worker(s) | {threads} dask thread(s) each")
    failed = []
    if workers == 1:
        for n in names:
            try:
                _run_city(n, args, out_root, threads)
            except Exception as e:  # noqa: BLE001  one bad city must not kill a long run
                print(f"!! {n} failed: {type(e).__name__}: {e}")
                failed.append(n)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_run_city, n, args, out_root, threads): n for n in names}
            for f in as_completed(futs):
                n = futs[f]
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    print(f"!! {n} failed: {type(e).__name__}: {e}")
                    failed.append(n)

    tabs = sorted((out_root / "cities").glob("*/table.csv.gz"))
    if tabs:
        alldf = pd.concat((pd.read_csv(t) for t in tabs), ignore_index=True)
        alldf.to_csv(out_root / "all_cities_training.csv.gz", index=False)
        print(f"\nStacked {len(tabs)} cities -> {len(alldf):,} rows -> {out_root / 'all_cities_training.csv.gz'}")
    if failed:
        print("Failed cities (rerun to retry):", failed)


if __name__ == "__main__":
    main()
