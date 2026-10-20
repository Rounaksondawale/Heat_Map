"""
Download the datasets the Pune UHI project is still missing.

    python get_missing_data.py              # everything
    python get_missing_data.py ghsl worldpop  # only some steps

Steps: ghsl, worldpop, sentinel2, osm, era5land
Outputs go to ./data. Each step is independent and skipped if its output exists.

NOT automated (need an account or manual export), see print_manual_steps():
    CPCB station data, ECOSTRESS LST.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------- config
BBOX = (73.65, 18.30, 74.10, 18.75)      # W, S, E, N  (same as the Landsat box)
CRS_UTM = "EPSG:32643"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

# GHSL 3-arcsec (~100 m) tile covering Pune. Grid is 10 deg x 10 deg:
# row = floor((90 - lat)/10) + 1 = 8, col = floor((lon + 180)/10) + 1 = 26
GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
GHSL_TILE = "R8_C26"
GHSL_LAYERS = {
    # name: product folder / file stem
    "built_h": "GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
               "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip",
    "built_s": "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
               "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip",
    "pop":     "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
               "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip",
}

WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif")

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_DATES = "2023-03-01/2024-05-31"
S2_MONTHS = (3, 4, 5)                    # hot dry season, matches the Landsat window
S2_MAX_CLOUD = 20

# ERA5-Land hourly for hot-season days (needs ~/.cdsapirc, same as test.ipynb)
ERA5_YEAR = "2024"
ERA5_MONTHS = ["04", "05"]
ERA5_AREA = [18.75, 73.65, 18.30, 74.10]  # N, W, S, E

OSM_TILES = 4                            # split the bbox into NxN tiles to keep Overpass happy


# ---------------------------------------------------------------- helpers
def done(path):
    if Path(path).exists():
        print(f"  [skip] {path} exists")
        return True
    return False


def clip_geotiff(src_path_or_url, dst, bbox=BBOX):
    """Window-read a (remote) GeoTIFF to the bbox and save as a small GeoTIFF."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(src_path_or_url) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=win)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2],
                       transform=src.window_transform(win), compress="deflate")
    with rasterio.open(dst, "w", **profile) as out:
        out.write(data)
    print(f"  saved {dst}  shape={data.shape}")


# ---------------------------------------------------------------- 1. GHSL
def get_ghsl():
    """GHS-BUILT-H (building height), GHS-BUILT-S (built-up m2/cell), GHS-POP (people/cell)."""
    print("GHSL ...")
    tmp = OUT / "_ghsl_tmp"
    tmp.mkdir(exist_ok=True)
    for name, pattern in GHSL_LAYERS.items():
        dst = OUT / f"ghsl_{name}_pune.tif"
        if done(dst):
            continue
        url = f"{GHSL_BASE}/{pattern.format(tile=GHSL_TILE)}"
        try:
            print(f"  downloading {url}")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                tif = next(n for n in z.namelist() if n.lower().endswith(".tif"))
                z.extract(tif, tmp)
            clip_geotiff(str(tmp / tif), dst)
        except Exception as e:  # noqa: BLE001
            print(f"  !! GHSL {name} failed: {e}\n"
                  f"     Check the tile id / path at https://human-settlement.emergency.copernicus.eu/download.php")


# ---------------------------------------------------------------- 2. WorldPop
def get_worldpop():
    """WorldPop 100 m population (second source, cross-check vs GHS-POP)."""
    print("WorldPop ...")
    dst = OUT / "worldpop_pop_2020_pune.tif"
    if done(dst):
        return
    try:
        clip_geotiff(WORLDPOP_URL, dst)      # HTTP range reads, avoids pulling the whole of India
    except Exception as e:  # noqa: BLE001
        print(f"  !! WorldPop failed: {e}\n     Download ind_ppp_2020.tif manually from worldpop.org and clip it.")


# ---------------------------------------------------------------- 3. Sentinel-2
def get_sentinel2():
    """10 m median composite of NDVI / NDBI / MNDWI / NDWI for the hot season, from Planetary Computer."""
    print("Sentinel-2 ...")
    dst = OUT / "s2_indices_pune.nc"
    if done(dst):
        return
    import planetary_computer as pc
    import xarray as xr
    from odc.stac import configure_rio, load
    from pystac_client import Client

    configure_rio(cloud_defaults=True)
    catalog = Client.open(PC_STAC, modifier=pc.sign_inplace)
    items = list(catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=S2_DATES,
        query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}}).items())
    items = [i for i in items if i.datetime.month in S2_MONTHS]
    print(f"  scenes: {len(items)}")
    if not items:
        return

    raw = load(items, bands=["B02", "B03", "B04", "B08", "B11", "SCL"],
               bbox=BBOX, crs=CRS_UTM, resolution=10, groupby="solar_day",
               chunks={"x": 1024, "y": 1024}, resampling={"SCL": "nearest", "*": "bilinear"})
    # SCL: 3 cloud shadow, 8/9 cloud, 10 cirrus, 1 saturated, 0 no data
    bad = raw["SCL"].isin([0, 1, 3, 8, 9, 10])
    sr = lambda b: (raw[b].where(raw[b] > 0) / 10000.0).where(~bad)   # noqa: E731
    green, red, nir, sw1 = sr("B03"), sr("B04"), sr("B08"), sr("B11")

    idx = xr.Dataset({
        "S2_NDVI":  ((nir - red) / (nir + red)).median("time"),
        "S2_NDBI":  ((sw1 - nir) / (sw1 + nir)).median("time"),
        "S2_MNDWI": ((green - sw1) / (green + sw1)).median("time"),
        "S2_NDWI":  ((green - nir) / (green + nir)).median("time"),
        "S2_nobs":  (~bad).sum("time"),
    }).compute()
    idx.drop_vars("spatial_ref", errors="ignore").to_netcdf(dst)
    print(f"  saved {dst}")


# ---------------------------------------------------------------- 4. OSM
def get_osm():
    """Whole-study-area OSM: buildings (+heights), roads, water, parks/green, landuse. Tiled for size."""
    print("OSM ...")
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd

    ox.settings.use_cache = True
    ox.settings.timeout = 300
    layers = {
        "buildings": {"building": True},
        "water":     {"natural": "water", "waterway": ["river", "canal", "stream"]},
        "green":     {"leisure": ["park", "garden", "playground", "pitch"],
                      "landuse": ["grass", "forest", "meadow", "recreation_ground", "village_green"],
                      "natural": ["wood", "scrub", "grassland"]},
        "landuse":   {"landuse": True},
    }
    W, S, E, N = BBOX
    xs = np.linspace(W, E, OSM_TILES + 1)
    ys = np.linspace(S, N, OSM_TILES + 1)

    for name, tags in layers.items():
        dst = OUT / f"osm_{name}_pune.gpkg"
        if done(dst):
            continue
        parts = []
        for i in range(OSM_TILES):
            for j in range(OSM_TILES):
                bb = (xs[i], ys[j], xs[i + 1], ys[j + 1])   # osmnx>=2: (W,S,E,N)
                try:
                    g = ox.features_from_bbox(bb, tags=tags)
                    g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])]
                    if len(g):
                        parts.append(g.reset_index(drop=True))
                except Exception as e:  # noqa: BLE001  (empty tiles raise InsufficientResponseError)
                    print(f"  {name} tile {i},{j}: {type(e).__name__}")
        if not parts:
            continue
        g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(CRS_UTM)
        keep = [c for c in ("geometry", "building", "height", "building:levels", "natural",
                            "leisure", "landuse", "waterway", "name") if c in g.columns]
        g = g[keep]
        for c in g.columns.drop("geometry"):
            g[c] = g[c].astype(str)                      # mixed types break GPKG writes
        g["area_m2"] = g.geometry.area
        g.to_file(dst, driver="GPKG")
        print(f"  saved {dst}  ({len(g)} features)")

    # roads as a graph -> edges
    dst = OUT / "osm_roads_pune.gpkg"
    if not done(dst):
        try:
            G = ox.graph_from_bbox(BBOX, network_type="drive")
            edges = ox.graph_to_gdfs(G, nodes=False).to_crs(CRS_UTM)
            keep = [c for c in ("geometry", "highway", "lanes", "length") if c in edges.columns]
            e = edges[keep].copy()
            for c in e.columns.drop("geometry"):
                e[c] = e[c].astype(str)
            e.to_file(dst, driver="GPKG")
            print(f"  saved {dst}  ({len(e)} edges)")
        except Exception as e:  # noqa: BLE001
            print(f"  !! roads failed: {e}")


# ---------------------------------------------------------------- 5. ERA5-Land hot season
def get_era5land():
    """Hourly ERA5-Land for April-May of ERA5_YEAR. Needs a CDS account and ~/.cdsapirc."""
    print("ERA5-Land ...")
    dst = OUT / f"era5land_{ERA5_YEAR}_{'_'.join(ERA5_MONTHS)}.grib"
    if done(dst):
        return
    import cdsapi
    req = {
        "variable": ["2m_temperature", "2m_dewpoint_temperature",
                     "10m_u_component_of_wind", "10m_v_component_of_wind",
                     "surface_net_solar_radiation", "surface_net_thermal_radiation",
                     "surface_sensible_heat_flux", "surface_latent_heat_flux",
                     "skin_temperature"],
        "year": [ERA5_YEAR],
        "month": ERA5_MONTHS,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": ERA5_AREA,
        "data_format": "grib",
        "download_format": "unarchived",
    }
    try:
        cdsapi.Client().retrieve("reanalysis-era5-land", req, str(dst))
        print(f"  saved {dst}")
    except Exception as e:  # noqa: BLE001
        print(f"  !! ERA5-Land failed: {e}")


# ---------------------------------------------------------------- manual
def print_manual_steps():
    print("""
MANUAL DOWNLOADS (cannot be scripted without your credentials)

1. CPCB station meteorology (ground truth for T, RH, wind)
   - https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/data
   - Select Maharashtra > Pune, pick stations, parameters AT / RH / WS / WD,
     1-hour averages, dates covering your study period, export CSV.
   - Save into data/cpcb/. Note station lat/lon from the station list.
   - Fallback: IMD Pune (Shivajinagar) hourly data, or Open-Meteo / Meteostat for Pune.

2. ECOSTRESS LST (70 m, varying overpass times)
   - Free NASA Earthdata account: https://urs.earthdata.nasa.gov
   - Product ECO_L2T_LSTE (tiled) via https://search.earthdata.nasa.gov, bbox as above,
     or in Python:  pip install earthaccess
         import earthaccess; earthaccess.login()
         r = earthaccess.search_data(short_name="ECO_L2T_LSTE", bounding_box=(73.65,18.30,74.10,18.75),
                                     temporal=("2023-03-01","2024-05-31"))
         earthaccess.download(r, "data/ecostress")
""")


STEPS = {"ghsl": get_ghsl, "worldpop": get_worldpop, "sentinel2": get_sentinel2,
         "osm": get_osm, "era5land": get_era5land}

if __name__ == "__main__":
    # worldpop is opt-in: server has no range reads, so it is a 1.7 GB full-India download
    chosen = sys.argv[1:] or [s for s in STEPS if s != "worldpop"]
    for s in chosen:
        if s not in STEPS:
            sys.exit(f"unknown step '{s}', choose from {list(STEPS)}")
        STEPS[s]()
    print_manual_steps()
