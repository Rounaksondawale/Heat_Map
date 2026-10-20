"""
Align every dataset onto ONE grid (the 30 m EPSG:32643 grid of pune_features.nc,
the one the Landsat LST target lives on) and write a model-ready table.

    python align_data.py

Outputs
    data/aligned_features.nc   all layers as (y, x) rasters
    data/training_table.csv.gz one row per valid pixel (features + LST target + coordinates)

Resampling rules
    continuous (NDVI, height, ...)  -> average when going finer->coarser, bilinear coarser->finer
    categorical (land cover)        -> nearest / mode
    extensive (population, built-up m2 per cell) -> converted to per-m2 density first, then resampled
Layers whose files are missing are skipped with a message, so you can rerun after more downloads.
"""
import re
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio import features
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt, uniform_filter

DATA = Path("data")
MASTER = DATA / "pune_features.nc"
CRS = "EPSG:32643"
RES = 30
SUB = 3                                  # rasterise vector data at 10 m, then average to 30 m


# ---------------------------------------------------------------- master grid
def load_master():
    ds = xr.open_dataset(MASTER).load()
    ds = ds.drop_vars("time", errors="ignore")
    ds = ds.rio.write_crs(CRS)
    return ds


def match(da, master, how):
    """Reproject a rioxarray DataArray onto the master grid."""
    return da.rio.reproject_match(master, resampling=how)


def read_tif(path, crs_fallback=None):
    da = rxr.open_rasterio(path, masked=True).squeeze(drop=True)
    if da.rio.crs is None and crs_fallback:
        da = da.rio.write_crs(crs_fallback)
    return da


# ---------------------------------------------------------------- raster layers
def add_ghsl(out, tmpl):
    """GHSL: 100 m, EPSG:4326. Height is intensive; built-up m2 and population are extensive."""
    p = DATA / "ghsl_built_h_pune.tif"
    if p.exists():
        out["GHSL_HEIGHT"] = match(read_tif(p), tmpl, Resampling.bilinear)
    # built-up m2 per cell -> fraction (cell area differs with latitude, so use actual area)
    for key, name, unit in (("built_s", "GHSL_BUILT_FRAC", "frac"), ("pop", "GHSL_POP_DENS", "pop")):
        p = DATA / f"ghsl_{key}_pune.tif"
        if not p.exists():
            continue
        da = read_tif(p)
        # cell area in m2 of the source 4326 grid at the study latitude
        lat = float(da.y.mean())
        dx = abs(float(da.x[1] - da.x[0])) * 111_320 * np.cos(np.radians(lat))
        dy = abs(float(da.y[1] - da.y[0])) * 110_540
        cell_m2 = dx * dy
        dens = da / cell_m2                               # per m2
        dens = match(dens, tmpl, Resampling.bilinear)
        out[name] = dens.clip(0, 1) if unit == "frac" else dens * (RES * RES)   # pop per 30 m cell
    return out


def add_sentinel2(out, tmpl):
    p = DATA / "s2_indices_pune.nc"
    if not p.exists():
        print("  - Sentinel-2 missing, skipped")
        return out
    s2 = xr.open_dataset(p).rio.write_crs(CRS)
    for v in ("S2_NDVI", "S2_NDBI", "S2_MNDWI", "S2_NDWI"):
        if v in s2:
            out[v] = match(s2[v], tmpl, Resampling.average)     # 10 m -> 30 m
    return out


# ---------------------------------------------------------------- vector layers
def _grid(tmpl):
    """Transform / shape of a SUB-times finer grid aligned with the master."""
    t = tmpl.rio.transform()
    ft = t * t.scale(1 / SUB, 1 / SUB)
    return ft, (tmpl.sizes["y"] * SUB, tmpl.sizes["x"] * SUB)


def _coarsen(arr, tmpl, name):
    a = arr.reshape(tmpl.sizes["y"], SUB, tmpl.sizes["x"], SUB).mean(axis=(1, 3))
    return xr.DataArray(a.astype("float32"), coords={"y": tmpl.y, "x": tmpl.x},
                        dims=("y", "x"), name=name)


def _num(x):
    m = re.search(r"[\d.]+", str(x))
    return float(m.group()) if m else np.nan


def add_osm(out, tmpl):
    import geopandas as gpd
    ft, shape = _grid(tmpl)

    def rast(gdf, vals=1.0, dtype="float32"):
        shapes = ((g, v) for g, v in zip(gdf.geometry, np.broadcast_to(vals, len(gdf))))
        return features.rasterize(shapes, out_shape=shape, transform=ft, fill=0,
                                  dtype=dtype, all_touched=False)

    p = DATA / "osm_buildings_pune.gpkg"
    if p.exists():
        b = gpd.read_file(p).to_crs(CRS)
        b = b[b.geometry.notna() & ~b.geometry.is_empty]
        out["BLDG_FRAC"] = _coarsen(rast(b), tmpl, "BLDG_FRAC")
        h = b["height"].map(_num) if "height" in b else np.nan
        lv = b["building:levels"].map(_num) if "building:levels" in b else np.nan
        b["h"] = np.where(~np.isnan(h), h, lv * 3.0)
        bh = b[b["h"].notna()]
        # mean height over buildings that HAVE a height, weighted by footprint
        hsum = _coarsen(rast(bh, bh["h"].values), tmpl, "hs")
        cover = _coarsen(rast(bh), tmpl, "cv")
        out["OSM_HEIGHT"] = (hsum / cover.where(cover > 0)).astype("float32")
        out["OSM_HEIGHT_COVER"] = cover                      # how much of the pixel has known heights
    else:
        print("  - OSM buildings missing, skipped")

    for key, name in (("water", "WATER_FRAC"), ("green", "GREEN_FRAC")):
        p = DATA / f"osm_{key}_pune.gpkg"
        if p.exists():
            g = gpd.read_file(p).to_crs(CRS)
            out[name] = _coarsen(rast(g), tmpl, name)
        else:
            print(f"  - OSM {key} missing, skipped")

    p = DATA / "osm_roads_pune.gpkg"
    if p.exists():
        r = gpd.read_file(p).to_crs(CRS)
        r["geometry"] = r.geometry.buffer(4)                  # ~8 m wide carriageway
        out["ROAD_FRAC"] = _coarsen(rast(r), tmpl, "ROAD_FRAC")
    else:
        print("  - OSM roads missing, skipped")
    return out


# ---------------------------------------------------------------- derived
def add_derived(out):
    """Neighbourhood context + distance features; heat is driven by surroundings, not just the pixel."""
    def nbr(a, px):
        v = np.nan_to_num(a.values, nan=0.0)
        return xr.DataArray(uniform_filter(v, size=px, mode="nearest").astype("float32"),
                            coords=a.coords, dims=a.dims)

    for src, tag in (("BLDG_FRAC", "BLDG"), ("GHSL_BUILT_FRAC", "BUILT"), ("S2_NDVI", "NDVI"), ("NDVI", "NDVI30")):
        if src in out:
            out[f"{tag}_nbr150"] = nbr(out[src], 5)           # 150 m window
            out[f"{tag}_nbr450"] = nbr(out[src], 15)          # 450 m window

    if "WATER_FRAC" in out:
        water = out["WATER_FRAC"].values > 0.3
        if water.any():
            out["DIST_WATER_m"] = xr.DataArray(
                (distance_transform_edt(~water) * RES).astype("float32"), coords=out["WATER_FRAC"].coords,
                dims=("y", "x"))
    # distance to nearest green (NDVI > 0.4) as a cooling-access proxy
    nd = "S2_NDVI" if "S2_NDVI" in out else "NDVI"
    green = np.nan_to_num(out[nd].values, nan=0) > 0.4
    if green.any():
        out["DIST_GREEN_m"] = xr.DataArray((distance_transform_edt(~green) * RES).astype("float32"),
                                           coords=out[nd].coords, dims=("y", "x"))
    # simple terrain terms
    if "ELEV" in out:
        gy, gx = np.gradient(out["ELEV"].values.astype("float64"), RES)
        out["SLOPE_deg"] = xr.DataArray(np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32"),
                                        coords=out["ELEV"].coords, dims=("y", "x"))
    return out


# ---------------------------------------------------------------- main
def main():
    master = load_master()
    tmpl = master["LST"]
    out = xr.Dataset({k: master[k] for k in master.data_vars})
    out = out.rio.write_crs(CRS)

    print("GHSL ...");       out = add_ghsl(out, tmpl)
    print("Sentinel-2 ...");  out = add_sentinel2(out, tmpl)
    print("OSM ...");         out = add_osm(out, tmpl)
    print("Derived ...");     out = add_derived(out)

    out = out.drop_vars("spatial_ref", errors="ignore")
    for v in out.data_vars:
        out[v] = out[v].astype("float32")
    out.to_netcdf(DATA / "aligned_features.nc")
    print(f"\nSaved data/aligned_features.nc with {len(out.data_vars)} layers on a "
          f"{out.sizes['y']}x{out.sizes['x']} grid ({RES} m, {CRS})")

    # ---- training table: one row per pixel with a valid target
    df = out.to_dataframe().reset_index()
    n0 = len(df)
    df = df[df["LST"].notna() & df["NDVI"].notna()]
    df["is_urban"] = (df["LANDCOVER"] == 50).astype("int8")
    df["lon_x"], df["lat_y"] = df["x"], df["y"]               # UTM metres, used for spatial CV blocks
    df.to_csv(DATA / "training_table.csv.gz", index=False)
    print(f"Training table: {len(df):,} valid pixels of {n0:,} -> data/training_table.csv.gz")

    print("\nNaN share per layer (valid pixels only):")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
