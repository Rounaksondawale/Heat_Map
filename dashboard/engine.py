"""
Dashboard engine: everything the app computes, as plain functions (no Streamlit in here, so it can be tested on its own).

    load_model(run, patches)                      trained U-Net + feature statistics
    list_cities(grids)                            cities that have <City>_features.nc and <City>_temp.nc
    load_city(city, M, grids, ...)                predicts the whole city once (cached on disk) and finds the Gi* hotspots
    box_from_lonlat(state, west, south, east, north)   a rectangle drawn on the map -> rows / columns of the city grid
    analyse_area(state, M, box)                   one rectangle: gap to its neighbours, hotspot share, drivers, cooling doses
    dose_now(S, veg, roof)                        the area's gap to its neighbours after converting a given share

It reuses models/hotspot_report.py and models/simulate_cooling.py, so the numbers match what those scripts print.
"""
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("models", "pre_processing", "extraction"):
    sys.path.insert(0, str(ROOT / sub))
from hotspot_report import (GROUPS, block_mean, driver_phi, gi_star, group_channels, hotspot_labels,  # noqa: E402
                            predict_grid, prepare_city)
from simulate_cooling import area_stats, build_simulation, dose_for, dose_table, find_hotspot, load_context  # noqa: E402
from train_unet import UNet  # noqa: E402

RES, PIXEL_HA = 30, 0.09
MIN_SIDE, MAX_SIDE = 8, 350              # rectangle side in pixels (240 m .. 10.5 km)


def load_model(run, patches):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(run) / "best.pt", map_location=dev, weights_only=False)
    meta = json.load(open(Path(patches) / "meta.json"))
    model = UNet(len(ck["features"]), len(ck["landcover_codes"]), ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "ck": ck, "meta": meta, "dev": dev, "run": str(run)}


def list_cities(grids):
    g = Path(grids)
    return sorted(p.name[:-len("_features.nc")] for p in g.glob("*_features.nc") if (g / p.name.replace("_features", "_temp")).exists())


def load_city(city, M, grids, block=20, cache_dir=None):
    """Predict the whole city once. The prediction is cached as <cache_dir>/<city>_<run>_pred.npy (rebuilt when best.pt is newer)."""
    ck, meta, dev = M["ck"], M["meta"], M["dev"]
    P = prepare_city(city, grids, ck, meta)
    pred = None
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"{city}_{Path(M['run']).name}_pred.npy"
        if cache.exists() and cache.stat().st_mtime >= (Path(M["run"]) / "best.pt").stat().st_mtime:
            pred = np.load(cache)
            if pred.shape != (P["H"], P["W"]):
                pred = None
    if pred is None:
        pred = predict_grid(M["model"], P["Z"], P["lci"], dev, ck["mu"], ck["sd"]) + P["add"]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, pred.astype("float32"))
    for k in ("X", "gf"):                    # large raw arrays that are not needed once the input stack Z is built (saves about 0.7 GB)
        P.pop(k, None)
    bp = block_mean(pred, P["usable"], block)
    bo = block_mean(P["lst"], P["valid"], block)
    return {"city": city, "P": P, "pred": pred, "B": block, "bp": bp, "bo": bo, "zp": gi_star(bp), "zo": gi_star(bo),
            "grids": str(grids), "ctx": None}


def hotspot_mask(state, z, min_blocks=2):
    """(block labels, pixel mask of the whole grid) of the Gi* hotspots at this z-score."""
    lab, n = hotspot_labels(state["zp"], z, min_blocks)
    B, H, W = state["B"], state["P"]["H"], state["P"]["W"]
    px = np.kron(lab > 0, np.ones((B, B), bool))
    return lab, np.pad(px, ((0, H - px.shape[0]), (0, W - px.shape[1])))


# ------------------------------------------------------------------ coordinates
def _transformer(epsg):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True), Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def city_bounds(state):
    """(south, west, north, east) of the city grid in degrees (the UTM box is drawn as its lon/lat envelope)."""
    P = state["P"]
    _, back = _transformer(P["epsg"])
    xs, ys = P["xs"], P["ys"]
    corners = [back.transform(x, y) for x in (xs.min(), xs.max()) for y in (ys.min(), ys.max())]
    lons, lats = zip(*corners)
    return min(lats), min(lons), max(lats), max(lons)


def box_from_lonlat(state, west, south, east, north):
    """Rectangle in degrees -> (row0, row1, col0, col1) of the city grid. Raises ValueError if it is outside the city or the wrong size."""
    P = state["P"]
    fwd, _ = _transformer(P["epsg"])
    pts = [fwd.transform(lo, la) for lo in (west, east) for la in (south, north)]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    xs, ys = P["xs"], P["ys"]
    c0, c1 = int(np.floor((x0 - xs[0]) / RES)), int(np.ceil((x1 - xs[0]) / RES))
    r0, r1 = int(np.floor((ys[0] - y1) / RES)), int(np.ceil((ys[0] - y0) / RES))
    return check_box((r0, r1, c0, c1), P["H"], P["W"])


def check_box(box, H, W):
    r0, r1, c0, c1 = box
    r0, r1, c0, c1 = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)
    h, w = r1 - r0, c1 - c0
    if h < 1 or w < 1:
        raise ValueError("the rectangle is outside this city's 40 km box")
    if min(h, w) < MIN_SIDE:
        raise ValueError(f"the rectangle is too small ({h * RES} x {w * RES} m); draw at least {MIN_SIDE * RES} m on each side")
    if max(h, w) > MAX_SIDE:
        raise ValueError(f"the rectangle is too large ({h * RES / 1000:.1f} x {w * RES / 1000:.1f} km); keep each side under {MAX_SIDE * RES / 1000:.1f} km")
    return r0, r1, c0, c1


def box_to_lonlat(state, box):
    """(south, west, north, east) of a grid box, for drawing it on the map."""
    P = state["P"]
    _, back = _transformer(P["epsg"])
    r0, r1, c0, c1 = box
    xs, ys = P["xs"], P["ys"]
    pts = [back.transform(xs[min(c, len(xs) - 1)], ys[min(r, len(ys) - 1)]) for c in (c0, c1) for r in (r0, r1)]
    lons, lats = zip(*pts)
    return min(lats), min(lons), max(lats), max(lons)


def top_hotspot_box(state, M, size=20, ring=30, min_built=0.0):
    """Box of the strongest size x size block compared with its ring (same search as simulate_cooling.py --auto)."""
    ctx = get_ctx(state, M)
    r0, c0 = find_hotspot(ctx["lst"], ctx["valid"], size, ring, ctx["gf"]["GHSL_BUILT_FRAC"].values, min_built)
    return r0, r0 + size, c0, c0 + size


# ------------------------------------------------------------------ map layers
def _png_url(rgba):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def temperature_layer(arr, valid, vmin=None, vmax=None, cmap="inferno", alpha=0.78, max_px=1200):
    """Colour a temperature grid as a transparent PNG data URL for the map (invalid pixels are transparent)."""
    import matplotlib
    a = np.where(valid, arr, np.nan).astype("float32")
    step = max(1, int(np.ceil(max(a.shape) / max_px)))
    if step > 1:
        a = a[::step, ::step]
    lo, hi = np.nanpercentile(a, [2, 98]) if vmin is None else (vmin, vmax)
    rgba = (matplotlib.colormaps[cmap]((np.clip(a, lo, hi) - lo) / max(hi - lo, 1e-6)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(a), int(alpha * 255), 0)
    return _png_url(rgba), (float(lo), float(hi))


def mask_layer(mask, color=(227, 73, 72), alpha=0.5, max_px=1200):
    step = max(1, int(np.ceil(max(mask.shape) / max_px)))
    m = mask[::step, ::step]
    rgba = np.zeros(m.shape + (4,), "uint8")
    rgba[..., :3] = color
    rgba[..., 3] = np.where(m, int(alpha * 255), 0)
    return _png_url(rgba)


# ------------------------------------------------------------------ one rectangle
def get_ctx(state, M):
    if state["ctx"] is None:
        a = SimpleNamespace(grids=state["grids"])
        state["ctx"] = load_context(state["city"], a, M["ck"], M["meta"])
    return state["ctx"]


def analyse_area(state, M, box, ring=30, steps=10, z=2.58, min_blocks=2, max_built=0.6, roof_albedo=0.15):
    """Everything about one rectangle (rows r0:r1, columns c0:c1): gap to its neighbours, hotspot share, drivers, cooling doses."""
    P = state["P"]
    ck, meta, dev, model = M["ck"], M["meta"], M["dev"], M["model"]
    ctx = get_ctx(state, M)
    a = SimpleNamespace(ring=ring, max_built=max_built, roof_albedo=roof_albedo, seed=0)
    S = build_simulation(ctx, box, a, model, ck, meta, dev)
    sl = S["sl"]
    # where the area sits in the city's hotspot map
    lab, hotpx = hotspot_mask(state, z, min_blocks)
    r0, r1, c0, c1 = S["r0"], S["r1"], S["c0"], S["c1"]
    inside = P["usable"][r0:r1, c0:c1]
    hot_share = float(hotpx[r0:r1, c0:c1][inside].mean()) if inside.any() else 0.0
    B = state["B"]
    zb = state["zp"][r0 // B:max(r1 // B, r0 // B + 1), c0 // B:max(c1 // B, c0 // B + 1)]
    mean_z = float(np.nanmean(zb)) if np.isfinite(zb).any() else float("nan")
    # drivers: swap each feature group for the neighbours' average (exact Shapley values over 6 groups)
    Zc, lc_c, us = P["Z"][:, sl[0], sl[1]], P["lci"][sl[0], sl[1]], P["usable"][sl[0], sl[1]]
    phi, _ = driver_phi(model, ck, Zc, lc_c, us, S["reg"], S["ring"] & us, group_channels(P["names"]), P["add"], dev)
    doses, rows = dose_table(S, steps)
    t = pd.DataFrame(rows)
    need = {}
    for tag, n in (("veg", len(S["sim"].idx_g)), ("roof", len(S["sim"].idx_r)), ("both", None)):
        d = dose_for(doses, t[f"{tag}_gap_left"].values) if S["pred_gap"] > 0 else 0.0
        need[tag] = d
    ha = S["n_px"] * PIXEL_HA
    return {"S": S, "box": (r0, r1, c0, c1), "ha": ha, "sides_m": ((r1 - r0) * RES, (c1 - c0) * RES), "obs_gap": S["obs_gap"],
            "pred_gap": S["pred_gap"], "unexplained": S["obs_gap"] - S["pred_gap"], "obs_area": S["o_reg"], "obs_ring": S["o_ring"],
            "pred_area": S["b_reg"], "pred_ring": S["b_ring"], "city_mean": ctx["city_mean"], "hot_share": hot_share, "mean_z": mean_z,
            "phi": dict(zip(GROUPS, [float(x) for x in phi])), "ref_mismatch": S["pred_gap"] - float(np.sum(phi)),
            "table": t, "doses": doses, "need": need, "n_green": len(S["sim"].idx_g), "n_roof": len(S["sim"].idx_r),
            "tree_ref": ctx["targets"], "drift": S["drift"]}


def dose_now(S, veg, roof):
    """(cooling, gap left) of the area after converting a share `veg` of its greenable pixels and `roof` of its built pixels."""
    sim = S["sim"]
    feats, _, _ = sim.features(veg, roof)
    reg_s, ring_s, _ = area_stats(sim.predict(feats), sim.ok, S["reg"], S["ring"])
    return S["b_reg"] - reg_s, reg_s - ring_s
