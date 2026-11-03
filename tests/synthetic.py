"""
A small synthetic city plus a tiny random-weight model, so the dashboard code can be tested anywhere (no real data, no internet).

    make_assets(folder)  ->  folder/best.pt, folder/meta.json, folder/grids/Synth_features.nc, folder/grids/Synth_temp.nc

The numbers mean nothing (the model is untrained); the files only have the right names, shapes and value ranges.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
for sub in ("models", "pre_processing", "extraction"):
    sys.path.insert(0, str(ROOT / sub))
from export_city_grids import FEATURES  # noqa: E402
from make_patches import LC_CODES, REL_FLOOR, REL_VARS  # noqa: E402
from train_unet import UNet  # noqa: E402

N, EPSG, X0, Y0 = 200, 32643, 375000.0, 2052000.0          # 6 km box near Pune
RANGES = {"NDVI": (-0.05, 0.7), "NDBI": (-0.3, 0.3), "MNDWI": (-0.5, 0.2), "ALBEDO": (0.08, 0.3), "ELEV": (550, 620),
          "SLOPE_deg": (0, 8), "GHSL_HEIGHT": (0, 25), "GHSL_BUILT_FRAC": (0, 1), "GHSL_POP": (0, 40), "DIST_WATER_m": (0, 3000),
          "DIST_GREEN_m": (0, 600)}


def _field(rng, lo, hi, sigma=12):
    f = gaussian_filter(rng.normal(size=(N, N)), sigma)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    return (lo + (hi - lo) * f).astype("float32")


def make_assets(folder, seed=0):
    """Base layers are random smooth fields; every derived feature (neighbourhood, distances, slope, surroundings) comes from the
    project's own code, so the files are as self-consistent as real ones."""
    import build_dataset as B
    from export_city_grids import export_dataset
    folder = Path(folder)
    (folder / "grids").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = X0 + 30.0 * np.arange(N)
    y = Y0 - 30.0 * np.arange(N)
    f = _field(rng, 0, 1)
    base = {k: _field(rng, *RANGES[k]) for k in ("NDVI", "NDBI", "MNDWI", "ALBEDO", "ELEV", "GHSL_HEIGHT", "GHSL_BUILT_FRAC", "GHSL_POP")}
    base["LANDCOVER"] = np.select([f < 0.15, f < 0.4, f < 0.6, f < 0.97], [10, 30, 40, 50], 80).astype("float32")
    base["LST"] = (38 + 5 * base["GHSL_BUILT_FRAC"] - 4 * base["NDVI"] + gaussian_filter(rng.normal(size=(N, N)), 1.5)).astype("float32")
    base["n_obs"] = np.full((N, N), 20, "float32")
    coords = {"y": ("y", y, {"units": "metre"}), "x": ("x", x, {"units": "metre", "crs": f"EPSG:{EPSG}"})}
    ds = xr.Dataset({k: (("y", "x"), v) for k, v in base.items()}, coords=coords)
    ds = B.add_derived(ds)
    export_dataset(ds, "Synth", folder / "grids")

    numeric = [v for v in FEATURES if v != "LANDCOVER"]
    names = numeric + [f"{v}_rel" for v in REL_VARS]
    feats = xr.open_dataset(folder / "grids" / "Synth_features.nc")
    meta = {"features": names, "mean": [0.0] * len(names), "std": [1.0] * len(names), "rel": {"vars": REL_VARS, "floor": REL_FLOOR},
            "min_pass": 0.6, "max_few_share": 3.0, "landcover_codes": LC_CODES, "size": 64}
    for i, n in enumerate(numeric):                              # standardise with the real spread of each synthetic feature
        v = feats[n].values
        meta["mean"][i], meta["std"][i] = float(np.nanmean(v)), float(np.nanstd(v) + 1e-6)
    feats.close()
    json.dump(meta, open(folder / "meta.json", "w"))
    torch.manual_seed(seed)
    model = UNet(len(names), len(LC_CODES), 8)
    torch.save({"model": model.state_dict(), "mu": 0.0, "sd": 2.5, "features": names, "target": "anomaly",
                "landcover_codes": LC_CODES, "base": 8}, folder / "best.pt")
    return folder
