"""
Pre-flight check for the dashboard: are the packages, the trained model and the city files all in place?

    python dashboard/check_setup.py

Looks in the same places as the app (the Deploy folder, see dashboard/paths.py; override with HEATMAP_RUN, HEATMAP_PATCHES, HEATMAP_GRIDS). Prints OK / MISSING lines and,
at the end, either the command to start the dashboard or the list of things to fix. Changes nothing.
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

DEF = paths.default_paths()
run, grids, patches = Path(DEF["run"]), Path(DEF["grids"]), Path(DEF["patches"])
problems = []


def line(ok, text):
    print(f"[{'OK     ' if ok else 'MISSING'}] {text}")
    if not ok:
        problems.append(text)


print(f"Python {sys.version.split()[0]}\n\nPackages")
for mod, pip in (("streamlit", "streamlit"), ("folium", "folium"), ("streamlit_folium", "streamlit-folium"), ("torch", "torch"),
                 ("xarray", "xarray"), ("netCDF4", "netCDF4"), ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                 ("matplotlib", "matplotlib"), ("pyproj", "pyproj"), ("PIL", "pillow"),
                 ("rioxarray", "rioxarray"), ("rasterio", "rasterio"), ("odc.stac", "odc-stac"), ("pystac_client", "pystac-client"),
                 ("planetary_computer", "planetary-computer"), ("requests", "requests")):
    try:
        m = importlib.import_module(mod)
        line(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception:  # noqa: BLE001
        line(False, f"{mod} is not installed: python -m pip install {pip}")

try:
    import torch
    print(f"        torch device: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (works, a bit slower)'}")
except Exception:  # noqa: BLE001
    torch = None

print("\nModel")
ck = None
f = run / "best.pt"
line(f.exists(), f"{f}" + (f"  ({f.stat().st_size / 1e6:.1f} MB)" if f.exists() else "  <- put best.pt here (from the VM: data/unet_run3/best.pt)"))
if f.exists() and torch is not None:
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        print(f"        {len(ck['features'])} input channels, target {ck.get('target', 'absolute')}, base {ck['base']}")
    except Exception as e:  # noqa: BLE001
        line(False, f"best.pt cannot be loaded: {e}")
m = patches / "meta.json"
line(m.exists(), f"{m}" + ("" if m.exists() else "  <- put meta.json here (from the VM: pre_processing/patches/meta.json)"))
if m.exists() and ck is not None:
    meta = json.load(open(m))
    ok = ck["features"] == meta["features"][:len(ck["features"])]
    line(ok, "meta.json matches the model's input features" if ok else "meta.json does not match best.pt (they must come from the same run)")

print("\nPrepared city files (optional: only the 'Prepared cities' mode needs them; the live mode downloads its own data)")
if grids.exists():
    feats = {p.name[:-len("_features.nc")] for p in grids.glob("*_features.nc")}
    temps = {p.name[:-len("_temp.nc")] for p in grids.glob("*_temp.nc")}
    both = sorted(feats & temps)
    print(f"[{'OK     ' if both else 'none   '}] {len(both)} city/cities in {grids}: {', '.join(both) if both else 'none'}")
    for c in sorted(feats ^ temps):
        line(False, f"{c}: only one of the two files is there")
    for c in both:
        mb = (grids / f"{c}_features.nc").stat().st_size / 1e6
        print(f"        {c}: {mb:.0f} MB features")
else:
    print(f"[none   ] {grids} does not exist (fine for the live mode)")

try:
    import psutil
    free = psutil.virtual_memory().available / 1e9
    print(f"\nFree memory: {free:.1f} GB" + ("  (a 40 km city needs about 2 to 3 GB; close other programs)" if free < 4 else ""))
except Exception:  # noqa: BLE001
    pass

print()
if problems:
    print(f"{len(problems)} thing(s) to fix before the dashboard can start (see MISSING above).")
    sys.exit(1)
print("All set. Start the dashboard with:\n    streamlit run dashboard/app.py")
