"""
Where the dashboard looks for its files.

If <repo>/Deploy/best.pt exists, the Deploy folder is the self-contained home of everything the dashboard needs:

    Deploy/best.pt        the trained model
    Deploy/meta.json      the feature means / spreads the model was trained with
    Deploy/grids/         <City>_features.nc and <City>_temp.nc for each city you want to offer
    Deploy/cache/         made by the app (the predicted temperature of each city, so it is only computed once)

Otherwise the old project layout is used (data/unet_run3, pre_processing/patches, pre_processing/grids). The environment variables
HEATMAP_RUN, HEATMAP_PATCHES, HEATMAP_GRIDS and HEATMAP_CACHE override either.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_paths(root=ROOT, env=None):
    env = os.environ if env is None else env
    deploy = Path(root) / "Deploy"
    if (deploy / "best.pt").exists():
        d = {"run": deploy, "patches": deploy, "grids": deploy / "grids", "cache": deploy / "cache"}
    else:
        r = Path(root)
        d = {"run": r / "data" / "unet_run3", "patches": r / "pre_processing" / "patches", "grids": r / "pre_processing" / "grids",
             "cache": r / "data" / "dashboard_cache"}
    return {k: str(env.get(f"HEATMAP_{k.upper()}", v)) for k, v in d.items()}
