# Deploy folder

Everything the dashboard (`dashboard/app.py`) needs to run, in one place:

| File | What it is | Where it comes from |
|---|---|---|
| `best.pt` | the trained U-Net (about 8 MB) | `data/unet_run3/best.pt` on the Azure VM |
| `meta.json` | feature means and spreads the model was trained with (a few KB) | `pre_processing/patches/meta.json` on the VM |
| `grids/<City>_features.nc`, `grids/<City>_temp.nc` | only for the "Prepared cities" mode: the two files of each city (about 200 MB per city) | `pre_processing/grids/` on the VM |
| `cache/` | made by the app (predictions, live areas, building tiles), safe to delete | |

`best.pt` and `meta.json` must come from the same training run. The live mode (a rectangle anywhere in India) needs only these two files and
internet access; commit both so the container build can include them.

Check it, then start the dashboard from the repo root:

```bash
python dashboard/check_setup.py
streamlit run dashboard/app.py
```
