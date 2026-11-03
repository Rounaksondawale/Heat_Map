# Deploy folder

Everything the dashboard (`dashboard/app.py`) needs to run, in one place:

| File | What it is | Where it comes from |
|---|---|---|
| `best.pt` | the trained U-Net (about 8 MB) | `data/unet_run3/best.pt` on the Azure VM |
| `meta.json` | feature means and spreads the model was trained with (a few KB) | `pre_processing/patches/meta.json` on the VM |
| `grids/<City>_features.nc`, `grids/<City>_temp.nc` | the two files of each city you want to offer (about 200 MB per city) | `pre_processing/grids/` on the VM |
| `cache/` | made by the app on first use, safe to delete | |

`best.pt` and `meta.json` must come from the same training run.

Check it, then start the dashboard from the repo root:

```bash
python dashboard/check_setup.py
streamlit run dashboard/app.py
```
