# 02 Data inventory: Heat_Map results (Azure Data Agent)

Discovery pass, 2026-11-19. Repo: `c:\Hackathon\Heat_Map` (branch `main`, clean, HEAD `6e20bf7`). No Azure access in this environment: no `az` CLI and no `AZURE_*` variables.

**Main finding.** The "results on Azure" are **files on the disk of an Azure ML compute instance (VM)**, not in a Blob container that the code reads. Nothing in the repo uses Azure Storage: no SDK, no `abfs://`, `wasb://` or `*.blob.core.windows.net` URLs, and no connection strings. Every script writes plain files under repo-relative paths (`data/...`, `pre_processing/...`) on that VM. So we need **files copied off the VM**, either as a zip or uploaded to a container we can read. Credentials for an existing storage account won't help.

---

## (a) Azure references in code

| Where | What | Kind |
|---|---|---|
| `.github/workflows/ci.yml:8-9, 95-112` | Optional `deploy-azure` job: `azure/login@v2` with OIDC, then `az containerapp update`. Repo **variables** `DEPLOY_TARGET` (= `azure`), `AZURE_CONTAINER_APP`, `AZURE_RESOURCE_GROUP`. Repo **secrets** `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`. | Deployment of the app image (ghcr.io) to an Azure Container App. Has nothing to do with result data. |
| `README.md:96-105` | Same variables and secrets. One-time setup: `az group create -n heat-rg -l centralindia`, `az containerapp env create -n heat-env`, `az containerapp create -n urban-heat ...` (Container App, 2 CPU / 4 GiB) | Names for deployment only |
| `README.md:263` | "Developed on a 4-core Azure ML instance" | Where the data was produced |
| `extraction/run_new_cities.sh:17-18` | `/anaconda/envs/azureml_py38`, the conda env of an **Azure ML compute instance** | Confirms that the pipeline ran on AML compute |
| `Deploy/README.md:7-9` | `best.pt` comes from "`data/unet_run3/best.pt` on the Azure VM", `meta.json` from "`pre_processing/patches/meta.json` on the VM", grids from "`pre_processing/grids/` on the VM" | **The only place that names the result locations.** They are paths on the VM, relative to the repo clone. |
| `.gitignore:1` | `# Data (too large for GitHub; regenerate or store in Azure)`. It ignores `data/`, `*.csv`, `*.nc`, `*.tif`, `*.png` (except `docs/`), `*.log`, `.env`, `Deploy/grids/`, `Deploy/cache/` | Explains why no results were committed. Note that `*.csv` is ignored everywhere. |
| `.dockerignore` | excludes `Deploy/grids`, `Deploy/cache`, `data`, `docs`, `tests`, `notebooks` | none |
| `dashboard/paths.py:12-29` | env overrides `HEATMAP_RUN`, `HEATMAP_PATCHES`, `HEATMAP_GRIDS`, `HEATMAP_CACHE` | Local path overrides, not Azure |
| `extraction/build_dataset.py:104,161` | `https://planetarycomputer.microsoft.com/api/stac/v1`: public STAC. The "SAS tokens" there are anonymous, short-lived Planetary Computer signing tokens that `planetary_computer.sign` fetches at run time. | Public input data. No secret involved. |
| `extraction/build_dataset.py:105,351` | JRC GHSL FTP and `archive-api.open-meteo.com` | Public input data |
| `.github/workflows/ci.yml:93` | `https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}`: a secret **reference**, not a value | Hugging Face deploy |
| `.streamlit/config.toml` | Theme only. No `secrets.toml` and no `st.secrets` use. | none |

Not found anywhere: `BlobServiceClient`, `azure-storage-*`, `adlfs`, `fsspec` Azure, `abfs(s)://`, `wasb(s)://`, `account_url`, `connection_string`, `sas_token`, `mlflow`, `azureml` SDK, `databricks`, `os.getenv`/`os.environ` reads of credentials, and `.env` files.
**No committed keys, tokens or SAS strings were found** in the working tree or in the file list of the git history. Nothing needs rotating.

---

## (b) Local data inventory

Every file under the repo was listed, including gitignored ones (skipping `.git` and `__pycache__`). There are no venvs and no `data/`, `pre_processing/grids/`, `pre_processing/patches/`, `Deploy/grids/` or `Deploy/cache/` folders. The sibling folders in `c:\Hackathon\` (BAH, FloatChat, h4h, hack4hack, FV, 3D bone mapping, ...) and `~/Downloads` were searched for `*.nc`, `*.npz`, `*.npy`, `*.pt`, `*.parquet`, `meta.json` and other result names. Everything found there belongs to other projects (GOES ABI satellite frames, ARGO parquet, etc.). **None of it is Heat_Map data.**

| Path | Size | Format | Schema / contents | What it is |
|---|---|---|---|---|
| `Deploy/best.pt` | 7.8 MB (7,804,252 B) | PyTorch checkpoint (dict) | keys `model` (93 tensors, **1,944,863 params**), `mu` = 1.92e-06, `sd` = 2.929, `features` (**51** names: 45 base and context features + 6 `*_rel`), `target` = `"anomaly"`, `landcover_codes` = [10,20,30,40,50,60,70,80,90,95,100] (ESA WorldCover), `base` = 32 | The trained final U-Net (`data/unet_run3/best.pt` on the VM). **The only real result artifact on disk.** |
| `README.md` (tables) | 19 KB | Markdown | Per-city metrics for the 10 validation cities (8 cols), first vs final headline, 3-model comparison, error by land cover (7 classes), hotspot detection (Delhi/Jaipur), cooling-dose table (3 areas), data and training facts, group-importance and vegetation-check numbers in prose | Numbers **printed by** `evaluate_unet.py`, `compare_runs.py`, `group_importance.py`, `hotspot_report.py`, `simulate_cooling.py` and transcribed by the authors. The mean row matches the mean of the 10 city rows (checked: R2_anomaly 0.745 vs 0.74, hot_AUC 0.936 vs 0.93). |
| `docs/fig_*.png` (7 files) | 52 KB to 704 KB, 1.19 MB total | PNG | `fig_cooling_curves`, `fig_delhi_drivers`, `fig_delhi_map` (crop of a real `simulate_cooling.py` figure), `fig_group_importance`, `fig_model_comparison`, `fig_per_city_r2`, `fig_vegetation_check` | Static figures "redrawn from the numbers printed by our own evaluation" (README:309). Can go into the dashboard as is. |
| `extraction/build_dataset.py` `CITIES` | (code) | Python dict | 50 cities: lon, lat, climate label, max cloud %, each a 40 km box | City config, good for the map layer |
| `pre_processing/make_patches.py` `VAL_DEFAULT`, `LC_CODES` | (code) | Python lists | 10 validation cities; 11 land-cover codes | Train/val split (script default) |
| `notebooks/data_audit.ipynb` | 12 KB | Notebook | One stored output: "using C:\Hackathon\Heat_Map\data\smoke_test, 8,100 rows x 39 columns". That folder **no longer exists** (deleted in `f2cb492`). | Smoke-test only, no results |
| `tests/synthetic.py` | (code) | builds a fake city, `meta.json` and `best.pt` | **Synthetic, do not use as results** |
| git history: `models/artifacts/lgbm_lst_anom_config.json` (removed in `c609cc0`) | – | JSON | Old LightGBM Pune smoke-test config (features, `n_trees` = 54) | Obsolete prototype, not a result of the final model |

**Missing locally and needed even by the existing app:** `Deploy/meta.json`. CI's `image` job fails without it (`ci.yml:52-54`), and the live and prepared-city modes cannot normalise inputs without it.

---

## (c) Expected pipeline outputs vs present

Paths are relative to the repo root on the VM (the script defaults; README:266-288 gives the exact commands used, e.g. `--out data/unet_run3`).

| Script | Writes | Format / schema | Local? |
|---|---|---|---|
| `extraction/build_dataset.py` | `data/multi_city/cities/<City>/aligned.nc`, `table.csv.gz`, `meta.json`; `data/multi_city/all_cities_training.csv.gz` | NetCDF stack; pixel table; `ds.attrs` (n_scenes, epsg, weather...) | No |
| `extraction/repair_dataset.py` | rewrites the same files | – | No |
| `extraction/finalize_dataset.py` | `data/multi_city/all_cities_clean.parquet` (`.csv.gz` without `--no-csv`) | ~9.25 M rows | No |
| `extraction/add_context_features.py` | `data/multi_city/all_cities_clean_ctx.parquet` | + 28 context columns | No |
| `extraction/check_dataset.py` | `data/multi_city/quality_report.csv` | per city OK/warn/bad | No |
| `extraction/grid_inventory.py` | `data/multi_city/grid_inventory.csv` (+ png) | per-city grid stats | No |
| `extraction/cell_inventory.py` | `data/multi_city/cell_inventory_1km.csv`, `india_data_coverage.png` | city, cx, cy, samples, mean_obs, mean_LST, lon, lat, coverage_% | No |
| `pre_processing/export_city_grids.py` | `pre_processing/grids/<City>_features.nc`, `<City>_temp.nc` (LST, n_obs), `grid_summary.csv` | ~1350x1350 at 30 m; ~200 MB per city (both files) | No |
| `pre_processing/make_patches.py` | `pre_processing/patches/{train,val}/<City>.npz` (X, lc, y, mask, pos), `patch_summary.csv`, **`meta.json`** (features, mean, std, rel, landcover_codes, size, train_cities, val_cities) | ~21k train and ~4.3k val 64x64 patches | **No (meta.json is critical)** |
| `pre_processing/pack_patches.py` | `pre_processing/patches/packed/{split}_X.npy`, `{split}_rest.npz` | float16 memmap, GBs | No (not needed) |
| `models/train_unet.py --out data/unet_run3` | `best.pt`, `metrics.csv` (MAE, RMSE, R2_pooled, R2_city, epoch, train_loss, seconds), `val_per_city.csv` (city, MAE, RMSE, R2) | – | `best.pt` **yes** (as `Deploy/best.pt`); CSVs no |
| `models/evaluate_unet.py --run data/unet_run3` | `data/unet_run3/eval_val/eval_per_city.csv` (city, pixels, LST_mean, LST_std, MAE, RMSE, bias, MAE_citymean, skill, R2_within, R2_anomaly, spearman, hot_AUC, hot_F1, hot_IoU, hot_bias), `eval_by_landcover.csv` (landcover_code, pixels, MAE, bias), `map_<City>.png`, `eval_summary.png` | – | No (README has a subset of the columns) |
| `models/compare_runs.py` | `data/model_comparison.csv` (rank, run, target, channels, base, 7 metric means, cities_R2_within<0, epochs_run), `data/model_comparison_per_city.csv` | – | No (README has the summary) |
| `models/group_importance.py` | `data/unet_run3/group_importance.csv` (group, channels, R2_inside_city, R2_drop, spearman_drop, hot_AUC_drop) | – | No (README prose has 3 values) |
| `models/hotspot_report.py --city Delhi Jaipur` | `data/hotspots/<City>_hotspots.csv` (cluster, rank, rows, cols, lon, lat, area_ha, main_landcover, obs_C, obs_gap, pred_gap, not_explained, mean_z, drv_<group>..., drv_ref_mismatch, top_drivers), `<City>_hotspots.png` | – | No |
| `models/simulate_cooling.py` | `data/simulations/sim_<City>_<r0>_<c0>.csv` (dose_%, {veg,roof,both}_cooling, _gap_left, _hot10_gap_left), `.png` | – | No (README has the 3-row summary) |
| `dashboard/engine.py` | `<cache>/<City>_<run>_pred.npy` (float32 HxW predicted LST) | ~7 MB per city | No |
| `dashboard/live.py` | `<cache>/live/<name>_features.nc`, `_temp.nc`, `_info.json` | – | No |
| run logs | `pipeline.log`, `run_log.txt`, `*.log` (gitignored) | text | No |

---

## (d) Dashboard sections that can be filled with real data now

Generated in `dashboard-project/data/` by `python dashboard-project/data/build_data.py` (standard library only; torch is optional for `model.json`). Every file has a `_source` field. The script **automatically prefers the raw CSVs** listed in (c) when they appear in the repo or in `dashboard-project/azure_raw/` (same relative paths), so it only needs re-running after the Azure download.

| File | Size | Contents | Source |
|---|---|---|---|
| `metrics.json` | 5.1 KB | headline (first vs final), 3-model comparison, error by land cover, hotspot detection Delhi/Jaipur, group importance (3 values plus "rest below 0.1"), vegetation check, data and training facts | README.md tables (line ranges in `_source`) |
| `per_city.json` | 2.6 KB | 10 validation cities x MAE, R2_within, R2_anomaly, spearman, hot_AUC, hot_F1, hot_bias, plus the mean row | README.md lines 152-164 |
| `cooling.json` | 0.8 KB | 3 areas: obs_gap, pred_gap, dose text, vegetation dose % | README.md lines 48-52 |
| `cities.json` | 7.8 KB | 50 cities: lon, lat, climate, cloud_max, train/val split | `extraction/build_dataset.py`, `pre_processing/make_patches.py` |
| `model.json` | 1.3 KB | checkpoint metadata: 1.94 M params, 51 feature names, target, base, LC codes | `Deploy/best.pt` |
| `figures.json` | 0.7 KB | list of the 7 `docs/fig_*.png` | `docs/` |

**Ready now:** KPI header (R² 0.74, Spearman 0.86, AUC 0.93, F1 0.60, 0 cities worse than baseline), first-vs-final model comparison, per-city bar or table for the 10 held-out cities, land-cover error bars, hotspot-quality card for Delhi/Jaipur, cooling-dose summary cards, India map of the 50 cities coloured by split (and the 10 validation cities by R²), model card (architecture, features), the static README figures.
**Not possible yet:** any temperature or hotspot **map** (no grid or prediction locally), the full cooling **curves** (only 3 summary rows), hotspot cluster lists and drivers per cluster, the training curve, the full group-importance table, per-city metrics for the 40 training cities, data-quality per city.
**No downsampled grid was built**, because no `.nc` grid or `*_pred.npy` exists locally.

---

## (e) What we still need from Azure, and the minimum access to ask for

The source is the Azure ML compute instance where the repo was run (the repo clone, typically under `~/cloudfiles/code/Users/<user>/Heat_Map`, which sits on the workspace's Azure file share). The items below are relative to that repo root.

**Priority 1: small, a few MB in total. Covers every remaining section except maps.**
```
pre_processing/patches/meta.json            # also fixes Deploy/meta.json for the app and CI
pre_processing/patches/patch_summary.csv
pre_processing/grids/grid_summary.csv
data/unet_run3/metrics.csv
data/unet_run3/val_per_city.csv
data/unet_run3/group_importance.csv
data/unet_run3/eval_val/eval_per_city.csv
data/unet_run3/eval_val/eval_by_landcover.csv
data/model_comparison.csv
data/model_comparison_per_city.csv
data/hotspots/*_hotspots.csv
data/simulations/sim_*.csv
data/multi_city/quality_report.csv
data/multi_city/grid_inventory.csv
data/multi_city/cell_inventory_1km.csv
data/multi_city/cities/*/meta.json
```
**Priority 2: for maps. About 7 MB per city, or about 200 MB per city for grids.** `data/dashboard_cache/*_pred.npy` or `Deploy/cache/*_pred.npy` (predicted LST per city) if they exist, **plus** `pre_processing/grids/Delhi_temp.nc` and `Jaipur_temp.nc` (observed LST). Better still, `Delhi_features.nc`/`_temp.nc` and the Jaipur pair: with `best.pt` and `meta.json` we can then recompute predictions, hotspots and doses locally with the existing `dashboard/engine.py`. Optional: `data/unet_run3/eval_val/*.png`, `data/hotspots/*.png`, `data/simulations/*.png`, `pipeline.log`.

**Simplest request (option A, recommended: no credentials shared).** Ask the user to run this on the compute instance terminal, from the repo root:
```bash
tar czf heatmap_results.tgz $(ls -d pre_processing/patches/meta.json pre_processing/patches/patch_summary.csv \
  pre_processing/grids/grid_summary.csv data/unet_run3/*.csv data/unet_run3/eval_val/*.csv data/model_comparison*.csv \
  data/hotspots/*.csv data/simulations/*.csv data/multi_city/*.csv data/multi_city/cities/*/meta.json \
  data/dashboard_cache/*_pred.npy Deploy/cache/*_pred.npy 2>/dev/null)
```
They download it through AML Studio (Notebooks file browser, right-click, Download) and drop it into `c:\Hackathon\Heat_Map\dashboard-project\azure_raw\` (extract there). The size is likely under 400 MB with the pred caches, and a few MB without.

**Option B, if they prefer a link.** Upload the same tarball, or the folders, to a Blob container. Then share a **container-scoped SAS URL with `sp=rl` (read and list only), HTTPS-only (`spr=https`) and an expiry of 7 days or less**, e.g. `https://<account>.blob.core.windows.net/<container>?sv=...&sp=rl&se=...&spr=https&sig=...`. We would keep it in a local env var such as `HEATMAP_SAS_URL`, never in the repo. No account keys, no connection strings, no write/delete permission and no subscription-level role are needed. If they would rather grant RBAC, **Storage Blob Data Reader** on that one container is the minimum.

Also please confirm **which run folder is final** (`data/unet_run3` per the README, though `simulate_cooling.py` defaults to `data/unet_run2`), and whether `Deploy/best.pt` is that run's checkpoint. `best.pt` and `meta.json` must come from the same run.
