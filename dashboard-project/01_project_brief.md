# 01 Project brief: Heat_Map (Urban Heat Hotspots for Indian Cities)

Source: local clone `c:\Hackathon\Heat_Map`, branch `main`, HEAD `6e20bf7` (2026-11-03). Every claim cites `file:line`. "Inferred" marks anything not stated verbatim. Figures in `docs/` were read visually; values quoted from them are the printed labels unless marked "read off bar".

Files read: README.md, Deploy/README.md, all .py in dashboard/, extraction/, pre_processing/, models/, tests/; extraction/run_new_cities.sh; notebooks/data_audit.ipynb (code + stored outputs); requirements.txt, requirements-app.txt, dashboard/requirements.txt; Dockerfile; .github/workflows/ci.yml; .streamlit/config.toml; .gitignore, .dockerignore, pre_processing/.gitignore; the 7 PNGs in docs/. Deploy/best.pt was opened read-only with `torch.load(weights_only=True)` to read its metadata keys (no code executed from it).

Not present on disk: `Deploy/meta.json`, `Deploy/grids/`, `data/`, `pre_processing/grids/`, `pre_processing/patches/`, and every metrics CSV the scripts write (`*.csv` and non-docs `*.png` are gitignored, `.gitignore:7,10-11`). So all result numbers below come from README text/tables, the docs figures, the notebook output and the checkpoint metadata.

---

## 1. Purpose / problem statement

- Tagline: "Which parts of a city are cooking, why, and how much green would it take to cool them down?" (`README.md:3`).
- Built "a satellite dataset of 50 Indian cities, trained a U-Net to predict how heat is spread *inside* a city", and turned it into three tools: hotspot finder, driver explainer, cooling what-if simulator (`README.md:5-6`, `README.md:24-41`).
- The model predicts the within-city pattern of land surface temperature (LST), not absolute temperature; the city mean must be supplied (`README.md:231`, `models/train_unet.py:8-12`).
- Target quantity is LST at the Landsat pass (late morning, clear days, March-May), not air temperature (`README.md:236`, `models/hotspot_report.py:21-22`).
- Drivers and doses are explicitly "the model's attribution, not proof of cause" (`README.md:233-234`, `models/simulate_cooling.py:16`).

## 2. Study area(s), resolution, time period

- **50 Indian cities**, defined in `extraction/build_dataset.py:49-102` as `name: (lon, lat, climate class, max cloud % for Landsat scenes)`:
  - Original 14 (`build_dataset.py:50-63`): Pune, Delhi, Jaipur, Ahmedabad, Nagpur, Hyderabad, Bengaluru, Chennai, Mumbai, Kolkata, Lucknow, Bhubaneswar, Guwahati, Chandigarh.
  - Added "to widen climate / terrain coverage", 19 cities (`build_dataset.py:64-83`): Shimla, Dehradun, Ranchi, Shillong, Visakhapatnam, Surat, Kochi, Thiruvananthapuram, Mangaluru, Jodhpur, Indore, Bhopal, Amritsar, Ludhiana, Patna, Raipur, Coimbatore, Vadodara, Kanpur.
  - "batch 3: reaches 50 cities", 17 cities (`build_dataset.py:84-101`): Srinagar, Jammu, Agartala, Imphal, Puducherry, Panaji, Madurai, Bikaner, Ajmer, Udaipur, Rajkot, Nashik, Aurangabad, Agra, Varanasi, Jabalpur, Gwalior.
  - `extraction/run_new_cities.sh:2,24` builds "the 36 new cities (to reach 50)".
  - Climate labels used: semi-arid, semi-arid/subtrop, tropical savanna, tropical coastal, tropical humid, humid subtropical, humid subtropical (hill), arid, temperate (hill) (`build_dataset.py:50-101`). Cloud limits per city range 10-30 % (same lines).
- **Box**: square of half-size 20 km, i.e. 40 km box per city (`build_dataset.py:526`, `README.md:218`), "about 1,350 x 1,350 pixels; about 89 million pixels in total" (`README.md:218`).
- **Grid**: 30 m (`build_dataset.py:114`), local UTM zone computed from lon/lat (`build_dataset.py:119-120`, `433`).
- **Time window**: Landsat scenes from 2022-2025 (`build_dataset.py:527`), months March, April, May only (`build_dataset.py:528`, `154`); up to 40 clearest scenes per city sorted by cloud cover (`build_dataset.py:155-156`, `529`). README: "mean land surface temperature of the clearest Landsat 8/9 passes in March to May 2022-2025, cloud-masked" (`README.md:248`).
- **ESA WorldCover** year 2021 (`build_dataset.py:220`); **GHSL** epochs 2018 (height) and 2020 (built-up, population) (`build_dataset.py:107-112`).
- **Train / validation split by city**: 10 validation cities by default = Jaipur, Lucknow, Nagpur, Chennai, Bengaluru, Kolkata, Bhubaneswar, Dehradun, Amritsar, Coimbatore, "spread over climates; the hardest cities stay in training" (`pre_processing/make_patches.py:38-39`); 40 training cities (`README.md:221`). Same 10 listed at `README.md:111-112`.
- **Live mode**: any rectangle 1-10 km per side whose centre lies in the India bounding box (68.0-97.5 E, 6.5-37.5 N) (`dashboard/live.py:31-32`, `48-61`), processed in a square window of half-size max(8 km, half the longest side + 3 km) (`dashboard/live.py:33`, `64-68`).

## 3. Inputs / data sources

**Platform**: Microsoft Planetary Computer STAC API `https://planetarycomputer.microsoft.com/api/stac/v1` (`build_dataset.py:104`, `138-141`), loaded with `odc.stac` (`build_dataset.py:146-147`, `178-180`). **Google Earth Engine is not used anywhere** (no `ee` import; searched all .py). GHSL is downloaded directly from the JRC FTP `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL` (`build_dataset.py:105`, `247`). Weather from Open-Meteo archive `https://archive-api.open-meteo.com/v1/archive` (`build_dataset.py:351`).

| Layer | Product / band | Processing | Citation |
|---|---|---|---|
| LST (target) | `landsat-c2-l2`, platforms landsat-8, landsat-9, band `lwir11` | `DN*0.00341802 + 149.0 - 273.15` (deg C); per-pixel mean over scenes | `build_dataset.py:150-153`, `178`, `184`, `201` |
| Cloud mask | `qa_pixel` bits 0-4 (fill, dilated cloud, cirrus, cloud, shadow) | pixels flagged are dropped per scene | `build_dataset.py:181` |
| Surface reflectance | `blue, green, red, nir08, swir16, swir22` | `DN*0.0000275 - 0.2` | `build_dataset.py:178`, `182-183` |
| NDVI | (nir-red)/(nir+red) | scene mean | `build_dataset.py:187` |
| NDBI | (swir1-nir)/(swir1+nir) | scene mean | `build_dataset.py:188` |
| MNDWI | (green-swir1)/(green+swir1) | scene mean | `build_dataset.py:189` |
| ALBEDO | 0.356 blue + 0.130 red + 0.373 nir + 0.085 swir1 + 0.072 swir2 - 0.0018 (inferred: Liang-type broadband albedo; the repo does not name it) | scene mean | `build_dataset.py:190` |
| LST_std, n_obs | per-pixel std of LST, count of valid passes | pixels with fewer than min(5, n_scenes) passes set to NaN | `build_dataset.py:201-207` |
| ELEV | Copernicus DEM `cop-dem-glo-30`, band `data` | bilinear to grid | `build_dataset.py:216-218` |
| SLOPE_deg | from ELEV gradient | degrees | `build_dataset.py:398-400` |
| LANDCOVER | ESA WorldCover `esa-worldcover` 2021, band `map` | mode resampling; categorical, 11 codes 10..100, 80 = water | `build_dataset.py:220-222`, `make_patches.py:34-35` |
| GHSL_HEIGHT | GHS_BUILT_H ANBH E2018 R2023A, 3 arcsec | bilinear | `build_dataset.py:107-108`, `270-272` |
| GHSL_BUILT_FRAC | GHS_BUILT_S E2020 R2023A, 3 arcsec | m2 per cell / cell area, clipped 0-1 | `build_dataset.py:109-110`, `273-278` |
| GHSL_POP | GHS_POP E2020 R2023A, 3 arcsec | people per 30 m cell | `build_dataset.py:111-112`, `279-280` |
| Neighbourhood | `NDVI_nbr150/450`, `BUILT_nbr150/450`, `HEIGHT_nbr150/450` | uniform filter 5 px (150 m) and 15 px (450 m) | `build_dataset.py:384-387` |
| DIST_WATER_m | distance to water (MNDWI > 0.1 or WorldCover 80) | EDT x 30 m; 5000 m fallback if no water | `build_dataset.py:389-394` |
| DIST_GREEN_m | distance to NDVI > 0.4 | EDT x 30 m; 5000 m fallback | `build_dataset.py:395-397` |
| Surroundings ("context") | for NDVI, NDBI, ALBEDO, MNDWI, GHSL_BUILT_FRAC, GHSL_HEIGHT: NaN-aware window means `_c90/_c330/_c930/_c2010` (3/11/31/67 px), plus `NDVI_sd330/930`, `GHSL_BUILT_FRAC_sd330/930` | 28 features | `extraction/add_context_features.py:7-9`, `24-26`, `29-48`; `README.md:252` |
| City-relative copies (patches only) | `NDVI_rel, NDBI_rel, ALBEDO_rel, GHSL_BUILT_FRAC_rel, GHSL_HEIGHT_rel, ELEV_rel` | (x - city mean)/max(city std, 0.25 x training std) | `make_patches.py:36-37`, `42-45`, `91-96` |
| Weather (table only, **not a model input**) | Open-Meteo ERA5-based hourly T2M, RH, wind, shortwave, precipitation at 10-11 h IST on scene dates; `MET_T2M, MET_RH, MET_WIND, MET_SW, MET_RAIN7, MET_RAIN30, MET_T2M_SEASON_MAX` | city-level constants | `build_dataset.py:343-375`; excluded: `pre_processing/export_city_grids.py:11`; `README.md:250` "(tables only)" |
| OSM (optional `--osm`, **not used by model**) | buildings, water, green | `BLDG_FRAC, OSM_HEIGHT, WATER_FRAC, GREEN_FRAC` | `build_dataset.py:285-339`, `531`; not in `export_city_grids.py:28-33` |
| Labels (not model inputs) | `LST_z`, `SUHI` (LST minus mean LST of WorldCover 30/40 rural pixels), `IS_URBAN` (WorldCover 50 or built >= 0.15), `HOTSPOT` (LST >= 90th pct of urban pixels) | | `build_dataset.py:404-420`; excluded as leakage: `export_city_grids.py:11` |

**Model input set** (`export_city_grids.py:28-33`): 18 BASE layers (incl. LANDCOVER) + 28 CONTEXT = 46 features; 45 numeric + LANDCOVER; patches add 6 `_rel` channels = 51 numeric channels + land-cover class (`README.md:220`, `make_patches.py:161-169`). Confirmed in `Deploy/best.pt`: `features` list has 51 names ending with the 6 `_rel` names; first conv weight shape `[32, 55, 3, 3]` = 51 numeric + 4 embedding channels (read from checkpoint).

README credits list: USGS/NASA Landsat 8/9 + Microsoft Planetary Computer; Copernicus DEM GLO-30; ESA WorldCover; JRC GHSL R2023A; Open-Meteo (ERA5-based) (`README.md:316-317`).

## 4. Pipeline steps in order

Documented order: `README.md:266-288`; one-shot script for the 36 added cities: `extraction/run_new_cities.sh:37-47`.

| # | Script | Reads | Writes | Notes |
|---|---|---|---|---|
| 1 | `extraction/build_dataset.py` | PC STAC (Landsat C2 L2, cop-dem-glo-30, esa-worldcover), JRC GHSL zips, Open-Meteo | `<out>/cities/<City>/aligned.nc` (all layers), `table.csv.gz` (every 3rd pixel, valid LST+NDVI), `meta.json` (attrs incl. `n_scenes`, `hotspot_threshold_C`, MET_*), `<out>/all_cities_training.csv.gz`; GHSL cache `<out>/_cache/ghsl` | `build_dataset.py:21-25`, `464-480`, `568-572`; stride 3 `530`; `block_id` = 3 km spatial blocks `115`, `475-476`; resumable, parallel workers `536-566` |
| 2 | `extraction/repair_dataset.py` | city folders | rebuilds cities with missing layers / patches weather, restacks | `repair_dataset.py:8-17`, `134-163` |
| 3 | `extraction/finalize_dataset.py` | `all_cities_training.csv.gz` | `all_cities_clean.parquet` (+ csv unless `--no-csv`) | drops NaN rows (default all cities), drops NDVI/NDBI/MNDWI/ELEV outside physical range; PASS/WARN/FAIL checks; writes only if no FAIL (`finalize_dataset.py:1-11`, `66-103`, `105-199`) |
| 4 | `extraction/add_context_features.py` | clean parquet + each `aligned.nc` | `all_cities_clean_ctx.parquet` | context features computed on full 30 m grid (`add_context_features.py:1-11`, `98-101`) |
| 5 | `extraction/check_dataset.py` | city folders | `<root>/quality_report.csv` | read-only audit OK/WARN/BAD (`check_dataset.py:1-15`) |
| (aux) | `extraction/grid_inventory.py`, `extraction/cell_inventory.py` | `aligned.nc` / clean table | `grid_inventory.csv`; `cell_inventory_<km>km.csv`, `india_data_coverage.png` | read-only inventories (`grid_inventory.py:1-11`, `cell_inventory.py:1-14`) |
| 6 | `pre_processing/export_city_grids.py` | `cities/<City>/aligned.nc` | `pre_processing/grids/<City>_features.nc` (46 features), `<City>_temp.nc` (LST, n_obs), `grid_summary.csv` | no leakage columns, no weather (`export_city_grids.py:1-11`, `45-54`, `67`) |
| 7 | `pre_processing/make_patches.py` | grids | `pre_processing/patches/{train,val}/<City>.npz` (X float16 (N,51,64,64), lc, y=LST deg C, mask, pos), `meta.json` (features, mean, std, rel, min_pass, max_few_share, landcover_codes, size, train_stride, train/val cities), `patch_summary.csv` | `make_patches.py:7-21`, `113-118`, `160-167` |
| 8 | `pre_processing/check_patches.py` | patches | stdout PASS/WARN/FAIL | `check_patches.py:1-10` |
| 9 | `pre_processing/pack_patches.py` (optional) | patches | `patches/packed/<split>_X.npy`, `<split>_rest.npz` for memory-mapped training | `pack_patches.py:1-9` |
| 10 | `models/train_unet.py` | patches (+ packed) | `<out>/best.pt`, `metrics.csv`, `val_per_city.csv` | `train_unet.py:15-17`, `220-227`; README run: `--base 32 --epochs 7 --out data/unet_run3` (`README.md:280`) |
| 11 | `models/evaluate_unet.py` | `best.pt`, val patches | `<run>/eval_val/eval_per_city.csv`, `eval_by_landcover.csv`, `eval_summary.png`, `map_<City>.png` | `evaluate_unet.py:7-18`, `151-156`, `200-202` |
| 12 | `models/compare_runs.py` | every `data/*/best.pt` | `data/model_comparison.csv`, `model_comparison_per_city.csv` | `compare_runs.py:1-14`, `98-99` |
| 13 | `models/group_importance.py` | `best.pt`, val patches | `<run>/group_importance.csv` | `group_importance.py:1-12`, `78-79` |
| 14 | `models/hotspot_report.py` | `best.pt`, grids, `patches/meta.json` | `data/hotspots/<City>_hotspots.csv`, `.png` | `hotspot_report.py:1-23`, `283-285`, `320` |
| 15 | `models/simulate_cooling.py` | `best.pt`, grids, `patches/meta.json` | `data/simulations/sim_<City>_<r0>_<c0>.csv`, `.png` | `simulate_cooling.py:1-17`, `293-296`, `353` |
| 16 | `dashboard/app.py` (+ `engine.py`, `live.py`, `plots.py`, `paths.py`, `check_setup.py`) | `Deploy/best.pt`, `Deploy/meta.json`, optionally `Deploy/grids/`; live mode downloads data | `Deploy/cache/` (predictions `<city>_<run>_pred.npy`, live grids, GHSL tiles) | `dashboard/paths.py:4-12`, `20-29`; `engine.py:51-68`; `app.py:41`; `Deploy/README.md:5-10` |

Live pipeline reuses steps 1 and 6 in-process (`dashboard/live.py:7-11`, `83-124`).

## 5. Models / algorithms

**Architecture** (`models/train_unet.py:65-97`):
- U-Net, 3 encoder levels + bottleneck; each block = 2 x (Conv3x3 no bias, BatchNorm, ReLU), optional Dropout2d (`train_unet.py:65-70`).
- Channels `[base, 2b, 4b, 8b]`; final model `base=32` (`README.md:223`; checkpoint `base: 32`) so 32/64/128/256 (inferred from code + base).
- Max-pool x2 downsampling, ConvTranspose2d x2 upsampling, skip concatenation, 1x1 output conv to 1 channel (`train_unet.py:79-97`).
- Dropout in bottleneck and the two deeper decoder blocks (`train_unet.py:80`, `83`).
- Land cover: learned `nn.Embedding(n_lc + 1, 4)` (last index = unknown/255), concatenated to inputs (`train_unet.py:76`, `86-88`). Checkpoint: embedding weight `[12, 4]`, 11 land-cover codes `[10,20,30,40,50,60,70,80,90,95,100]`.
- "about 2 M parameters" (`README.md:222`).
- Inputs clipped to +-10 standard deviations (`train_unet.py:62`, `199`; `README.md:254`).

**Target and loss**:
- `--target anomaly` (default) = LST minus the mean LST of that patch's city (over usable pixels); `absolute` alternative (`train_unet.py:8-12`, `53-59`, `151`). Final model: anomaly (`README.md:178`, `225`; checkpoint `target: anomaly`).
- Target standardised by training mean/std `mu, sd` (`train_unet.py:179-181`), stored in checkpoint (`train_unet.py:224-225`). Checkpoint values: `mu = 1.92e-06`, `sd = 2.929` (read from `Deploy/best.pt`).
- Loss: `F.smooth_l1_loss` (Huber) on `mask == 1` pixels only (`train_unet.py:205`). Mask = LST present, every feature present, not water, not few-pass (`make_patches.py:13-17`, `80-89`) and 0 < LST < 75 deg C (`train_unet.py:46`).

**Training setup** (`train_unet.py:147-230`; README run `README.md:223`, `280`):
- AdamW, lr 2e-3, weight decay 1e-2, OneCycleLR (`train_unet.py:156-157`, `188-189`).
- Batch 64 (`train_unet.py:155`), dropout 0.1 (`158`), EMA of weights decay 0.998 with warm-up; EMA weights are validated and saved (`159`, `109-115`, `211-225`).
- Each epoch uses a random 50 % of training patches (`--epoch-frac 0.5`, `train_unet.py:153`, `186`).
- Augmentation: random 90-degree rotations and horizontal flip (`train_unet.py:100-106`).
- During training, validation uses every 2nd val patch (`--val-stride 2`, `train_unet.py:154`, `174-175`); best epoch by validation MAE is saved (`train_unet.py:222-227`).
- Final run: `--base 32 --epochs 7`, "on a 4-core CPU" (`README.md:223`). Defaults in code are `--base 16`, `--epochs 12` (`train_unet.py:152`, `160`).

**Patches / normalisation** (`pre_processing/make_patches.py`):
- 64 x 64 patches (1.9 km) (`make_patches.py:125`; `README.md:221`). Training stride = round(64 x 0.85) (15 % overlap); validation stride 64 (no overlap) (`make_patches.py:126`, `143`, `153-154`).
- Patches with < 50 % usable pixels dropped (`make_patches.py:127`, `103-105`).
- Feature mean/std from training cities only (every 7th usable pixel) (`make_patches.py:61-66`, `147-152`; `README.md:253`).
- Few-pass masking: pixels with `n_obs < 0.6 x city median` masked, but only in cities where they are <= 3 % of usable pixels (`make_patches.py:84-89`, `128-130`); README: masked groups were Srinagar, Hyderabad, Jammu (`README.md:256`).

**Split strategy**: whole cities, never by patch; 40 train / 10 validation (`make_patches.py:20`, `38-39`, `138-139`; `README.md:221`, `253`). There is no separate test set: the 10 validation cities also selected the best epoch, "so every score here is slightly optimistic" (`README.md:166-167`, `230`; `evaluate_unet.py:17-18`; `compare_runs.py:14`). The dataset builder also writes 3 km `block_id`s "ready for leave-one-city-out and spatial-block cross-validation" (`build_dataset.py:12-13`, `115`), but no script uses `block_id` for the U-Net (inferred from searching all model scripts).

**Models compared** (`README.md:174-180`; `models/compare_runs.py`):
- First model: absolute target, 45 inputs, 30 epochs.
- Smaller anomaly model: anomaly target, 45 inputs, 12 epochs.
- Final model: anomaly target, 51 inputs, 7 epochs.
- Older models use the first 45 of the 51 channels (`README.md:180`; `compare_runs.py:7-8`, `58-61`).
- Trivial baseline: "every pixel = the city's true mean" (`MAE_citymean`, `skill`, R2_within < 0 count) (`evaluate_unet.py:9-11`, `56-58`; `README.md:119`).
- Not compared: no non-deep-learning baseline (random forest, regression) exists in the repo (inferred from search). No ablation of the three changes (`README.md:129`).

**Metrics** (`evaluate_unet.py:7-15`, `44-63`), per validation city on usable pixels, with the city's true mean added back for anomaly models (`evaluate_unet.py:131-132`):
- MAE, RMSE, bias (pred - true).
- `R2_within` = 1 - SSE / within-city variance.
- `R2_anomaly` = same after removing each map's own mean ("R² offset removed").
- Spearman rank correlation.
- `hot_AUC` (rank AUC of true top-10 % pixels; `--hot-share 0.10`), `hot_F1`, `hot_IoU` (predicted top 10 % vs true top 10 %), `hot_bias` (bias on true hot pixels).
- `skill` = 1 - MAE/MAE_citymean.
- Error by land cover (`evaluate_unet.py:138-141`, `153-156`).
- Pooled R2 is printed but flagged as not comparable for anomaly runs (`evaluate_unet.py:162-164`).
- `compare_runs.py` ranks models by mean `R2_anomaly` (`compare_runs.py:11-12`, `38`).

## 6. Outputs

**Prediction**: a per-pixel LST anomaly map; at inference, the city's observed mean LST over valid pixels is added back (`hotspot_report.py:8`, `179-181`, `193`). Whole-city inference uses 512-px tiles with a 32-px margin (`hotspot_report.py:64-73`); inputs are padded to multiples of 8 (`simulate_cooling.py:66-71`).

**Hotspot definitions** (three different ones exist in the repo):
1. Tool / dashboard hotspot (Getis-Ord Gi*):
   - Computed on block means of the predicted LST, block = 20 px = 600 m, with >= 50 % usable pixels (`hotspot_report.py:9-11`, `76-80`, `198`, `331`).
   - Gi* over a 3 x 3 block window (`hotspot_report.py:83-90`); hot if z >= `--z`, default 2.58 (`hotspot_report.py:332`). The docstring still says 1.96 (`hotspot_report.py:10`).
   - 8-connected clusters of >= 2 blocks (`hotspot_report.py:113-120`, `335`), kept only if predicted gap to the neighbour ring is >= 1.0 deg C (`hotspot_report.py:252`, `333-334`).
   - Neighbour ring = 30 px (900 m) around the cluster, excluding other hotspots (`hotspot_report.py:249`, `336`).
   - Per cluster: `obs_gap`, `pred_gap`, `not_explained = obs_gap - pred_gap` (`hotspot_report.py:258-261`).
   - Dashboard: z slider over {1.645, 1.96, 2.58, 3.29}, default 2.58 (`dashboard/app.py:167`). Block = 20 px for prepared cities, 10 px (300 m) for live windows (`app.py:270`, `281`).
2. Evaluation hotspot: the hottest 10 % of each city's usable pixels (`evaluate_unet.py:14-15`, `50-52`, `105`; `README.md:150`). This is the one behind "Hotspot AUC 0.93".
3. Dataset label `HOTSPOT`: LST >= 90th percentile of urban pixels, urban only (`build_dataset.py:417-419`). Used only in tables and audits, never as a model input or target (`export_city_grids.py:11`).

**Hotspot validation**: predicted-block hotspots compared with Gi* hotspots on the observed LST (precision / recall / IoU on blocks, block-mean correlation) (`hotspot_report.py:205-212`).

**Vegetation sanity check**: the model's mean LST change for +0.1 NDVI on all NDVI channels, compared with the satellite NDVI-LST slope, both raw and within strata of land cover x built-up bins x elevation quintile x distance-to-water bins (groups >= 200 px) (`hotspot_report.py:214-234`).

**Driver analysis**:
- *Per hotspot, Shapley.* Features fall into 6 groups: vegetation (NDVI*, DIST_GREEN), built-up (NDBI*, GHSL*, BUILT*, HEIGHT*), albedo, water (MNDWI*, DIST_WATER), terrain (ELEV*, SLOPE), land cover (`hotspot_report.py:44-61`). For each of the 2^6 = 64 subsets, the absent groups take the ring's mean value (land cover takes the ring's modal class) and the model predicts again. Exact Shapley values are reported in deg C above "a neighbour-like version of the cluster". `observed gap = sum of groups + reference mismatch + not explained` (`hotspot_report.py:14-17`, `93-101`, `127-142`, `268-281`). The dashboard does the same for the user's rectangle (`dashboard/engine.py:198-200`).
- *Global, permutation.* `models/group_importance.py:7-12`, `47-66` shuffles each group between patches of the same validation city (150 patches per city by default, `group_importance.py:32`) and reports the drop in R2_anomaly, Spearman and hot AUC.

**Cooling simulation ("doses")** (`models/simulate_cooling.py:1-17`, `74-139`, `222-236`; dashboard reuse `engine.py:182-221`):
- Area = user box or `--auto N` (block with highest observed LST minus its ring; `--min-built` filter) (`simulate_cooling.py:42-62`, `242-246`). Neighbours = ring of 30 px (900 m) (`simulate_cooling.py:244`; dashboard slider 150-1500 m, `app.py:169`).
- Lever "vegetation":
  - Eligible pixels are non-water with NDVI < 0.4 and built fraction < 0.6 (`simulate_cooling.py:93`, `250`).
  - Converted pixels get the city's median NDVI, NDBI, MNDWI and ALBEDO of its tree pixels (WorldCover 10 and NDVI > 0.4), and land cover is set to 10 (tree) (`simulate_cooling.py:111-113`, `167-173`).
- Lever "cool roofs":
  - Eligible pixels are WorldCover 50 or built fraction >= 0.3 (`simulate_cooling.py:94`).
  - ALBEDO is raised by 0.15, capped at 0.6 (`simulate_cooling.py:114`, `251`).
  - The lever is "not applicable" if < 30 built pixels (`simulate_cooling.py:307-309`; `app.py:465`).
- **Dose** = share (0-100 %) of the eligible pixels converted, in random but fixed order (seed 0), for veg, roof and both at once (`simulate_cooling.py:95-97`, `105-108`, `228-229`). Steps of 10 % by default (`simulate_cooling.py:252`); the dashboard uses 5 steps for boxes > 90,000 px (`app.py:418`).
- After each edit:
  - nbr150/450, all 90 m-2 km context windows, DIST_GREEN and the `_rel` channels are recomputed (`simulate_cooling.py:115-121`, `131-133`).
  - The model re-predicts, and the area's mean is compared with the ring's (`simulate_cooling.py:142-145`, `230-234`).
- The dose that closes the gap is found by linear interpolation to gap = 0; `None` if never reached (`simulate_cooling.py:148-155`). Results are reported as % and hectares (0.09 ha per pixel) (`simulate_cooling.py:38`, `311-319`).
- A stricter target is also reported: the hottest 10 % of the area brought to the ring level (`simulate_cooling.py:144`, `320-321`).
- A drift check (features recomputed with no edits) warns if > 0.05 sd (`simulate_cooling.py:210`, `287`).
- Dashboard verdict:
  - "Not hotter" if `pred_gap <= 0.2`.
  - "A hotspot" if `pred_gap >= 1.0` and >= 50 % of the area lies in Gi* hot blocks.
  - Otherwise "warmer" (`app.py:425-437`).
- Sliders let the user try a custom mix (`app.py:487-496`).

## 7. Results actually present in the repo

No metrics files, logs or `meta.json` from training are committed. README says the charts "were redrawn from the numbers printed by our own evaluation, comparison, importance and simulator runs" and the Delhi map "is a crop of a real `simulate_cooling.py` figure" (`README.md:309-310`).

### 7.1 Headline numbers (README)
| Value | Meaning | Source |
|---|---|---|
| 50 cities, 30 m, ~89 M pixels, 46 features each | dataset | `README.md:19`, `218` |
| R² 0.74 "inside the city", on 10 unseen cities | final model (equals the "offset removed" mean, see 7.3) | `README.md:20`, `116` |
| Hotspot AUC 0.93 | final model | `README.md:21`, `118` |
| Delhi 600 m block 4.5 °C hotter than surroundings | observed | `README.md:10` |
| gap closes at ~45 % of open ground planted (~15 ha), or nearly all roofs reflective, or 35 % of each | simulator | `README.md:11-12` |
| ~2.4 °C of observed gap not explained | Delhi block | `README.md:13` |

### 7.2 First vs final model (10 held-out cities) — `README.md:114-119`
| Metric | First | Final |
|---|---|---|
| R² inside the city | 0.54 | 0.74 |
| Rank correlation inside the city | 0.78 | 0.86 |
| Hotspot AUC / F1 | 0.88 / 0.50 | 0.93 / 0.60 |
| Cities worse than city-average guess | 1 | 0 |

- First U-Net pooled R² 0.85 (`README.md:125`).
- "Kolkata turned out worse than guessing the city average", Bengaluru and Bhubaneswar barely better (`README.md:125-127`).
- Coimbatore and Dehradun "slipped a little" (`README.md:128`).

### 7.3 Final model per city — `README.md:152-164`
| City | MAE °C | R² inside | R² offset removed | Spearman | Hot AUC | Hot F1 | Bias hot px °C |
|---|---|---|---|---|---|---|---|
| Amritsar | 0.54 | 0.77 | 0.77 | 0.88 | 0.97 | 0.76 | -0.52 |
| Bengaluru | 0.90 | 0.75 | 0.76 | 0.86 | 0.93 | 0.60 | -0.53 |
| Bhubaneswar | 1.00 | 0.70 | 0.73 | 0.90 | 0.93 | 0.58 | -0.18 |
| Chennai | 0.93 | 0.76 | 0.79 | 0.85 | 0.90 | 0.54 | -1.21 |
| Coimbatore | 1.74 | 0.68 | 0.75 | 0.91 | 0.95 | 0.65 | -1.09 |
| Dehradun | 1.96 | 0.67 | 0.69 | 0.86 | 0.94 | 0.61 | -3.99 |
| Jaipur | 0.98 | 0.78 | 0.79 | 0.89 | 0.95 | 0.59 | -1.07 |
| Kolkata | 0.82 | 0.78 | 0.79 | 0.91 | 0.95 | 0.62 | -1.87 |
| Lucknow | 1.23 | 0.64 | 0.64 | 0.74 | 0.92 | 0.56 | -1.83 |
| Nagpur | 1.06 | 0.71 | 0.74 | 0.83 | 0.92 | 0.54 | -1.18 |
| **Mean** | **1.12** | **0.72** | **0.74** | **0.86** | **0.93** | **0.60** | **-1.35** |

Notes:
- MAE is with the city mean given, i.e. pattern error (`README.md:166`).
- "Hot" = hottest 10 % (`README.md:150`).
- Peaks are underestimated by ~1.35 °C on average and 4 °C in Dehradun (`README.md:232`).
- "Think about 0.7, not 0.74" (`README.md:230`).

### 7.4 All trained models, same validation pixels — `README.md:174-178`
| Model | Target | Inputs | Epochs | R² offset removed | Spearman | Hot AUC | Hot F1 | Bias hot px | Cities worse than city mean |
|---|---|---|---|---|---|---|---|---|---|
| First | absolute | 45 | 30 | 0.54 | 0.78 | 0.88 | 0.50 | -1.45 °C | 1 |
| Smaller anomaly | anomaly | 45 | 12 | 0.53 | 0.74 | 0.89 | 0.47 | -2.45 °C | 0 |
| Final | anomaly | 51 | 7 | 0.74 | 0.86 | 0.93 | 0.60 | -1.35 °C | 0 |

### 7.5 Error by land cover (final model) — `README.md:187-195`
| Land cover | Pixels | MAE °C | Bias °C |
|---|---|---|---|
| Cropland | 6.74 M | 0.92 | +0.04 |
| Tree cover | 4.98 M | 1.37 | +0.43 |
| Built-up | 3.40 M | 1.12 | -0.07 |
| Grassland | 2.13 M | 1.16 | +0.39 |
| Shrubland | 0.36 M | 1.06 | +0.19 |
| Bare / sparse | 0.18 M | 2.21 | -0.83 |
| Wetland | 0.02 M | 1.05 | +0.24 |

### 7.6 Hotspot detection quality (Gi* z >= 2.58, clusters >= 1 °C) — `README.md:202-207`
| | Delhi | Jaipur |
|---|---|---|
| Blocks flagged (600 m) | 11.1 % | 18.9 % |
| Clusters kept after 1 °C filter | 7 (3.7 % of city) | 5 (16.0 % of city) |
| Precision / recall / IoU vs observed-LST hotspots | 0.57 / 0.50 / 0.36 | 0.75 / 0.78 / 0.62 |
| Correlation of block means obs vs pred | 0.85 | 0.91 |

### 7.7 Cooling doses — `README.md:48-52`
| Area | Hotter than neighbours | What closes the gap |
|---|---|---|
| Jaipur, strongest 600 m hotspot | 4.4 °C observed, 2.2 °C model | plant 40 % of open pixels (14 ha) |
| Delhi, dense 600 m block | 4.5 °C observed, 2.0 °C model | plant 45 % (15 ha), or roofs 99 %, or 35 % each |
| Delhi, 1.2 x 1.8 km | 1.3 °C observed, 1.0 °C model | plant 30 % (57 ha), roofs 48 % (76 ha), or 21 % each |

### 7.8 Vegetation check and importance
- +0.1 NDVI: model cools 1.0 °C (Delhi), 2.0 °C (Jaipur); satellite matched-group estimate 0.8 and 1.8 °C; the model is 10-30 % steeper (`README.md:135-137`).
- Group permutation R² drop: vegetation 0.89, water 0.39, built-up 0.31, others < 0.1 (`README.md:141`).

### 7.9 Data and training facts — `README.md:216-225`
- 10.08 M rows stacked; 9.25 M after cleaning; quality audit 42 cities OK, 8 warnings, 0 bad (`README.md:219`).
- ~21,000 training patches (~15 % overlap), ~4,300 validation patches; 40 train / 10 val cities (`README.md:221`).
- U-Net, 3 down-sampling levels, ~2 M parameters (`README.md:222`).
- Live-mode test, fresh Pune area (`README.md:75-78`):
  - ~4 minutes in total on a laptop connection.
  - 26 Landsat scenes, ~2.5 min; building layers ~1 min the first time.
  - Against the original build of the same place: median feature correlation 0.998, temperature correlation 1.000.
- Live mode needs ~4 GB RAM (`README.md:81`). Training reproduction needs ~32 GB RAM, 4-core Azure ML instance (`README.md:263`).
- Prepared grids ~200 MB per city (`README.md:70`; `Deploy/README.md:9`); `best.pt` ~8 MB (`Deploy/README.md:7`); GHSL tile ~85 MB per layer per tile (`dashboard/live.py:16`).

### 7.10 Figures in docs/ (values as printed on the figure)
- **`docs/fig_cooling_curves.png`**: "How much change does it take to cool a hotspot down to its surroundings?" Two panels of predicted gap "Area minus neighbours (°C)" vs "Share of the changeable surface converted (%)", lines Plant / Cool roofs / Both, dots at the zero crossing.
  - Left, "Delhi, dense built-up 600 m block (36 ha)": gap starts at ~2 °C at 0 % (read off axis). Labelled doses: Both 35 %, Plant 45 %, Cool roofs 99 %. At 100 %, Plant ~-2.2 and Both ~-2.3 (read off axis).
  - Right, "Delhi, 1.2 x 1.8 km area (211 ha)": gap starts at ~1.0 °C. Labelled doses: Both 21 %, Plant 30 %, Cool roofs 48 %. All lines reach ~-2.5 at 100 % (read off axis).
- **`docs/fig_delhi_drivers.png`**: "Why Delhi's biggest hotspot is hot". Subtitle: "2,556 ha cluster in south-west Delhi. Predicted gap to its neighbours: +2.2 °C (observed +3.2 °C)". Shapley bars (°C above a neighbour-like version): Vegetation +2.11, Built-up +1.10, Terrain +0.03, Land cover -0.04, Albedo -0.40, Water -0.63.
- **`docs/fig_delhi_map.png`**: "Delhi: the model finds the same hotspot the satellite sees". Left: "Satellite: observed temperature (°C)", colour bar ticks 38-44. Right: "Model: predicted (°C vs the city average)", colour bar ticks -4 to +4. Cyan box = area studied, dashed = neighbour ring, white = water left out. No numeric results beyond the scales.
- **`docs/fig_group_importance.png`**: "What the model leans on", "Shuffle one feature group at a time ... (10 held-out cities)". Drop in R² inside the city: Vegetation 0.89, Water 0.39, Built-up 0.31, Terrain 0.07, Albedo 0.03, Land cover 0.03.
- **`docs/fig_model_comparison.png`**: "Three trained models, scored on the same validation pixels", mean over 10 held-out cities. First / Smaller anomaly / Final:
  - R² inside the city 0.54 / 0.53 / 0.74
  - Spearman 0.78 / 0.74 / 0.86
  - Hotspot AUC 0.88 / 0.89 / 0.93
  - Hotspot F1 0.50 / 0.47 / 0.60
- **`docs/fig_per_city_r2.png`**: "The three failing cities are fixed; two strong cities slipped". R² inside each city (constant offset removed), first vs final, "Mean 0.54 to 0.74".
  - Labelled: Bengaluru 0.09 to 0.76, Bhubaneswar 0.12 to 0.73, Kolkata 0.14 to 0.79.
  - Final-model bars match `README.md:153-163` "offset removed" column (read off bar).
  - First-model bars for the other cities are unlabelled. Approximate visual reads, not data: Chennai ~0.59, Lucknow ~0.66, Jaipur ~0.72, Amritsar ~0.73, Nagpur ~0.74, Dehradun ~0.75, Coimbatore ~0.87.
- **`docs/fig_vegetation_check.png`**: "Is the vegetation effect real?" Cooling for +0.1 NDVI (°C), Model / Satellite no controls / Satellite matched groups: Delhi 1.03 / 0.73 / 0.77; Jaipur 2.00 / 1.15 / 1.79. Subtitle: model "10 to 30 % steeper than the matched-group estimate".

### 7.11 Notebook output — `notebooks/data_audit.ipynb`
- Only cell 1 has a stored output: `using C:\Hackathon\Heat_Map\data\smoke_test` / `8,100 rows x 39 columns, 4 MB`. This was a smoke-test folder, not the 50-city dataset.
- All other cells have no stored outputs. `EXPECTED_CITIES` lists only the original 14 cities (cell 1).

### 7.12 Checkpoint metadata — `Deploy/best.pt` (7,804,252 bytes on disk)
- Keys: `model, mu, sd, features, target, landcover_codes, base`.
- `target = "anomaly"`, `base = 32`, `mu = 1.920494e-06`, `sd = 2.929306`.
- 51 feature names (45 numeric + 6 `_rel`); 11 land-cover codes.
- Parameter count computed by me from the state_dict (not stated in the repo): 1,942,033 trainable weights. The README says "about 2 M" (`README.md:222`).

## 8. Tech stack, deployment, live mode

**Training/data environment** (`requirements.txt:1-20`, pinned):
- numpy 2.4.2, pandas 2.3.3, scipy 1.15.3, xarray 2026.9.0, dask 2026.8.0, netCDF4 1.7.2, h5netcdf 1.8.1.
- rasterio 1.5.2, rioxarray 0.23.0, pyproj 3.7.2, shapely 2.1.2, geopandas 1.1.1, osmnx 2.1.1.
- planetary-computer 1.0.0, pystac-client 0.9.0, odc-stac 0.5.3.
- requests 2.32.5, pyarrow 21.0.0, cfgrib 0.9.15.1, matplotlib 3.10.3.
- PyTorch installed separately (`README.md:263`); README says Python 3.10 (`README.md:263`). Linux conda env `azureml_py38` is referenced in `extraction/run_new_cities.sh:17-18`.

**App environment** (`requirements-app.txt:1-20`, ranges):
- numpy >=2.0,<2.5; pandas >=2.2,<3; scipy >=1.13,<1.17; xarray >=2024.6; dask >=2024.6; netCDF4 >=1.7.
- matplotlib >=3.8,<3.11; pyproj >=3.6,<3.8; pillow >=10,<12; requests >=2.31.
- rasterio >=1.4; rioxarray >=0.17; odc-stac >=0.3.11; pystac-client >=0.8; planetary-computer >=1.0.
- streamlit >=1.49,<2; folium >=0.17; streamlit-folium >=0.24.
- torch 2.5.1 CPU wheel (`requirements-app.txt:1-2`; `Dockerfile:11`; `ci.yml:35`).
- `dashboard/requirements.txt:1-5` (older): streamlit >=1.35, folium >=0.15, streamlit-folium >=0.20, pillow.

**Streamlit** (`dashboard/app.py`):
- Two modes, "Anywhere in India (live)" (default) and "Prepared cities" (`app.py:138-141`).
- Folium map with OSM + Esri imagery, rectangle draw tool (`app.py:116-126`, `203-210`).
- Metrics cards; tabs "Why is it hot?", "How much cooling is needed?", "Maps", "Numbers" (`app.py:449`); CSV download of the dose table (`app.py:508`).
- Shareable query params `?city=&box=` / `?rect=` (`app.py:225`, `294`, `512-514`).
- Theme: light, primary `#2a78d6`, usage stats off (`.streamlit/config.toml:1-10`).
- Paths: `Deploy/` if `Deploy/best.pt` exists, else `data/unet_run3`, `pre_processing/patches`, `pre_processing/grids`. Overridable by env vars `HEATMAP_RUN`, `HEATMAP_PATCHES`, `HEATMAP_GRIDS`, `HEATMAP_CACHE` (`dashboard/paths.py:11-12`, `20-29`).
- Pre-flight check: `dashboard/check_setup.py`.

**Live mode** (`dashboard/live.py`; `README.md:56-81`):
- Rectangle of 1-10 km per side inside India (`live.py:31-32`, `48-61`).
- Window half-size >= 8 km with a 3 km pad (`live.py:33`, `64-68`).
- Landsat cloud limit tried in steps 20 → 40 → 70 % (`live.py:34`, `98-105`); default 20 scenes, adjustable 8-40 (`live.py:81`; `app.py:171`).
- Same years and months as training (`live.py:81`).
- One build at a time, enforced by a lock (`live.py:36`, `87-88`).
- Output cached as `live_<sha1>` grids (`live.py:71-74`).
- Stated lower reliability: fewer scenes; city-relative inputs and Gi* are relative to the window (`live.py:13-16`; `README.md:79-80`).

**Docker** (`Dockerfile:1-24`):
- Base `python:3.12-slim`; installs torch 2.5.1 CPU + `requirements-app.txt`.
- Copies dashboard, extraction, pre_processing, models, .streamlit, Deploy.
- Port 8501, healthcheck `/_stcore/health`. "Plan for 4 GB of RAM or more" (`Dockerfile:4`).
- `.dockerignore` excludes docs, tests, notebooks, data, Deploy/grids, Deploy/cache (`.dockerignore:1-11`).

**CI/CD** (`.github/workflows/ci.yml`):
- Triggers: push to main, PRs, manual dispatch (`ci.yml:12-16`).
- Job `test` (Python 3.12): compileall syntax check, then `python -m unittest discover -s tests -v` (`ci.yml:23-40`). Tests use a synthetic 200 x 200 px city near Pune and an untrained model, with no network (`tests/synthetic.py:1-6`, `24`; `tests/test_app.py:1-4`).
- Job `image` (main only): fails if `Deploy/best.pt` or `Deploy/meta.json` is missing; builds and pushes `ghcr.io/<repo>:latest` and `:<sha>` (`ci.yml:42-71`).
- Optional deploys selected by repo variable `DEPLOY_TARGET`:
  - `huggingface`: pushes to a Docker Space (`ci.yml:73-93`).
  - `azure`: `az containerapp update` (`ci.yml:95-113`).
- README one-time Azure setup: `az group create -n heat-rg -l centralindia`, a containerapp env `heat-env`, and app `urban-heat` with 2 CPU / 4Gi / port 8501 / min 1 replica (`README.md:101-105`).

## 9. Azure references (names only)

Search terms: azure, blob, abfs, wasb, AZURE, SAS, storage (case-insensitive) over all tracked files.
- **No Azure Storage usage**: no storage account, container, blob path, `abfs://`/`wasb://` URL, connection string, or Azure SDK import (`azure.*`, `adlfs`, `fsspec`) anywhere in code or requirements.
- GitHub Actions Azure deploy (`.github/workflows/ci.yml:8-9`, `95-113`; `README.md:96`):
  - Repository variables: `DEPLOY_TARGET` (value `azure`), `AZURE_CONTAINER_APP`, `AZURE_RESOURCE_GROUP`.
  - Repository secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` (OIDC federated credential, `azure/login@v2`, `id-token: write`).
- README Azure CLI names: resource group `heat-rg`, region `centralindia`, env `heat-env`, app `urban-heat` (`README.md:102-104`).
- Other mentions:
  - "Developed on a 4-core Azure ML instance" (`README.md:263`).
  - `best.pt` / `meta.json` / grids come from "the Azure VM" / "the VM" paths `data/unet_run3/best.pt`, `pre_processing/patches/meta.json`, `pre_processing/grids/` (`Deploy/README.md:7-9`).
  - `.gitignore:1` comment: "Data (too large for GitHub; regenerate or store in Azure)".
  - conda env path `/anaconda/envs/azureml_py38` (`extraction/run_new_cities.sh:17-18`).
- "SAS" appears only for Planetary Computer's own expiring SAS tokens on Landsat asset URLs, re-signed via `planetary_computer.sign_inplace` (`build_dataset.py:161-176`). These are not user credentials.
- Other secret/var names: `HF_TOKEN` (secret), `HF_SPACE` (variable) (`ci.yml:80-82`); `GITHUB_TOKEN` (`ci.yml:62`).

## 10. Ambiguities and open questions for the author

1. **Which R² is the headline?**
   - "R² 0.74 inside the city" (`README.md:20`, `116`) equals the mean of "R² offset removed" (R2_anomaly, `README.md:164`, `178`). The "R² inside" (as scored, R2_within) mean is 0.72 (`README.md:164`).
   - `fig_per_city_r2.png` and `fig_model_comparison.png` both say "R² inside the city" for the offset-removed values. Confirm the label for the dashboard.
2. **No independent test set.** The 10 "held-out" cities also picked the best epoch, and `compare_runs` ranks models on them (`README.md:166-167`, `230`; `compare_runs.py:14`), yet `README.md:20` says "10 cities the model never saw". How should the methodology phrase this?
3. **First-model failure story.** README says Kolkata was worse than the city average (`README.md:125-126`). The figure shows Kolkata at 0.14 and Bengaluru lowest at 0.09 (R² offset removed). Which metric shows Kolkata < 0 (presumably R2_within, inferred)? The per-city table for the first and smaller models is not in the repo.
4. **Which checkpoint produced the hotspot/simulator numbers?**
   - `simulate_cooling.py` defaults to `--run data/unet_run2` (`simulate_cooling.py:247`), while `hotspot_report`, `group_importance` and the dashboard default to `unet_run3` (`hotspot_report.py:328`, `group_importance.py:30`, `paths.py:27`).
   - `README.md:45` runs the simulator without `--run`. Were the README dose numbers (`README.md:50-52`) and the cooling figure produced with the final model?
5. **Gi* default threshold.** Docstring says z 1.96 (`hotspot_report.py:10`); code default is 2.58 (`hotspot_report.py:332`; `README.md:200` uses 2.58).
6. **Three hotspot definitions** (Gi* clusters; top-10 % pixels for AUC/F1; dataset `HOTSPOT` label = top-10 % of urban pixels). The headline "Hotspot AUC 0.93" uses the top-10 % pixel definition, not the Gi* definition used by the tool. Which should the dashboard call "hotspot"?
7. **Delhi example consistency.**
   - The 600 m block is "4.5 °C observed, 2.0 °C model" (`README.md:51`), but `README.md:13` says ~2.4 °C unexplained; the difference of the rounded values would be 2.5 (inferred). Is the 2.4 from unrounded values?
   - The drivers figure is a different object (2,556 ha cluster, obs +3.2 / pred +2.2). Its pixel coordinates and run settings are not given.
   - The cooling figure labels the block "36 ha" while the planting dose is "15 ha" (= share of greenable pixels, not of the area; inferred from `simulate_cooling.py:319`).
8. **Live-mode test used 26 Landsat scenes** (`README.md:75`), but the live default is 20 (`live.py:81`; `README.md:79`). Was `max_scenes` raised for that test?
9. **`Deploy/meta.json` is missing.**
   - The CI `image` job will fail (`ci.yml:50-53`), and the dashboard cannot start without it (`app.py:146-153`).
   - Feature means/stds for the deployed model are unavailable to us.
   - Is it available on the Azure VM?
10. **Raw evaluation outputs are absent** (`eval_per_city.csv`, `metrics.csv` training curves, `model_comparison*.csv`, `group_importance.csv`, `*_hotspots.csv`, `sim_*.csv`, `patch_summary.csv`, `quality_report.csv`, `grid_inventory.csv`). Figures were "redrawn" from printed numbers (`README.md:309`). Can the author share them for traceability, plus RMSE, skill and IoU, which are computed but never reported?
11. **Hyperparameters of the earlier models** (base width, lr, epochs actually run vs selected epoch) are not recorded. The README epochs (30, 12, 7) may be epochs run rather than the best epoch. What was the best epoch of the final run?
12. **Exact counts** for patches (~21,000 / ~4,300, `README.md:221`), training/validation pixels, and which cities triggered the "8 with warnings" audit (`README.md:219`) are not in the repo.
13. **Land-cover error table** (`README.md:187-195`): which split? Inferred validation, from the `evaluate_unet.py:103` default. Classes like Snow/Mangrove/Moss are absent; was that due to zero pixels?
14. **Python version.** README says 3.10 (`README.md:263`); Docker/CI use 3.12 (`Dockerfile:5`, `ci.yml:29`). `requirements.txt` pins very recent versions (e.g. numpy 2.4.2, xarray 2026.9.0); confirm the training environment versions.
15. **Unused prepared data.** Weather features, OSM layers and 3 km spatial blocks are built but not used by the U-Net (`export_city_grids.py:11`; `build_dataset.py:12-13`). Confirm, and say whether any spatial-block CV or leave-one-city-out was ever run.
16. **Validation-city choice.** "The hardest cities stay in training" (`make_patches.py:39`). Which cities were considered hardest, and could this bias the scores?
17. **Uncertainty.** Single training seed and a single split; no confidence intervals or variance across runs are reported anywhere.
18. **Vegetation check settings.** Exact settings (z, grid, run) behind 1.03/0.73/0.77 and 2.00/1.15/1.79 are not given. README rounds the satellite values to 0.8/1.8 (`README.md:135-136`).
19. **Dashboard copy.** It says "U-Net trained on 40 Indian cities" (`app.py:181`), while the README headline says 50 cities. Both are true (50 built, 40 trained, 10 validated), but the dashboard should state this consistently.
20. **Notebook is stale.** It only covers the original 14 cities and a smoke-test output (`notebooks/data_audit.ipynb` cell 1). Is there an audit of the 50-city table?
21. **Azure.** Training ran on an Azure ML instance / VM, but no Azure Storage location for `data/`, grids or patches is documented. Where do the full datasets live, if the dashboard team needs them?
