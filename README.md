# Urban Heat Hotspots for Indian Cities

Satellite data for 50 Indian cities, a U-Net that predicts how heat is distributed inside a city, and tools that use it to
**find hotspots**, **say what drives them**, and **estimate how much greening or cool roofing would bring them down to the level of
their surroundings**.

> Built for a hackathon problem on urban heat: identify hotspots, their drivers, and scenario-based cooling interventions.
> Everything below was measured on held-out cities, but it is a research prototype: read [Limitations](#limitations) before relying on any number.

## What it does

| Question | How it is answered | Script |
|---|---|---|
| Where is heat concentrated? | The model predicts the land surface temperature of every 30 m pixel; Getis-Ord Gi\* on 600 m blocks marks areas that are significantly hotter than the rest of the city. There is no fixed "top 10 %": a city can have many hotspots or none. | `models/hotspot_report.py` |
| What drives each hotspot? | Features are split into six groups (vegetation, built-up, albedo, water, terrain, land cover). Each group inside a hotspot is swapped for its neighbours' average and the model predicts again; all 64 combinations give exact Shapley values in degrees C. | `models/hotspot_report.py`, `models/group_importance.py` |
| What would cool it down? | A what-if simulator converts a growing share of an area to tree cover and/or raises roof albedo, recomputes every neighbourhood feature, re-predicts, and reports the dose at which the area is as cool as its neighbours. | `models/simulate_cooling.py` |

## Results

Final model (`models/train_unet.py --base 32 --epochs 7`), scored on **10 cities never used for training** (Jaipur, Lucknow, Nagpur,
Chennai, Bengaluru, Kolkata, Bhubaneswar, Dehradun, Amritsar, Coimbatore), on usable pixels only. The model predicts the pattern
*inside* a city, so scores are computed with each city's mean temperature given.

| Measure (mean over the 10 cities) | Earlier model (absolute target) | Final model |
|---|---|---|
| R² inside the city (constant offset removed) | 0.54 | **0.75** |
| Spearman rank correlation inside the city | 0.78 | **0.86** |
| Hotspot AUC (hottest 10 % of each city) | 0.88 | **0.93** |
| Hotspot F1 | 0.50 | **0.60** |
| Cities where the model loses to "every pixel = city mean" | 1 | **0** |
| MAE, city mean given | | 1.1 °C |
| Bias on the hottest pixels | -1.45 °C | -1.35 °C |

Per city, R² inside the city is 0.64 to 0.78, Spearman 0.74 to 0.91 and hotspot AUC 0.90 to 0.97. The best epoch was chosen on these same
10 cities, so the numbers are slightly optimistic. The MAE is not comparable with the earlier model's absolute-temperature error because
the final model gets the city mean as an input to the scoring.

**Example outputs** (model's view, land surface temperature at the Landsat pass):

| Area | Gap to neighbours | To close the gap |
|---|---|---|
| Jaipur, strongest 600 m hotspot | observed +4.4 °C, model +2.2 °C | vegetation: 40 % of the open pixels (14 ha) |
| Delhi, dense built-up 600 m block | observed +4.5 °C, model +2.0 °C | vegetation 45 % (15 ha), or cool roofs 99 % of built pixels, or 35 % of each |
| Delhi, 1.2 x 1.8 km area | observed +1.3 °C, model +1.0 °C | vegetation 30 % (57 ha), cool roofs 48 % (76 ha), or 21 % of each |

In these three cases 0.2 to 2.4 °C of the observed gap was **not explained by the model's features**, and the levers cannot remove that part.

## How it works

```
Landsat 8/9, DEM, WorldCover, GHSL, weather  ->  extraction/        50 city grids at 30 m (aligned.nc), cleaned tables
                                              ->  pre_processing/     feature grids + temperature grids -> 64x64 patches
                                              ->  models/             U-Net, evaluation, hotspots, drivers, simulator
```

**Data.** 50 cities, each a 40 km box at 30 m resolution (about 1,350 x 1,350 pixels, about 89 million pixels in total).
The target is the mean land surface temperature of the clearest Landsat 8/9 passes in March to May 2022-2025 (cloud-masked). Inputs come
from public sources: Landsat Collection 2 Level-2, Copernicus DEM and ESA WorldCover through Microsoft Planetary Computer, JRC GHSL
(building height, built-up fraction, population) and Open-Meteo weather (used in the tables only).

**Features.** 46 layers per pixel: NDVI, NDBI, MNDWI, albedo, elevation, slope, land cover, building height, built-up fraction,
population, distance to water and to green space, 150 m / 450 m neighbourhood means, and 28 "surroundings" features (window means from
90 m to 2 km and local variation). The patches add six city-relative copies (NDVI, NDBI, albedo, built-up fraction, building height and
elevation relative to the pixel's own city), giving 51 numeric channels plus the land-cover class.

**Patches.** 64 x 64 pixels (1.9 km); training patches overlap by about 15 %, validation patches do not. Cities are split as a whole
(40 train, 10 validation), never by patch, because neighbouring patches look alike. Feature statistics come from training cities only.
Water, missing values and pixels with unusually few satellite passes are masked out of the loss.

**Model.** A U-Net (three down-sampling levels, about 2 M parameters, a learned embedding for land cover) trained on the *anomaly*:
the pixel's temperature minus its city's mean. Smooth-L1 loss on usable pixels, AdamW with weight decay, dropout, a running average of the
weights, random rotations and flips, and features clipped at +-10 standard deviations.

**What made the difference** (found by inspecting the data and the first runs): the first model learned city-to-city differences well but
ranked pixels inside a city poorly, and failed in three cities (Kolkata, Bengaluru, Bhubaneswar). Adding city-relative inputs, masking
few-pass pixels and using the larger network together lifted inside-city R² from 0.54 to 0.75 and fixed those cities. We did not run the
ablation that would separate the three changes.

## Checks that shaped the work

- **Data audit.** A tile-offset bug in the GHSL reader left several cities with missing building data; it was found by comparing downloaded
  tiles and fixed. Cleaning rules (index ranges, water pixels, implausible temperatures) are documented in `extraction/finalize_dataset.py`.
- **Target noise.** Each pixel averages a different set of cloud-free days. We measured the effect against the land features: large blocks
  with fewer passes are fine, while tiny groups of few-pass pixels (Srinagar, Hyderabad, Jammu) are cold outliers, so only those are masked.
- **Is the vegetation effect real?** The model's change in temperature for +0.1 NDVI is -1.0 °C (Delhi) and -2.0 °C (Jaipur); the satellite
  data, compared within groups of equal land cover, built-up share, elevation and distance to water, gives -0.77 and -1.79 °C. The sign
  agrees and the model is 10 to 30 % steeper, so cooling doses are best read as the optimistic end.
- **What does the model rely on?** Shuffling feature groups between patches of the same city drops inside-city R² by 0.89 for vegetation,
  0.39 for water, 0.31 for built-up and under 0.1 for terrain, albedo and land cover. The model leans heavily on vegetation.

## Limitations

- **The scores come from 10 validation cities** that also picked the best epoch. Treat 0.75 as "about 0.7 to 0.75" until tested on new cities.
- **The model predicts the pattern inside a city, not absolute temperature.** It needs the city's mean temperature, so a city with no
  satellite temperature needs another way to supply it. Relative inputs need the whole city grid, not just a small box.
- **Hotspot temperatures are underestimated** by about 1.35 °C on average for the hottest pixels; the ranking is good, the peak values are low.
- **Drivers and cooling doses are the model's attributions, not measured causes.** Features such as vegetation, built-up share and albedo are
  correlated, and the doses assume random pixels are changed at 30 m, whereas real planting comes as parks and tree rows.
- **Part of the heat is outside the data** (traffic, industry, air conditioning, roof materials); it shows up as the "not explained" gap.
- **Surface temperature at the satellite pass** (late morning, clear days, March to May) is not air temperature or night-time heat.
- Large arid cities such as Jaipur give very large hot zones rather than compact hotspots, and their driver split is less precise.

## Repository layout

```
extraction/            data pipeline: build, repair, clean, context features, audits
  build_dataset.py       one aligned 30 m grid per city (Landsat, DEM, land cover, GHSL, weather)
  repair_dataset.py      fills missing layers / weather
  finalize_dataset.py    cleaning and validation checks
  add_context_features.py  surroundings features (90 m - 2 km windows)
  check_dataset.py, grid_inventory.py, cell_inventory.py   read-only audits and inventories
  run_new_cities.sh      build -> repair -> clean -> context for a list of cities
  align_data.py, get_missing_data.py, extract_data.ipynb, get_ERA5.ipynb   early single-city (Pune) prototypes
pre_processing/        feature / temperature grids per city and the 64 x 64 patches
  export_city_grids.py, make_patches.py, check_patches.py, pack_patches.py
models/                U-Net and everything built on it
  train_unet.py, evaluate_unet.py, compare_runs.py
  hotspot_report.py, group_importance.py, simulate_cooling.py
notebooks/data_audit.ipynb
```

Data, patches and trained weights are large and are not in the repository (`data/` and `pre_processing/grids`, `pre_processing/patches` are git-ignored).

## Reproduce

Python 3.10, a machine with about 32 GB of RAM, and `pip install -r requirements.txt` plus PyTorch (`pip install torch`, CPU or CUDA).
Developed on a 4-core Azure ML compute instance; on Linux you may need the conda C++ runtime on the library path
(`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`).

```bash
# 1. Data (several hours, network bound; resumable; every source is public)
python extraction/build_dataset.py --out data/multi_city --workers 4
python extraction/repair_dataset.py --out data/multi_city
python extraction/finalize_dataset.py --src data/multi_city/all_cities_training.csv.gz --no-csv
python extraction/add_context_features.py --root data/multi_city --no-csv
python extraction/check_dataset.py --root data/multi_city          # per-city quality report

# 2. Pre-processing (minutes to tens of minutes)
python pre_processing/export_city_grids.py --workers 4
python pre_processing/make_patches.py --workers 4
python pre_processing/check_patches.py

# 3. Model
python models/train_unet.py --base 32 --epochs 7 --out data/unet_run3
python models/evaluate_unet.py --run data/unet_run3                 # per-city scores and maps
python models/compare_runs.py                                       # rank all saved runs on the same pixels
python models/group_importance.py --run data/unet_run3

# 4. Hotspots, drivers and cooling what-ifs
python models/hotspot_report.py --city Delhi Jaipur
python models/simulate_cooling.py --city Delhi --run data/unet_run3 --box 600 640 1040 1100
python models/simulate_cooling.py --city Delhi --run data/unet_run3 --auto 20 --min-built 0.4   # strongest dense urban hotspot
```

A quick smoke test of the data builder on one small box: `python extraction/build_dataset.py --half 8 --cities Pune --out data_test`.

## Data credits

USGS / NASA Landsat 8 and 9 Collection 2 Level-2 and Microsoft Planetary Computer; Copernicus DEM GLO-30; ESA WorldCover; European
Commission JRC Global Human Settlement Layer (GHSL R2023A); Open-Meteo (ERA5-based weather archive). Check each source's licence before reuse.
