# 🔥 Urban Heat Hotspots for Indian Cities

**Which parts of a city are cooking, why, and how much green would it take to cool them down?**

We built a satellite dataset of 50 Indian cities, trained a U-Net to predict how heat is spread *inside* a city, and turned it into
three tools: a **hotspot finder**, a **"what is driving this heat?" explainer** and a **cooling what-if simulator**.

## The 30-second version

> A dense, built-up 600 m block in Delhi runs **4.5 °C hotter** than the streets around it.
> Our simulator says the gap closes if about **45 % of its open ground is planted** (roughly 15 hectares),
> *or* nearly all its roofs are made reflective, *or* a bit of both (35 % of each).
> About 2.4 °C of the observed gap is heat our data cannot explain (traffic, industry, air conditioning...), and the tool says so out loud.

| | |
|---|---|
| 🛰️ **50 cities**, 30 m pixels | about 89 million pixels, 46 features each |
| 🎯 **R² 0.75** inside the city | on 10 cities the model never saw |
| 🔎 **Hotspot AUC 0.93** | hot pixels ranked above cool ones |
| 🌳 **Cooling doses** | "how much green closes the gap?" for any area you pick |

## Three tools

**1. Where are the hotspots?** `models/hotspot_report.py`
The model predicts the temperature of every pixel, and Getis-Ord Gi\* flags blocks that are *significantly* hotter than the rest of the
city. No "top 10 %" quota: some cities have many hotspots, some have none.

**2. What is driving them?** `models/hotspot_report.py`, `models/group_importance.py`
Features are grouped (vegetation, built-up, albedo, water, terrain, land cover). For each hotspot we swap one group at a time for its
neighbours' average and ask the model again, which gives each group's exact share of the extra heat (Shapley values), in °C.

**3. What would cool it down?** `models/simulate_cooling.py`
Pick an area (draw a box or let it find the worst one). The simulator converts a growing share of it to tree cover and/or reflective roofs,
recomputes every neighbourhood feature, re-predicts, and tells you the dose at which the area is as cool as its surroundings.

```
python models/hotspot_report.py --city Delhi
python models/simulate_cooling.py --city Delhi --auto 20 --min-built 0.4
```

## More examples

| Area | Hotter than neighbours by | What closes the gap (model's view) |
|---|---|---|
| Jaipur, strongest 600 m hotspot | 4.4 °C observed, 2.2 °C seen by the model | plant 40 % of the open pixels (14 ha) |
| Delhi, dense 600 m block | 4.5 °C observed, 2.0 °C seen by the model | plant 45 % (15 ha), or reflective roofs on 99 % of buildings, or 35 % of each |
| Delhi, 1.2 x 1.8 km area | 1.3 °C observed, 1.0 °C seen by the model | plant 30 % (57 ha), roofs 48 % (76 ha), or 21 % of each |

## How well does it work?

Scored on **10 cities held out from training** (Jaipur, Lucknow, Nagpur, Chennai, Bengaluru, Kolkata, Bhubaneswar, Dehradun, Amritsar,
Coimbatore), on usable pixels, with each city's mean temperature given (the model predicts the pattern *inside* a city).

| | First model | Final model |
|---|---|---|
| R² inside the city | 0.54 | **0.75** |
| Rank correlation inside the city | 0.78 | **0.86** |
| Hotspot AUC / F1 | 0.88 / 0.50 | **0.93 / 0.60** |
| Cities worse than "every pixel = city average" | 1 | **0** |

### The plot twist

Our first U-Net scored a shiny R² of 0.85 and we nearly stopped there. Then we scored it *inside* each city, and Kolkata turned out
**worse than guessing the city average**, with Bengaluru and Bhubaneswar barely better. Most of that 0.85 was just "Jaipur is hotter than
Dehradun". Chasing it down led to the changes that mattered: giving the model each feature *relative to its own city*, masking pixels with
suspiciously few satellite passes, and a larger network. Those fixed all three cities. (We did not run the ablation that separates the three
changes, so we can't say which one did the most.)

### Did the model just make up the vegetation effect?

We checked. For +0.1 NDVI the model predicts a cooling of 1.0 °C in Delhi and 2.0 °C in Jaipur; the satellite data itself, compared across
places with the same land cover, built-up share, elevation and distance to water, shows 0.8 and 1.8 °C. Same direction, same range. The model
is 10 to 30 % steeper, so read the doses as the optimistic end. It leans on vegetation heavily: shuffling that group drops R² by 0.89, against
0.39 for water, 0.31 for built-up and under 0.1 for the rest.

## The honest part

- **Scores are slightly flattering.** The 10 cities also picked the best training epoch. Think "about 0.7", not "0.75".
- **It predicts patterns, not absolute temperature.** It needs the city's average temperature to be supplied, and it needs the whole city grid.
- **Peaks are underestimated** by about 1.35 °C on average. The ranking is good, the hottest values come out low.
- **Drivers and doses are the model's attribution, not proof of cause.** Vegetation, built-up share and albedo move together, and real planting
  comes as parks and tree rows, not scattered pixels.
- **Some heat is outside our data.** It shows up as the "not explained" gap.
- **This is land surface temperature** at the Landsat pass (late morning, clear days, March to May), not air temperature or how it feels at night.
- **Huge arid cities (Jaipur) give big hot zones**, not compact hotspots.

<details>
<summary><b>How it works (technical details)</b></summary>

```
Landsat 8/9, DEM, WorldCover, GHSL, weather  ->  extraction/       50 city grids at 30 m, cleaned tables
                                             ->  pre_processing/   feature + temperature grids -> 64x64 patches
                                             ->  models/           U-Net, evaluation, hotspots, drivers, simulator
```

- **Data.** 50 cities, each a 40 km box at 30 m (about 1,350 x 1,350 pixels). Target: mean land surface temperature of the clearest Landsat 8/9
  passes in March to May 2022-2025, cloud-masked. Public sources: Landsat Collection 2 Level-2, Copernicus DEM and ESA WorldCover (via Microsoft
  Planetary Computer), JRC GHSL (building height, built-up fraction, population), Open-Meteo weather (tables only).
- **Features.** 46 per pixel: NDVI, NDBI, MNDWI, albedo, elevation, slope, land cover, building height, built-up fraction, population, distance to
  water and green space, 150 m / 450 m neighbourhood means, and 28 "surroundings" features (window means from 90 m to 2 km, local variation).
  The patches add six city-relative copies, for 51 numeric channels plus the land-cover class.
- **Patches.** 64 x 64 pixels (1.9 km), about 15 % overlap in training and none in validation. Cities are split whole (40 train, 10 validation),
  never by patch. Statistics come from training cities only. Water, missing values and few-pass pixels are masked out of the loss.
- **Model.** U-Net, three down-sampling levels, about 2 M parameters, a learned embedding for land cover. It predicts each pixel's temperature
  minus its city's mean. Smooth-L1 loss on usable pixels, AdamW with weight decay, dropout, an averaged copy of the weights, random rotations and
  flips, features clipped at +-10 standard deviations. Trained with `--base 32 --epochs 7` on a 4-core CPU.
- **Data findings.** A tile-offset bug in the GHSL reader left several cities without building data (fixed). Each pixel averages a different set of
  cloud-free days; blocks with fewer passes are fine, but tiny few-pass groups (Srinagar, Hyderabad, Jammu) are cold outliers, so only those are masked.

</details>

<details>
<summary><b>Run it yourself</b></summary>

Python 3.10, about 32 GB of RAM, `pip install -r requirements.txt` plus PyTorch (`pip install torch`). Developed on a 4-core Azure ML instance; on
Linux you may need `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`. Data, patches and trained weights are not in the repo.

```bash
# 1. Data (hours, network bound, resumable, all sources public)
python extraction/build_dataset.py --out data/multi_city --workers 4
python extraction/repair_dataset.py --out data/multi_city
python extraction/finalize_dataset.py --src data/multi_city/all_cities_training.csv.gz --no-csv
python extraction/add_context_features.py --root data/multi_city --no-csv
python extraction/check_dataset.py --root data/multi_city

# 2. Pre-processing
python pre_processing/export_city_grids.py --workers 4
python pre_processing/make_patches.py --workers 4
python pre_processing/check_patches.py

# 3. Model
python models/train_unet.py --base 32 --epochs 7 --out data/unet_run3
python models/evaluate_unet.py --run data/unet_run3        # per-city scores and maps
python models/compare_runs.py                              # rank all saved runs on the same pixels
python models/group_importance.py --run data/unet_run3

# 4. Hotspots, drivers, cooling
python models/hotspot_report.py --city Delhi Jaipur
python models/simulate_cooling.py --city Delhi --run data/unet_run3 --box 600 640 1040 1100
```

Smoke test of the data builder on one small box: `python extraction/build_dataset.py --half 8 --cities Pune --out data_test`.

</details>

<details>
<summary><b>What is in the repo</b></summary>

```
extraction/       data pipeline: build_dataset, repair_dataset, finalize_dataset, add_context_features, audits, run_new_cities.sh
                  (align_data.py, get_missing_data.py and two notebooks are early single-city Pune prototypes)
pre_processing/   export_city_grids, make_patches, check_patches, pack_patches
models/           train_unet, evaluate_unet, compare_runs, hotspot_report, group_importance, simulate_cooling
notebooks/        data_audit.ipynb
```

</details>

## Credits

USGS / NASA Landsat 8 and 9 and Microsoft Planetary Computer; Copernicus DEM GLO-30; ESA WorldCover; European Commission JRC Global
Human Settlement Layer (GHSL R2023A); Open-Meteo (ERA5-based weather). Check each source's licence before reuse.
