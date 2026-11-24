# 04 Insights: Heat_Map results (Data Analyst)

Inputs: `02_data_inventory.md`, `data/{metrics,per_city,cooling,cities,model,figures}.json`, `README.md`, and, for metric definitions only, `models/evaluate_unet.py`, `models/compare_runs.py`, `models/group_importance.py` and `models/simulate_cooling.py`. Every value below comes from those files. Values from figures are used only where the figure prints the number as text, and they are marked **[fig text]**. Derived values show their formula. Calculations were run with `python -I` on `per_city.json`, `metrics.json` and `cooling.json`.

How to read the metrics (definitions from `models/evaluate_unet.py:9-14, 44-63`):
- `MAE` is mean |pred - obs| in °C, with the city's true mean supplied. It measures pattern error, not absolute-temperature error (README:166).
- `R2_within` = 1 - SSE / SST inside the city. Below 0 means the model does worse than "every pixel = city mean".
- `R2_anomaly` is the same after removing each map's own mean, so a constant offset is ignored. By construction R2_anomaly >= R2_within.
- `hot_*`: "hot" means the hottest 10 % of the city's pixels, in both truth and prediction. `hot_bias` = mean(pred - obs) on truly hot pixels, so a negative value means peaks come out too low.
- Reference points (derived from these definitions, not printed in the repo): a random ranking gives AUC 0.5. Because the predicted and true hot sets are both 10 % of pixels, random overlap gives precision = recall = 0.10, so F1 = 0.10.

---

## 1. Headline KPIs (hero row)

| # | Label | Value | Plain-English meaning | Source |
|---|---|---|---|---|
| 1 | R² inside the city (offset removed) | **0.74** | On 10 cities it never saw, the model explains about three quarters of the temperature variation between places inside a city. | `per_city.json` `mean.R2_anomaly`; `metrics.json` `headline.rows[0].final`, `models.rows[2].R2_anomaly`; README:116, 164, 178 |
| 2 | Hotspot AUC | **0.93** | Pick a random truly hot pixel and a random other pixel: 93 % of the time the model ranks the hot one higher (random = 0.50). | `per_city.json` `mean.hot_AUC`; `metrics.json` `headline.rows[2].final`; README:118, 164 |
| 3 | Rank correlation (Spearman) | **0.86** | The model orders the neighbourhoods from coolest to hottest very close to the satellite's order. | `per_city.json` `mean.spearman`; `metrics.json` `headline.rows[1].final`; README:117, 164 |
| 4 | Pattern error (MAE) | **1.12 °C** | Given a city's average temperature, a typical pixel's predicted temperature is about 1.1 °C off. | `per_city.json` `mean.MAE`; README:164, 166 |
| 5 | Held-out cities worse than the city average | **0 of 10** (first model: 1) | In every test city the model beats the trivial guess "every pixel = city mean". | `metrics.json` `headline.rows[4]`; README:119; check: `per_city.json` min `R2_within` = 0.64 > 0 |
| 6 | Coverage | **50 cities, about 89 M pixels at 30 m** | Scale of the dataset: 40 training and 10 held-out cities. | `metrics.json` `facts.rows[0]`; `cities.json` `n_cities`, `n_val`; README:218, 221 |

Alternates if a slot frees up: **Hotspot F1 0.60** (vs 0.10 random; `per_city.json` `mean.hot_F1`; README:118), **peaks underestimated by 1.35 °C** (`per_city.json` `mean.hot_bias` = -1.35; README:232), and **Delhi dense block: planting 45 % of open ground closes a 2.0 °C modelled gap** (`cooling.json` `rows[1]`; README:51).

**Labelling warning for KPI 1.** README:116 and the hero table (README:20) label 0.74 as "R² inside the city", but the per-city table's "R² inside" column averages **0.72** (`per_city.json` `mean.R2_within`), and 0.74 is the "offset removed" column. Show "0.74 (offset removed)" or "0.72 (as scored)". See section 6.

---

## 2. Distributions across the 10 held-out cities (`per_city.json` `cities[]`, README:152-164)

| Metric | Best city | Worst city | Min | Median | Max | Range (max - min) | Mean of 10 rows | README mean |
|---|---|---|---|---|---|---|---|---|
| MAE (°C, lower is better) | Amritsar 0.54 | Dehradun 1.96 | 0.54 | 0.99 | 1.96 | 1.42 | 11.16/10 = 1.116 | 1.12 |
| R² inside, as scored (`R2_within`) | Jaipur, Kolkata 0.78 | Lucknow 0.64 | 0.64 | 0.73 | 0.78 | 0.14 | 7.24/10 = 0.724 | 0.72 |
| R² offset removed (`R2_anomaly`) | Chennai, Jaipur, Kolkata 0.79 | Lucknow 0.64 | 0.64 | 0.755 | 0.79 | 0.15 | 7.45/10 = 0.745 | 0.74 |
| Spearman | Coimbatore, Kolkata 0.91 | Lucknow 0.74 | 0.74 | 0.87 | 0.91 | 0.17 | 8.63/10 = 0.863 | 0.86 |
| Hot AUC | Amritsar 0.97 | Chennai 0.90 | 0.90 | 0.935 | 0.97 | 0.07 | 9.36/10 = 0.936 | 0.93 (see section 6) |
| Hot F1 | Amritsar 0.76 | Chennai, Nagpur 0.54 | 0.54 | 0.595 | 0.76 | 0.22 | 6.05/10 = 0.605 | 0.60 |
| Bias on hot pixels (°C, closer to 0 is better) | Bhubaneswar -0.18 | Dehradun -3.99 | -3.99 | -1.135 | -0.18 | 3.81 | -13.47/10 = -1.347 | -1.35 |

Median = mean of the 5th and 6th sorted values, e.g. MAE (0.98 + 1.00)/2 = 0.99.

Thresholds the README defines:
- **Worse than the city mean** (`R2_within` < 0; README:119, `evaluate_unet.py:169`): **0 of 10** cities.
- **"Think about 0.7"** (README:230). `R2_within` >= 0.7 holds in **7 of 10** cities: Amritsar, Bengaluru, Bhubaneswar, Chennai, Jaipur, Kolkata, Nagpur. Below are Lucknow 0.64, Dehradun 0.67 and Coimbatore 0.68. `R2_anomaly` >= 0.7 holds in **8 of 10**: below are Lucknow 0.64 and Dehradun 0.69.
- **Hot AUC >= 0.90**: all **10 of 10** (minimum is Chennai at 0.90).
- **Hot F1 vs random 0.10** (derived reference): every city is at least 0.54 / 0.10 = 5.4x random.
- **Peaks underestimated by more than 1 °C** (`hot_bias` < -1): **7 of 10** cities (Chennai, Coimbatore, Dehradun, Jaipur, Kolkata, Lucknow, Nagpur). Without Dehradun the mean is -9.48/9 = **-1.05 °C**, against -1.35 with it. One city pulls the mean down by 0.30 °C, so the median (-1.135) describes a typical city better.
- **MAE above the 1.12 °C mean**: 3 cities, Coimbatore 1.74, Dehradun 1.96 and Lucknow 1.23.

Offset effect (`R2_anomaly - R2_within`) is 0.00 to 0.03 in nine cities. In **Coimbatore it is 0.75 - 0.68 = 0.07**, the largest constant offset of the ten.

City profiles:
- **Amritsar** is the all-round best: lowest MAE (0.54), best AUC (0.97) and best F1 (0.76).
- **Lucknow** is the weakest on pattern: last on R² (0.64 / 0.64) and on Spearman (0.74).
- **Dehradun and Coimbatore** have the two highest MAEs (1.96, 1.74) yet rank well (Spearman 0.86, 0.91). Their error is in level and peaks, not in ordering. Dehradun's peak bias is -3.99 °C.
- **Chennai** has the lowest hotspot scores (AUC 0.90, F1 0.54) despite a high R² (0.79).

---

## 3. Model comparison (`metrics.json` `models.rows`, README:174-178; same validation pixels, `compare_runs.py`)

| Metric | First (absolute target, 45 inputs, 30 epochs) | Smaller anomaly (45 inputs, 12 epochs) | Final (anomaly, 51 inputs, 7 epochs) | Final - First (abs / rel) | Final - Smaller (abs / rel) |
|---|---|---|---|---|---|
| R² offset removed | 0.54 | 0.53 | **0.74** | +0.20 / +37.0 % | +0.21 / +39.6 % |
| Spearman | 0.78 | 0.74 | **0.86** | +0.08 / +10.3 % | +0.12 / +16.2 % |
| Hot AUC | 0.88 | 0.89 | **0.93** | +0.05 / +5.7 % | +0.04 / +4.5 % |
| Hot F1 | 0.50 | 0.47 | **0.60** | +0.10 / +20.0 % | +0.13 / +27.7 % |
| Bias on hot pixels (°C) | -1.45 | -2.45 | **-1.35** | +0.10 (6.9 % smaller underestimate) | +1.10 (44.9 % smaller underestimate) |
| Cities with `R2_within` < 0 | 1 | 0 | **0** | -1 | 0 |

Relative change = (final - other) / |other|, e.g. R² (0.74 - 0.54)/0.54 = +37.0 %.

Error-reduction view, which is fairer near the ceiling:
- Unexplained variance (1 - R²) falls from 0.46 to 0.26: (0.46 - 0.26)/0.46 = **43 % less**.
- AUC shortfall (1 - AUC) falls from 0.12 to 0.07: (0.12 - 0.07)/0.12 = **42 % less**.

What it says:
- The **anomaly target alone did not lift the pattern scores**. The smaller anomaly model is level with or below the first model on R² (0.53 vs 0.54), Spearman (0.74 vs 0.78) and F1 (0.47 vs 0.50). It did remove the one failing city (1 to 0).
- The big jump arrives only with the final model, which changed several things at once: 6 city-relative inputs (45 to 51 channels), masking of few-pass pixels and a larger network (README:127-128).
- README:129 says the ablation was not run, so **no single change can be credited**.

Per-city first to final, from the printed labels **[fig text]** in `docs/fig_per_city_r2.png` (R² offset removed); the final values match `per_city.json` `R2_anomaly`:
- Bengaluru 0.09 to 0.76 (+0.67)
- Bhubaneswar 0.12 to 0.73 (+0.61)
- Kolkata 0.14 to 0.79 (+0.65)

README:128 says Coimbatore and Dehradun "slipped a little". The figure does not print their first-model values, so the size of that slip is not available as data.

Pooled vs inside-city: the first model's pooled R² was 0.85 (README:125), while its inside-city R² was 0.54. Most of that pooled score came from differences *between* cities.

---

## 4. Error by land cover, hotspot quality, cooling doses, drivers

### 4a. Error by land cover (`metrics.json` `landcover.rows`, README:187-195)

| Land cover | Pixels (M) | Share of pixels = px / 17.81 | MAE (°C) | MAE / 1.12 mean | Bias (°C) |
|---|---|---|---|---|---|
| Cropland | 6.74 | 37.8 % | **0.92** (best) | 0.82 | +0.04 |
| Tree cover | 4.98 | 28.0 % | 1.37 | 1.22 | **+0.43** |
| Built-up | 3.40 | 19.1 % | 1.12 | 1.00 | -0.07 |
| Grassland | 2.13 | 12.0 % | 1.16 | 1.04 | +0.39 |
| Shrubland | 0.36 | 2.0 % | 1.06 | 0.95 | +0.19 |
| Bare / sparse | 0.18 | 1.0 % | **2.21** (worst) | 1.97 | **-0.83** |
| Wetland | 0.02 | 0.1 % | 1.05 | 0.94 | +0.24 |

Total pixels = 6.74 + 4.98 + 3.40 + 2.13 + 0.36 + 0.18 + 0.02 = 17.81 M.
Pixel-weighted MAE = Σ(px × MAE)/17.81 = **1.13 °C**, which agrees with the city-mean MAE of 1.12. Pixel-weighted bias = **+0.16 °C**.

What it says:
- **Built-up pixels, the ones that matter for urban heat, are predicted at the average error** (1.12 °C) with almost no bias (-0.07).
- The model **runs warm over vegetation**: tree cover +0.43 and grassland +0.39. It **runs cold over bare ground** (-0.83), with nearly double the average error (2.21). Bare ground is only 1 % of pixels.
- Wetland (0.02 M pixels) and bare ground (0.18 M) are too small for firm conclusions.

### 4b. Hotspot detection quality (`metrics.json` `hotspot_detection`, README:202-209; Gi* z >= 2.58, clusters >= 1 °C above neighbours)

| | Delhi | Jaipur |
|---|---|---|
| Split in `cities.json` | **train** | val (held out) |
| Blocks flagged | 11.1 % | 18.9 % |
| Clusters kept | 7 (3.7 % of city) | 5 (16.0 % of city) |
| Precision / recall / IoU | 0.57 / 0.50 / 0.36 | 0.75 / 0.78 / 0.62 |
| Derived F1 = 2PR/(P+R) | 2·0.57·0.50/1.07 = 0.53 | 2·0.75·0.78/1.53 = 0.76 |
| Consistency check: IoU = F1/(2 - F1) | 0.533/1.467 = 0.36 ✓ | 0.765/1.235 = 0.62 ✓ |
| Correlation of block means, observed vs predicted | 0.85 | 0.91 |
| Average cluster size = area % / clusters | 3.7/7 = 0.53 % of city | 16.0/5 = 3.2 % of city |

What it says:
- Block-level temperatures track the satellite well in both cities (r = 0.85 and 0.91).
- Matching exact hotspot footprints is harder: Delhi recovers half of the observed hotspot area (recall 0.50), and 57 % of what it flags is right.
- Jaipur scores higher, but README:209 says its flagged area is "two very large zones (the dry outskirts)". Its average cluster is about 6x larger in city share (3.2 / 0.53). Large zones are easier to overlap, so **Jaipur's 0.62 IoU is not directly comparable to Delhi's 0.36**.
- The precision, recall and IoU values agree with each other (check row above).

### 4c. Cooling doses (`cooling.json` `rows`, README:48-52; model's view, not measured effects)

| Area | Observed gap (°C) | Modelled gap (°C) | Not explained = obs - model | Share the model sees = model / obs | Planting dose | Roof dose | Both |
|---|---|---|---|---|---|---|---|
| Jaipur, strongest 600 m hotspot (val city) | 4.4 | 2.2 | 2.2 | 50 % | 40 % (14 ha) | not given | not given |
| Delhi, dense 600 m block (train city) | 4.5 | 2.0 | 2.5 | 44 % | 45 % (15 ha) | 99 % | 35 % each |
| Delhi, 1.2 x 1.8 km area (train city) | 1.3 | 1.0 | 0.3 | 77 % | 30 % (57 ha) | 48 % (76 ha) | 21 % each |

Area sizes **[fig text]** (`docs/fig_cooling_curves.png`): Delhi block 36 ha, Delhi area 211 ha. Implied pool of eligible pixels = ha / dose:
- Jaipur greenable: 14/0.40 = 35 ha
- Delhi block greenable: 15/0.45 = 33.3 ha
- Delhi area greenable: 57/0.30 = 190 ha
- Delhi area built (for roofs): 76/0.48 = 158 ha

What it says:
- **On the two hottest 600 m blocks the model explains only about half of the observed excess** (50 % and 44 %). The other 2.2 to 2.5 °C is heat the inputs cannot account for (traffic, industry, AC; README:13, 235). Doses close the *modelled* gap only.
- **Planting beats cool roofs in the dense Delhi block**: 45 % planting does what roofs on 99 % of buildings barely do. In the larger mixed area roofs become workable (48 %).
- Combining the two levers lowers each dose below either one alone: 35 % each vs 45 % / 99 %, and 21 % each vs 30 % / 48 %.
- The model's vegetation effect is steeper than the data's: model/data = 1.0/0.8 = 1.25 (Delhi) and 2.0/1.8 = 1.11 (Jaipur), i.e. 11 to 25 % steeper (`metrics.json` `vegetation_check`; README:135-137). Read the doses as the **optimistic end** (README:137).

### 4d. Drivers and feature reliance

Group importance (`metrics.json` `group_importance`; README:141), measured as the drop in R² (offset removed) when a feature group is shuffled between patches of the same city (`group_importance.py:7-9`):

| Group | R² drop |
|---|---|
| Vegetation | 0.89 |
| Water | 0.39 |
| Built-up | 0.31 |
| Terrain | 0.07 **[fig text]** |
| Albedo | 0.03 **[fig text]** |
| Land cover | 0.03 **[fig text]** |

The terrain, albedo and land-cover values are printed in `docs/fig_group_importance.png`; README gives only "under 0.1 for the rest".
- **Uncertain derivation:** the vegetation drop (0.89) is larger than the final R² (0.74). If the unshuffled baseline is 0.74, shuffled R² ≈ 0.74 - 0.89 = **-0.15**, i.e. worse than the city-mean guess. The baseline the script printed is not available, and the script can subsample patches (`--per-city`), so this is an inference.

Delhi's biggest hotspot **[fig text]** (`docs/fig_delhi_drivers.png`, a 2,556 ha cluster in south-west Delhi):
- Gaps: predicted +2.2 °C, observed +3.2 °C, so not explained = 3.2 - 2.2 = 1.0 °C.
- Shapley shares: vegetation +2.11, built-up +1.10, terrain +0.03, land cover -0.04, albedo -0.40, water -0.63.
- Check: the shares sum to +2.17 ≈ +2.2 ✓.

---

## 5. Top insights

1. **The model reads heat patterns well in cities it never saw: R² 0.74 (offset removed), Spearman 0.86, hotspot AUC 0.93, and no city below the city-mean baseline.**
   Source: `per_city.json` `mean`; `metrics.json` `headline`; README:114-119, 164.
2. **The final model cut unexplained variation by 43 % and lifted hotspot F1 by 20 % over the first model, but the anomaly target alone did nothing for pattern skill.**
   - R² 0.54 to 0.74 (+0.20); (0.46 - 0.26)/0.46 = 43 %.
   - F1 0.50 to 0.60 = +20 %.
   - The smaller anomaly model scored R² 0.53 and F1 0.47.
   - Source: `metrics.json` `models.rows`; README:174-178.
3. **The three cities that failed under the first model are now in the top half.** Bengaluru went from 0.09 to 0.76, Bhubaneswar from 0.12 to 0.73 and Kolkata from 0.14 to 0.79, and Kolkata now ties for best.
   Source: **[fig text]** `docs/fig_per_city_r2.png`; `per_city.json` `R2_anomaly`; README:125-128.
4. **Ranking is strong but peaks come out low: hot pixels are underestimated by 1.35 °C on average, and Dehradun alone accounts for much of that.**
   - Dehradun's bias is -3.99 °C.
   - 7 of 10 cities are below -1 °C.
   - The mean without Dehradun is -1.05 °C and the median is -1.135 °C.
   - Source: `per_city.json` `hot_bias`; README:232.
5. **City quality varies: Amritsar is best on three metrics, Lucknow is weakest on pattern, and Dehradun/Coimbatore have high MAE despite good ranking.**
   - Amritsar: MAE 0.54, AUC 0.97, F1 0.76.
   - Lucknow: R² 0.64, Spearman 0.74.
   - Dehradun: MAE 1.96, Spearman 0.86. Coimbatore: MAE 1.74, Spearman 0.91.
   - MAE range 1.42 °C, against an R² range of only 0.14.
   - Source: `per_city.json` `cities`.
6. **For the hottest 600 m blocks, the model explains only about half of the observed excess heat, so cooling doses address the explainable half.**
   - Jaipur: 2.2 of 4.4 °C (50 %). Delhi block: 2.0 of 4.5 °C (44 %).
   - Planting 40 to 45 % of open ground closes the modelled gap.
   - Source: `cooling.json` `rows`; README:48-52.
7. **Vegetation dominates both the model and its fixes, and the effect is backed by the data but slightly exaggerated.**
   - Shuffling vegetation drops R² by 0.89 (vs 0.39 water, 0.31 built-up).
   - In the dense Delhi block, planting needs 45 % where cool roofs need 99 %.
   - The model's cooling per +0.1 NDVI is 11 to 25 % steeper than the satellite data's (1.0 vs 0.8 °C, 2.0 vs 1.8 °C).
   - Source: `metrics.json` `group_importance`, `vegetation_check`; `cooling.json` `rows[1]`; README:135-141.

---

## 6. Caveats and inconsistencies

### What the numbers do not show
- **Only 10 held-out cities.** Spreads and "best/worst" rest on n = 10. There are no per-city metrics for the 40 training cities (inventory section d).
- **The scores are slightly optimistic.** The best epoch was picked on the same 10 cities (README:167, 230), and the README itself says "think about 0.7".
- **Every metric is transcribed from README tables**, not read from raw CSVs (`eval_per_city.csv`, `model_comparison.csv`, etc. are still on the Azure VM; inventory sections c and e). The numbers are rounded to 2 decimals, so derived means and ratios carry about ±0.005 rounding noise.
- **No spatial data locally**: no grids, predictions or hotspot polygons. Maps, full cooling curves (only 3 summary rows), per-cluster driver tables and the training curve cannot be built (inventory section d).
- **Delhi is a training city** (`cities.json` split "train"; it is not in the held-out list at README:111). The Delhi hotspot-quality numbers, the Delhi cooling doses and the Delhi drivers are therefore in-sample. Only Jaipur's numbers are on a held-out city.
  - Caveat to this caveat: `cities.json` takes the split from the script default (`make_patches.py` `VAL_DEFAULT`). The authoritative split is in `meta.json`, which is missing (`cities.json` `_note`). The 10 default validation cities do match README:111.
- **Doses and drivers are the model's attribution, not causal effects** (README:233). The vegetation effect is 11 to 25 % steeper than the data's.
- **The target is land surface temperature** at the late-morning Landsat pass in March to May (README:236). It is not air temperature and not night-time heat.
- **MAE is pattern error** with the city mean supplied (README:166). It is not an absolute-temperature error.
- **The hotspot-quality table covers 2 cities only**, and the cooling table covers 3 areas in 2 cities.

### Internal inconsistencies and checks
1. **"R² inside the city" is used for two different numbers.**
   - README:20 and README:116 (`metrics.json` `headline.rows[0]`) show 0.74 under that label.
   - The per-city table's "R² inside" column (`R2_within`) averages **0.72** (README:164). 0.74 is the "R² offset removed" column.
   - The model table (README:174) and `fig_per_city_r2.png` label it correctly as "offset removed".
   - The dashboard should label it explicitly.
2. **Hot AUC mean, minor rounding issue.** The 10 rows average 9.36/10 = 0.936, which rounds to 0.94, but README:164 shows 0.93. This is possible only if the unrounded mean is below 0.935. It is within rounding tolerance, not an error, but the displayed rows do not reproduce the displayed mean.
   - Similarly, R2_anomaly averages 0.745 and hot_F1 0.605, both exactly on a rounding boundary, shown as 0.74 and 0.60.
   - All other columns reproduce exactly: MAE 1.116 → 1.12, R2_within 0.724 → 0.72, Spearman 0.863 → 0.86, hot_bias -1.347 → -1.35.
   - The inventory's "hot_AUC 0.936 vs 0.93 ✓" glosses over this.
3. **The "not explained" gap for the Delhi block is 2.5 °C, not 2.4.** README:13 says "about 2.4 °C of the observed gap is heat our data cannot explain", but README:51 gives 4.5 - 2.0 = **2.5 °C**. This is probably rounding of the unrounded inputs; it is not checkable without `sim_*.csv`.
4. **The cooling-dose hectares do not fit inside the area if the pools are disjoint.**
   - For the 211 ha Delhi area **[fig text]**, the implied greenable pool (57/0.30 = 190 ha) plus the built pool (76/0.48 = 158 ha) is 348 ha, more than 211 ha.
   - `simulate_cooling.py:10-12, 110` defines greenable as NDVI < 0.4 and built fraction < max-built, and roofs as built pixels, with overlap resolved at run time (`roof &= ~green`). So the two pools likely overlap on thinly built pixels.
   - **Uncertain** until the simulation CSV is available.
   - Also, the README calls the area "1.2 x 1.8 km", which is 216 ha geometrically, while the figure says 211 ha (probably usable pixels only).
5. **Pixel counts are slightly off.**
   - Land-cover pixels total 17.81 M for the 10 validation cities. "About 89 M pixels" for 50 cities (README:218) gives 89/50 × 10 = 17.8 M for 10 cities. So the land-cover table counts essentially every pixel, although metrics are said to use "usable pixels" with water and few-pass pixels masked (README:112, 253). Either the validation boxes are slightly larger or the land-cover table uses a different pixel set. **Uncertain.**
   - "About 1,350 x 1,350" × 50 = 91.1 M, not 89 M. 40 km / 30 m = 1,333 pixels, and 1,333² × 50 = 88.9 M matches 89 M, so "1,350" is a loose rounding.
6. **Feature counts are consistent, but the wording can mislead.** "46 features" (README:219-220) includes the categorical land-cover class: 45 numeric + 6 relative = 51 channels. This matches `model.json` `n_input_channels` = 51 and README:180 ("older models use the first 45 of the 51"). The parameter count of 1,944,863 (`model.json`) matches "about 2 M" (README:222).
7. **The Kolkata first-model values look contradictory but are not.**
   - README:126 says Kolkata was "worse than guessing the city average", i.e. `R2_within` < 0.
   - The figure shows its first-model R² (offset removed) as 0.14.
   - The two are compatible, because the absolute-target first model can carry a large constant offset that only `R2_within` penalises.
8. **The vegetation check is consistent.** The README's "10 to 30 % steeper" (README:137) brackets the computed 11 % and 25 %.
9. **The hotspot table is internally consistent.** IoU = F1/(2 - F1) reproduces both cities' IoU from their precision and recall.
10. **Run identity is unconfirmed.** README trains `data/unet_run3`, but `simulate_cooling.py` defaults to `unet_run2` (inventory section e). It is not confirmed that the cooling numbers and `Deploy/best.pt` come from the same run as the evaluation tables.
