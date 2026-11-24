<div align="center">

# 🌴 HEAT_MAP 🔥
### `W A N T E D   L E V E L :   ★ ★ ★ ★ ★   H E A T S T R O K E`

**A U-Net hunts the hottest blocks in 50 Indian cities... from space. 🛰️**

`🌅 SUNSET MODE: ON` · `🛰️ LANDSAT 8/9` · `🧠 U-NET` · `☁️ AZURE (UAE NORTH)` · `🇮🇳 50 CITIES`

![status](https://img.shields.io/badge/MISSION-COMPLETE-ff2e93?style=for-the-badge)
![cities](https://img.shields.io/badge/CITIES-50-00e5ff?style=for-the-badge)
![held-out](https://img.shields.io/badge/HELD--OUT-10-ffb300?style=for-the-badge)
![model](https://img.shields.io/badge/MODEL-U--NET-7c4dff?style=for-the-badge)
![r2](https://img.shields.io/badge/R%C2%B2_inside_city-~0.7-ff6d00?style=for-the-badge)

</div>

---

## 📼 LOADING SCREEN

```
   ╔══════════════════════════════════════════════════════════╗
   ║   ░▒▓ H E A T _ M A P ▓▒░          [ 🌴 NEON EDITION ]   ║
   ║                                                          ║
   ║   ☀️  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ║
   ║   🏙️  ▁▂▃▅▆▇█▇▆▅▃▂▁  ← the city's heat profile          ║
   ║                                                          ║
   ║   LOADING SATELLITE PASS ........ ██████████ 100%        ║
   ║   LOADING 30 m PIXELS ........... ██████████ 100%        ║
   ║   LOADING COOLING PLAN .......... ██████████ 100%        ║
   ╚══════════════════════════════════════════════════════════╝
```

> *"Which parts of a city are cooking, why, and how much green would it take to cool them down?"*

Heat isn't spread evenly inside a city. Some blocks are boiling, some are chill.
**HEAT_MAP** is our answer to an **ISRO problem statement on urban heat**: a model that reads
Landsat + land-cover layers and predicts which parts of a city run **hotter or cooler than that
city's own average** (late-morning pass, **March to May**).

🎮 **Live results page:** [Heat_Map Results](https://claude.ai/artifact/YRye9qJcAnkZ4KuYPcUK1L)
📦 **Repo:** `Rounaksondawale/Heat_Map`

---

## 🗺️ THE MAP (a.k.a. the stats you flex)

| 🎯 STAT | 💥 VALUE |
|---|---|
| 🏙️ Cities scanned | **50** across India |
| 🎓 Training cities | **40** (the model learned here) |
| 🧪 Held-out cities | **10** (the model never trained on these) |
| 📐 Map tile | **40 km square**, **30 m** pixels |
| 🧠 Model | **U-Net**, **51 input channels** |
| 🌡️ What it predicts | Land surface temperature (LST) *inside* the city, not air temp |
| 📊 Headline score | R² inside the city, think **"about 0.7"** (README's own words) |

> ⚠️ **Honest-dev disclaimer:** one split, one run, no confidence intervals. The 10 held-out cities
> also picked the best epoch, so scores are *slightly optimistic*. Respect.

---

## 🎬 MISSIONS (the 3 tools)

### 🚨 MISSION 1: *"Where are the hotspots?"*
The model predicts every pixel. **Getis-Ord Gi\*** then flags blocks that are *significantly* hotter
than the rest of the city. No "top 10 %" quota, only real clusters.
- Pixels averaged into **600 m blocks**
- Flag blocks at **z ≥ 2.58**
- Touching blocks join into clusters of **2+**, kept only if **≥ 1 °C** above a 900 m ring of neighbours
- 🔧 `models/hotspot_report.py`

### 🔍 MISSION 2: *"What is driving them?"*
Six crews (feature groups) get questioned: **vegetation · built-up · albedo · water · terrain · land cover**.
Switch one crew off at a time (swap in the neighbours' values), re-predict, and each group gets its
exact **Shapley share in °C**. All **2⁶ = 64** combos get checked.
- 🔧 `models/hotspot_report.py`
- 📝 *These are the model's attributions, not proven causes.*

### 🌳 MISSION 3: *"What would cool it down?"*
The what-if simulator is basically a city-builder cheat menu:
- 🌲 **Trees:** converts non-water pixels with NDVI < 0.4 to the city's median tree values
- 🏠 **Cool roofs:** raises built-pixel albedo by **+0.15** (capped at **0.6**)
- 🔁 Features recomputed, model re-predicts, repeat
- 🏁 **The "dose"** = share converted when the hot area finally matches its surroundings
- 🔧 `models/simulate_cooling.py`

---

## 🔧 THE GARAGE (pipeline, in 4 stages)

```
 🛰️ SATELLITES ──▶ 🧮 FEATURES ──▶ 🧠 U-NET ──▶ 🛠️ TOOLS
 Landsat, DEM,      NDVI, NDBI,     predicts the   Gi* hotspots,
 WorldCover, GHSL   MNDWI, albedo   heat pattern   Shapley drivers,
                    + neighbours    per pixel      cooling doses
```

### 🛰️ Loot table (data sources)

| 🎒 Source | 🎁 What it drops | 📏 Resolution |
|---|---|---|
| **Landsat 8/9 C2 L2** | Thermal band → LST (the target); reflectance → NDVI, NDBI, MNDWI, albedo; QA → cloud mask | 30 m |
| **Copernicus DEM GLO-30** | Elevation → slope | 30 m |
| **ESA WorldCover** | Land-cover class (water = code 80) | 10 m → 30 m |
| **JRC GHSL R2023A** | Building height, built-up surface, population | ~90 m → 30 m |
| **Open-Meteo (ERA5)** | Air temp, humidity, wind (audit only, *not* a model input) | per city |
| **OpenStreetMap** *(optional)* | Building, water, green fractions | 30 m |

Scenes: **Mar-May, 2022-2025**, up to **40 clearest scenes per city**, served via
**Microsoft Planetary Computer STAC**. Google Earth Engine: **not used.**

### 🧠 The ride (model + training)
- **Inputs (per pixel):** surface indices · terrain · land cover · urban form · distances to water/green ·
  neighbourhood means · city-relative copies
- **Loss:** Huber on standardised anomalies, usable pixels only
- **Optimiser:** AdamW (lr `2e-3`, wd `1e-2`), one-cycle schedule, batch 64
- **Tricks:** random 90° rotations + flips, dropout 0.1, EMA weights (0.998)
- **Final run:** `--base 32 --epochs 7` on a **4-core CPU** (yes, really 😎)
- **Split:** whole cities only, so no pixel from a held-out city ever leaks into training

---

## 🏆 LEADERBOARD (what the results page shows)

| 🥇 Check | 🎮 What it means |
|---|---|
| **R² inside the city** | How well the model reads the heat pattern within each held-out city |
| **Spearman** | Does it order blocks cool → hot like the satellite does? |
| **Hot AUC** | Can it find each city's hottest 10 % of pixels? (random = 0.50) |
| **Hot F1** | Hotspot detection balance of precision and recall |
| **Model comparison** | 3 models on the same pixels: did the final one improve, and which change gets the credit? |
| **Permutation importance** | Which input group the model leans on most (reliance, not causation) |
| **Error by ground type** | Where it runs warm (+) or cold (−) |
| **Delhi vs Jaipur** | Gi\* hotspot clusters, predicted vs observed |

> 🧪 *Delhi numbers come from a training city, so they're in-sample. Held-out scores are the real flex.*

---

## ☁️ THE HIDEOUT (infrastructure)

- 🏝️ **Azure region:** UAE North
- 💸 Cost is tracked per component, with a per-city figure (total ÷ 50) and the share spent on compute
- 🌐 App hosting (Container Apps or Hugging Face) has **no line in this bill**
- 📊 Full cost donut lives on the [results page](https://claude.ai/artifact/YRye9qJcAnkZ4KuYPcUK1L)

---

## ⚠️ KNOWN GLITCHES (limitations, said out loud)

- 🎲 One split, one run, no confidence intervals
- 🧪 Held-out cities also chose the best epoch, so scores skew a bit high
- 🧱 No non-neural baseline and no ablation study
- 🌡️ Predicts **surface** temperature, not what you feel on the street
- 🔎 Shapley drivers show what the *model* relies on, **not** proven causes
- 🖼️ Some preview views on the page are placeholders until the predicted grid is copied off the compute instance

---

## 🕹️ CONTROLS (quick start)

```bash
git clone https://github.com/Rounaksondawale/Heat_Map.git
cd Heat_Map

# 1. build the dataset  (extraction/build_dataset.py)
# 2. make patches       (pre_processing/make_patches.py)
# 3. train the U-Net    (models/train_unet.py --base 32 --epochs 7)
# 4. hotspots + drivers (models/hotspot_report.py)
# 5. cooling what-ifs   (models/simulate_cooling.py)
```

---

<div align="center">

### 🌅 `THANKS FOR PLAYING` 🌅

*Stay cool out there. Plant a tree. Paint a roof white.* 🌳🏠

**HEAT_MAP** · built on Landsat, Copernicus, ESA, JRC and Open-Meteo data

</div>
