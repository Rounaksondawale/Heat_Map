# 05 Visualization spec: Heat_Map results dashboard (Visualization Designer)

Inputs: `04_insights.md`, `02_data_inventory.md`, `data/{insights,metrics,per_city,cooling,cities,model,figures}.json`, `docs/fig_group_importance.png`, `docs/fig_cooling_curves.png`.
Method: the `dataviz` skill (form first, colour last, validator run on every categorical/ordinal palette, mark specs, hover layer by default, selected dark mode).
Output: one self-contained HTML file, plain JS, inline SVG plus one `<canvas>`. **No charting library**: every chart here is a dot plot, a bar chart, a scatter or a 60x60 raster. Hand-rolled SVG covers them in about the same amount of code as configuring a library, and it keeps theming on CSS tokens. Nothing loads from the network; all JSON is inlined (see section 6).

Lead decisions applied throughout:
- Headline R² is **0.72 (as scored, `per_city.json mean.R2_within`)**. **0.74 (offset removed, `mean.R2_anomaly`)** appears as a secondary line.
- Hotspot AUC is "finding the **hottest 10 % of pixels**". It is never described as finding the Gi* hotspot clusters, which are a separate analysis (V7).
- Held-out cities: always "**held out from training (also used to pick the best epoch)**". The words "never seen" and "unseen" must not appear. Note that `insights.json kpis[0].meaning` says "10 unseen cities". The frontend **must not render that string**. Use the copy given in V1.
- **Delhi = training city (in-sample); Jaipur = held out.** Every place a Delhi or Jaipur number appears carries a split badge (section 3.6).
- The heat map is a **placeholder**, labelled in the chart (watermark + badge + caption) and swappable (V2, section 6).

---

## 1. Design tokens

All colours are CSS custom properties on `:root`. Dark values are declared twice: under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and under `:root[data-theme="dark"]`. The header has a 3-way theme toggle (Auto / Light / Dark) that sets `data-theme` on `<html>`. Wrap any `localStorage` read/write for the remembered choice in try/catch. `body` gets an explicit `background: var(--page)`.

### 1.1 Colour tokens

| Token | Role | Light | Dark |
|---|---|---|---|
| `--page` | page plane (body background) | `#f9f9f7` | `#0d0d0d` |
| `--surface` | card / chart surface | `#fcfcfb` | `#1a1a19` |
| `--surface-raised` | tooltip, popover, sticky table header | `#ffffff` | `#242422` |
| `--border` | hairline ring around cards, tooltip | `rgba(11,11,11,0.10)` | `rgba(255,255,255,0.10)` |
| `--grid` | gridlines (1px solid, never dashed) | `#e1e0d9` | `#2c2c2a` |
| `--axis` | baselines, axis lines, zero line, reference lines | `#c3c2b7` | `#383835` |
| `--text-primary` | titles, values, table cells | `#0b0b0b` | `#f0efec` |
| `--text-secondary` | subtitles, legend labels, axis titles, captions | `#52514e` | `#c3c2b7` |
| `--text-muted` | tick labels, footnotes, source lines (text-safe) | `#6b6a65` | `#9d9b94` |
| `--ink-muted` | **non-text only** (icons, disabled marks). Any text, including SVG tick and graticule labels, uses `--text-muted` | `#898781` | `#898781` |
| `--accent` | links, focus ring, selected control | `#256abf` | `#86b6ef` |
| `--held-out` | categorical slot 1: held-out cities, and the default single-series colour (all aggregates are on held-out cities) | `#2a78d6` | `#3987e5` |
| `--train` | categorical slot 2: training cities (incl. Delhi) | `#eb6834` | `#d95926` |
| `--model-1` | ordinal step: First model | `#86b6ef` | `#1c5cab` |
| `--model-2` | ordinal step: Smaller anomaly model | `#2a78d6` | `#3987e5` |
| `--model-3` | ordinal step: Final model (emphasis) | `#104281` | `#9ec5f4` |
| `--fig-text` | bars whose value was read off a docs figure (V9) | `#86b6ef` | `#1c5cab` |
| `--unexplained` | "not explained by the model" segment (V8a), with 45° texture | `#e1e0d9` | `#383835` |
| `--unexplained-ink` | texture lines on `--unexplained` | `#c3c2b7` | `#52514e` |
| `--bias-warm` | positive bias (model runs warm) | `#b13f3c` | `#c74845` |
| `--bias-cool` | negative bias (model runs cold) | `#256abf` | `#2a78d6` |
| `--status-warning` | placeholder badge fill/icon (fixed, never themed) | `#fab219` | `#fab219` |
| `--warn-bg` | placeholder caption banner background | `#fdf3dc` | `#3a2f12` |
| `--warn-text` | text on `--warn-bg` | `#6b4600` | `#fab219` |
| `--success-text` | positive delta text (with ▲ icon) | `#006300` | `#0ca30c` |
| `--critical` | error/empty-state icon | `#d03b3b` | `#d03b3b` |

Diverging anomaly scale stops are in section 3.2.

### 1.2 Type

- Font stack: `system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`. No display or serif face. No web font.
- Scale (px / weight / line-height): hero value 48 / 600 / 1.0; tile value 32 / 600 / 1.1; page title 28 / 700 / 1.2; section title 20 / 650 / 1.3; chart title 16 / 600 / 1.35; body 15 / 400 / 1.5; label, legend and tooltip 13 / 400 / 1.4; tick and footnote 12 / 400 / 1.4. Nothing below 12 px.
- `font-variant-numeric: tabular-nums` on table cells, axis ticks and tooltip values only. Hero and tile values use proportional figures.
- Minus sign: render negatives with U+2212 "−", not a hyphen. Signed anomalies and biases always show a sign ("+2.4 °C", "−1.35 °C"). A space goes between number and unit ("1.12 °C", "45 %"), matching the README.

### 1.3 Spacing, radii, elevation, motion

- Spacing scale: 4, 8, 12, 16, 24, 32, 48 px (`--s1` … `--s7`). Card padding 24 (16 under 640 px). Grid gap 24 (16 under 640 px). Page side gutter 16 at phone width, max content width 1200 px, centred.
- Radii: `--r-mark` 4 px (bar data-ends; the baseline end stays square), `--r-control` 8 px (buttons, chips, tooltip), `--r-card` 12 px.
- Elevation: cards are flat with a 1px `--border` ring. Only the tooltip gets a shadow: `0 4px 16px rgba(0,0,0,0.12)` light, `0 4px 16px rgba(0,0,0,0.5)` dark.
- Motion: 150 ms ease-out on hover/selection opacity, 250 ms on sort re-order (transform only). Everything is off under `prefers-reduced-motion: reduce`.

### 1.4 Mark specs (all charts)

Bars are at most 24 px thick, with 4 px rounded data-ends and a square baseline end. Adjacent touching fills get a 2 px `--surface` gap. Dots are r 5 (r 6 for emphasis) with a 2 px `--surface` ring. Lines are 2 px with round caps. Gridlines are 1 px `--grid` and the zero/baseline is 1 px `--axis`. Reference lines (means, random baselines, thresholds) are 1 px `--text-muted`, solid, with a 12 px label at the end. **Text never takes a series colour**: identity comes from a swatch, dot or line-key beside text that is in a text token. Value labels sit **outside** bar ends. White-on-blue fails WCAG AA (4.42:1 on `#2a78d6`), so no labels go inside blue fills.

### 1.5 Texture (accessibility channel)

One "lines" fill at 45° (and its 135° mirror), drawn as an SVG `<pattern>` with 1.5 px strokes, 6 px spacing, tone-on-tone. It is used by default in only one place, the "not explained" segment in V8a, because there it carries meaning (a different kind of quantity, not a series). Under `forced-colors: active` and `@media print`, it is also applied to `--train` marks at 135°, so split identity survives without hue.

---

## 2. Contrast check (WCAG 2.x AA, computed, not eyeballed)

Ratios were computed with the sRGB relative-luminance formula (`python -I` script in the scratchpad, same formula as the skill's `contrast()`). AA thresholds: 4.5:1 for normal text, 3:1 for large text (≥ 24 px, or ≥ 18.67 px bold) and for non-text marks.

### 2.1 Text tokens on backgrounds

| Text token | Light: on `--surface` `#fcfcfb` | on `--page` `#f9f9f7` | on `--surface-raised` `#ffffff` | Dark: on `--surface` `#1a1a19` | on `--page` `#0d0d0d` | on `--surface-raised` `#242422` | Verdict |
|---|---|---|---|---|---|---|---|
| `--text-primary` | 19.17 | 18.67 | 19.68 | 15.15 | 16.90 | 13.52 | AA ✓ |
| `--text-secondary` | 7.73 | 7.53 | 7.94 | 9.72 | 10.85 | 8.68 | AA ✓ |
| `--text-muted` (`#6b6a65` / `#9d9b94`) | 5.28 | 5.14 | 5.42 | 6.26 | 6.99 | 5.59 | AA ✓ |
| palette muted `#898781` (rejected for text) | **3.50** | **3.41** | **3.59** | 4.85 | 5.41 | **4.33** | Fails AA for normal text in light mode and on dark raised → kept only as `--ink-muted` (non-text) |
| `--accent` (`#256abf` / `#86b6ef`) | 5.26 | 5.12 | 5.39 | 8.25 | 9.21 | 7.37 | AA ✓ |
| `--success-text` (`#006300` / `#0ca30c`) | 7.35 | 7.15 | 7.54 | 5.19 | 5.79 | 4.64 | AA ✓ |
| `--critical` `#d03b3b` (icon + label) | 4.68 | 4.56 | 4.80 | (dark uses label in `--text-primary`) | | | AA ✓ light |
| `--warn-text` on `--warn-bg` | `#6b4600` on `#fdf3dc`: **7.61** | | | `#fab219` on `#3a2f12`: **7.18** | | | AA ✓ |
| `--text-primary` on `--warn-bg` | 17.84 | | | 11.45 | | | AA ✓ |

### 2.2 Labels inside fills (where used)

| Label / fill | Ratio | Use |
|---|---|---|
| `#0b0b0b` on `--train` light `#eb6834` | 6.15 | allowed (white would be 3.20 ✗) |
| `#0b0b0b` on `--train` dark `#d95926` | 5.07 | allowed (white 3.88 ✗) |
| any label on `--held-out` light `#2a78d6` | white 4.42 / black 4.46 | **✗: never label inside blue**, put the label outside |
| `#0b0b0b` on `--unexplained` light `#e1e0d9` | 14.87 | allowed |
| `#f0efec` on `--unexplained` dark `#383835` | 10.23 | allowed |
| heat-map cells | n/a | no text inside cells; the readout panel carries values |

### 2.3 Non-text marks vs surface (3:1)

| Mark | Light | Dark | Note |
|---|---|---|---|
| `--held-out` | 4.30 ✓ | 4.79 ✓ | |
| `--train` | 3.12 ✓ | 4.48 ✓ | |
| `--model-1` | 2.06 | 2.63 | ordinal light end, ≥ 2:1 rule ✓. Relief: direct value labels + table view |
| `--model-3` | 9.66 ✓ | 9.74 ✓ | |
| `--fig-text` | 2.06 | 2.63 | relief: every bar carries its printed value |
| `--bias-warm` / `--bias-cool` | ✓ (validator) | ✓ | |
| `--axis` vs surface | 1.75 / 1.48 | | decorative chrome. Data never depends on it |

---

## 3. Colour scales

### 3.1 Validator log (skill `scripts/validate_palette.js`)

| Palette | Command | Result |
|---|---|---|
| Split (held-out, train), light | `"#2a78d6,#eb6834" --mode light --pairs all` | ALL PASS. CVD ΔE 24.7, normal ΔE 33.6, contrast ≥ 3:1 |
| Split, dark | `"#3987e5,#d95926" --mode dark --surface "#1a1a19" --pairs all` | ALL PASS. CVD ΔE 26.8, normal ΔE 31.8 |
| Model ordinal, light | `"#86b6ef,#2a78d6,#104281" --ordinal --mode light` | ALL PASS. Light end 2.06:1, hue spread 3° |
| Model ordinal, dark | `"#9ec5f4,#3987e5,#1c5cab" --ordinal --mode dark --surface "#1a1a19"` | ALL PASS. Dark end 2.63:1 |
| Bias sign, light | `"#256abf,#b13f3c" --mode light --pairs all` | ALL PASS. CVD ΔE 20.6 |
| Bias sign, dark | `"#2a78d6,#c74845" --mode dark --pairs all` | ALL PASS. CVD ΔE 22.3. (A first try with `#659fea,#e07b74` FAILED the dark lightness band, so it was re-stepped.) |
| Rejected: cooling-lever trio aqua/violet/magenta | dark `--pairs all` | **FAIL**: magenta↔aqua deutan ΔE 1.6. This is why levers are direct-labelled rows, not colours (V8) |

`--pairs all` was used for the split palette because the city map is a scatter. Only 2 categorical slots are ever on screen together, well inside the 3-slot all-pairs cap.

### 3.2 Diverging temperature-anomaly scale (V2)

Blue (cooler than the city mean) ↔ neutral gray ↔ red (hotter than the city mean). The blue arm is the palette's blue ramp. The red arm is **OKLCH-matched**: same L and C as its blue partner, hue 25° (the palette red's hue), and gamut-clamped. This makes the two arms perceptually symmetric. Equal OKLCH ΔL per step means perceptually uniform lightness. Blue↔red stays separable under protan/deutan simulation (blue vs yellow-brown), and the sign also shows as "+"/"−" in the readout.

Domain: **fixed, symmetric −4 … +4 °C**, centred at 0 °C. Values outside are clamped to the end colour, and the legend ends get ◂ ▸ caps labelled "≤ −4" / "≥ +4". A fixed domain keeps colours comparable when the real grid (or several cities) is swapped in. Interpolate **in OKLab** between adjacent stops, not in sRGB, to keep lightness steps even. Stop positions are at every 1 °C.

| °C | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| **Light** | `#104281` | `#256abf` | `#5598e7` | `#9ec5f4` | `#f0efec` | `#f1aea8` | `#dd716a` | `#b13f3c` | `#762221` |
| OKLCH L | 0.385 | 0.527 | 0.671 | 0.812 | 0.952 | 0.812 | 0.672 | 0.527 | 0.385 |
| **Dark** | `#a0c4f4` | `#659fea` | `#2a78d6` | `#1656a1` | `#383835` | `#952f2e` | `#c74845` | `#e07b74` | `#f1aea8` |
| OKLCH L | 0.812 | 0.694 | 0.576 | 0.458 | 0.340 | 0.458 | 0.576 | 0.694 | 0.812 |

- Light: the extremes are darkest (most ink on a light surface). ΔL is 0.140 to 0.144 per step.
- Dark: the anchor flips. The midpoint is the dark neutral `#383835` and the extremes are lightest, so extremes stay most salient on a dark surface. ΔL is 0.118 per step. The dark arms stop at L 0.81 rather than 0.90 because at 0.90 the two poles' chroma falls to 0.04 and hot and cold become hard to tell apart.
- Midpoint gray is the palette's documented diverging neutral (light `#f0efec`, dark `#383835`). No hue sits at 0.
- Texture fallback (`forced-colors` / print): positive cells get 45° lines and negative cells get 135° lines. Line density steps with |value| (none for |v| < 0.5, 1 px for < 2, 2 px for ≥ 2).

### 3.3 Split (identity: which cities a number is about)

- **Held out** = slot 1 blue (`--held-out`). **Training** = slot 2 orange (`--train`).
- This is the **only** categorical pair in the dashboard. It follows the entity everywhere: city map dots (V3), per-city dots (V4, all held out), Jaipur vs Delhi (V7), cooling areas (V8). Aggregates computed on held-out cities (V5 model rows, V6 land cover, V9 importance) use the held-out blue family. That is honest, because they come from the 10 held-out cities.
- A filter never repaints: hiding the training cities fades them and leaves the held-out cities blue.

### 3.4 Model ordinal (V5)

First → Smaller anomaly → Final is a chronological sequence, so it gets **ordinal** encoding: one blue hue, getting stronger toward the final model (light `--model-1..3`; dark reversed in lightness so the final model is still the most salient). This keeps orange free to mean "training".

### 3.5 Bias sign (V6)

Positive bias ("runs warm") is `--bias-warm`, negative ("runs cold") is `--bias-cool`. Sign also shows as text ("+0.43 °C"). This is the same warm/cool convention as the anomaly scale.

### 3.6 Split badge (text component, not colour-only)

An inline pill placed after a city name: a 8 px dot in the split colour plus text in `--text-secondary`: "**held out**" or "**training city · in-sample**". Every Delhi number carries "training city · in-sample", and every Jaipur number carries "held out". `aria-label` repeats the text.

### 3.7 Status

Status colours are used only for the placeholder warning (`--status-warning` + ⚠ icon + the word "PLACEHOLDER") and the empty/error state (`--critical` + icon + label). They are never used as series colours.

---

## 4. Visuals

Common to every chart card: a title (the question it answers, in sentence case), a one-line subtitle, the chart, a legend when there are ≥ 2 series, a source line (`--text-muted`, 12 px, naming the exact JSON key path), and a "Table" toggle button (top-right of the card, `aria-pressed`) that swaps the chart for an accessible `<table>` holding the same numbers. Every SVG has `role="img"` and an `aria-label` summary sentence. Marks are keyboard-focusable via one roving `tabindex` per chart: arrow keys move, and focus shows the same tooltip as hover. Tooltip spec: section 5.

---

### V1. KPI tile row

**Question:** How good is the model, in five numbers?
**Form:** stat tiles, not a chart. One hero tile (R²) at 48 px and four standard tiles at 32 px. Tiles are not interactive charts, so there is no hover tooltip. Each tile has an ⓘ button that opens a popover with the definition and source keys. The popover is click/Enter/Space-toggled and Esc-dismissed.

| # | Label (sentence case) | Value | Sublabel line 1 | Sublabel line 2 / delta | Binding |
|---|---|---|---|---|---|
| 1 (hero) | R² inside the city | **0.72** | 0.74 with each city's constant offset removed | ▲ +0.20 vs first model (0.54), offset removed | `per_city.json mean.R2_within`; `mean.R2_anomaly`; `metrics.json models.rows[0].R2_anomaly` |
| 2 | Hotspot AUC (hottest 10 % of pixels) | **0.93** | Finds each city's hottest 10 % of pixels · random = 0.50 | ▲ +0.05 vs first model (0.88) | `per_city.json mean.hot_AUC`; `metrics.json headline.rows[2].first` |
| 3 | Rank correlation (Spearman) | **0.86** | Orders neighbourhoods cool → hot almost as the satellite does | ▲ +0.08 vs first model (0.78) | `per_city.json mean.spearman`; `metrics.json headline.rows[1].first` |
| 4 | Pattern error (MAE) | **1.12 °C** | Typical pixel error, given the city's mean temperature | no delta (first-model MAE not available) | `per_city.json mean.MAE` |
| 5 | Held-out cities worse than the city-mean guess | **0 of 10** | Beats "every pixel = city mean" in every held-out city | First model: 1 of 10 | `metrics.json headline.rows[4].final` / `.first` |

- Row caption under the tiles (`--text-secondary`, 13 px): "Averages over 10 cities held out from training (also used to pick the best epoch), so scores are slightly optimistic. 50 cities in total, about 89 M pixels at 30 m." Bindings: `cities.json n_cities`, `n_val`; `metrics.json facts.rows[0].value`.
- ⓘ popover text for tile 1: "R² inside the city: 1 − SSE/SST within each city, averaged over the 10 held-out cities. 0 = no better than painting every pixel with the city mean. 0.74 removes each predicted map's constant offset, so only the pattern is scored." For tile 2: "AUC for telling the hottest 10 % of a city's pixels from the rest. This is pixel-level and is not the Gi* hotspot-cluster analysis further down."
- Delta styling: "▲" icon plus text in `--success-text`. The icon and the sign carry direction, not colour alone. The comparison is always named ("vs first model").
- Do not round 0.936 up: the AUC shown is 0.93 (README mean), per `04_insights.md` §6.2. The ⓘ popover footnotes it: "The 10 rounded city values average 0.936. The README reports 0.93 from unrounded values."
- Layout: 5 columns at ≥ 1024 px (the hero tile spans 1.4 fr). At 640–1023 px, the hero is full width and the other four form a 2x2 grid. Under 640 px, everything is 1 column with the hero first.
- Empty state: if a value is missing, show "—" with the sublabel "not available in data", never 0.

---

### V2. Interactive heat map (PLACEHOLDER)

**Question:** (once real data exists) Where inside a city is it hotter or cooler than the city average?
**Form:** a raster heat map (one 60x60 grid of 30 m cells = 1.8 x 1.8 km, the size of one model patch) on `<canvas>`, with an SVG overlay for the watermark, hover ring and hottest-10 % outline. A raster is the right form because the data is a continuous field on a regular grid. Any aggregation (contours, hexbins) would hide the cell-level pattern the model predicts.

**Data binding:** `data/placeholder_grid.json` → `{_placeholder, _source, units, rows, cols, cell_m, values[rows][cols]}`. It is inlined in the HTML as `<script type="application/json" id="grid-data">`. Values are °C anomaly (cell minus grid mean). The placeholder's range is −3.4 … +3.7 °C, p90 = +1.5 °C.

**Placeholder labelling (mandatory while `_placeholder === true`):**
1. **Watermark** in the SVG overlay: the text "PLACEHOLDER" twice, rotated −30°, 700 weight, 44 px (scales with width, min 24 px), fill `--text-primary` at 14 % opacity, `pointer-events: none`. It is drawn above the cells, so it cannot be cropped out of a screenshot.
2. **Badge** in the overlay's top-left corner, inside the plot: a ⚠ icon in `--status-warning` and the text "PLACEHOLDER · synthetic" in `--warn-text` on `--warn-bg`, radius 8, padding 4 8.
3. **Caption** directly under the plot, in a `--warn-bg` banner with ⚠ icon, `--text-primary`, 13 px: "**PLACEHOLDER: synthetic pattern, not model output.** Smooth noise plus a few warm and cool blobs, generated only to lay out this view. No predicted temperature grid is available locally yet (see Data inventory §d)."
4. The card title gets the suffix "(placeholder)". The readout panel's value line ends with "· placeholder".
5. The table view is **disabled** for placeholder data (button tooltip: "No real data to tabulate"), so synthetic numbers are never presented as a data table.

**Encodings:** x = column (west → east), y = row (north → south), fill = anomaly through the diverging scale (section 3.2), interpolated in OKLab. Each cell is a square. Render the 60x60 into an offscreen ImageData once, then `drawImage` it scaled up with `imageSmoothingEnabled = false` and the canvas backing store sized by `devicePixelRatio`. Repaint on theme change, because the dark stops differ.

**Axes:** no tick axes (it is a map). A scale bar sits bottom-right, outside the plot: a 2 px `--text-secondary` line with "500 m" over ⌈500 / cell_m⌉ cells. A small "N ↑" sits top-right outside the plot. Under the plot: "1.8 × 1.8 km · 30 m cells" (computed from `rows`, `cols`, `cell_m`).

**Legend:** a horizontal gradient bar 240 x 12 px (full width on mobile) with a 1 px `--border` ring and ticks at −4, −2, 0, +2, +4 °C (12 px, `--text-muted`, tabular). End caps read "≤ −4" / "≥ +4". Below the ends sit two captions in `--text-secondary`: "◂ cooler than the city mean" and "hotter than the city mean ▸". Legend title: "Temperature anomaly (°C)".

**Hover readout:**
- A persistent readout panel sits beside the map (right of it at ≥ 1024 px, below it otherwise), so the value does not depend on a floating tooltip:
  - Line 1: "**+2.4 °C** vs grid mean · placeholder"
  - Line 2: "Cell row 23, col 41 · 1.23 km E, 0.69 km S of the NW corner"
  - Line 3 (if the hottest-10 % overlay is on): "In the hottest 10 % of cells" / "Not in the hottest 10 %"
  - At rest: "Hover or tap a cell to read its value."
- On the canvas, the hovered cell gets a 2 px `--surface` ring plus a 1 px `--text-primary` outer ring, so it shows on both pale and dark cells.
- The hit target is the cell itself, which is at least 6 px at 360 px width (360/60). Under 8 px cell size, the hit radius snaps to the nearest cell centre with a 12 px tolerance.

**Interactions:**
- Toggle chip "Outline hottest 10 %": computes the 90th-percentile threshold of `values` and outlines the region boundary. For each in-cell, draw each edge that borders an out-cell, as a 2 px `--text-primary` line. This links visually to the AUC KPI definition. Caption when on: "Outline = cells at or above the 90th percentile (+1.5 °C here)".
- Keyboard: the canvas is focusable. Arrow keys move a cursor cell, Shift+arrow moves 5 cells, and the readout updates (`aria-live="polite"`, throttled to 1 update/250 ms).
- Touch: tap pins the readout to a cell. Drag scrubs.
- No zoom/pan (60x60 fits).

**Empty state:** if the grid JSON is missing or malformed (rows·cols ≠ values length, or non-numeric values), show a `--surface` panel with ⚠ `--critical` and "Heat map unavailable: no grid loaded." The legend is hidden.

**Swap to real data:** see section 6. When `_placeholder !== true`, items 1 to 5 above are removed automatically and the caption becomes "Predicted temperature anomaly, {city}, {_source}". The real grid must also carry `city` and `split`, and the split badge appears after the city name.

**Mobile:** the canvas is full card width (square), the readout panel moves below it, the legend goes full width, and the watermark font scales to stay legible.

---

### V3. Map of the 50 cities

**Question:** Where are the cities, and which 10 are held out?
**Form:** an SVG scatter in plain lon/lat. There is **no outline, coastline or tiles**, so nothing is fetched and no borders are drawn (which avoids disputed-boundary issues). A light 5° graticule gives geographic orientation.

**Data binding:** `cities.json cities[]` → `city, lon, lat, climate, cloud_max_pct, split`. For held-out cities, join `per_city.json cities[]` on `city` for tooltip metrics.

**Projection:** equirectangular with a cos-latitude correction: `x = (lon − 69.5) · cos(21.5°) · k`, `y = (35.5 − lat) · k`. Domain lon 69.5–95.5 °E, lat 7.5–35.5 °N (the data spans 70.80–93.94 °E, 8.52–34.08 °N, plus about 1° padding). viewBox 560 x 650 (aspect ≈ 0.86). `preserveAspectRatio="xMidYMid meet"`.

**Encodings:**
- Position = city centre.
- Colour = split: held out `--held-out`, training `--train`.
- Size is also redundant with split: held out r 6, training r 4.5. Both get a 2 px `--surface` ring.
- Shape stays a circle. Under the forced-colors/print texture mode, training dots become hollow rings.
- Draw order: training dots first, then held-out dots on top.

**Axes / graticule:** lines every 5° (70, 75 … 95 °E; 10, 15 … 35 °N), 1 px `--grid`. Labels "75°E", "20°N" along the bottom and left edges, 12 px `--text-muted`. No axis titles needed.

**Labels:** direct labels for the 10 held-out cities plus Delhi (11 labels), 12 px `--text-primary`, with label anchor offsets chosen to avoid collisions:

| City | Side | City | Side |
|---|---|---|---|
| Amritsar | right | Kolkata | right |
| Dehradun | right | Bhubaneswar | right |
| Delhi | left (+ "training" in `--text-secondary`) | Nagpur | right |
| Jaipur | left | Bengaluru | left |
| Lucknow | right | Chennai | right |
| | | Coimbatore | left |

Labels get a 3 px `--surface` halo (`paint-order: stroke`). Under 480 px width, label only Delhi, Jaipur and the city currently selected in V4, and leave the rest to tooltips.

**Legend** (top-left, inside the plot's empty Arabian Sea area): "● Held out from training (10)" and "● Training (40)". Each dot is drawn at its map size and the text is `--text-secondary`. Footnote under the legend: "Held-out cities were also used to pick the best epoch."

**Tooltip** (hover/focus on a dot):
- Title: "**Jaipur**" + split badge.
- Rows:
  - Climate: semi-arid
  - Centre: 26.91 °N, 75.79 °E
  - Box: 40 km × 40 km (from `_note`)
  - Max cloud allowed: 10 %
  - Held-out cities only: R² 0.78 (offset removed 0.79) · MAE 0.98 °C · Hot AUC 0.95
- Training cities get the line "No per-city scores for training cities."

**Interactions:**
- A filter chip row above the chart (single-select): All · Held out · Training. Unselected groups fade to 15 % opacity and stay in place, unrepainted. Faded dots are not focusable.
- Click/Enter on a held-out dot selects that city in V4 (cross-highlight). V4's selection also highlights the dot here with a 2 px `--text-primary` outer ring.
- Click on Delhi or Jaipur scrolls to V7.

**Caption:** "City centres from `cities.json`. Split is the code default (`make_patches.py`). The run's authoritative `meta.json` split is not available locally, but the 10 held-out cities match the README."

**Empty state:** if `cities.json` is missing, show "City list unavailable."
**Mobile:** full width; reduced labels (above); hit target r = 14 px invisible circles.
**Table view:** City · Split · Climate · Lat · Lon · Cloud max %, sortable.

---

### V4. Per-city metrics for the 10 held-out cities

**Question:** How does performance vary across the held-out cities, and which ones are weak?
**Form:** (a) a **sortable table** for exact values, plus (b) a **Cleveland dot plot**, one metric at a time, chosen with a selector. A dot plot fits here because it needs no zero baseline: the values cluster (for example R² 0.64–0.79), and bars would waste most of their length. One metric at a time keeps it to one axis.

**Data binding:** `per_city.json cities[]` (10 rows) and `per_city.json mean` (the README mean row; show it as given, **not recomputed**). Medians come from `insights.json per_city_stats.<metric>.median`.

**Metric selector:** a segmented control (single-select, keyboard arrow-navigable, `role="radiogroup"`):

| Option label | Key | Domain (x) | Better | Reference lines |
|---|---|---|---|---|
| MAE (°C) | `MAE` | 0 – 2.2 | lower | mean 1.12 |
| R² as scored | `R2_within` | 0.5 – 1.0 | higher | mean 0.72; 0.70 "rule of thumb (README)" |
| R² offset removed | `R2_anomaly` | 0.5 – 1.0 | higher | mean 0.74; 0.70 |
| Spearman | `spearman` | 0.5 – 1.0 | higher | mean 0.86 |
| Hot AUC | `hot_AUC` | 0.5 – 1.0 | higher | axis starts at 0.50 = random (label it); mean 0.93 |
| Hot F1 | `hot_F1` | 0 – 1.0 | higher | 0.10 = random (derived); mean 0.60 |
| Peak bias (°C) | `hot_bias` | −4.5 – +0.5 | closer to 0 | 0 (1 px `--axis`); mean −1.35; median −1.14 |

Default selection: **R² as scored**.

**Dot plot encodings:**
- y = city (band), **sorted best → worst for the selected metric** (MAE ascending, bias by |value| ascending, others descending). Re-sorting animates 250 ms.
- x = metric value. Dot r 5, `--held-out` (all 10 are held out, so a single series and no legend box; the title says it).
- A thin 1 px `--grid` leader line runs from the axis start to each dot to help the eye.
- Value labels: only for the best and worst city (direct label right of the dot, 12 px `--text-primary`). Everything else goes in the tooltip/table.
- Reference lines: vertical 1 px `--text-muted`, labelled at the top ("mean 0.72", "random 0.50", "rule of thumb 0.70").
- For peak bias, the dot sits on a 2 px stem from 0. The axis label says "Peak bias on hottest 10 % of pixels (°C) · negative = peaks predicted too low".

**Axis labels (x):** "MAE, °C (city mean supplied: pattern error)" · "R² inside the city (as scored)" · "R² inside the city (offset removed)" · "Spearman rank correlation" · "Hotspot AUC (hottest 10 % of pixels)" · "Hotspot F1 (hottest 10 % of pixels)" · "Peak bias, °C". Ticks are 12 px tabular, with 4–6 clean ticks.

**Tooltip (dot):**
- Title: "**Dehradun**" + "held out" badge.
- Selected metric in bold: "Peak bias −3.99 °C".
- Rank among the 10: "Rank 10 of 10".
- Delta vs mean: "2.64 °C below the mean (−1.35)".
- A divider, then the other six metrics in `--text-secondary`.

**Table:**
- Columns: City · MAE (°C) · R² as scored · R² offset removed · Spearman · Hot AUC · Hot F1 · Peak bias (°C).
- Headers are buttons. Clicking sorts and toggles asc/desc, with `aria-sort` set and a ▲/▼ glyph.
- The default sort matches the chart's metric, and the column for the selected metric is highlighted with a `--grid` background.
- The mean row is **pinned** as a `<tfoot>`, labelled "Mean (README)", in 600 weight.
- Footnote: "The README mean row is computed from unrounded values. The 10 rounded rows average 0.936 for AUC (shown 0.93), 0.745 for R² offset removed (shown 0.74)."
- Cells are right-aligned and tabular, with 2 decimals (°C to 2 decimals as in the source).
- Row hover highlights the matching dot (and the map dot in V3). Row click selects.

**Linked selection:** the selected city (from table row, dot or map) gets a 2 px `--text-primary` ring on its dot and a `--grid` row background, and syncs with V3. Esc clears it.

**Callouts** (text under the chart, from `04_insights.md` §2):
- Amritsar is best on MAE, AUC and F1.
- Lucknow is weakest on pattern (R² 0.64, Spearman 0.74).
- Dehradun and Coimbatore have high MAE (1.96, 1.74) but rank well (Spearman 0.86, 0.91). Their error is in level and peaks.
- Dehradun's peak bias −3.99 °C pulls the mean from −1.05 to −1.35.

**Empty state:** "Per-city scores unavailable."
**Mobile:** the selector becomes a horizontally scrollable chip row inside its own container (no page scroll), or a native `<select>` under 400 px. The table scrolls horizontally inside its wrapper, with the City column sticky.

---

### V5. Model comparison: First vs Smaller anomaly vs Final

**Question:** Did the final model actually improve, and which change gets the credit?
**Form:** a dot plot with three dots per metric row on a shared 0.4–1.0 scale, plus a separate small panel for the °C bias (different units → separate axis, never a dual axis). A row of three dots shows both the jump and the "smaller anomaly model didn't help" story better than grouped bars, whose long stems would be mostly empty from 0 to 0.4.

**Data binding:** `metrics.json models.rows[0..2]` → `model, target, inputs, epochs, R2_anomaly, spearman, hot_AUC, hot_F1, hot_bias, cities_worse_than_city_mean`. Relative changes come from `insights.json model_comparison.final_vs_first` / `final_vs_smaller`.

**Encodings:**
- Panel A rows: R² offset removed · Spearman · Hot AUC · Hot F1. x = score, domain 0.4–1.0. The axis label reads "Score on the 10 held-out cities (same pixels for all three models)".
- Panel B: "Peak bias on hottest 10 % of pixels (°C)", domain −3.0 to 0, with a 0 line in `--axis`.
- Colour = model (ordinal `--model-1/2/3`). Each row has 3 sub-lanes offset −7 / 0 / +7 px, so near-ties (AUC 0.88 vs 0.89) never overlap. Dots are r 5 (Final r 6) with 2 px `--surface` rings.
- A 2 px line in `--grid` connects First → Final in each row to show the gain.
- Direct label on the Final dot only ("0.74"). The gain label "+0.20" sits in `--text-secondary` beyond the dot.

**Legend** (above the panels, since there are 3 series):
- ● First model: absolute target, 45 inputs, 30 epochs
- ● Smaller anomaly model: anomaly target, 45 inputs, 12 epochs
- ● Final model: anomaly target, 51 inputs, 7 epochs

**Under the panels**, a one-line fact row: "Held-out cities worse than the city-mean guess: 1 → 0 → 0".

**Tooltip (dot):**
- Title: model name.
- Metric value in bold.
- "vs First: +0.20 (+37.0 %)". For the Final dot also "vs Smaller: +0.21 (+39.6 %)". For bias, "6.9 % smaller underestimate".
- Footer in `--text-muted`: target · inputs · epochs.

**Annotations** (text below, `--text-secondary`):
- "The anomaly target alone did not lift pattern scores. The Smaller anomaly model ties or trails the First model on R², Spearman and F1."
- "Final changed several things at once (6 city-relative inputs, masking of few-pass pixels, larger network). No ablation was run (README:129), so no single change gets the credit."
- "Unexplained variance fell 43 % (1 − R²: 0.46 → 0.26)."

**Interactions:** hovering a legend item highlights that model's dots in both panels and fades the others to 25 %. Click toggles a persistent highlight. The table view lists the three models as rows and the metrics as columns.
**Empty state:** "Model comparison unavailable."
**Mobile:** the panels stack, the legend wraps to 3 lines, and the sub-lane offset stays at ±7 px.

---

### V6. Error by land cover

**Question:** On which kinds of ground is the model least accurate, and does it run warm or cold there?
**Form:** **two aligned panels sharing one row order**: (a) horizontal bars of MAE, and (b) a diverging lollipop of bias centred at 0. They are two charts side by side with their own x-axes, not a dual-axis chart.

**Data binding:** `metrics.json landcover.rows[]` → `landcover, pixels_millions, MAE, bias`. Pixel share = `pixels_millions / Σ pixels_millions` (Σ = 17.81 M), computed in JS and shown to 1 decimal.

**Row order:** by pixel share descending (Cropland, Tree cover, Built-up, Grassland, Shrubland, Bare / sparse, Wetland), so the rows that matter most come first. A "Sort by MAE" toggle re-orders both panels together.

**Row label:** "Built-up · 19.1 % of pixels" (`--text-primary` name + `--text-secondary` share). **Built-up** is bold, with a small "urban heat" tag. Rows under 2 % of pixels (Bare / sparse 1.0 %, Wetland 0.1 %) get a "small sample" tag in `--text-muted`.

**Panel (a) MAE:**
- Bars ≤ 20 px, `--held-out`, square at 0, 4 px rounded end, with a 2 px gap between rows.
- x domain 0 – 2.5 °C, axis label "MAE (°C)".
- Reference line at the overall mean, 1.12 °C, labelled "all-pixel mean 1.12".
- Value label outside each bar end, 12 px `--text-primary`.

**Panel (b) Bias:**
- A 2 px stem from 0 to the value, with an r 5 dot.
- Colour by sign: `--bias-warm` above 0 ("runs warm"), `--bias-cool` below ("runs cold").
- x domain −1.0 – +0.6 °C, 0 line in `--axis`. Axis label "Bias, predicted − observed (°C)", with end captions "◂ runs cold" / "runs warm ▸".
- Value labels for all 7 rows outside the dot (only 7, all meaningful), with the sign shown.

**Legend:** panel (a) is a single series and needs none. Panel (b) has a two-swatch key: "● runs warm (+)", "● runs cold (−)".

**Tooltip (row; hovering anywhere on the row highlights both panels):**
- Title: "Bare / sparse".
- Rows: MAE 2.21 °C (1.97× the mean) · Bias −0.83 °C (runs cold) · Pixels 0.18 M (1.0 %).
- "Small sample: treat with caution" where tagged.

**Caption:**
- "Built-up pixels, the ones that matter for urban heat, are predicted at the average error with almost no bias."
- "The model runs warm over vegetation and cold over bare ground."
- "10 held-out cities, final model."

**Empty state:** "Land-cover breakdown unavailable."
**Mobile:** the panels stack vertically, both keep the same row order, and the row labels repeat above panel (b).

---

### V7. Hotspot quality: Delhi (training) vs Jaipur (held out)

**Question:** When the model maps hotspot *clusters*, how well do they match the satellite's clusters?
**Form:** a dumbbell / paired-dot chart (two dots per metric row on a shared 0–1 axis) plus a compact fact table. These are 0–1 scores of mixed kinds, compared across two cities, so dots on one axis read cleanly.

**Data binding:** `metrics.json hotspot_detection.cities.Delhi|Jaipur` → `precision, recall, IoU, block_mean_corr, blocks_flagged_pct, clusters_kept, clusters_area_pct`. F1 is **derived** = 2PR/(P+R) (Delhi 0.53, Jaipur 0.76), labelled "F1 (derived)". Average cluster size is derived = `clusters_area_pct / clusters_kept`.

**Title block:**
- Title: "Hotspot clusters: Delhi vs Jaipur".
- Subtitle: "Gi* clusters (z ≥ 2.58, ≥ 1 °C above neighbours), predicted vs observed. This is a different test from the 'hottest 10 % of pixels' AUC above."

**Encodings:**
- Rows: Precision · Recall · IoU · F1 (derived) · Block-mean correlation (r).
- x domain 0 – 1.0, axis label "Score (0–1)".
- Jaipur dot `--held-out` r 6, Delhi dot `--train` r 6, both with 2 px `--surface` rings, joined by a 2 px `--grid` connector.
- Value labels at both dots: Delhi's value on the left of the pair, Jaipur's on the right, 12 px `--text-primary`.

**Legend:** "● Jaipur · held out" and "● Delhi · training city, in-sample" (split badges).

**Fact table (right of chart; below it on mobile):**

| | Delhi · training city | Jaipur · held out |
|---|---|---|
| Blocks flagged | 11.1 % | 18.9 % |
| Clusters kept | 7 | 5 |
| Cluster area | 3.7 % of city | 16.0 % of city |
| Avg cluster size (derived) | 0.53 % of city | 3.2 % of city |

**Tooltip (dot):**
- Title: "Delhi" + "training city · in-sample".
- "Recall 0.50: recovers half the observed hotspot area".
- For F1: "Derived from precision and recall (not printed in README)".

**Caveat callout** (bordered note in `--text-secondary`, with an ⓘ icon):
- "Delhi is a training city, so its scores are in-sample. Jaipur is held out."
- "Jaipur's clusters are two very large zones on the dry outskirts (avg cluster 6× Delhi's share). Large zones overlap more easily, so Jaipur's IoU 0.62 is not directly comparable to Delhi's 0.36."

**Interactions:** hover a row to highlight both dots and show the tooltip. Legend hover fades the other city. The table view holds all 9 numbers.
**Empty state:** "Hotspot-cluster comparison unavailable."
**Mobile:** full width, with the fact table below.

---

### V8. Cooling doses per area (vegetation vs cool roofs)

**Question:** How much of each hot area's excess heat does the model explain, and how much planting or cool roofing would close that modelled gap?
**Form:** two linked small charts, with the area as the row in both:
- **(a)** a **segmented bar** per area: observed gap = modelled gap + not explained (°C).
- **(b)** a **dose bar chart**: for each area, three thin rows labelled Plant / Cool roofs / Both (each), with length = % of the eligible surface converted.

The levers are **direct-labelled rows, not colours**: a 3-colour lever palette failed the CVD gate (section 3.1), and colour is already spent on the split.

**Data binding:**
- `cooling.json rows[]` → `area, obs_gap_C, pred_gap_C, veg_dose_pct, dose_text`.
- Derived: `not_explained = obs_gap_C − pred_gap_C`, `share_seen = pred/obs` (equal to `insights.json cooling_derived.rows[]`).
- Split per area: Jaipur → `val`, Delhi → `train`, from `insights.json cooling_derived.rows[].split`.
- Roof and "both" doses and the hectare figures are **not separate fields** in `cooling.json`. They exist only inside `dose_text`. Ship them as an inline constant in the HTML with a source comment, and add a build-time assert that each number appears in the matching `dose_text`:

```js
// Source: cooling.json rows[].dose_text (README.md:50-52). null = not reported.
const DOSE_EXTRA = {
  "Jaipur, strongest 600 m hotspot": { veg_ha: 14, roof_pct: null, roof_ha: null, both_pct: null },
  "Delhi, dense 600 m block":        { veg_ha: 15, roof_pct: 99,   roof_ha: null, both_pct: 35 },
  "Delhi, 1.2 x 1.8 km area":        { veg_ha: 57, roof_pct: 48,   roof_ha: 76,   both_pct: 21 }
};
```

**Row labels (both panels):**
- "Jaipur · strongest 600 m hotspot" + "held out" badge
- "Delhi · dense 600 m block (36 ha)" + "training city · in-sample" badge
- "Delhi · 1.2 × 1.8 km area (211 ha)" + "training city · in-sample" badge

The 36 ha and 211 ha come from `docs/fig_cooling_curves.png`. Mark them "[fig]" in the tooltip.

**(a) Gap decomposition encodings:**
- Tall segmented bar, 24 px high, one per area, on a shared x axis 0 – 5 °C, axis label "Excess over neighbours (°C)".
- Segment 1 = modelled gap, filled with the area's split colour.
- Segment 2 = not explained, filled `--unexplained` with 45° texture in `--unexplained-ink`.
- 2 px `--surface` gap between segments; 4 px corners on the outer ends.
- Segment values print **below** the bar in text tokens ("model sees 2.2 · not explained 2.2"), because labels inside blue fail AA. The total "observed 4.4 °C" goes at the bar end.
- Legend: "▮ explained by the model (colour = city split)" and "▨ not explained by the inputs (traffic, industry, AC …)".

**(b) Dose encodings:**
- Per area, 3 bars ≤ 14 px thick, 2 px gap, in the area's split colour.
- x 0 – 100 %, axis label "Share of the changeable surface converted (%)". Light gridlines at 25 / 50 / 75 %.
- Direct label right of each bar: "Plant 45 % (15 ha)", "Cool roofs 99 %", "Both 35 % each".
- `null` → no bar. Instead an empty 1 px `--grid` track with the text "not reported" in `--text-muted` (e.g. Jaipur roofs and both).

**Tooltip:**
- On (a): area + badge; observed 4.5 °C; modelled 2.0 °C (44 % of observed); not explained 2.5 °C.
- On (b): lever, dose %, hectares if known, and "closes the **modelled** 2.0 °C gap, not the observed 4.5 °C".

**Interactions:** hovering an area row in either panel highlights the same area in the other. The table view combines both panels (Area · Split · Observed · Modelled · Not explained · Plant % (ha) · Roofs % (ha) · Both %).

**Caveat line (mandatory, under the panels):**
- "Model's what-if, not measured effects."
- "The model's cooling per +0.1 NDVI is 11–25 % steeper than the satellite data's (Delhi 1.0 vs 0.8 °C, Jaipur 2.0 vs 1.8 °C; `metrics.json vegetation_check`), so read doses as the optimistic end."
- "Delhi is a training city."

Optional: a "See full dose curves" link that opens `docs/fig_cooling_curves.png` (relative path; works when the HTML sits at repo level, otherwise hide it).

**Empty state:** "Cooling scenarios unavailable."
**Mobile:** (a) and (b) stack, and direct labels wrap under the bar when the label would overflow the card.

---

### V9. Feature-group importance

**Question:** Which inputs does the model lean on most?
**Form:** horizontal bars, single series, sorted descending. This is a magnitude comparison with a true zero, so bars are right.

**Data binding:**
- Bars 1–3: `metrics.json group_importance.rows[]` → `group, R2_drop` (vegetation 0.89, water 0.39, built-up 0.31).
- Bars 4–6: **not in any JSON**. They are printed on `docs/fig_group_importance.png` (terrain 0.07, albedo 0.03, land cover 0.03; see `04_insights.md` §4d). Ship them as an inline constant with source `docs/fig_group_importance.png`, and assert each is < `metrics.json group_importance.rest_below` (0.1).
- If the frontend chooses not to use fig-text values, render a single muted row "Terrain, albedo, land cover: each < 0.1" instead.

**Encodings:**
- y = group (Vegetation, Water, Built-up, Terrain, Albedo, Land cover).
- x = drop in R² (offset removed) when the group is shuffled between patches of the same city, domain 0 – 1.0, axis label "Drop in R² inside the city when the group is shuffled".
- Bars ≤ 20 px. JSON-sourced bars use `--held-out`. Fig-text bars use `--fig-text` (lighter step) with a "from figure" tag in `--text-muted` after the value.
- Value labels at all bar ends (6 bars, all meaningful).

**Legend:** none for a single series. A small key under the chart reads "▮ from metrics.json   ▮ read from docs figure". Swatches sit next to text in `--text-secondary`.

**Tooltip:** group name; "R² drop 0.89"; for vegetation also "Larger than the final R² (0.74): shuffled vegetation probably leaves R² below 0 (worse than the city-mean guess). Baseline not available, so this is an inference."; source line.

**Caveat:** "Importance is measured on the 10 held-out cities. It shows reliance, not causation."

**Fallback:** if the bars cannot be rendered, embed `docs/fig_group_importance.png` with alt text listing all six values.

**Empty state:** "Feature importance unavailable."
**Mobile:** full width; value labels stay outside the bars (domain up to 1.0 leaves room).

---

## 5. Interaction rules (shared)

**Tooltip component:**
- `--surface-raised` background, 1 px `--border`, radius 8, shadow (section 1.3), padding 8 × 12, max-width 280 px, 13 px text.
- Title line in 600 weight, `--text-primary`. Then rows of [optional swatch 8 px] label (`--text-secondary`) … value (`--text-primary`, tabular, right-aligned).
- Position: 12 px from the pointer, flipping at viewport edges. Appears after 0 ms and hides 100 ms after leave.
- `role="tooltip"`, `aria-live="polite"` for keyboard focus.
- Touch: tap shows and pins it, and tapping outside dismisses it.

**Hit targets:** at least 24 × 24 px (invisible enlarged circles/rects around dots and thin bars). For bars, the whole row band is the target.

**Highlight:** hover/focus emphasises the target and fades siblings to 30 % (never recolours them). Selection persists until Esc or clicking empty space.

**Filters:** only in one row above the chart they control (V3 split chips, V4 metric selector, V6 sort toggle, V2 overlay toggle). There is no global filter, since the datasets don't share a filterable dimension beyond the split.

**Table view:** every chart except V2-in-placeholder-mode has one, with the same numbers at full source precision.

**Focus:** a visible 2 px `--accent` outline with 2 px offset on all controls and focused marks.

**Print:** force light theme, enable textures, expand tables under charts, hide toggles.

---

## 6. Data embedding and placeholder swap

- **Single file, no fetch:** `fetch()` fails on `file://`, so each JSON is inlined as `<script type="application/json" id="data-<name>">…</script>` (metrics, per_city, cooling, cities, insights, grid) and read with `JSON.parse(el.textContent)`. Total inline JSON is about 25 KB plus the grid at 16.6 KB.
- **Grid contract** (`data/placeholder_grid.json` now; a real file later):

```json
{ "_placeholder": true, "_source": "...", "units": "degC anomaly (pixel minus grid mean)",
  "rows": 60, "cols": 60, "cell_m": 30, "values": [[...60 numbers...], ...60 rows] }
```

  - A real grid replaces the file with `_placeholder: false`, plus `city`, `split` and a `_source` naming the run.
  - Values are predicted anomaly (°C, prediction minus city mean). NaN/masked pixels are `null`, rendered as `--surface` with a 45° `--grid` hatch and the readout "masked (water or few satellite passes)".
  - Any `rows × cols` up to about 400 × 400 works. Cell size is computed from the container width.
  - Candidate source per `02_data_inventory.md` §e: `Deploy/cache/<City>_pred.npy` minus its mean, downsampled.
- The renderer decides everything from `_placeholder`. No other code changes are needed to swap.

---

## 7. Page layout (12-column grid, max 1200 px)

| Order | Section | ≥ 1024 px | 640–1023 px | < 640 px |
|---|---|---|---|---|
| 0 | Header: title "Heat_Map: urban heat from satellite data, 50 Indian cities", subtitle "U-Net predicting land-surface-temperature anomaly (Landsat late-morning pass, March–May) · 40 training + 10 held-out cities", theme toggle | 12 | 6 | 1 |
| 1 | V1 KPI row + caption | 12 | 6 | 1 |
| 2 | V2 heat map (placeholder) · V3 city map | 7 · 5 | 6 · 6 stacked | stacked |
| 3 | V4 per-city (dot plot 7 · table 5) | 7 · 5 | stacked | stacked |
| 4 | V5 model comparison · V9 feature importance | 7 · 5 | stacked | stacked |
| 5 | V6 land cover | 12 | 6 | 1 |
| 6 | V7 hotspot quality · V8 cooling | 6 · 6 | stacked | stacked |
| 7 | Caveats & sources: the `04_insights.md` §6 bullets (short form), run identity note (run3 vs run2), "metrics transcribed from README tables, rounded to 2 decimals" | 12 | 6 | 1 |

There is no horizontal page scroll at any width. Wide content (tables, metric chips) scrolls inside its own wrapper.

---

## 8. Copy rules (enforced in review)

- Use "held out from training (also used to pick the best epoch)" on first mention per section, and "held out" after that. Never "never seen" / "unseen".
- Name AUC as "Hotspot AUC (hottest 10 % of pixels)". Name Gi* results as "hotspot clusters".
- Every Delhi number carries "training city · in-sample".
- Present doses as "model's what-if". Use "closes the modelled gap", never "cools the area by".
- MAE is "pattern error (city mean supplied)", not absolute temperature error.
- Land surface temperature, not air temperature.

---

## Appendix A. Placeholder grid generator (run with `python -I script.py <out_path>`)

Produced `data/placeholder_grid.json`: 16,612 bytes, 60 × 60, `cell_m` 30, min −3.4, max +3.7, p2 −2.6, p98 +2.5, p90 +1.5 °C, mean 0.

```python
"""Synthetic placeholder anomaly grid for dashboard layout only (NOT model output)."""
import json, math, random, sys
out = sys.argv[1]
R = C = 60
rng = random.Random(20261119)
waves = [(rng.uniform(0.04, 0.16), rng.uniform(0.04, 0.16), rng.uniform(0, 2*math.pi), rng.uniform(0.3, 0.8)) for _ in range(6)]
blobs = [(14, 42, 6.0, 4.2), (38, 18, 8.0, 3.4), (47, 47, 4.5, 3.0), (22, 12, 5.0, -3.2), (30, 34, 7.0, -2.4), (52, 8, 4.0, -2.0)]
vals = []
for r in range(R):
    row = []
    for c in range(C):
        v = sum(a*math.sin(fx*c + fy*r + p) for fx, fy, p, a in waves)
        for br, bc, s, amp in blobs:
            v += amp*math.exp(-((r-br)**2 + (c-bc)**2)/(2*s*s))
        v += rng.gauss(0, 0.15)
        row.append(v)
    vals.append(row)
mean = sum(map(sum, vals))/(R*C)
vals = [[round(v-mean, 1) + 0.0 for v in row] for row in vals]
doc = {"_placeholder": True,
       "_source": "synthetic, generated by dashboard-project/05_viz_spec.md generator (make_placeholder_grid.py, seed 20261119: 6 low-frequency sinusoids + 3 warm and 3 cool Gaussian blobs + small noise, mean removed) for layout only. NOT model output, NOT satellite data.",
       "units": "degC anomaly (pixel minus grid mean)", "rows": R, "cols": C, "cell_m": 30, "values": vals}
open(out, "w", encoding="utf-8").write(json.dumps(doc, separators=(",", ":")))
```

## Appendix B. Data gaps that shaped the spec

- No real temperature grid → V2 is a placeholder.
- No per-city metrics for training cities → V3 tooltips show scores for held-out cities only.
- Roof/both doses exist only as text → the `DOSE_EXTRA` constant with asserts.
- Terrain/albedo/land-cover importance exist only on a figure → the `--fig-text` treatment.
- First-model MAE is absent → no delta on the MAE tile.
- First-model per-city values exist only for 3 cities as figure text (`insights.json model_comparison.first_to_final_per_city_R2_anomaly_fig_text`) → not charted. They are mentioned as a text callout in V5 if space allows ("Bengaluru 0.09 → 0.76, Bhubaneswar 0.12 → 0.73, Kolkata 0.14 → 0.79, read from figure").
