"""
How much vegetation / cool roofing would make a hot area as cool as its neighbours? (what-if simulation on the trained U-Net)

    python models/simulate_cooling.py --city Delhi --auto 20                  # strongest hotspot of 20x20 pixels (600 m)
    python models/simulate_cooling.py --city Delhi --box 640 660 300 320     # your own area: rows 640-660, cols 300-320

Area and neighbours: the area is a block of the city grid; its neighbours are the ring of --ring pixels (30 m each) around it.
--auto picks the block whose observed LST is highest compared with its ring.
Levers (applied to a growing share of the pixels they apply to, the same pixels in every run):
    vegetation   open / thinly built pixels (NDVI < 0.4, built fraction < --max-built) take the city's typical tree-cover values
                 (median NDVI, NDBI, MNDWI, ALBEDO of its tree pixels), land cover = tree
    cool roofs   built pixels get ALBEDO + --roof-albedo (capped at 0.6)
After each edit the neighbourhood and surroundings features (nbr150/450, 90 m - 2 km windows, distance to green) are recomputed
exactly as in the dataset build, the U-Net predicts again, and the area's predicted temperature is compared with the ring's.
Prints, for 0 - 100 % of each lever: cooling of the area and the gap to its neighbours that is left, then the dose that closes it.
Also writes a CSV and a figure to --out. All results are the MODEL's view: associations it learned, not measured effects.
Temperatures are land surface temperature at the Landsat pass (late morning, clear days, Mar-May), not air temperature.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import distance_transform_edt, uniform_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extraction"))
sys.path.insert(0, str(HERE.parent / "pre_processing"))
from add_context_features import VARS, context_grids  # noqa: E402
from make_patches import REL_FLOOR, city_rel  # noqa: E402
from train_unet import CLIP, UNet  # noqa: E402

RES, WATER, PIXEL_HA = 30, 80, 0.09
EDIT = ("NDVI", "NDBI", "MNDWI", "ALBEDO", "LANDCOVER")


def find_hotspot(lst, valid, n, ring, built=None, min_built=0.0):
    """Top-left (row, col) of the n x n block whose mean LST is highest above the mean of its surrounding ring."""
    H, W = lst.shape
    integral = lambda a: np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))   # noqa: E731
    sv, cv = integral(np.where(valid, lst, 0.0).astype("float64")), integral(valid.astype("float64"))
    box = lambda S, i, j, h: S[i + h, j + h] - S[i, j + h] - S[i + h, j] + S[i, j]   # noqa: E731
    ii, jj = np.meshgrid(np.arange(ring, H - n - ring + 1), np.arange(ring, W - n - ring + 1), indexing="ij")
    if ii.size == 0:
        raise SystemExit("grid too small for this area size and ring")
    big = n + 2 * ring
    sw, cw = box(sv, ii, jj, n), box(cv, ii, jj, n)
    sb, cb = box(sv, ii - ring, jj - ring, big), box(cv, ii - ring, jj - ring, big)
    cr = cb - cw
    ok = (cw >= 0.8 * n * n) & (cr >= 0.6 * (big * big - n * n))
    if built is not None and min_built > 0:                       # only blocks that are (densely) built up
        ok &= box(integral(np.nan_to_num(built).astype("float64")), ii, jj, n) >= min_built * n * n
    gap = np.where(ok, sw / np.maximum(cw, 1) - (sb - sw) / np.maximum(cr, 1), -np.inf)
    if not np.isfinite(gap.max()):
        raise SystemExit("no block with enough valid pixels; use --box")
    k = np.unravel_index(np.argmax(gap), gap.shape)
    return int(ii[k]), int(jj[k])


@torch.no_grad()
def run_model(model, Z, lci, dev, mu, sd):
    h, w = lci.shape
    ph, pw = (-h) % 8, (-w) % 8                                   # the U-Net halves the size three times
    x = torch.from_numpy(np.pad(Z, ((0, 0), (0, ph), (0, pw)))).float().clamp(-CLIP, CLIP)[None].to(dev)
    lc = torch.from_numpy(np.pad(lci, ((0, ph), (0, pw)), constant_values=255))[None].to(dev)
    return (model(x, lc)[0].cpu().numpy() * sd + mu)[:h, :w]


class Simulator:
    def __init__(self, model, ck, meta, base, reg, box, targets, a, dev, rel=None):
        self.model, self.dev, self.a = model, dev, a
        self.mu, self.sd, self.codes = ck["mu"], ck["sd"], ck["landcover_codes"]
        self.names = list(base)                                    # all feature names of the grid file
        self.model_names = ck["features"]                          # the model's own inputs (old and new checkpoints differ)
        look = {n: (m, s) for n, m, s in zip(meta["features"], meta["mean"], meta["std"])}
        missing = [n for n in self.model_names if n not in look]
        if missing:
            raise SystemExit(f"patches/meta.json has no normalisation for the model inputs {missing[:3]}")
        self.mean = np.array([look[n][0] for n in self.model_names], "float32")[:, None, None]
        self.std = np.array([look[n][1] for n in self.model_names], "float32")[:, None, None]
        self.std_by_name = {n: look[n][1] for n in self.model_names}
        self.rel = rel or {}                                       # name -> (source feature, city mean, city std)
        self.base, self.reg, self.box, self.targets = base, reg, box, targets
        self.shape = reg.shape
        lc = base["LANDCOVER"]
        built = np.nan_to_num(base["GHSL_BUILT_FRAC"])
        water = (lc == WATER) | (base["MNDWI"] > 0.1)
        elig_g = reg & ~water & (np.nan_to_num(base["NDVI"], nan=1.0) < 0.4) & (built < a.max_built)
        elig_r = reg & ~water & ((lc == 50) | (built >= 0.3))
        rng = np.random.default_rng(a.seed)
        self.idx_g = rng.permutation(np.flatnonzero(elig_g.ravel()))
        self.idx_r = rng.permutation(np.flatnonzero(elig_r.ravel()))
        X = np.stack([base[n] for n in self.model_names])
        self.ok = np.isfinite(X).all(0) & (lc != WATER)

    def features(self, g, r):
        """Feature grids of the crop after greening a share g and cool-roofing a share r of the eligible pixels."""
        b = self.base
        f = {k: b[k].copy() for k in EDIT}
        green = np.zeros(self.shape, bool).ravel()
        green[self.idx_g[:int(round(g * len(self.idx_g)))]] = True
        roof = np.zeros(self.shape, bool).ravel()
        roof[self.idx_r[:int(round(r * len(self.idx_r)))]] = True
        green, roof = green.reshape(self.shape), roof.reshape(self.shape)
        roof &= ~green
        for k in ("NDVI", "NDBI", "MNDWI", "ALBEDO"):
            f[k][green] = self.targets[k]
        f["LANDCOVER"][green] = 10
        f["ALBEDO"][roof] = np.maximum(f["ALBEDO"][roof], np.minimum(f["ALBEDO"][roof] + self.a.roof_albedo, 0.6))
        new = {"NDVI_nbr150": uniform_filter(np.nan_to_num(f["NDVI"]), 5, mode="nearest"),
               "NDVI_nbr450": uniform_filter(np.nan_to_num(f["NDVI"]), 15, mode="nearest")}
        ds = xr.Dataset({v: (("y", "x"), (f[v] if v in f else b[v]).astype("float64")) for v in VARS})
        for k, arr in context_grids(ds).items():
            new[k] = arr.values
        dist = distance_transform_edt(~green) * RES if green.any() else np.full(self.shape, np.inf)
        new["DIST_GREEN_m"] = np.minimum(b["DIST_GREEN_m"], dist)
        out = {n: b[n].copy() for n in self.names}
        bs = self.box
        for k in EDIT:
            out[k] = f[k]
        for k, v in new.items():
            if k in out:
                out[k][bs] = v.astype("float32")[bs]
        return out, int(green.sum()), int(roof.sum())

    def predict(self, feats):
        for n, (v, m, s) in self.rel.items():                      # city-relative copies follow the edited features
            feats[n] = ((feats[v] - m) / s).astype("float32")
        X = np.stack([feats[n] for n in self.model_names])
        Z = np.where(np.isfinite(X), (X - self.mean) / self.std, 0).astype("float32")
        lci = np.full(self.shape, 255, "uint8")
        for i, c in enumerate(self.codes):
            lci[feats["LANDCOVER"] == c] = i
        return run_model(self.model, Z, lci, self.dev, self.mu, self.sd)


def area_stats(pred, ok, reg, ring):
    r, b = pred[reg & ok], pred[ring & ok]
    hot = np.sort(r)[-max(1, int(round(0.1 * len(r)))):].mean()
    return float(r.mean()), float(b.mean()), float(hot)


def dose_for(doses, gaps):
    """Smallest dose where the gap to the neighbours reaches 0 (linear interpolation); None if it never does."""
    for i, g in enumerate(gaps):
        if g <= 0:
            if i == 0:
                return 0.0
            return float(doses[i - 1] + (doses[i] - doses[i - 1]) * gaps[i - 1] / (gaps[i - 1] - g))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--auto", type=int, default=0, help="find the strongest hotspot of N x N pixels (30 m each)")
    ap.add_argument("--box", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"), help="rows R0:R1, columns C0:C1 of the city grid")
    ap.add_argument("--ring", type=int, default=30, help="width of the neighbour ring in pixels (30 = 900 m)")
    ap.add_argument("--min-built", type=float, default=0.0,
                    help="with --auto: only blocks whose mean built fraction is at least this (0.4 finds dense urban hotspots, where cool roofs apply)")
    ap.add_argument("--run", default="data/unet_run2", help="folder with best.pt")
    ap.add_argument("--grids", default="pre_processing/grids")
    ap.add_argument("--patches", default="pre_processing/patches", help="only meta.json (feature mean/std) is read")
    ap.add_argument("--max-built", type=float, default=0.6, help="vegetation only on pixels with a built fraction below this")
    ap.add_argument("--roof-albedo", type=float, default=0.15, help="albedo increase of a cool roof")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/simulations")
    a = ap.parse_args()
    if bool(a.auto) == bool(a.box):
        raise SystemExit("give exactly one of --auto N or --box R0 R1 C0 C1")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev, weights_only=False)
    meta = json.load(open(Path(a.patches) / "meta.json"))
    model = UNet(len(ck["features"]), len(ck["landcover_codes"]), ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    gf = xr.open_dataset(Path(a.grids) / f"{a.city}_features.nc")
    lst = xr.open_dataset(Path(a.grids) / f"{a.city}_temp.nc")["LST"].values.astype("float32")
    lc_full = gf["LANDCOVER"].values
    valid = np.isfinite(lst) & (lc_full != WATER)
    H, W = lst.shape
    if a.auto:
        r0, c0 = find_hotspot(lst, valid, a.auto, a.ring, gf["GHSL_BUILT_FRAC"].values, a.min_built)
        r1, c1 = r0 + a.auto, c0 + a.auto
        how = "found automatically as the strongest hotspot compared with its ring"
    else:
        r0, r1, c0, c1 = a.box
        how = "given by --box"
    r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, H), min(c1, W)
    # crop = area + ring + 34 pixels (the largest surroundings window is 67 pixels wide)
    pad = a.ring + 34
    R0, R1, C0, C1 = max(0, r0 - pad), min(H, r1 + pad), max(0, c0 - pad), min(W, c1 + pad)
    sl = (slice(R0, R1), slice(C0, C1))
    base = {n: gf[n].isel(y=sl[0], x=sl[1]).values.astype("float32") for n in gf.data_vars}
    reg = np.zeros(base["NDVI"].shape, bool)
    reg[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = True
    boxm = np.zeros_like(reg)
    boxm[max(r0 - a.ring, R0) - R0:min(r1 + a.ring, R1) - R0, max(c0 - a.ring, C0) - C0:min(c1 + a.ring, C1) - C0] = True
    ring = boxm & ~reg
    bs = (slice(max(r0 - a.ring, R0) - R0, min(r1 + a.ring, R1) - R0), slice(max(c0 - a.ring, C0) - C0, min(c1 + a.ring, C1) - C0))

    # what a "green" pixel of this city looks like
    ndvi, lcf = gf["NDVI"].values, lc_full
    tree = (lcf == 10) & (ndvi > 0.4)
    if tree.sum() < 500:
        tree = (ndvi > 0.4) & (lcf != WATER)
    if not tree.any():
        raise SystemExit("no green reference pixels in this city")
    targets = {k: float(np.nanmedian(gf[k].values[tree])) for k in ("NDVI", "NDBI", "MNDWI", "ALBEDO")}

    # city-relative inputs (checkpoints trained on the _rel channels): the city's own mean / spread of each feature, from the unedited grid
    rel = {}
    rel_names = [n for n in ck["features"] if n.endswith("_rel")]
    if rel_names:
        numeric = [n for n in ck["features"] if not n.endswith("_rel")]
        usable_city = np.isfinite(np.stack([gf[n].values for n in numeric])).all(0) & (lc_full != WATER)
        gstd = dict(zip(meta["features"], meta["std"]))
        for n in rel_names:
            v = n[:-4]
            m, s = city_rel(gf[v].values, usable_city, REL_FLOOR * gstd[v])
            rel[n] = (v, m, s)
            base[n] = ((base[v] - m) / s).astype("float32")

    sim = Simulator(model, ck, meta, base, reg, bs, targets, a, dev, rel)
    f0, _, _ = sim.features(0, 0)
    drift = max(float(np.nanmax(np.abs(f0[n][bs] - base[n][bs])) / sim.std_by_name.get(n, 1.0)) for n in sim.model_names)
    p0 = sim.predict(base)
    lst_c = lst[sl[0], sl[1]]
    okl = sim.ok & np.isfinite(lst_c)
    o_reg, o_ring = float(lst_c[reg & okl].mean()), float(lst_c[ring & okl].mean())
    city_mean = float(lst[valid].mean())
    b_reg, b_ring, b_hot = area_stats(p0, sim.ok, reg, ring)
    obs_gap, pred_gap = o_reg - o_ring, b_reg - b_ring
    n_px = int((reg & sim.ok).sum())
    print(f"\n{a.city}: area rows {r0}-{r1}, cols {c0}-{c1} ({r1 - r0} x {c1 - c0} pixels = {(r1 - r0) * RES} m x {(c1 - c0) * RES} m, "
          f"{n_px * PIXEL_HA:.1f} ha usable), {how}; neighbours = ring of {a.ring} pixels ({a.ring * RES} m)")
    print(f"observed LST: area {o_reg:.1f} C, neighbours {o_ring:.1f} C -> area is {obs_gap:+.1f} C vs its neighbours "
          f"({o_reg - city_mean:+.1f} C vs the city mean {city_mean:.1f} C)")
    print(f"model:        gap to neighbours {pred_gap:+.1f} C -> not explained by the features: {obs_gap - pred_gap:+.1f} C")
    print(f"levers: {len(sim.idx_g)} greenable pixels ({len(sim.idx_g) * PIXEL_HA:.1f} ha), {len(sim.idx_r)} built pixels for cool roofs "
          f"({len(sim.idx_r) * PIXEL_HA:.1f} ha) | tree reference: NDVI {targets['NDVI']:.2f}, ALBEDO {targets['ALBEDO']:.2f}")
    print(f"check: recomputing the features without any edit changes them by at most {drift:.3f} sd" + ("  <- WARNING" if drift > 0.05 else ""))
    if pred_gap <= 0:
        print("\nIn the model's view this area is not hotter than its neighbours, so there is no gap to close.")

    doses = np.linspace(0, 1, a.steps + 1)
    rows = []
    for d in doses:
        row = {"dose_%": round(d * 100)}
        for tag, (g, r) in {"veg": (d, 0), "roof": (0, d), "both": (d, d)}.items():
            feats, ng, nr = sim.features(g, r)
            reg_s, ring_s, hot_s = area_stats(sim.predict(feats), sim.ok, reg, ring)
            row[f"{tag}_cooling"] = b_reg - reg_s
            row[f"{tag}_gap_left"] = reg_s - ring_s
            row[f"{tag}_hot10_gap_left"] = hot_s - ring_s
        rows.append(row)
    t = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"sim_{a.city}_{r0}_{c0}"
    t.round(3).to_csv(out / f"{stem}.csv", index=False)

    print("\ndose = share of the eligible pixels converted | cooling = drop of the area's mean | gap left = area minus neighbours (C)")
    print(f"{'dose':>5} | {'vegetation':^20} | {'cool roofs':^20} | {'both together':^20}")
    print(f"{'':>5} | {'cooling':>9} {'gap left':>9} | {'cooling':>9} {'gap left':>9} | {'cooling':>9} {'gap left':>9}")
    for r in rows:
        print(f"{r['dose_%']:>4}% | {r['veg_cooling']:>9.2f} {r['veg_gap_left']:>9.2f} | {r['roof_cooling']:>9.2f} {r['roof_gap_left']:>9.2f} | "
              f"{r['both_cooling']:>9.2f} {r['both_gap_left']:>9.2f}")
    print(f"\nTo bring the area down to its neighbours' temperature (model's view, predicted gap {pred_gap:+.1f} C):")
    for tag, label, n in (("veg", "vegetation alone", len(sim.idx_g)), ("roof", "cool roofs alone", len(sim.idx_r)),
                          ("both", "both together", None)):
        if tag == "roof" and n < 30:
            print(f"  {label}: not applicable here, only {n} built pixels ({n * PIXEL_HA:.1f} ha) in this area "
                  f"(--auto N --min-built 0.4 looks for a dense urban hotspot)")
            continue
        dose = dose_for(doses, t[f"{tag}_gap_left"].values) if pred_gap > 0 else 0.0
        if dose is None:
            left = t[f"{tag}_gap_left"].iloc[-1]
            print(f"  {label}: NOT enough, even at 100% of the eligible pixels {left:+.1f} C of the gap is left")
        elif tag == "both":
            print(f"  {label}: {dose:.0%} of the greenable pixels ({dose * len(sim.idx_g) * PIXEL_HA:.1f} ha) and {dose:.0%} of the built "
                  f"pixels ({dose * len(sim.idx_r) * PIXEL_HA:.1f} ha)")
        else:
            print(f"  {label}: {dose:.0%} of the eligible pixels = {dose * n * PIXEL_HA:.1f} ha ({dose * n * PIXEL_HA / (n_px * PIXEL_HA):.0%} of the area)")
    print(f"  stricter target, the hottest 10% of the area at the neighbours' level: gap left with everything "
          f"{t['both_hot10_gap_left'].iloc[-1]:+.1f} C (now {b_hot - b_ring:+.1f} C)")
    for tag, label, n in (("veg", "vegetation", len(sim.idx_g)), ("roof", "cool roofs", len(sim.idx_r))):
        if n >= 30 and t[f"{tag}_cooling"].iloc[-1] < 0.05:
            print(f"  WARNING: the model predicts no cooling from {label} ({t[f'{tag}_cooling'].iloc[-1]:+.2f} C at 100%). It learned associations, "
                  f"not physics (e.g. bright surfaces are often hot, dry ground), so do not trust this lever's result")
    unexpl = obs_gap - pred_gap
    if unexpl > 0.5:
        print(f"  note: {unexpl:.1f} C of the observed gap is not explained by the model's features, so these levers cannot remove that part")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    lo, hi = np.nanpercentile(lst_c[okl], [2, 98])
    for a_, g, ttl in ((ax[0], np.where(okl, lst_c, np.nan), "observed LST (C)"), (ax[1], np.where(sim.ok, p0, np.nan), "model: predicted (C; relative to the city mean for an anomaly model)")):
        im = a_.imshow(g, cmap="inferno", vmin=lo if a_ is ax[0] else None, vmax=hi if a_ is ax[0] else None)
        a_.add_patch(Rectangle((c0 - C0 - .5, r0 - R0 - .5), c1 - c0, r1 - r0, fill=False, ec="cyan", lw=2))
        b0, b1 = bs
        a_.add_patch(Rectangle((b1.start - .5, b0.start - .5), b1.stop - b1.start, b0.stop - b0.start, fill=False, ec="white", lw=1.2, ls="--"))
        a_.set_title(f"{a.city}: {ttl}; cyan = area, dashed = ring")
        a_.set_xticks([])
        a_.set_yticks([])
        plt.colorbar(im, ax=a_, shrink=0.7)
    for tag, lab in (("veg", "vegetation"), ("roof", "cool roofs"), ("both", "both")):
        ax[2].plot(doses * 100, t[f"{tag}_gap_left"], "o-", label=lab)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set_xlabel("share of eligible pixels converted (%)")
    ax[2].set_ylabel("area minus neighbours (C), predicted")
    ax[2].set_title("gap left: 0 = as cool as its neighbours")
    ax[2].legend()
    fig.savefig(out / f"{stem}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out / (stem + '.csv')} and {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
