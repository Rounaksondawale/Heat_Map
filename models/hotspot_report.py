"""
Hotspots of a whole city, what drives each one, and how well the model finds them (read-only; nothing is retrained).

    python models/hotspot_report.py --city Delhi                       # uses data/unet_run3/best.pt
    python models/hotspot_report.py --city Delhi Jaipur --top 8 --z 2.58

1. The U-Net predicts the temperature of every pixel of the city (tiles of 512 pixels with a 32 pixel margin).
   Anomaly models predict the pattern inside the city, so the city's observed mean LST is added back.
2. Hotspots = Getis-Ord Gi* on blocks of --block pixels (default 20 = 600 m) of the PREDICTED temperature: a block is hot when it and its
   3 x 3 block neighbourhood (1.8 km) are significantly hotter than the city as a whole (z >= --z, default 1.96 = 95 %). There is no fixed
   share: a city may have many hotspots or none. Neighbouring hot blocks are merged into clusters (at least --min-blocks blocks).
3. Per cluster: area, observed and predicted temperature against its ring of neighbours (--ring pixels, hotspots excluded), and the part
   of the observed gap the model's features cannot explain (observed gap - predicted gap).
4. Drivers (the --top clusters): the features are split into groups (vegetation, built-up, albedo, water, terrain, land cover). Each
   group's values inside the cluster are swapped for the neighbours' average and the model predicts again; with 6 groups all 64
   combinations are tried and the contribution of each group is its exact Shapley value (degrees C above a neighbour-like version of
   the cluster). observed gap = sum of the groups + reference mismatch + not explained.
5. Checks: how well the predicted hotspots match hotspots found the same way on the OBSERVED temperature (precision / recall / IoU on
   blocks), and a vegetation sanity check: the model's response to +0.1 NDVI everywhere against the same effect fitted directly on
   the satellite data (both are associations, not proof of cause).
Writes <out>/<City>_hotspots.csv and <out>/<City>_hotspots.png. Temperatures are land surface temperature at the Landsat pass
(late morning, clear days, Mar-May), not air temperature. Drivers are what the MODEL attributes the heat to, not measured causes.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import distance_transform_edt, label, uniform_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pre_processing"))
from make_patches import REL_FLOOR, city_rel  # noqa: E402
from simulate_cooling import run_model  # noqa: E402
from train_unet import UNet  # noqa: E402

WATER, RES = 80, 30
GROUPS = ["vegetation", "built-up", "albedo", "water", "terrain", "land cover"]
NDVI_CHANNELS = ("NDVI", "NDVI_nbr150", "NDVI_nbr450", "NDVI_c90", "NDVI_c330", "NDVI_c930", "NDVI_c2010")


def group_of(name):
    if name == "LANDCOVER":
        return "land cover"
    if name.startswith("NDVI") or name == "DIST_GREEN_m":
        return "vegetation"
    if name.startswith(("NDBI", "GHSL", "BUILT", "HEIGHT")):
        return "built-up"
    if name.startswith("ALBEDO"):
        return "albedo"
    if name.startswith(("MNDWI", "DIST_WATER")):
        return "water"
    if name.startswith(("ELEV", "SLOPE")):
        return "terrain"
    raise SystemExit(f"feature {name} has no driver group")


def predict_grid(model, Z, lci, dev, mu, sd, tile=512, margin=32):
    C, H, W = Z.shape
    out = np.zeros((H, W), "float32")
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            r0, c0, r1, c1 = max(r - margin, 0), max(c - margin, 0), min(r + tile + margin, H), min(c + tile + margin, W)
            p = run_model(model, Z[:, r0:r1, c0:c1], lci[r0:r1, c0:c1], dev, mu, sd)
            h, w = min(r + tile, H) - r, min(c + tile, W) - c
            out[r:r + h, c:c + w] = p[r - r0:r - r0 + h, c - c0:c - c0 + w]
    return out


def block_mean(a, ok, B, min_share=0.5):
    Hb, Wb = a.shape[0] // B, a.shape[1] // B
    s = np.where(ok, a, 0.0)[:Hb * B, :Wb * B].reshape(Hb, B, Wb, B).sum((1, 3))
    n = ok[:Hb * B, :Wb * B].reshape(Hb, B, Wb, B).sum((1, 3))
    return np.where(n >= min_share * B * B, s / np.maximum(n, 1), np.nan)


def gi_star(block, win=3):
    valid = np.isfinite(block)
    w = valid.astype("float64")
    x = np.where(valid, block, 0.0)
    sx, sw = uniform_filter(x, win, mode="constant") * win * win, uniform_filter(w, win, mode="constant") * win * win
    n, xbar, s = valid.sum(), block[valid].mean(), block[valid].std()
    den = s * np.sqrt(np.maximum(n * sw - sw ** 2, 1e-9) / (n - 1))
    return np.where(valid, (sx - xbar * sw) / den, np.nan)


def shapley(v, k):
    phi = np.zeros(k)
    for g in range(k):
        for m in range(1 << k):
            if m >> g & 1:
                continue
            s = bin(m).count("1")
            phi[g] += math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k) * (v[m | 1 << g] - v[m])
    return phi


def lonlat(x, y, epsg):
    try:
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(x, y)
        return round(lon, 5), round(lat, 5)
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


def hotspot_labels(zp, z, min_blocks):
    """Clusters (labels 1..n) of blocks with Gi* z >= z that touch each other (8-neighbourhood) and have at least min_blocks blocks."""
    lab, ncl = label(np.nan_to_num(zp, nan=-9) >= z, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=ncl + 1)
    for i in range(1, ncl + 1):
        if sizes[i] < min_blocks:
            lab[lab == i] = 0
    return label(lab > 0, structure=np.ones((3, 3)))


def group_channels(names):
    return {g: [i for i, n in enumerate(names) if group_of(n) == g and n != "LANDCOVER"] for g in GROUPS}


def driver_phi(model, ck, Zc, lc_c, us, cm, ring, gidx, add, dev):
    """Shapley contribution (C) of each feature group to the mean predicted temperature of the pixels `cm`, against the neighbours `ring`."""
    ref = Zc[:, ring].mean(1)
    ref_lc = int(np.bincount(lc_c[ring & (lc_c < 255)]).argmax()) if (ring & (lc_c < 255)).any() else 255
    v = {}
    for m in range(1 << len(GROUPS)):
        Zm, lm = Zc.copy(), lc_c.copy()
        for gi, g in enumerate(GROUPS):
            if m >> gi & 1:
                continue                                           # this group keeps the cluster's own values
            for ch in gidx[g]:
                Zm[ch][cm] = ref[ch]
            if g == "land cover":
                lm[cm] = ref_lc
        v[m] = float(run_model(model, Zm, lm, dev, ck["mu"], ck["sd"])[cm & us].mean() + add)
    return shapley(v, len(GROUPS)), v


def prepare_city(city, grids, ck, meta):
    """Everything the model needs for one city: normalised input stack Z, land-cover indices, masks and the city's mean temperature."""
    gf = xr.open_dataset(Path(grids) / f"{city}_features.nc")
    tf = xr.open_dataset(Path(grids) / f"{city}_temp.nc")
    lst = tf["LST"].values.astype("float32")
    nobs = tf["n_obs"].values.astype("float32") if "n_obs" in tf else None
    lc = gf["LANDCOVER"].values
    xs, ys = gf["x"].values, gf["y"].values
    epsg = int(str(gf["x"].attrs.get("crs", "EPSG:0")).split(":")[-1] or 0)
    names = ck["features"]
    numeric = [n for n in names if not n.endswith("_rel")]
    look = {n: (m, s) for n, m, s in zip(meta["features"], meta["mean"], meta["std"])}
    H, W = lst.shape
    X = {n: gf[n].values.astype("float32") for n in numeric}
    finite = np.isfinite(np.stack([X[n] for n in numeric])).all(0)
    usable = finite & (lc != WATER)                                    # the model has an input everywhere here
    valid = usable & np.isfinite(lst)                                  # ... and the satellite has a temperature here
    if nobs is not None and meta.get("min_pass", 0) > 0:               # same few-pass rule as the training patches
        keep = np.isfinite(nobs) & (nobs >= meta["min_pass"] * float(np.median(nobs[valid & np.isfinite(nobs)])))
        if (valid & ~keep).sum() / max(valid.sum(), 1) * 100 <= meta.get("max_few_share", 3.0):
            valid &= keep
    chans, rel_s = [], {}
    for n in names:
        if n.endswith("_rel"):
            v = n[:-4]
            m, s = city_rel(X[v], usable, REL_FLOOR * look[v][1])
            rel_s[n] = s
            chans.append(np.where(np.isfinite(X[v]), (X[v] - m) / s, 0).astype("float32"))
        else:
            chans.append(np.where(np.isfinite(X[n]), (X[n] - look[n][0]) / look[n][1], 0).astype("float32"))
    Z = np.stack(chans)
    lci = np.full(lc.shape, 255, "uint8")
    for i, c in enumerate(ck["landcover_codes"]):
        lci[lc == c] = i
    city_mean = float(lst[valid].mean())
    anomaly = ck.get("target", "absolute") == "anomaly"
    add = city_mean if anomaly else 0.0
    return {"gf": gf, "lst": lst, "lc": lc, "xs": xs, "ys": ys, "epsg": epsg, "names": names, "look": look, "X": X,
            "usable": usable, "valid": valid, "Z": Z, "lci": lci, "rel_s": rel_s, "city_mean": city_mean, "anomaly": anomaly,
            "add": add, "H": H, "W": W}


def analyse(city, a, model, ck, meta, dev):
    P = prepare_city(city, a.grids, ck, meta)
    lst, lc, xs, ys, epsg, names, look, X = (P[k] for k in ("lst", "lc", "xs", "ys", "epsg", "names", "look", "X"))
    usable, valid, Z, lci, rel_s, city_mean, add, H, W = (P[k] for k in ("usable", "valid", "Z", "lci", "rel_s", "city_mean", "add", "H", "W"))
    pred = predict_grid(model, Z, lci, dev, ck["mu"], ck["sd"]) + add
    print(f"\n=== {city}: grid {H} x {W}, observed city mean {city_mean:.1f} C, {valid.mean() * 100:.0f}% of pixels usable ===")

    # ---- hotspots (Gi* on blocks of the predicted temperature)
    B = a.block
    bp, bo = block_mean(pred, usable, B), block_mean(lst, valid, B)
    zp, zo = gi_star(bp), gi_star(bo)
    lab, ncl = hotspot_labels(zp, a.z, a.min_blocks)
    hot = lab > 0
    nb = int(np.isfinite(bp).sum())
    print(f"hotspots (Gi* z >= {a.z} on {B * RES} m blocks): {int(hot.sum())} of {nb} blocks = {hot.sum() / max(nb, 1) * 100:.1f}% of the city, "
          f"{ncl} clusters of at least {a.min_blocks} blocks")
    both = np.isfinite(bp) & np.isfinite(bo)
    hot_o = np.nan_to_num(zo, nan=-9) >= a.z
    tp = int((hot & hot_o & both).sum())
    prec, rec = tp / max(int((hot & both).sum()), 1), tp / max(int((hot_o & both).sum()), 1)
    iou = tp / max(int(((hot | hot_o) & both).sum()), 1)
    r = float(np.corrcoef(bp[both], bo[both])[0, 1])
    print(f"vs hotspots found the same way on the OBSERVED temperature ({int((hot_o & both).sum())} blocks): precision {prec:.2f}  "
          f"recall {rec:.2f}  IoU {iou:.2f} | block-mean correlation observed vs predicted {r:.2f}")

    # ---- vegetation sanity check: model response to +0.1 NDVI everywhere vs regression on the satellite data
    d = 0.1
    Zs = Z.copy()
    for i, n in enumerate(names):
        if n in NDVI_CHANNELS:
            Zs[i] += d / look[n][1]
        elif n == "NDVI_rel":
            Zs[i] += d / rel_s[n]
    shift = predict_grid(model, Zs, lci, dev, ck["mu"], ck["sd"]) + add
    model_eff = float((shift - pred)[valid].mean())
    idx = np.flatnonzero(valid.ravel())[::5]
    df = pd.DataFrame({"lst": lst.ravel()[idx], "ndvi": X["NDVI"].ravel()[idx], "lc": lc.ravel()[idx],
                       "built": np.digitize(np.nan_to_num(X["GHSL_BUILT_FRAC"].ravel()[idx]), [0.1, 0.3, 0.6]),
                       "elev": pd.qcut(X["ELEV"].ravel()[idx], 5, labels=False, duplicates="drop"),
                       "wat": np.digitize(np.nan_to_num(X["DIST_WATER_m"].ravel()[idx], nan=5000.0), [300, 1500])})
    slope = lambda g: np.cov(g["ndvi"], g["lst"])[0, 1] / max(g["ndvi"].var(), 1e-9)   # noqa: E731
    parts = [(len(g), slope(g)) for _, g in df.groupby(["lc", "built", "elev", "wat"]) if len(g) >= 200]
    strat = sum(n * s for n, s in parts) / max(sum(n for n, _ in parts), 1) if parts else float("nan")
    print(f"vegetation check, change in LST for +0.1 NDVI: model {model_eff:+.2f} C | satellite data {slope(df) * d:+.2f} C without controls, "
          f"{strat * d:+.2f} C within groups of equal land cover, built-up share, elevation and distance to water "
          f"({sum(n for n, _ in parts) / len(df):.0%} of the pixels)")

    # ---- clusters
    rows = []
    ringpx = a.ring
    for cid in range(1, ncl + 1):
        bm = lab == cid
        pm = np.kron(bm, np.ones((B, B), bool))
        pm = np.pad(pm, ((0, H - pm.shape[0]), (0, W - pm.shape[1])))
        rr, cc = np.where(pm)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        hotpx = np.pad(np.kron(hot, np.ones((B, B), bool)), ((0, H - hot.shape[0] * B), (0, W - hot.shape[1] * B)))
        m0, m1, n0, n1 = max(r0 - ringpx - 40, 0), min(r1 + ringpx + 40, H), max(c0 - ringpx - 40, 0), min(c1 + ringpx + 40, W)
        sl = (slice(m0, m1), slice(n0, n1))
        cm, hp, us, vd = pm[sl], hotpx[sl], usable[sl], valid[sl]
        ring = (distance_transform_edt(~cm) <= ringpx) & ~hp & us
        if ring.sum() < 200 or (cm & us).sum() < 50:
            continue
        if float(pred[sl][cm & us].mean()) - float(pred[sl][ring].mean()) < a.min_gap:
            continue
        o_in, o_ring = float(lst[sl][cm & vd].mean()) if (cm & vd).any() else np.nan, float(lst[sl][ring & vd].mean()) if (ring & vd).any() else np.nan
        p_in, p_ring = float(pred[sl][cm & us].mean()), float(pred[sl][ring].mean())
        lonlat_c = lonlat(float(xs[(c0 + c1) // 2]), float(ys[(r0 + r1) // 2]), epsg)
        modes = np.bincount(lc[sl][cm & us].astype(int))
        row = {"cluster": cid, "rows": f"{r0}-{r1}", "cols": f"{c0}-{c1}", "lon": lonlat_c[0], "lat": lonlat_c[1],
               "area_ha": round(float(cm.sum()) * RES * RES / 1e4, 1), "main_landcover": int(modes.argmax()),
               "obs_C": round(o_in, 2), "obs_gap": round(o_in - o_ring, 2), "pred_gap": round(p_in - p_ring, 2),
               "not_explained": round((o_in - o_ring) - (p_in - p_ring), 2), "mean_z": round(float(np.nanmean(zp[bm])), 2)}
        rows.append((row, sl, cm, ring, p_in, p_ring))
    rows.sort(key=lambda t: -t[0]["pred_gap"])
    hot = np.isin(lab, [r[0]["cluster"] for r in rows])                # the map shows only the clusters that were kept
    print(f"kept {len(rows)} clusters with a predicted gap of at least {a.min_gap:g} C to their neighbours "
          f"({hot.sum() / max(nb, 1) * 100:.1f}% of the city's blocks)")

    # ---- drivers (Shapley over the feature groups) for the top clusters
    gidx = group_channels(names)
    out_rows = []
    for rank, (row, sl, cm, ring, p_in, p_ring) in enumerate(rows, 1):
        row["rank"] = rank
        if rank <= a.top:
            Zc, lc_c, us = Z[:, sl[0], sl[1]], lci[sl[0], sl[1]], usable[sl]
            phi, v = driver_phi(model, ck, Zc, lc_c, us, cm, ring, gidx, add, dev)
            for g, p in zip(GROUPS, phi):
                row[f"drv_{g}"] = round(float(p), 2)
            row["drv_ref_mismatch"] = round(row["pred_gap"] - float(phi.sum()), 2)
            order = np.argsort(-phi)
            row["top_drivers"] = ", ".join(f"{GROUPS[i]} {phi[i]:+.1f}" for i in order[:3] if abs(phi[i]) >= 0.05)
        out_rows.append(row)
    t = pd.DataFrame(out_rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"{city}_hotspots.csv", index=False)
    if t.empty:
        print("no hotspot clusters at this threshold")
    else:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        show = ["rank", "rows", "cols", "lat", "lon", "area_ha", "obs_gap", "pred_gap", "not_explained"]
        print("\nclusters ranked by predicted gap to their neighbours (C); not_explained = observed gap - predicted gap")
        print(t[show].head(a.top * 2).to_string(index=False))
        dcols = [c for c in t.columns if c.startswith("drv_")]
        if dcols:
            print("\ndrivers of the top clusters, C above a neighbour-like version of the cluster (positive = makes it hotter):")
            print(t[["rank"] + dcols + ["top_drivers"]].head(a.top).rename(columns=lambda c: c.replace("drv_", "")).to_string(index=False))
            top = t.iloc[0]
            print(f"\nnext: python models/simulate_cooling.py --city {city} --run {a.run} --box {top['rows'].replace('-', ' ')} {top['cols'].replace('-', ' ')}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    lo, hi = np.nanpercentile(bo, [2, 98])
    for a_, g, ttl, kw in ((ax[0], bo, "observed LST (C)", dict(cmap="inferno", vmin=lo, vmax=hi)),
                           (ax[1], bp, "predicted LST (C)", dict(cmap="inferno", vmin=lo, vmax=hi)),
                           (ax[2], zp, f"Gi* z of the prediction (hotspots: z >= {a.z})", dict(cmap="coolwarm", vmin=-4, vmax=4))):
        im = a_.imshow(g, **kw)
        a_.contour(hot.astype(float), levels=[0.5], colors="cyan", linewidths=1)
        a_.set_title(f"{city}: {ttl}")
        a_.set_xticks([])
        a_.set_yticks([])
        plt.colorbar(im, ax=a_, shrink=0.7)
    for row, *_ in rows[:a.top]:
        r0, r1 = (int(x) // B for x in row["rows"].split("-"))
        c0, c1 = (int(x) // B for x in row["cols"].split("-"))
        ax[2].text((c0 + c1) / 2, (r0 + r1) / 2, str(row["rank"]), color="k", fontsize=9, ha="center", va="center",
                   bbox=dict(facecolor="white", alpha=0.7, pad=1, lw=0))
    fig.savefig(out / f"{city}_hotspots.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out / (city + '_hotspots.csv')} and {out / (city + '_hotspots.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", required=True)
    ap.add_argument("--run", default="data/unet_run3", help="folder with best.pt")
    ap.add_argument("--grids", default="pre_processing/grids")
    ap.add_argument("--patches", default="pre_processing/patches", help="only meta.json (feature mean/std) is read")
    ap.add_argument("--block", type=int, default=20, help="block size in pixels (30 m each)")
    ap.add_argument("--z", type=float, default=2.58, help="Gi* z-score for a hotspot (1.645 = 90%, 1.96 = 95%, 2.58 = 99%, 3.29 = 99.9%)")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="keep only clusters whose predicted temperature is at least this many C above their neighbours")
    ap.add_argument("--min-blocks", type=int, default=2)
    ap.add_argument("--ring", type=int, default=30, help="width of the neighbour ring in pixels")
    ap.add_argument("--top", type=int, default=10, help="clusters that get a driver analysis")
    ap.add_argument("--out", default="data/hotspots")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev, weights_only=False)
    meta = json.load(open(Path(a.patches) / "meta.json"))
    model = UNet(len(ck["features"]), len(ck["landcover_codes"]), ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    for city in a.city:
        analyse(city, a, model, ck, meta, dev)


if __name__ == "__main__":
    main()
