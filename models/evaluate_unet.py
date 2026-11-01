"""
Evaluate a trained U-Net city by city (read-only; nothing is retrained).

    python models/evaluate_unet.py                                  # best.pt on the validation cities
    python models/evaluate_unet.py --split train --cities Delhi Pune   # check the gap to the training cities

Per city, on usable pixels only (mask == 1 and 0 < LST < 75):
    MAE, RMSE, bias           error in deg C (bias = mean(pred - true); a city-wide offset shows up here)
    MAE_citymean              MAE of the trivial guess "every pixel = the city's true mean" (a hard-to-beat oracle baseline)
    skill                     1 - MAE / MAE_citymean; above 0 means the model beats that guess inside the city
    R2_within                 1 - SSE / variation of the true LST inside the city; below 0 = worse than the city mean
    R2_anomaly                same after removing each map's own mean from prediction and truth (ignores a constant offset)
    spearman                  rank correlation inside the city: does it put the hot pixels in the right places?
    hot_AUC / hot_F1 / hot_IoU  hotspot test: true hotspot = hottest --hot-share (default 10 %) of the city's pixels;
                              AUC ranks all pixels, F1 and IoU compare the predicted hottest 10 % with the true one
Also writes the error by land-cover class, a scatter and training-curve figure, and a true / predicted / error map per city.
Outputs go to <run>/eval_<split>/. NOTE: best.pt was chosen as the best epoch on the validation cities, so these scores are
slightly optimistic; for a final number keep a few cities that were never used to pick the epoch.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_unet import CLIP, UNet  # noqa: E402


@torch.no_grad()
def predict(model, X, L, dev, mu, sd, bs=128):
    out = np.empty(L.shape, "float32")
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev).float().clamp(-CLIP, CLIP)
        lc = torch.from_numpy(L[i:i + bs]).to(dev)
        out[i:i + bs] = (model(x, lc) * sd + mu).cpu().numpy()
    return out


def city_metrics(y, p, mu, share):
    y, p = y.astype("float64"), p.astype("float64")
    e = p - y
    yc, pc = y - y.mean(), p - p.mean()
    sst = float((yc ** 2).sum())
    ry, rp = rankdata(y), rankdata(p)
    hot = y >= np.quantile(y, 1 - share)
    npos, nneg = int(hot.sum()), int((~hot).sum())
    pred_hot = p >= np.quantile(p, 1 - share)
    tp = int((pred_hot & hot).sum())
    prec, rec = tp / max(pred_hot.sum(), 1), tp / max(npos, 1)
    mae = float(np.abs(e).mean())
    mae_city = float(np.abs(yc).mean())
    return {"pixels": len(y), "LST_mean": y.mean(), "LST_std": y.std(), "MAE": mae, "RMSE": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "MAE_citymean": mae_city, "skill": 1 - mae / mae_city,
            "R2_within": 1 - float((e ** 2).sum()) / sst, "R2_anomaly": 1 - float(((pc - yc) ** 2).sum()) / sst,
            "spearman": float(np.corrcoef(ry, rp)[0, 1]),
            "hot_AUC": float((rp[hot].sum() - npos * (npos + 1) / 2) / (npos * nneg)),
            "hot_F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "hot_IoU": tp / max(int((pred_hot | hot).sum()), 1),
            "hot_bias": float(e[hot].mean())}


def stitch(d, y, p, m):
    size = d["X"].shape[-1]
    pos = d["pos"]
    H, W = pos[:, 0].max() + size, pos[:, 1].max() + size
    gy, gp = np.full((H, W), np.nan, "float32"), np.full((H, W), np.nan, "float32")
    for (r, c), yy, pp, mm in zip(pos, y, p, m):
        gy[r:r + size, c:c + size] = np.where(mm, yy, np.nan)
        gp[r:r + size, c:c + size] = np.where(mm, pp, np.nan)
    return gy, gp


def draw_map(city, gy, gp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.nanpercentile(gy, [2, 98])
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    for a, g, t in ((ax[0], gy, "true LST (deg C)"), (ax[1], gp, "predicted LST (deg C)")):
        im = a.imshow(g, cmap="inferno", vmin=lo, vmax=hi)
        a.set_title(f"{city}: {t}")
        plt.colorbar(im, ax=a, shrink=0.7)
    err = gp - gy
    lim = float(np.nanpercentile(np.abs(err), 98))
    im = ax[2].imshow(err, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title("error: predicted - true (red = too hot)")
    plt.colorbar(im, ax=ax[2], shrink=0.7)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/unet_run", help="folder holding best.pt and metrics.csv")
    ap.add_argument("--patches", default="pre_processing/patches")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--hot-share", type=float, default=0.10, help="share of a city's pixels counted as hotspot")
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    run, root = Path(a.run), Path(a.patches)
    out = run / f"eval_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)   # our own file
    model = UNet(len(ck["features"]), len(ck["landcover_codes"]), ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    mu, sd, codes = ck["mu"], ck["sd"], ck["landcover_codes"]
    files = sorted((root / a.split).glob("*.npz"))
    if a.cities:
        files = [f for f in files if f.stem in a.cities]
    print(f"{run / 'best.pt'} | {len(files)} {a.split} cities | device {dev}\n", flush=True)

    rows, lc_acc = [], np.zeros((len(codes), 3))
    sse = sae = n = sy = syy = 0.0
    sx, sp_ = [], []
    for f in files:
        t0 = time.time()
        d = np.load(f)
        y, m = d["y"], (d["mask"] == 1) & (d["y"] > 0) & (d["y"] < 75)
        p = predict(model, d["X"], d["lc"], dev, mu, sd)
        r = {"city": f.stem, **city_metrics(y[m], p[m], mu, a.hot_share)}
        rows.append(r)
        e = (p - y)[m].astype("float64")
        v = y[m].astype("float64")
        sse, sae, n, sy, syy = sse + (e ** 2).sum(), sae + np.abs(e).sum(), n + len(e), sy + v.sum(), syy + (v ** 2).sum()
        lci = d["lc"][m]
        ok = lci < len(codes)
        for j, w in enumerate((np.ones(ok.sum()), np.abs(e[ok]), e[ok])):
            lc_acc[:, j] += np.bincount(lci[ok], weights=w, minlength=len(codes))[:len(codes)]
        sx.append(v[::200])
        sp_.append((e + v)[::200])
        if not a.no_maps:
            gy, gp = stitch(d, y, p, m)
            draw_map(f.stem, gy, gp, out / f"map_{f.stem}.png")
        print(f"  {f.stem}: MAE {r['MAE']:.2f}  R2_within {r['R2_within']:.2f}  spearman {r['spearman']:.2f}  "
              f"hot_AUC {r['hot_AUC']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        d.close()

    t = pd.DataFrame(rows).set_index("city").sort_index()
    t.round(3).to_csv(out / "eval_per_city.csv")
    lc = pd.DataFrame({"landcover_code": codes, "pixels": lc_acc[:, 0], "MAE": lc_acc[:, 1] / np.maximum(lc_acc[:, 0], 1),
                       "bias": lc_acc[:, 2] / np.maximum(lc_acc[:, 0], 1)})
    lc = lc[lc["pixels"] > 0].round(3)
    lc.to_csv(out / "eval_by_landcover.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["MAE", "bias", "MAE_citymean", "skill", "R2_within", "R2_anomaly", "spearman", "hot_AUC", "hot_F1", "hot_bias"]
    print("\n" + t[cols].round(2).to_string())
    var = syy / n - (sy / n) ** 2
    print(f"\nPOOLED over all cities: MAE {sae / n:.2f} C | RMSE {np.sqrt(sse / n):.2f} C | R2 {1 - sse / n / var:.3f} "
          f"(includes hot-vs-cool differences between cities)")
    print(f"INSIDE a city (mean of cities): R2_within {t['R2_within'].mean():.2f} (median {t['R2_within'].median():.2f}) | "
          f"R2_anomaly {t['R2_anomaly'].mean():.2f} | spearman {t['spearman'].mean():.2f} | skill vs city-mean guess {t['skill'].mean():.2f}")
    print(f"HOTSPOTS (hottest {a.hot_share:.0%} of each city): AUC {t['hot_AUC'].mean():.2f} | F1 {t['hot_F1'].mean():.2f} | "
          f"IoU {t['hot_IoU'].mean():.2f} | mean bias on hot pixels {t['hot_bias'].mean():+.2f} C")
    print(f"cities where the model loses to the city-mean guess (R2_within < 0): "
          f"{', '.join(t.index[t['R2_within'] < 0]) or 'none'}")
    print(f"cities with a large offset (|bias| > 1 C): {', '.join(t.index[t['bias'].abs() > 1]) or 'none'}")
    print("\nerror by land cover (code: MAE / bias in C):\n" + lc.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hexbin(np.concatenate(sx), np.concatenate(sp_), gridsize=60, bins="log", cmap="viridis")
    lim = [np.concatenate(sx).min(), np.concatenate(sx).max()]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_xlabel("true LST (C)")
    ax[0].set_ylabel("predicted LST (C)")
    ax[0].set_title("all pixels, all cities")
    x = np.arange(len(t))
    ax[1].bar(x - 0.2, t["R2_within"], 0.4, label="R2 inside city")
    ax[1].bar(x + 0.2, t["spearman"], 0.4, label="spearman inside city")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(t.index, rotation=60, ha="right")
    ax[1].legend()
    ax[1].set_title("how well each city's hot and cool spots are ranked")
    mf = run / "metrics.csv"
    if mf.exists():
        mt = pd.read_csv(mf)
        ax[2].plot(mt["epoch"], mt["MAE"], "o-", label="validation MAE (C)")
        ax[2].plot(mt["epoch"], mt["train_loss"] * 20, "s-", label="training loss x 20")
        ax[2].set_xlabel("epoch")
        ax[2].legend()
        ax[2].set_title("training curve")
    fig.savefig(out / "eval_summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}/ (eval_per_city.csv, eval_by_landcover.csv, eval_summary.png" + ("" if a.no_maps else ", map_<City>.png") + ")")


if __name__ == "__main__":
    main()
