"""
Rank every trained U-Net found on this machine, all scored on the SAME validation pixels (read-only; nothing is retrained).

    python models/compare_runs.py                                   # every data/*/best.pt, scored on pre_processing/patches
    python models/compare_runs.py --runs data/unet_run data/unet_run2 data/unet_run3

Each run's best.pt may use a different number of input channels (older runs: the first 45 of the current patch channels); the
current validation patches are sliced to fit, so every model is judged on identical pixels. Per model, the mean over the
validation cities of: MAE, R2 inside the city (with and without the city's constant offset), Spearman rank correlation,
hotspot AUC / F1 (hottest --hot-share of each city) and the bias on hot pixels. Anomaly models get each city's true mean
added back before scoring, as in evaluate_unet.py. The best model is the one with the highest mean --rank-by (default
R2_anomaly: the pattern inside a city, fair to absolute and anomaly models alike).
Writes <data>/model_comparison.csv (one row per model) and <data>/model_comparison_per_city.csv.
NOTE: every run's best epoch was chosen on these same validation cities, so all scores are slightly optimistic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_unet import city_metrics, predict  # noqa: E402
from train_unet import UNet  # noqa: E402

COLS = ["MAE", "R2_within", "R2_anomaly", "spearman", "hot_AUC", "hot_F1", "hot_bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder searched for */best.pt")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--patches", default="pre_processing/patches")
    ap.add_argument("--hot-share", type=float, default=0.10)
    ap.add_argument("--rank-by", default="R2_anomaly", choices=COLS)
    a = ap.parse_args()
    data, root = Path(a.data), Path(a.patches)
    runs = [Path(r) for r in a.runs] if a.runs else sorted(p.parent for p in data.glob("*/best.pt"))
    runs = [r for r in runs if (r / "best.pt").exists()]
    if not runs:
        raise SystemExit(f"no best.pt found under {data}/*/")
    meta = json.load(open(root / "meta.json"))
    names = meta["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted((root / "val").glob("*.npz"))
    print(f"{len(runs)} runs | {len(files)} validation cities from {root} ({len(names)} channels) | device {dev}\n", flush=True)
    val = []
    for f in files:
        d = np.load(f)
        val.append((f.stem, d["X"], d["lc"], d["y"], (d["mask"] == 1) & (d["y"] > 0) & (d["y"] < 75)))

    summary, per_city = [], []
    for run in runs:
        ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)   # our own files
        nf = len(ck["features"])
        if ck["features"] != names[:nf]:
            print(f"  skip {run}: its input features are not the first {nf} channels of these patches\n")
            continue
        model = UNet(nf, len(ck["landcover_codes"]), ck["base"]).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        anomaly = ck.get("target", "absolute") == "anomaly"
        rows = []
        for city, X, L, y, m in val:
            p = predict(model, X[:, :nf], L, dev, ck["mu"], ck["sd"])
            if anomaly:
                p = p + float(y[m].mean())
            rows.append({"run": str(run), "city": city, **city_metrics(y[m], p[m], ck["mu"], a.hot_share)})
        t = pd.DataFrame(rows)
        per_city.append(t)
        s = {"run": str(run), "target": "anomaly" if anomaly else "absolute", "channels": nf, "base": ck["base"],
             **{c: t[c].mean() for c in COLS}, "cities_R2_within<0": int((t["R2_within"] < 0).sum())}
        mf = run / "metrics.csv"
        if mf.exists():
            s["epochs_run"] = len(pd.read_csv(mf))
        summary.append(s)
        print(f"  {run}: {s['target']}, {nf} channels | R2_anomaly {s['R2_anomaly']:.3f}  spearman {s['spearman']:.3f}  "
              f"hot_AUC {s['hot_AUC']:.3f}  MAE {s['MAE']:.2f}", flush=True)
    if not summary:
        raise SystemExit("no run could be scored")

    t = pd.DataFrame(summary).sort_values(a.rank_by, ascending=(a.rank_by in ("MAE",))).reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + t.round(3).to_string(index=False))
    best = t.iloc[0]
    print(f"\nBEST by mean {a.rank_by}: {best['run']}/best.pt  ({best['target']} target, {best['channels']} channels)")
    for c, hi in (("hot_AUC", True), ("hot_F1", True), ("spearman", True), ("MAE", False)):
        w = t.sort_values(c, ascending=not hi).iloc[0]
        print(f"  best {c:10s}: {w['run']}  ({w[c]:.3f})")
    if len(t) > 1:
        gap = t[a.rank_by].iloc[0] - t[a.rank_by].iloc[1]
        print(f"  lead over the runner-up: {gap:+.3f} {a.rank_by}" + ("  (small: within the epoch-to-epoch noise, treat as a tie)" if abs(gap) < 0.03 else ""))
    t.round(4).to_csv(data / "model_comparison.csv", index=False)
    pd.concat(per_city).round(4).to_csv(data / "model_comparison_per_city.csv", index=False)
    print(f"\nwrote {data / 'model_comparison.csv'} and {data / 'model_comparison_per_city.csv'}")


if __name__ == "__main__":
    main()
