"""
What does the model rely on? Group permutation importance on the validation cities (read-only; nothing is retrained).

    python models/group_importance.py --run data/unet_run3
    python models/group_importance.py --run data/unet_run3 --per-city 300

For each feature group (vegetation, built-up, albedo, water, terrain, land cover) the group's values are shuffled between the patches
of the same city (each patch keeps its own spatial structure, but now carries another place's values for that group), the model
predicts again, and the drop in score against the unshuffled run is reported: mean over the cities of R2 inside the city
(offset removed), Spearman and hotspot AUC. A big drop means the model leans on that group.
Caveats: groups that carry similar information (vegetation / built-up / albedo are all derived from the same satellite bands) can
stand in for one another, so a small drop does not prove a group is useless, and a large drop shows reliance, not cause.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_unet import city_metrics, predict  # noqa: E402
from hotspot_report import GROUPS, group_of  # noqa: E402
from train_unet import UNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/unet_run3")
    ap.add_argument("--patches", default="pre_processing/patches")
    ap.add_argument("--per-city", type=int, default=150, help="validation patches used per city (random)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev, weights_only=False)
    model = UNet(len(ck["features"]), len(ck["landcover_codes"]), ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    nf = len(ck["features"])
    anomaly = ck.get("target", "absolute") == "anomaly"
    gch = {g: [i for i, n in enumerate(ck["features"]) if group_of(n) == g] for g in GROUPS}
    rng = np.random.default_rng(a.seed)
    files = sorted((Path(a.patches) / "val").glob("*.npz"))
    print(f"{a.run}: {nf} channels, {len(files)} validation cities, {a.per_city} patches each\n", flush=True)
    res = {g: [] for g in ["none"] + GROUPS}
    for f in files:
        d = np.load(f)
        n = len(d["X"])
        sel = np.sort(rng.choice(n, min(a.per_city, n), replace=False))
        X, L, y = d["X"][sel][:, :nf], d["lc"][sel], d["y"][sel]
        m = (d["mask"][sel] == 1) & (y > 0) & (y < 75)
        perm = rng.permutation(len(sel))
        for g in ["none"] + GROUPS:
            Xg, Lg = X, L
            if g != "none":
                Xg = X.copy()
                if gch[g]:
                    Xg[:, gch[g]] = X[perm][:, gch[g]]
                if g == "land cover":
                    Lg = L[perm]
            p = predict(model, Xg, Lg, dev, ck["mu"], ck["sd"])
            if anomaly:
                p = p + float(y[m].mean())
            r = city_metrics(y[m], p[m], ck["mu"], 0.10)
            res[g].append((r["R2_anomaly"], r["spearman"], r["hot_AUC"]))
        print(f"  {f.stem} done", flush=True)
    base = np.mean(res["none"], axis=0)
    rows = []
    for g in GROUPS:
        v = np.mean(res[g], axis=0)
        rows.append({"group": g, "channels": len(gch[g]) if g != "land cover" else 1, "R2_inside_city": v[0],
                     "R2_drop": base[0] - v[0], "spearman_drop": base[1] - v[1], "hot_AUC_drop": base[2] - v[2]})
    t = pd.DataFrame(rows).sort_values("R2_drop", ascending=False)
    print(f"\nunshuffled: R2 inside city {base[0]:.3f}, Spearman {base[1]:.3f}, hotspot AUC {base[2]:.3f}")
    print("\nscore lost when the group is shuffled between patches of the same city (bigger = the model relies on it more):")
    print(t.round(3).to_string(index=False))
    out = Path(a.run) / "group_importance.csv"
    t.round(4).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
