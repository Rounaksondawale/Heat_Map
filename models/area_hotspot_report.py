"""
Score hotspot AREAS (connected patches), not single pixels, from the out-of-fold predictions a training run saved.
No retraining: it reads models/outputs/<tag>_oof_predictions.csv.gz.

    python models/area_hotspot_report.py --tag combined_v2
    python models/area_hotspot_report.py --tag combined_v2 --min-ha 10 --maps Pune Delhi Jaipur

Observed hotspot areas (from the satellite LST) and predicted hotspot areas (from the model, for a city it never saw)
are built with the same definition (see hotspots.py): smooth, hottest 10 % of urban pixels, connected patches of at
least --min-ha hectares. Reported per city:
    area_f1 / area_iou   pixel overlap between predicted and observed hotspot areas
    cluster_recall       share of observed hotspot patches the model finds
    cluster_precision    share of predicted patches that are real
Writes models/outputs/<tag>_area_hotspots.csv (per-city table) and, with --maps, one PNG per named city.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hotspots import area_scores, hotspot_areas  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def draw(g, truth_col, pred_col, city, path, **kw):
    t = hotspot_areas(g, truth_col, **kw) > 0
    p = hotspot_areas(g, pred_col, **kw) > 0
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for a_, title in ((ax[0], "observed hotspot areas (satellite)"), (ax[1], "predicted hotspot areas (unseen city)")):
        a_.scatter(g["x"], g["y"], c="#e8e8e8", s=1.2, marker="s", linewidths=0)
        a_.set_title(f"{city}: {title}")
        a_.set_aspect("equal")
        a_.set_xticks([])
        a_.set_yticks([])
    ax[0].scatter(g.loc[t, "x"], g.loc[t, "y"], c="#d7191c", s=3, marker="s", linewidths=0)
    both, only_p = p & t, p & ~t
    ax[1].scatter(g.loc[both, "x"], g.loc[both, "y"], c="#d7191c", s=3, marker="s", linewidths=0, label="also observed")
    ax[1].scatter(g.loc[only_p, "x"], g.loc[only_p, "y"], c="#fdae61", s=3, marker="s", linewidths=0, label="not observed")
    ax[1].scatter(g.loc[t & ~p, "x"], g.loc[t & ~p, "y"], c="#2c7bb6", s=3, marker="s", linewidths=0, label="missed")
    ax[1].legend(loc="lower left", markerscale=4, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="combined_v2")
    ap.add_argument("--pred-col", default="pred_lgbm")
    ap.add_argument("--min-ha", type=float, default=5.0, help="smallest patch that counts as a hotspot area")
    ap.add_argument("--win", type=int, default=3, help="smoothing window in table pixels")
    ap.add_argument("--overlap", type=float, default=0.3, help="share of a patch that must be covered to count as found")
    ap.add_argument("--maps", nargs="*", default=[], help="cities to draw")
    a = ap.parse_args()

    f = OUT / f"{a.tag}_oof_predictions.csv.gz"
    df = pd.read_csv(f)
    print(f"{len(df):,} predictions, {df['city'].nunique()} cities | patches >= {a.min_ha} ha, smoothing {a.win}x{a.win}")
    rows = []
    for c, g in df.groupby("city"):
        r = area_scores(g, a.pred_col, "LST", win=a.win, min_ha=a.min_ha, overlap=a.overlap)
        r["city"] = c
        rows.append(r)
        if c in a.maps:
            draw(g, "LST", a.pred_col, c, OUT / f"{a.tag}_areas_{c}.png", win=a.win, min_ha=a.min_ha)
    t = pd.DataFrame(rows).set_index("city").round(3)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(t.to_string())
    t.to_csv(OUT / f"{a.tag}_area_hotspots.csv")
    m = t[["area_f1", "area_iou", "cluster_recall", "cluster_precision"]].mean()
    print("\nMEAN over cities: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"median patches per city: observed {t['true_areas'].median():.0f}, predicted {t['pred_areas'].median():.0f}")


if __name__ == "__main__":
    main()
