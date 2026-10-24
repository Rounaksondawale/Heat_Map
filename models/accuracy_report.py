"""
Accuracy report for the presentation: one summary table + figures, built from the files the training scripts wrote.

    python models/accuracy_report.py                                        # per-city + global_ctx
    python models/accuracy_report.py --global-tag global_simple

Reads   models/artifacts/per_city_cv_metrics.csv           (train_per_city.py: held-out spatial blocks inside each city)
        models/artifacts/<global-tag>_cv_metrics.csv       (train_lgbm.py: leave-one-city-out, unseen cities)
        models/outputs/per_city_oof_predictions.csv.gz     (out-of-block predictions, for the scatter plot)
Writes  models/outputs/accuracy/summary.csv, fig1_hotspot_by_city.png, fig2_observed_vs_predicted.png,
        fig3_ranking_vs_baseline.png, and prints the headline sentences.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ART, OUT = HERE / "artifacts", HERE / "outputs"
C_CITY, C_GLOB, C_BASE = "#1b7837", "#2166ac", "#999999"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", default=str(ART / "per_city_cv_metrics.csv"))
    ap.add_argument("--global-tag", default="global_ctx")
    ap.add_argument("--oof", default=str(OUT / "per_city_oof_predictions.csv.gz"))
    a = ap.parse_args()
    out = OUT / "accuracy"
    out.mkdir(parents=True, exist_ok=True)

    pc = pd.read_csv(a.per_city, index_col="city")
    gl = pd.read_csv(ART / f"{a.global_tag}_cv_metrics.csv", index_col="city")
    cities = sorted(set(pc.index) & set(gl.index))

    # ---------------- summary table (means over cities)
    def m(df, col):
        return float(df.loc[cities, col].mean())
    rows = [
        ("Per-city model (held-out 3 km blocks, same city)", m(pc, "r2"), m(pc, "r2_rescaled"), m(pc, "spearman"), m(pc, "hot_auc"), m(pc, "hot_f1")),
        ("Global model (unseen city, leave-one-city-out)", m(gl, "r2"), m(gl, "r2_rescaled"), m(gl, "spearman"), m(gl, "hot_auc"), m(gl, "hot_f1")),
        ("Ridge baseline (unseen city)", m(gl, "ridge_r2"), np.nan, np.nan, m(gl, "ridge_hot_auc"), m(gl, "ridge_hot_f1")),
    ]
    summ = pd.DataFrame(rows, columns=["model", "R2", "R2_rescaled", "Spearman", "hotspot_AUC", "hotspot_F1"]).set_index("model").round(3)
    summ.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print(summ.to_string())

    # ---------------- fig 1: hotspot F1 + AUC by city
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(cities))
    for a_, col, ridge, title in ((ax[0], "hot_f1", "ridge_hot_f1", "Hotspot F1 (top 10 % of urban pixels)"),
                                  (ax[1], "hot_auc", "ridge_hot_auc", "Hotspot AUC (ranking quality, 0.5 = random)")):
        a_.bar(x - 0.27, pc.loc[cities, col], 0.27, color=C_CITY, label="per-city model")
        a_.bar(x, gl.loc[cities, col], 0.27, color=C_GLOB, label="global model (unseen city)")
        a_.bar(x + 0.27, gl.loc[cities, ridge], 0.27, color=C_BASE, label="Ridge baseline (unseen city)")
        a_.set_xticks(x, cities, rotation=60, ha="right")
        a_.set_title(title)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].axhline(0.1, color="k", lw=0.8, ls="--")
    ax[0].text(len(cities) - 0.5, 0.11, "random guess", ha="right", fontsize=8)
    ax[1].set_ylim(0.5, 1)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig1_hotspot_by_city.png", dpi=150)
    plt.close(fig)

    # ---------------- fig 2: observed vs predicted (per-city, out-of-block)
    if Path(a.oof).exists():
        o = pd.read_csv(a.oof, usecols=["city", "TARGET", "pred_lgbm"]).dropna()
        o = o.sample(min(len(o), 400_000), random_state=0)
        r2 = 1 - ((o.TARGET - o.pred_lgbm) ** 2).sum() / ((o.TARGET - o.TARGET.mean()) ** 2).sum()
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        hb = ax.hexbin(o.TARGET, o.pred_lgbm, gridsize=70, bins="log", cmap="viridis", mincnt=1)
        lim = np.nanpercentile(np.abs(o.TARGET), 99.5)
        ax.plot([-lim, lim], [-lim, lim], color="w", lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("observed temperature anomaly (C)")
        ax.set_ylabel("predicted (held-out blocks)")
        ax.set_title(f"Per-city models, all cities pooled: R2 = {r2:.2f}")
        plt.colorbar(hb, label="pixels (log)")
        fig.tight_layout()
        fig.savefig(out / "fig2_observed_vs_predicted.png", dpi=150)
        plt.close(fig)

    # ---------------- fig 3: ranking quality vs baseline on unseen cities
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Spearman", "hotspot AUC", "hotspot F1"]
    vals = [(m(gl, "spearman"), m(gl, "hot_auc"), m(gl, "hot_f1")), (np.nan, m(gl, "ridge_hot_auc"), m(gl, "ridge_hot_f1")),
            (m(pc, "spearman"), m(pc, "hot_auc"), m(pc, "hot_f1"))]
    for i, (name, col, v) in enumerate((("global model (unseen city)", C_GLOB, vals[0]), ("Ridge (unseen city)", C_BASE, vals[1]),
                                        ("per-city (same city, new blocks)", C_CITY, vals[2]))):
        ax.bar(np.arange(3) + (i - 1) * 0.27, v, 0.27, color=col, label=name)
        for j, t in enumerate(v):
            if not np.isnan(t):
                ax.text(j + (i - 1) * 0.27, t + 0.01, f"{t:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_ranking_vs_baseline.png", dpi=150)
    plt.close(fig)

    # ---------------- headline sentences
    best, worst = pc.loc[cities, "r2"].idxmax(), pc.loc[cities, "r2"].idxmin()
    print(f"""
HEADLINES (copy into the slides)
- Per city: models predict the heat pattern of unseen 3 km blocks with mean R2 {m(pc,'r2'):.2f} (range {pc.loc[cities,'r2'].min():.2f}-{pc.loc[cities,'r2'].max():.2f}; best {best}, weakest {worst}).
- Hotspots (top 10 % of urban pixels): per-city F1 {m(pc,'hot_f1'):.2f}, AUC {m(pc,'hot_auc'):.2f} (a random guess gives F1 about 0.10).
- One model for unseen cities: Spearman {m(gl,'spearman'):.2f}, hotspot AUC {m(gl,'hot_auc'):.2f}, F1 {m(gl,'hot_f1'):.2f}; it ranks pixels well but gives less reliable absolute values.
- Honest caveat: held-out blocks border training blocks, so the per-city scores are somewhat optimistic.
Figures -> {out}""")


if __name__ == "__main__":
    main()
