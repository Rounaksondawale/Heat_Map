"""
Terminal version of explore_data.ipynb: prints the summaries and saves every plot as a PNG.

    python notebooks/explore_data.py                                   # smoke-test table
    python notebooks/explore_data.py data/multi_city/all_cities_training.csv.gz
    python notebooks/explore_data.py <table> --city Delhi --out outputs/plots

Plots are written to --out (default outputs/plots); open them from the file explorer.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

EXCLUDE = ["LST", "LST_z", "SUHI", "HOTSPOT", "x", "y", "lon", "lat", "n_obs", "LST_std"]


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="data/smoke_test/all_cities_training.csv.gz")
    ap.add_argument("--city", default=None, help="city for the maps (default: first in the table)")
    ap.add_argument("--out", default="outputs/plots")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table)
    print(f"{args.table}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 1. overview
    print("\nPer city:")
    print(df.groupby("city").agg(rows=("LST", "size"), LST=("LST", "mean"),
                                 NDVI=("NDVI", "mean"), hotspot=("HOTSPOT", "mean")).round(2))
    miss = df.isna().mean()
    print("\nColumns with missing values:", miss[miss > 0].round(3).to_dict() or "none")

    # 2. LST distribution and hotspot share
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for c, g in df.groupby("city"):
        ax[0].hist(g["LST"], bins=60, alpha=.5, label=c, density=True)
    ax[0].set_xlabel("LST (°C)"); ax[0].set_title("LST distribution per city")
    if df["city"].nunique() <= 15:
        ax[0].legend(fontsize=7)
    df.groupby("city")["HOTSPOT"].mean().sort_values().plot.barh(ax=ax[1])
    ax[1].set_title("Hotspot share per city")
    save(fig, out, "1_lst_and_hotspots.png")

    # 3. maps for one city
    city = args.city or df["city"].iloc[0]
    g = df[df.city == city]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a, (col, cmap) in zip(ax, [("LST", "inferno"), ("NDVI", "YlGn"),
                                    ("GHSL_BUILT_FRAC", "Greys"), ("HOTSPOT", "Reds")]):
        s = a.scatter(g.lon, g.lat, c=g[col], s=3, cmap=cmap)
        a.set_title(f"{city}: {col}"); a.set_aspect("equal"); plt.colorbar(s, ax=a, shrink=.8)
    save(fig, out, f"2_maps_{city}.png")

    # 4. what relates to LST
    feat = [c for c in df.select_dtypes("number").columns if c not in EXCLUDE and df[c].nunique() > 1]
    corr = df[feat + ["LST"]].corr()["LST"].drop("LST").sort_values()
    print("\nCorrelation with LST:\n", corr.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, max(4, .25 * len(corr))))
    corr.plot.barh(ax=ax); ax.axvline(0, c="k", lw=.5); ax.set_title("Pearson r with LST")
    save(fig, out, "3_correlation_with_lst.png")

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    samp = df.sample(min(len(df), 20000), random_state=0)
    for a, col in zip(ax, ["NDVI", "NDBI", "ALBEDO", "GHSL_HEIGHT"]):
        a.hexbin(samp[col], samp["LST"], gridsize=40, cmap="viridis", mincnt=1)
        a.set_xlabel(col); a.set_ylabel("LST")
    save(fig, out, "4_feature_vs_lst.png")

    # 5. land cover
    print("\nBy land cover (10 tree, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare, 80 water):")
    print(df.groupby("LANDCOVER").agg(n=("LST", "size"), LST=("LST", "mean"),
                                      hotspot=("HOTSPOT", "mean")).round(2))

    # 6. feature correlation heatmap
    c = df[feat].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat))); ax.set_xticklabels(feat, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feat))); ax.set_yticklabels(feat, fontsize=7)
    plt.colorbar(im, shrink=.7)
    save(fig, out, "5_feature_correlation.png")

    # 7. baseline model (spatial-block split)
    X, y, grp = df[feat].fillna(df[feat].median()), df["LST"], df["block_id"]
    tr, te = next(GroupShuffleSplit(test_size=.25, random_state=0).split(X, y, grp))
    rf = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X.iloc[tr], y.iloc[tr])
    p = rf.predict(X.iloc[te])
    print(f"\nBaseline RF (held-out blocks): R2 {r2_score(y.iloc[te], p):.3f} | "
          f"MAE {mean_absolute_error(y.iloc[te], p):.2f} °C")
    fig, ax = plt.subplots(figsize=(7, 5))
    pd.Series(rf.feature_importances_, feat).sort_values().tail(15).plot.barh(ax=ax)
    ax.set_title("RF feature importance")
    save(fig, out, "6_rf_feature_importance.png")


if __name__ == "__main__":
    main()
