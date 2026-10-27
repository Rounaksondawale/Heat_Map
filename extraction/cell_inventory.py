"""
How much data do we have per MAP GRID CELL (not per 30 m pixel)?

    python extraction/cell_inventory.py --root data/multi_city                    # 1 km cells
    python extraction/cell_inventory.py --root data/multi_city --cell-km 2 --city Delhi

Uses the cleaned training table (every training row is one 90 m sample, 0.81 ha). For each cell of --cell-km:
    samples      number of training rows in the cell
    coverage_%   share of the cell covered by samples (100 % = fully covered)
    mean_obs     average number of satellite passes behind each pixel
    mean_LST     average land surface temperature
Writes <root>/cell_inventory_<km>km.csv (all cities), a per-city summary, an India-wide map of data coverage
(<root>/india_data_coverage.png, 0.1 degree cells) and, with --city, a cell map of that city.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIXEL_HA = 0.81            # one table row = one 90 m x 90 m sample


def load(root):
    for name in ("all_cities_clean_ctx.parquet", "all_cities_clean.parquet"):
        f = root / name
        if f.exists():
            cols = ["city", "x", "y", "lon", "lat", "LST", "n_obs"]
            return pd.read_parquet(f, columns=cols), name
    f = root / "all_cities_training.csv.gz"
    return pd.read_csv(f, usecols=["city", "x", "y", "lon", "lat", "LST", "n_obs"]), f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multi_city")
    ap.add_argument("--cell-km", type=float, default=1.0)
    ap.add_argument("--city", default=None, help="also draw the cell map of this city")
    a = ap.parse_args()
    root = Path(a.root)
    df, src = load(root)
    size = a.cell_km * 1000
    print(f"{len(df):,} rows from {src} | {df['city'].nunique()} cities | cells of {a.cell_km:g} km x {a.cell_km:g} km")

    df["cx"] = np.floor(df["x"] / size).astype(int)
    df["cy"] = np.floor(df["y"] / size).astype(int)
    cell_ha = a.cell_km ** 2 * 100
    cells = (df.groupby(["city", "cx", "cy"])
               .agg(samples=("x", "size"), mean_obs=("n_obs", "mean"), mean_LST=("LST", "mean"),
                    lon=("lon", "mean"), lat=("lat", "mean")).reset_index())
    cells["coverage_%"] = (cells["samples"] * PIXEL_HA / cell_ha * 100).clip(upper=100).round(1)
    cells = cells.round({"mean_obs": 1, "mean_LST": 2, "lon": 4, "lat": 4})
    out_csv = root / f"cell_inventory_{a.cell_km:g}km.csv"
    cells.to_csv(out_csv, index=False)

    per = cells.groupby("city").agg(cells=("samples", "size"), samples=("samples", "sum"),
                                    full_cells=("coverage_%", lambda s: int((s >= 90).sum())),
                                    median_samples=("samples", "median"), min_samples=("samples", "min"),
                                    mean_coverage=("coverage_%", "mean")).round(1)
    per["thin_cells_%"] = cells.groupby("city")["coverage_%"].apply(lambda s: round(float((s < 25).mean() * 100), 1))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(per.to_string())
    print(f"\nTOTAL: {len(cells):,} cells of {a.cell_km:g} km with data | {int(cells['samples'].sum()):,} samples | "
          f"median {cells['samples'].median():.0f} samples per cell ({cells['samples'].median() * PIXEL_HA / cell_ha:.0%} of a cell) | "
          f"{(cells['coverage_%'] >= 90).mean():.0%} of cells are at least 90% covered, {(cells['coverage_%'] < 25).mean():.0%} under 25%")

    # ---- India-wide coverage map: 0.1 degree cells (about 11 km)
    g = df.assign(glon=(df["lon"] // 0.1) * 0.1, glat=(df["lat"] // 0.1) * 0.1).groupby(["glon", "glat"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(8.5, 9))
    sc = ax.scatter(g["glon"] + 0.05, g["glat"] + 0.05, c=g["n"], s=22, marker="s", cmap="viridis", linewidths=0)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="training samples per 0.1 degree cell (about 11 km)")
    for c, d in df.groupby("city"):
        ax.annotate(c, (d["lon"].mean(), d["lat"].mean()), fontsize=6, ha="center", va="bottom", color="k")
    ax.set_xlim(67, 98)
    ax.set_ylim(6, 37)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Where we have data: {df['city'].nunique()} city boxes, {len(df):,} samples")
    ax.grid(alpha=0.3)
    fig.savefig(root / "india_data_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if a.city:
        c = cells[cells["city"] == a.city]
        if c.empty:
            print(f"no data for {a.city}")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            sc = ax.scatter(c["cx"], c["cy"], c=c["samples"], s=max(10, 4000 / len(c) ** 0.5), marker="s", cmap="viridis")
            plt.colorbar(sc, ax=ax, shrink=0.8, label=f"samples per {a.cell_km:g} km cell")
            ax.set_aspect("equal")
            ax.set_title(f"{a.city}: samples per {a.cell_km:g} km cell (max {int(cell_ha / PIXEL_HA)} = full cell)")
            fig.savefig(root / f"cell_inventory_{a.city}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"city map -> {root / f'cell_inventory_{a.city}.png'}")
    print(f"\nwrote {out_csv}, {root / 'india_data_coverage.png'}")


if __name__ == "__main__":
    main()
