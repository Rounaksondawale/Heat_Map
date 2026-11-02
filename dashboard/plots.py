"""Charts for the dashboard (matplotlib, same palette as the README figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SURF, INK, INK2, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": BASE})


def _style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def drivers_fig(phi):
    """Shapley contribution of each feature group (C above a neighbour-like version of the area)."""
    items = sorted(phi.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    ax.barh(y, vals, 0.62, color=[RED if v > 0 else BLUE for v in vals], zorder=3)
    ax.axvline(0, color=BASE, lw=1.2, zorder=4)
    span = max(max(abs(v) for v in vals), 0.3)
    for yy, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03) * span / 1.0, yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([k.capitalize() for k, _ in items], fontsize=10)
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.set_xlabel("°C above a neighbour-like version of this area   (red = hotter, blue = cooler)", fontsize=9)
    _style(ax, "x")
    fig.tight_layout()
    return fig


def dose_fig(table, need, roof_ok=True):
    """Gap to the neighbours against the share of surface converted, for planting, cool roofs and both."""
    doses = table["dose_%"].values
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    for key, name, c in (("veg", "Plant", BLUE), ("roof", "Cool roofs", ORANGE), ("both", "Both", AQUA)):
        if key == "roof" and not roof_ok:
            continue
        ax.plot(doses, table[f"{key}_gap_left"].values, color=c, lw=2.2, zorder=3, label=name)
        d = need.get(key)
        if d is not None and d > 0:
            ax.plot([d * 100], [0], "o", ms=8, mfc=c, mec=SURF, mew=2, zorder=5)
            ax.annotate(f"{d * 100:.0f} %", (d * 100, 0), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the changeable surface converted (%)", fontsize=9)
    ax.set_ylabel("Area minus neighbours (°C)", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="best")
    _style(ax)
    fig.tight_layout()
    return fig


def area_maps_fig(res, state):
    """Observed and predicted temperature around the area (cyan = the area, dashed white = its neighbour ring)."""
    S = res["S"]
    sl, reg, ring = S["sl"], S["reg"], S["ring"]
    obs = np.where(S["okl"], S["lst_c"], np.nan)
    pred = np.where(S["sim"].ok, state["pred"][sl], np.nan)
    lo, hi = np.nanpercentile(obs, [2, 98]) if np.isfinite(obs).any() else (np.nanmin(pred), np.nanmax(pred))
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.3))
    r0, r1, c0, c1, R0, C0 = S["r0"], S["r1"], S["c0"], S["c1"], S["R0"], S["C0"]
    b0, b1 = S["bs"]
    for ax, g, ttl in ((axs[0], obs, "Satellite: observed (°C)"), (axs[1], pred, "Model: predicted (°C)")):
        im = ax.imshow(g, cmap="inferno", vmin=lo, vmax=hi)
        ax.add_patch(Rectangle((c0 - C0 - .5, r0 - R0 - .5), c1 - c0, r1 - r0, fill=False, ec="cyan", lw=2))
        ax.add_patch(Rectangle((b1.start - .5, b0.start - .5), b1.stop - b1.start, b0.stop - b0.start, fill=False, ec="white", lw=1.2, ls="--"))
        ax.set_title(ttl, fontsize=10.5, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout()
    return fig
