"""
Turn per-pixel predictions into hotspot *areas*: smooth the predicted heat over neighbouring pixels, take the hottest
10 % of urban pixels, and keep only connected clusters (isolated single pixels are not hotspots).

    from hotspots import hotspot_flag
    flag = hotspot_flag(city_df, "pred_lgbm")        # boolean Series aligned with city_df

city_df needs x, y (grid coordinates, one city), IS_URBAN and the prediction column. Pixels missing from the table
(water, cloud gaps, nodata) are treated as holes and ignored when smoothing.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter


def _grid(g):
    xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
    step = min(np.diff(np.unique(xs)).min(), np.diff(np.unique(ys)).min())
    ix = np.round((xs - xs.min()) / step).astype(int)
    iy = np.round((ys - ys.min()) / step).astype(int)
    return ix, iy, (iy.max() + 1, ix.max() + 1)


def smooth(g, col, win=3):
    """Mean of `col` over a win x win neighbourhood on the table grid, ignoring missing pixels."""
    ix, iy, shape = _grid(g)
    a = np.zeros(shape)
    m = np.zeros(shape)
    a[iy, ix] = g[col].to_numpy()
    m[iy, ix] = 1.0
    num, den = uniform_filter(a, win, mode="constant"), uniform_filter(m, win, mode="constant")
    return pd.Series(num[iy, ix] / np.maximum(den[iy, ix], 1e-9), index=g.index)


def hotspot_flag(g, pred_col, q=0.9, win=3, min_size=4):
    ix, iy, shape = _grid(g)
    s = smooth(g, pred_col, win)
    urban = g["IS_URBAN"].to_numpy() == 1
    flag = (s >= s[urban].quantile(q)).to_numpy() & urban
    grid = np.zeros(shape, bool)
    grid[iy[flag], ix[flag]] = True
    lab, n = label(grid, structure=np.ones((3, 3)))          # 8-connected clusters
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return pd.Series(keep[lab[iy, ix]] & flag, index=g.index)
