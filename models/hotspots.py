"""
Hotspots as AREAS, not pixels.

A hotspot is a connected patch of the city that is much hotter than the rest of the urban area. One warm pixel is
noise (a metal roof, a bare lot); a patch of several hectares is something a planner can act on.

Definition used everywhere (observed and predicted alike):
    1. smooth the heat value over a small neighbourhood (win x win pixels on the table grid, default 3 x 3 = 270 m)
    2. keep the hottest 10 % of URBAN pixels
    3. group touching pixels (8-neighbourhood) into clusters and keep clusters of at least `min_ha` hectares

    from hotspots import hotspot_areas, cluster_table, area_scores
    ids = hotspot_areas(city_df, "pred_lgbm")            # Series: 0 = not a hotspot, 1..n = cluster id
    table = cluster_table(city_df, ids, "pred_lgbm")     # one row per hotspot: area, mean heat, centre
    area_scores(city_df, "pred_lgbm", "LST")             # how well predicted areas match observed areas

city_df needs x, y (UTM metres, one city), lon, lat, IS_URBAN and the heat column. Pixels missing from the table
(water, cloud gaps, nodata) are holes: ignored when smoothing and never part of a cluster.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter


def _grid(g):
    xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
    step = min(np.diff(np.unique(xs)).min(), np.diff(np.unique(ys)).min())
    ix = np.round((xs - xs.min()) / step).astype(int)
    iy = np.round((ys - ys.min()) / step).astype(int)
    return ix, iy, (iy.max() + 1, ix.max() + 1), float(step)


def smooth(g, col, win=3):
    """Mean of `col` over a win x win neighbourhood on the table grid, ignoring missing pixels."""
    ix, iy, shape, _ = _grid(g)
    a = np.zeros(shape)
    m = np.zeros(shape)
    a[iy, ix] = g[col].to_numpy()
    m[iy, ix] = 1.0
    num, den = uniform_filter(a, win, mode="constant"), uniform_filter(m, win, mode="constant")
    return pd.Series(num[iy, ix] / np.maximum(den[iy, ix], 1e-9), index=g.index)


def hotspot_areas(g, col, q=0.9, win=3, min_ha=5.0):
    """Cluster id per pixel (0 = no hotspot). Clusters smaller than `min_ha` hectares are dropped."""
    ix, iy, shape, step = _grid(g)
    s = smooth(g, col, win)
    urban = g["IS_URBAN"].to_numpy() == 1
    flag = (s >= s[urban].quantile(q)).to_numpy() & urban
    grid = np.zeros(shape, bool)
    grid[iy[flag], ix[flag]] = True
    lab, n = label(grid, structure=np.ones((3, 3)))          # 8-connected clusters
    min_px = max(1, int(np.ceil(min_ha * 1e4 / step ** 2)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    new_id = np.zeros(n + 1, int)
    new_id[keep] = np.arange(1, keep.sum() + 1)               # renumber 1..k
    return pd.Series(new_id[lab[iy, ix]] * flag, index=g.index)


def hotspot_flag(g, col, q=0.9, win=3, min_size=4):
    """Boolean version (kept for older callers): True where a pixel belongs to a hotspot area."""
    step = _grid(g)[3]
    return hotspot_areas(g, col, q, win, min_ha=min_size * step ** 2 / 1e4) > 0


def cluster_table(g, ids, col):
    """One row per hotspot: id, pixels, hectares, mean / max of `col`, centre (lon, lat)."""
    step = _grid(g)[3]
    d = g.assign(_id=ids.to_numpy())
    d = d[d["_id"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["cluster_id", "pixels", "area_ha", "mean_heat", "max_heat", "lon", "lat"])
    t = d.groupby("_id").agg(pixels=("x", "size"), mean_heat=(col, "mean"), max_heat=(col, "max"),
                             lon=("lon", "mean"), lat=("lat", "mean"))
    t["area_ha"] = t["pixels"] * step ** 2 / 1e4
    t = t.reset_index().rename(columns={"_id": "cluster_id"})
    return t.sort_values("mean_heat", ascending=False)[["cluster_id", "pixels", "area_ha", "mean_heat", "max_heat", "lon", "lat"]]


def _hit_share(ids_a, mask_b):
    """For each cluster in ids_a, the share of its pixels that fall inside mask_b."""
    a = np.asarray(ids_a)
    k = a.max()
    if k == 0:
        return np.array([])
    inside = np.bincount(a[mask_b & (a > 0)], minlength=k + 1)[1:]
    total = np.bincount(a[a > 0], minlength=k + 1)[1:]
    return inside / np.maximum(total, 1)


def area_scores(g, pred_col, truth_col="LST", q=0.9, win=3, min_ha=5.0, overlap=0.3):
    """Compare predicted hotspot areas with observed ones (both built with the same definition).
    area_f1 / area_iou   pixel overlap of the two area masks
    cluster_recall       share of observed hotspots that the prediction finds (at least `overlap` of the patch covered)
    cluster_precision    share of predicted hotspots that are real (same test)"""
    true_ids = hotspot_areas(g, truth_col, q, win, min_ha).to_numpy()
    pred_ids = hotspot_areas(g, pred_col, q, win, min_ha).to_numpy()
    t, p = true_ids > 0, pred_ids > 0
    inter = int((t & p).sum())
    prec, rec = inter / max(p.sum(), 1), inter / max(t.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = inter / max(int((t | p).sum()), 1)
    found = _hit_share(true_ids, p) >= overlap
    real = _hit_share(pred_ids, t) >= overlap
    return {"true_areas": int(true_ids.max()), "pred_areas": int(pred_ids.max()),
            "area_f1": f1, "area_iou": iou,
            "cluster_recall": float(found.mean()) if len(found) else np.nan,
            "cluster_precision": float(real.mean()) if len(real) else np.nan}
