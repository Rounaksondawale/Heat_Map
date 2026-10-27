"""
Check the patches from make_patches.py before training (read-only).

    python pre_processing/check_patches.py                       # pre_processing/patches
    python pre_processing/check_patches.py --patches path/to/patches

Checks: disk size, patches per city (low counts), split is by city and has no overlap between train and val, array shapes and
dtypes, NaN/inf, usable share, temperature range, standardised features (mean ~0, std ~1 on usable training pixels, no huge
values), landcover indices, patch positions, and that no patch is a duplicate. Prints PASS / WARN / FAIL per check.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

res = []


def report(level, name, msg):
    res.append(level)
    print(f"[{level}] {name}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="pre_processing/patches")
    a = ap.parse_args()
    root = Path(a.patches)
    meta = json.load(open(root / "meta.json"))
    size, nf = meta["size"], len(meta["features"])
    files = {s: {p.stem: p for p in sorted((root / s).glob("*.npz"))} for s in ("train", "val")}
    gb = sum(p.stat().st_size for s in files.values() for p in s.values()) / 1e9
    print(f"{len(files['train'])} train cities, {len(files['val'])} val cities, {nf} numeric features, {size}x{size} patches, {gb:.1f} GB on disk\n")

    report("PASS" if not set(files["train"]) & set(files["val"]) else "FAIL", "split by city",
           "no city in both train and val" if not set(files["train"]) & set(files["val"]) else str(set(files["train"]) & set(files["val"])))
    missing = [c for c in meta["train_cities"] if c not in files["train"]] + [c for c in meta["val_cities"] if c not in files["val"]]
    report("PASS" if not missing else "WARN", "cities with no patches", ", ".join(missing) if missing else "none (every city has patches)")

    rows, xs, bad, warn = [], [], [], []
    hist = np.zeros(256, "int64")
    for split, fs in files.items():
        seen = set()
        for city, p in fs.items():
            d = np.load(p)
            X, lc, y, m, pos = d["X"], d["lc"], d["y"], d["mask"], d["pos"]
            n = len(X)
            ok = m == 1
            r = {"city": city, "split": split, "patches": n, "usable_%": round(float(ok.mean() * 100), 1),
                 "LST_min": float(np.nanmin(y[ok])) if ok.any() else np.nan, "LST_max": float(np.nanmax(y[ok])) if ok.any() else np.nan,
                 "LST_mean": float(np.nanmean(y[ok])) if ok.any() else np.nan}
            if X.shape[1:] != (nf, size, size) or lc.shape != (n, size, size) or y.shape != (n, size, size) or m.shape != (n, size, size):
                bad.append(f"{city}: shapes X{X.shape} lc{lc.shape} y{y.shape} mask{m.shape}")
            if not np.isfinite(X).all():
                bad.append(f"{city}: NaN/inf in X")
            if (y[ok] <= 0).any() or (y[ok] >= 75).any():
                warn.append(f"{city}: {int(((y <= 0) | (y >= 75))[ok].sum())} LST pixels outside (0, 75)")
            xm = float(np.abs(X.astype("float32")).max())
            if xm > 60:
                warn.append(f"{city}: |X| max {xm:.0f}")
            if (lc[ok] == 255).any():
                bad.append(f"{city}: usable pixels with unknown landcover")
            if len({tuple(q) for q in pos}) != n:
                bad.append(f"{city}: duplicate patch positions")
            hist += np.bincount(lc.ravel(), minlength=256)
            if split == "train":
                xs.append(np.moveaxis(X[:, :, ::4, ::4], 1, -1)[ok[:, ::4, ::4]][::5].astype("float32"))   # subsample usable pixels
            rows.append(r)
            d.close()
    t = pd.DataFrame(rows)
    print()
    report("PASS" if not warn else "WARN", "outliers (LST outside 0-75 C, standardised feature > 60 sd)",
           "none" if not warn else f"{len(warn)} cities, e.g. " + "; ".join(warn[:4]) + " | train_unet.py masks bad LST and clips features to +-10 sd")
    report("PASS" if not bad else "FAIL", "array sanity (shape, NaN, landcover, positions)",
           "all files pass" if not bad else "; ".join(bad[:8]))

    pc = t.groupby("split")["patches"].describe()[["min", "50%", "max"]]
    low = t[t["patches"] < 0.5 * t.groupby("split")["patches"].transform("median")]
    report("PASS" if low.empty else "WARN", "patches per city",
           f"train min/median/max {int(pc.loc['train','min'])}/{int(pc.loc['train','50%'])}/{int(pc.loc['train','max'])}, "
           f"val {int(pc.loc['val','min'])}/{int(pc.loc['val','50%'])}/{int(pc.loc['val','max'])}" +
           ("" if low.empty else " | low: " + ", ".join(f"{r.city} {r.patches}" for r in low.itertuples())))
    lowu = t[t["usable_%"] < 85]
    report("PASS" if lowu.empty else "WARN", "usable share in kept patches",
           f"mean {t['usable_%'].mean():.1f}%" + ("" if lowu.empty else " | below 85%: " + ", ".join(f"{c} {u}%" for c, u in zip(lowu["city"], lowu["usable_%"]))))

    S = np.concatenate(xs)
    mu, sd = S.mean(0), S.std(0)
    off = [(n, round(float(a), 2), round(float(b), 2)) for n, a, b in zip(meta["features"], mu, sd) if abs(a) > 0.3 or not 0.5 < b < 2]
    report("PASS" if not off else "WARN", "standardised training features (mean ~0, std ~1)",
           "all in range" if not off else f"{len(off)} off: " + ", ".join(f"{n} mean {a} std {b}" for n, a, b in off[:6]))
    const = [n for n, b in zip(meta["features"], sd) if b < 1e-3]
    report("PASS" if not const else "WARN", "constant features", "none" if not const else ", ".join(const))

    names = {i: c for i, c in enumerate(meta["landcover_codes"])}
    share = {names[i]: round(hist[i] / hist[:255].sum() * 100, 1) for i in names if hist[i]}
    print(f"\nlandcover share of all pixels (code: %): {share}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print("\n" + t.sort_values(["split", "patches"]).round(1).to_string(index=False))
    print(f"\n{res.count('PASS')} PASS, {res.count('WARN')} WARN, {res.count('FAIL')} FAIL")


if __name__ == "__main__":
    main()
