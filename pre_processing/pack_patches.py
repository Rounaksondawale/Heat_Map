"""
Pack the per-city patch files into a few big arrays that train_unet.py can read straight from disk (memory-mapped),
so training needs little RAM instead of ~10 GB. Run once after make_patches.py (or after copying the patches to a small PC).

    python pre_processing/pack_patches.py                      # reads pre_processing/patches/{train,val}/*.npz

Writes <patches>/packed/<split>_X.npy (all feature patches, float16) and <split>_rest.npz (lc, y, mask, city per patch).
train_unet.py uses <patches>/packed automatically when it exists. Needs about as much free disk as the patches, uncompressed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="pre_processing/patches")
    a = ap.parse_args()
    root = Path(a.patches)
    meta = json.load(open(root / "meta.json"))
    nf, size = len(meta["features"]), meta["size"]
    out = root / "packed"
    out.mkdir(exist_ok=True)
    for split in ("train", "val"):
        files = sorted((root / split).glob("*.npz"))
        if not files:
            raise SystemExit(f"no files in {root / split}")
        n = sum(len(np.load(f)["pos"]) for f in files)
        print(f"{split}: {len(files)} cities, {n:,} patches -> {n * nf * size * size * 2 / 1e9:.1f} GB", flush=True)
        X = np.lib.format.open_memmap(out / f"{split}_X.npy", mode="w+", dtype="float16", shape=(n, nf, size, size))
        lc = np.empty((n, size, size), "uint8")
        y = np.empty((n, size, size), "float32")
        mask = np.empty((n, size, size), "uint8")
        city = []
        i = 0
        for f in files:
            d = np.load(f)
            k = len(d["pos"])
            X[i:i + k], lc[i:i + k], y[i:i + k], mask[i:i + k] = d["X"], d["lc"], d["y"], d["mask"]
            city += [f.stem] * k
            i += k
            d.close()
        X.flush()
        del X
        np.savez(out / f"{split}_rest.npz", lc=lc, y=y, mask=mask, city=np.array(city))
        print(f"  wrote {split}_X.npy and {split}_rest.npz", flush=True)
    print(f"\npacked arrays in {out}; train_unet.py will use them automatically")


if __name__ == "__main__":
    main()
