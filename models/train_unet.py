"""
Train a U-Net: 64x64 feature patches in -> 64x64 land surface temperature out.

    python models/train_unet.py                                 # patches in pre_processing/patches, 30 epochs
    python models/train_unet.py --epochs 2 --max-patches 2000   # quick test

Input per patch: the 45 standardised numeric features (clipped to +-10) + LANDCOVER (learned 4-channel embedding). Target: LST in degrees C
(standardised with the training mean/std inside the loss). The loss is computed on mask == 1 pixels only.
Validation uses the held-out cities (no overlap between patches). Each epoch prints train loss, validation MAE (deg C),
RMSE, R2 and the time; the best model by validation MAE is saved to <out>/best.pt, with metrics.csv and val_per_city.csv.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_split(root, split, max_patches=0, seed=0):
    X, L, Y, M, C = [], [], [], [], []
    for p in sorted((root / split).glob("*.npz")):
        d = np.load(p)
        X.append(d["X"]); L.append(d["lc"]); Y.append(d["y"]); M.append(d["mask"])
        C += [p.stem] * len(d["X"])
    X, L, Y, M, C = np.concatenate(X), np.concatenate(L), np.concatenate(Y), np.concatenate(M), np.array(C)
    M = M * ((Y > 0) & (Y < 75)).astype("uint8")                    # drop implausible LST pixels from the loss (e.g. Kochi > 75 C)
    if max_patches and len(X) > max_patches:
        i = np.sort(np.random.default_rng(seed).choice(len(X), max_patches, replace=False))
        X, L, Y, M, C = X[i], L[i], Y[i], M[i], C[i]
    return X, L, Y, M, C


CLIP = 10.0        # standardised features beyond +-10 sd (rare extreme pixels) are clipped


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, n_num, n_lc, base=32, emb=4):
        super().__init__()
        self.emb = nn.Embedding(n_lc + 1, emb)                     # last index = unknown (255)
        self.n_lc = n_lc
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([block(n_num + emb, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(c[2] * 2, c[2]), block(c[1] * 2, c[1]), block(c[0] * 2, c[0])])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x, lc):
        lc = torch.where(lc == 255, torch.full_like(lc, self.n_lc), lc).long()
        x = torch.cat([x, self.emb(lc).permute(0, 3, 1, 2)], 1)
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.out(x)[:, 0]


def augment(x, lc, y, m):
    k = int(torch.randint(0, 4, (1,)))
    if k:
        x, lc, y, m = (torch.rot90(t, k, (-2, -1)) for t in (x, lc, y, m))
    if torch.rand(1) < 0.5:
        x, lc, y, m = (torch.flip(t, (-1,)) for t in (x, lc, y, m))
    return x, lc, y, m


@torch.no_grad()
def evaluate(model, data, dev, mu, sd, bs=256):
    X, L, Y, M, C = data
    model.eval()
    per = {}
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev).float().clamp(-CLIP, CLIP)
        lc = torch.from_numpy(L[i:i + bs]).to(dev)
        y = torch.from_numpy(Y[i:i + bs]).to(dev)
        m = torch.from_numpy(M[i:i + bs]).to(dev).bool()
        p = model(x, lc) * sd + mu
        for c in np.unique(C[i:i + bs]):
            k = torch.from_numpy(C[i:i + bs] == c).to(dev)
            e = (p[k] - y[k])[m[k]].double()
            v = y[k][m[k]].double()
            s = per.setdefault(c, np.zeros(5))
            s += [float((e ** 2).sum()), float(e.abs().sum()), float(len(e)), float(v.sum()), float((v ** 2).sum())]
    tot = sum(per.values())
    mse, mae, n, sy, syy = tot[0] / tot[2], tot[1] / tot[2], tot[2], tot[3], tot[4]
    var = syy / n - (sy / n) ** 2
    rows = []
    for c, s in per.items():
        v = s[4] / s[2] - (s[3] / s[2]) ** 2
        rows.append({"city": c, "MAE": s[1] / s[2], "RMSE": np.sqrt(s[0] / s[2]), "R2": 1 - (s[0] / s[2]) / v if v > 0 else np.nan})
    return {"MAE": mae, "RMSE": float(np.sqrt(mse)), "R2": 1 - mse / var}, pd.DataFrame(rows).sort_values("city")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="pre_processing/patches")
    ap.add_argument("--out", default="data/unet_run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32, help="channels of the first U-Net level")
    ap.add_argument("--max-patches", type=int, default=0, help="use only this many training patches (quick test)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.patches), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads or os.cpu_count() or 1)
    print(f"device {dev}, {torch.get_num_threads()} threads | loading patches ...", flush=True)
    t0 = time.time()
    tr = load_split(root, "train", a.max_patches)
    va = load_split(root, "val", a.max_patches // 5 if a.max_patches else 0)
    print(f"train {len(tr[0]):,} patches, val {len(va[0]):,} patches, loaded in {time.time() - t0:.0f}s", flush=True)
    ok = tr[3] == 1
    mu, sd = float(tr[2][ok].mean()), float(tr[2][ok].std())
    print(f"train LST mean {mu:.2f} C, std {sd:.2f} C", flush=True)

    model = UNet(tr[0].shape[1], len(meta["landcover_codes"]), a.base).to(dev)
    print(f"U-Net with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    nb = len(tr[0]) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb)
    rng = np.random.default_rng(0)
    log, best = [], 1e9
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(tr[0]))
        tot = cnt = 0.0
        for b in range(nb):
            i = np.sort(perm[b * a.batch:(b + 1) * a.batch])
            x = torch.from_numpy(tr[0][i]).to(dev).float().clamp(-CLIP, CLIP)
            lc = torch.from_numpy(tr[1][i]).to(dev)
            y = torch.from_numpy(tr[2][i]).to(dev)
            m = torch.from_numpy(tr[3][i]).to(dev)
            x, lc, y, m = augment(x, lc, y, m)
            m = m.bool()
            loss = F.smooth_l1_loss(model(x, lc)[m], ((y - mu) / sd)[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            cnt += 1
            if (b + 1) % 50 == 0 or b + 1 == nb:
                print(f"\r  epoch {ep}/{a.epochs}  batch {b + 1}/{nb}  loss {tot / cnt:.4f}  {time.time() - t0:.0f}s", end="", flush=True)
        met, per = evaluate(model, va, dev, mu, sd)
        met.update({"epoch": ep, "train_loss": tot / cnt, "seconds": round(time.time() - t0)})
        log.append(met)
        pd.DataFrame(log).to_csv(out / "metrics.csv", index=False)
        flag = ""
        if met["MAE"] < best:
            best = met["MAE"]
            torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "features": meta["features"],
                        "landcover_codes": meta["landcover_codes"], "base": a.base}, out / "best.pt")
            per.round(3).to_csv(out / "val_per_city.csv", index=False)
            flag = "  <- best"
        print(f"\r  epoch {ep}/{a.epochs}  loss {tot / cnt:.4f} | val MAE {met['MAE']:.2f} C  RMSE {met['RMSE']:.2f} C  "
              f"R2 {met['R2']:.3f} | {met['seconds']}s{flag}", flush=True)
    print(f"\nbest val MAE {best:.2f} C -> {out / 'best.pt'}; per-city scores in {out / 'val_per_city.csv'}")


if __name__ == "__main__":
    main()
