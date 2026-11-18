"""Label-blind comparison of slice options for DECISION_RULES.md (every 4th session, no diagnosis used).
Compares coronal vs axial, full 176 px field vs a fixed crop to the pooled brain bounding box.
Patch-14 grid (16 x 16) at 224 px, patch counted if >= 25% inside the region."""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = sorted(glob.glob(os.path.join(ROOT, "data/t88_nii/OAS1_*.nii.gz")))

# pooled brain bounding box (label-blind), over all sessions
lo, hi = np.array([999] * 3), np.array([0] * 3)
for p in paths:
    nz = np.argwhere(np.asanyarray(nib.load(p).dataobj) > 0)
    lo, hi = np.minimum(lo, nz.min(0)), np.maximum(hi, nz.max(0))
print("pooled brain bounding box (x, y, z):", lo.tolist(), hi.tolist())


def count(sl, th=0.25, grid=16):
    n = sl.shape[0]
    idx = np.floor(np.arange(224) * n / 224).astype(int)
    big = sl[np.ix_(idx, idx)].astype(np.float32)
    k = 224 // grid
    return int((big.reshape(grid, k, grid, k).mean(axis=(1, 3)) >= th).sum())


def square(a0, a1, b0, b1):
    """Square box covering [a0,a1] x [b0,b1]."""
    side = max(a1 - a0, b1 - b0) + 1
    ca, cb = (a0 + a1) // 2, (b0 + b1) // 2
    return ca - side // 2, cb - side // 2, side


rows = []
for p in paths[::4]:
    sid = os.path.basename(p)[:13]
    m = {r: np.asanyarray(nib.load(os.path.join(ROOT, f"data/masks/{r}/{sid}_{r}.nii.gz")).dataobj) > 0
         for r in ("mtl", "ventricles", "hippocampus")}
    brain = np.asanyarray(nib.load(p).dataobj) > 0
    # coronal: slice axis 1, plane (x, z); axial: slice axis 2, plane (x, y)
    xa, za, sc = square(lo[0], hi[0], lo[2], hi[2])
    xb, yb, sa = square(lo[0], hi[0], lo[1], hi[1])
    for y in range(96, 116, 4):
        for r, mk in m.items():
            full = mk[:, y, :]
            crop = np.pad(full, sc)[xa + sc: xa + 2 * sc, za + sc: za + 2 * sc]
            rows.append(dict(sid=sid, plane="coronal", pos=y, region=r, full=count(full), crop=count(crop)))
    for z in range(60, 80, 4):
        for r, mk in m.items():
            full = np.pad(mk[:, :, z], ((16, 16), (0, 0)))   # 176 x 208 -> 208 x 208 square
            crop = np.pad(mk[:, :, z], sa)[xb + sa: xb + 2 * sa, yb + sa: yb + 2 * sa]
            rows.append(dict(sid=sid, plane="axial", pos=z, region=r, full=count(full), crop=count(crop)))
    # brain patch count, for scale
    rows.append(dict(sid=sid, plane="coronal", pos=104, region="brain", full=count(brain[:, 104, :]),
                     crop=count(np.pad(brain[:, 104, :], sc)[xa + sc: xa + 2 * sc, za + sc: za + 2 * sc])))

d = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results", "geometry"), exist_ok=True)
d.to_csv(os.path.join(ROOT, "results", "geometry", "slice_options.csv"), index=False)
print(d.groupby(["plane", "region", "pos"])[["full", "crop"]].median().to_string())
