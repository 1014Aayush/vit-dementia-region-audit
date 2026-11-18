"""Label-blind mask geometry used to fix the slice range and patch threshold in DECISION_RULES.md.

Pools all 436 sessions (no diagnosis used). Coronal = axis 1 of the LAS T88 images (176 x 208 x 176).
Writes results/geometry/*.csv.
"""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry")
os.makedirs(OUT, exist_ok=True)


def patch_fraction(sl, grid):
    """Fraction of each patch inside the mask after resizing a 176x176 slice to 224 px."""
    idx = np.floor(np.arange(224) * sl.shape[0] / 224).astype(int)   # same rule as torch 'nearest'
    big = sl[np.ix_(idx, idx)].astype(np.float32)
    k = 224 // grid                                                    # 14 (patch-14) or 16 (patch-16)
    return big.reshape(grid, k, grid, k).mean(axis=(1, 3))


rows, per_slice = [], []
for p in sorted(glob.glob(os.path.join(ROOT, "data/masks/mtl/*_mtl.nii.gz"))):
    sid = os.path.basename(p)[:13]
    mtl = np.asanyarray(nib.load(p).dataobj) > 0
    ven = np.asanyarray(nib.load(p.replace("mtl", "ventricles")).dataobj) > 0
    hip = np.asanyarray(nib.load(p.replace("mtl", "hippocampus")).dataobj) > 0
    area = mtl.sum(axis=(0, 2))
    ys = np.nonzero(area)[0]
    rows.append(dict(sid=sid, mtl_y_min=ys.min(), mtl_y_max=ys.max(),
                     mtl_y_centroid=float((area * np.arange(len(area))).sum() / area.sum()),
                     hip_y_centroid=float(np.nonzero(hip)[1].mean())))
    for y in range(60, 140):
        rec = dict(sid=sid, y=y)
        for name, m in (("mtl", mtl), ("hippocampus", hip), ("ventricles", ven)):
            sl = m[:, y, :]
            for grid, tag in ((16, "p14"), (14, "p16")):
                fr = patch_fraction(sl, grid)
                for th in (0.25, 0.5):
                    rec[f"{name}_{tag}_n{int(th * 100)}"] = int((fr >= th).sum())
        per_slice.append(rec)

subj = pd.DataFrame(rows)
sl = pd.DataFrame(per_slice)
subj.to_csv(os.path.join(OUT, "mtl_extent_per_session.csv"), index=False)
sl.to_csv(os.path.join(OUT, "patch_counts_per_slice.csv"), index=False)

print("MTL coronal extent (y index), median over sessions: min %d, max %d, centroid %.1f; hippocampus centroid %.1f"
      % (subj.mtl_y_min.median(), subj.mtl_y_max.median(), subj.mtl_y_centroid.median(), subj.hip_y_centroid.median()))
print("centroid 5th-95th pct: %.1f - %.1f" % tuple(np.percentile(subj.mtl_y_centroid, [5, 95])))
med = sl.groupby("y").median(numeric_only=True)
cols = [c for c in med.columns if c != "sid"]
print(med.loc[range(84, 116, 2), [c for c in cols if "p14" in c]].to_string())
print(med.loc[range(84, 116, 2), [c for c in cols if "p16" in c]].to_string())
