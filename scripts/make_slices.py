"""ViT inputs fixed by DECISION_RULES.md §3-4: 5 coronal slices per subject (y = 96..112, step 4),
fixed crop x 14-158, z 7-151 (145 x 145), intensity clipped at the 99.5th percentile of brain voxels
and scaled to [0, 1]. Region masks cropped the same way (resizing and patch fractions happen on Kaggle).

    python scripts/make_slices.py
Writes data/vit_input/{images,mtl,ventricles,hippocampus,brain}.npy, meta.csv, and a zip for Kaggle
with these plus cohort.csv, folds.csv and the vitprobe code.
"""
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "vit_input")
VITPROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vitprobe")
YS = [96, 100, 104, 108, 112]
X0, Z0, SIDE = 14, 7, 145
REGIONS = ["mtl", "ventricles", "hippocampus"]

os.makedirs(OUT, exist_ok=True)
cohort = pd.read_csv(os.path.join(ROOT, "results", "cohort.csv"))


def crop(sl):
    out = np.zeros((SIDE, SIDE), sl.dtype)
    a = sl[X0:X0 + SIDE, Z0:Z0 + SIDE]
    out[:a.shape[0], :a.shape[1]] = a
    # rows = x (left-right), cols = z (inferior-superior); transpose + flip so the image is upright
    return out.T[::-1]


imgs, masks, meta = [], {r: [] for r in REGIONS + ["brain"]}, []
for sid in cohort["sid"]:
    vol = np.asanyarray(nib.load(os.path.join(ROOT, "data", "t88_nii", sid + ".nii.gz")).dataobj).astype(np.float32)
    top = np.percentile(vol[vol > 0], 99.5)
    vol = np.clip(vol / top, 0, 1)
    reg = {r: np.asanyarray(nib.load(os.path.join(ROOT, "data", "masks", r, f"{sid}_{r}.nii.gz")).dataobj) > 0
           for r in REGIONS}
    for y in YS:
        imgs.append(crop(vol[:, y, :]).astype(np.float16))
        masks["brain"].append(crop(vol[:, y, :] > 0))
        for r in REGIONS:
            masks[r].append(crop(reg[r][:, y, :]))
        meta.append(dict(sid=sid, y_slice=y))

np.save(os.path.join(OUT, "images.npy"), np.stack(imgs))
for r, m in masks.items():
    np.save(os.path.join(OUT, f"{r}.npy"), np.stack(m).astype(np.uint8))
pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
print("slices:", len(meta), "image array", np.stack(imgs).shape)

# one zip for Kaggle (Kaggle unpacks it): inputs + cohort/folds + vitprobe code
zpath = os.path.join(ROOT, "data", "oasis1_vit_input.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT):
        z.write(os.path.join(OUT, f), f)
    for f in ("cohort.csv", "folds.csv"):
        z.write(os.path.join(ROOT, "results", f), f)
    z.writestr("vitprobe/__init__.py", "")
    for f in ("models.py", "masks.py"):
        z.write(os.path.join(VITPROBE, f), "vitprobe/" + f)
print("wrote", zpath, "%.0f MB" % (os.path.getsize(zpath) / 1e6))
