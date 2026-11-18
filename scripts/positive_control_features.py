"""Positive control (DECISION_RULES.md §12.1): planted signal with a known location, synthetic labels.
Features are computed locally on CPU with the same attention-cut code (vitprobe) and seeds as the Kaggle run (§7).

    python scripts/positive_control_features.py calibrate --scenario plant_mtl
    python scripts/positive_control_features.py features  --scenario plant_mtl --alpha 0.2 --models dinov2_s deit_s

Outputs in results/positive_control/:
  calibration.csv                      AUC on `full` per scenario and alpha (DINOv2-S)
  features/<model>__neg.npz            synthetic-negative slices (unchanged images; shared by both scenarios)
  features/<model>__<scenario>.npz     all 990 slices, every condition, same layout as results/vit_features/*.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402
from vitprobe.masks import edge_keep_mask, isolate_mask, n_cross_pairs, token_labels  # noqa: E402
from vitprobe.models import ProbedViT  # noqa: E402

OUT = os.path.join(ROOT, "results", "positive_control")
FEAT = os.path.join(OUT, "features")
INP = os.path.join(ROOT, "data", "vit_input")
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40)
CAL_AUC = 0.75
PERM_SEED = 12345
REGIONS = ["mtl", "ventricles"]            # same order as the Kaggle run, so the random-cut seeds match (§7)
TH, BRAIN_TH, SEEDS, BATCH = 0.25, 0.25, (0, 1, 2), 32
PLANT = {"plant_mtl": "mtl", "plant_global": "brain", "plant_mtl_weak": "mtl"}   # weak: change log 2026-11-18
torch.set_num_threads(max(1, os.cpu_count() - 1))


def synthetic_labels():
    """§12.1: the real labels permuted once. Returns subject labels (198,), slice labels (990,)."""
    cohort = pd.read_csv(os.path.join(ROOT, "results", "cohort.csv"))
    meta = pd.read_csv(os.path.join(INP, "meta.csv"))
    ys = np.random.default_rng(PERM_SEED).permutation(cohort["y"].to_numpy())
    return ys, meta["sid"].map(dict(zip(cohort["sid"], ys))).to_numpy()


def load_inputs():
    images = torch.from_numpy(np.load(os.path.join(INP, "images.npy")).astype(np.float32))
    masks = {r: torch.from_numpy(np.load(os.path.join(INP, f"{r}.npy")).astype(np.float32))
             for r in ("mtl", "ventricles", "brain")}
    return images, masks


def planted(images, masks, scenario, alpha, idx):
    """Images of slices idx with the signal planted (all idx are synthetic-positive)."""
    return images[idx] * (1 - alpha * masks[PLANT[scenario]][idx])


def to_input(x, net):
    x = F.interpolate(x[:, None], size=(224, 224), mode="bilinear", align_corners=False).clamp(0, 1)
    x = x.repeat(1, 3, 1, 1)
    return (x - torch.tensor(net.mean).view(1, 3, 1, 1)) / torch.tensor(net.std).view(1, 3, 1, 1)


def patch_frac(m, idx, patch):
    return F.avg_pool2d(F.interpolate(m[idx][:, None], size=(224, 224), mode="nearest"), patch).flatten(1)


# --- identical to kaggle/make_vit_nb.py (seeding included) ---
def gens(seed, idx):
    return [torch.Generator().manual_seed(seed * 1_000_003 + int(i)) for i in idx]


def tok(patch_side, n_prefix):
    return token_labels(patch_side, n_prefix, "fg")


def random_patches(region, brain, idx, seed):
    sel = torch.zeros_like(region)
    for b, g in enumerate(gens(seed, idx)):
        cand = torch.nonzero(brain[b] & ~region[b]).flatten()
        k = min(int(region[b].sum()), len(cand))
        if k:
            sel[b, cand[torch.randperm(len(cand), generator=g)[:k]]] = True
    return sel


def conditions(idx, fr, brain, n_prefix, full_only):
    yield "full", None
    if full_only:
        return
    for ri, r in enumerate(REGIONS):
        R = fr[r] >= TH
        lab = tok(~R, n_prefix)
        yield f"remove_{r}", isolate_mask(lab)
        for s in SEEDS:
            base = 100 * ri + s
            yield (f"random_links_{r}#{s}",
                   edge_keep_mask(lab, None, "random", generators=gens(base, idx), n_remove=n_cross_pairs(lab)))
            yield f"random_patches_{r}#{s}", isolate_mask(tok(~random_patches(R, brain, idx, base + 10), n_prefix))
# --- end identical part ---


def extract(net, images, masks, idx_all, full_only=False, plant=None):
    """plant: None or (scenario, alpha). Returns feats {cond: (len(idx_all), d)} and counts."""
    feats, counts = {}, {"n_mtl": [], "n_ventricles": [], "n_brain": []}
    t0 = time.time()
    for s in range(0, len(idx_all), BATCH):
        idx = idx_all[s:s + BATCH]
        x = to_input(planted(images, masks, *plant, idx) if plant else images[idx], net)
        fr = {r: patch_frac(masks[r], idx, net.patch) for r in REGIONS}
        brain = patch_frac(masks["brain"], idx, net.patch) >= BRAIN_TH
        for r in REGIONS:
            counts[f"n_{r}"] += (fr[r] >= TH).sum(1).tolist()
        counts["n_brain"] += brain.sum(1).tolist()
        for name, keep in conditions(idx, fr, brain, net.n_prefix, full_only):
            net.ctrl.keep, net.ctrl.layers = keep, None
            feats.setdefault(name, []).append(net.features(x).float().numpy())
        net.ctrl.keep = None
        print(f"    {min(s + BATCH, len(idx_all))}/{len(idx_all)} slices, {time.time() - t0:.0f} s", flush=True)
    return {k: np.concatenate(v) for k, v in feats.items()}, {k: np.array(v) for k, v in counts.items()}


def merge(neg, pos, neg_idx, pos_idx, n):
    out = {}
    for k in neg:
        a = np.zeros((n,) + neg[k].shape[1:], neg[k].dtype)
        a[neg_idx], a[pos_idx] = neg[k], pos[k]
        out[k] = a
    return out


def cv_auc_full(full, y_slice):
    """Protocol B gate AUC with the §6 folds, using synthetic labels."""
    from sklearn.metrics import roc_auc_score
    from analyze_vit import context, slice_logits, subject_mean
    ctx = context()
    lg = slice_logits({"full": full}, y_slice, ctx.slice_subj, ctx.fold_of, ["full"])["full"]
    ys, _ = synthetic_labels()
    s, _ = subject_mean(lg, ctx.slice_subj, ctx.n)
    return float(np.mean([roc_auc_score(ys, v) for v in s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "features"])
    ap.add_argument("--scenario", choices=list(PLANT), required=True)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--models", nargs="+", default=["dinov2_s"])
    a = ap.parse_args()
    os.makedirs(FEAT, exist_ok=True)
    images, masks = load_inputs()
    ys, y_slice = synthetic_labels()
    pos_idx, neg_idx = np.where(y_slice == 1)[0], np.where(y_slice == 0)[0]
    n = len(y_slice)

    if a.stage == "calibrate":                      # §12.1: `full` only, DINOv2-S, smallest alpha with AUC >= 0.75
        net = ProbedViT("dinov2_s")
        neg_path = os.path.join(FEAT, "dinov2_s__neg_full.npz")
        if os.path.exists(neg_path):
            neg = dict(np.load(neg_path))
        else:
            neg, _ = extract(net, images, masks, neg_idx, full_only=True)
            np.savez(neg_path, **neg)
        cal_path = os.path.join(OUT, "calibration.csv")
        rows = pd.read_csv(cal_path).to_dict("records") if os.path.exists(cal_path) else []
        for alpha in ALPHAS:
            pos, _ = extract(net, images, masks, pos_idx, full_only=True, plant=(a.scenario, alpha))
            auc = cv_auc_full(merge(neg, pos, neg_idx, pos_idx, n)["full"], y_slice)
            rows.append(dict(scenario=a.scenario, alpha=alpha, model="dinov2_s", auc_full=auc))
            pd.DataFrame(rows).to_csv(cal_path, index=False)
            print(f"{a.scenario} alpha={alpha}: AUC {auc:.3f}", flush=True)
            if auc >= CAL_AUC:
                print(f"chosen alpha = {alpha}")
                break
        return

    for key in a.models:
        net = ProbedViT(key)
        neg_path = os.path.join(FEAT, f"{key}__neg.npz")
        if os.path.exists(neg_path):
            z = np.load(neg_path)
            neg = {k: z[k] for k in z.files if not k.startswith("n_")}
            neg_cnt = {k: z[k] for k in z.files if k.startswith("n_")}
        else:
            print(f"{key}: synthetic-negative slices", flush=True)
            neg, neg_cnt = extract(net, images, masks, neg_idx)
            np.savez(neg_path, **neg, **neg_cnt)
        print(f"{key}: {a.scenario} alpha={a.alpha}, synthetic-positive slices", flush=True)
        pos, pos_cnt = extract(net, images, masks, pos_idx, plant=(a.scenario, a.alpha))
        np.savez(os.path.join(FEAT, f"{key}__{a.scenario}.npz"),
                 **merge(neg, pos, neg_idx, pos_idx, n), **merge(neg_cnt, pos_cnt, neg_idx, pos_idx, n),
                 alpha=np.array(a.alpha))


if __name__ == "__main__":
    main()
