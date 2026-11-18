"""Positive control analysis (DECISION_RULES.md §12.1). Synthetic labels; otherwise §5-8 unchanged
(Protocol B L2 probe, same folds, margins, d vs each control, paired bootstrap). Holm within the 16 tests.

    python scripts/positive_control_analysis.py
Outputs in results/positive_control/: tests.csv, condition_auc.csv, verdict.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_vit import context, d_vs, slice_logits, subject_mean
from common import boot_mean, holm
from positive_control_features import OUT, synthetic_labels

SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(OUT, "kaggle", "pc_features"))
    a = ap.parse_args()
    ctx = context()
    real_y = ctx.y.copy()
    ys, y_slice = synthetic_labels()
    ctx.y, ctx.sign, ctx.y_slice = ys, 2 * ys - 1, y_slice
    tests, aucs = [], []
    paths = sorted(glob.glob(os.path.join(a.features, "*__plant_*.npz")))
    assert paths, f"no feature files in {a.features}"
    for path in paths:
        model, scenario = os.path.basename(path)[:-4].split("__")
        z = np.load(path)
        feats = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("n_") and k != "alpha"}
        counts = {k: z[k] for k in z.files if k.startswith("n_")}
        alpha = float(z["alpha"])
        print(model, scenario, "alpha", alpha, flush=True)
        lg = slice_logits(feats, y_slice, ctx.slice_subj, ctx.fold_of, list(feats))
        for c in lg:
            s, _ = subject_mean(lg[c], ctx.slice_subj, ctx.n)
            aucs.append(dict(model=model, scenario=scenario, alpha=alpha, condition=c,
                             auc_mean=np.mean([roc_auc_score(ys, v) for v in s])))
        for r in ("mtl", "ventricles"):
            for ctrl in ("random_links", "random_patches"):
                dd, n_used, n_no = d_vs(lg, counts, ctx, r, f"remove_{r}", [f"{ctrl}_{r}#{s}" for s in SEEDS])
                mean, lo, hi, p = boot_mean(dd[~np.isnan(dd)])
                tests.append(dict(model=model, scenario=scenario, alpha=alpha, region=r, control=ctrl,
                                  n_subjects=n_used, mean_d=mean, ci_lo=lo, ci_hi=hi, p=p))
    df = pd.DataFrame(tests)
    df["family"] = np.where(df["scenario"] == "plant_mtl_weak", "weak (added, change log)", "pre-specified §12.1")
    df["p_holm"] = df.groupby("family")["p"].transform(lambda p: pd.Series(holm(p.to_numpy()), index=p.index))
    df["supported"] = (df["p_holm"] < 0.05) & (df["mean_d"] < 0)
    df.to_csv(os.path.join(OUT, "tests.csv"), index=False)
    pd.DataFrame(aucs).to_csv(os.path.join(OUT, "condition_auc.csv"), index=False)

    m = df[df.region == "mtl"]
    pm = m[m.scenario == "plant_mtl"]
    pg = m[(m.scenario == "plant_global") & (m.control == "random_patches")]
    n_models = df["model"].nunique()
    crit1 = bool(len(pm) == 2 * n_models and pm["supported"].all())
    crit2 = bool(len(pg) == n_models and not pg["supported"].any())
    verdict = pd.DataFrame([
        dict(item=f"plant_mtl: MTL dependence vs both controls in all {n_models} models", value=crit1),
        dict(item=f"plant_global: no MTL dependence vs random patches in any of {n_models} models", value=crit2),
        dict(item="method validated (both criteria)", value=crit1 and crit2),
        dict(item="corr(synthetic label, real label), subjects", value=float(np.corrcoef(ys, real_y)[0, 1])),
    ])
    verdict.to_csv(os.path.join(OUT, "verdict.csv"), index=False)
    print(df.round(4).to_string(index=False))
    print(pd.DataFrame(aucs).pivot_table(index="condition", columns=["model", "scenario"], values="auc_mean").round(3))
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()
