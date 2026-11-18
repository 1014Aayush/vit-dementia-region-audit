"""Protocol A (DECISION_RULES.md §5, exploratory): the probe is refit on each condition's own features,
same folds (20 repeats x 5). Answers whether "region alone gives near-chance AUC" (Protocol B) is lost
information or only a probe that never saw region-only features.

    python scripts/protocol_a.py
Output: results/vit/protocol_a.csv (AUC per model and condition, Protocol A next to Protocol B)
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_vit import OUT, context, load, subject_mean
from common import N_FOLDS, ROOT, boot_auc_ci, probe

CONDS = ["full", "remove_brain", "only_mtl", "only_ventricles", "only_hippocampus",
         "remove_mtl", "remove_ventricles", "remove_hippocampus", "readout_only_mtl", "readout_only_ventricles"]


def main():
    ctx = context()
    b = pd.read_csv(os.path.join(OUT, "condition_auc.csv"))
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "vit_features", "*.npz"))):
        key = os.path.basename(path)[:-4]
        feats, _ = load(path)
        for c in CONDS:
            lg = np.zeros((ctx.fold_of.shape[0], len(ctx.y_slice)))
            for r in range(ctx.fold_of.shape[0]):
                fs = ctx.fold_of[r][ctx.slice_subj]
                for f in range(N_FOLDS):
                    te = fs == f
                    lg[r, te] = probe().fit(feats[c][~te], ctx.y_slice[~te]).decision_function(feats[c][te])
            s, _ = subject_mean(lg, ctx.slice_subj, ctx.n)
            lo, hi = boot_auc_ci(ctx.y, s.mean(0))
            pb = b[(b.model == key) & (b.condition == c)]["auc_mean"]
            rows.append(dict(model=key, condition=c, auc_protocol_a=np.mean([roc_auc_score(ctx.y, v) for v in s]),
                             auc_a_ci_lo=lo, auc_a_ci_hi=hi,
                             auc_protocol_b=float(pb.iloc[0]) if len(pb) else np.nan))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "protocol_a.csv"), index=False)


if __name__ == "__main__":
    main()
