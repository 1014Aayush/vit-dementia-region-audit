"""Readout sensitivity (DECISION_RULES.md §12.2, exploratory): the §5-8 analysis with a sparse (L1) probe.
C is chosen inside each outer training fold by inner 5-fold subject-grouped CV on `full` (slice-level AUC).
Real labels, existing Kaggle features, Protocol B, same folds, gate and tests as analyze_vit.py.

    python scripts/l1_probe.py
Outputs in results/vit_l1/: gate.csv, confirmatory.csv (Holm within the 20 tests), condition_auc.csv, probe.csv
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_vit import context, gate, lesion, load
from common import N_FOLDS, ROOT, holm

CS = (0.01, 0.03, 0.1, 0.3)
OUT = os.path.join(ROOT, "results", "vit_l1")


def l1(c):
    return make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=1.0, C=c, solver="liblinear", max_iter=5000))


def choose_c(X, y, groups, seed):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    split = list(cv.split(X, y, groups))
    score = [np.mean([roc_auc_score(y[va], l1(c).fit(X[tr], y[tr]).decision_function(X[va])) for tr, va in split])
             for c in CS]
    return CS[int(np.argmax(score))]


def slice_logits_l1(feats, ctx, conds):
    n_rep = ctx.fold_of.shape[0]
    out = {c: np.zeros((n_rep, len(ctx.y_slice))) for c in conds}
    info = []
    for r in range(n_rep):
        fold_slice = ctx.fold_of[r][ctx.slice_subj]
        for f in range(N_FOLDS):
            te = fold_slice == f
            Xtr, ytr, gtr = feats["full"][~te], ctx.y_slice[~te], ctx.slice_subj[~te]
            c = choose_c(Xtr, ytr, gtr, 1000 * r + f)
            m = l1(c).fit(Xtr, ytr)
            info.append(dict(repeat=r, fold=f, C=c, n_nonzero=int((m[-1].coef_ != 0).sum())))
            for k in conds:
                out[k][r, te] = m.decision_function(feats[k][te])
        print(f"  repeat {r} done", flush=True)
    return out, info


def main():
    ctx = context()
    os.makedirs(OUT, exist_ok=True)
    gate_rows, conf, auc_rows, probe_rows = [], [], [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "vit_features", "*.npz"))):
        key = os.path.basename(path)[:-4]
        feats, counts = load(path)
        print(key, flush=True)
        lg, info = slice_logits_l1(feats, ctx, list(feats))
        g, _, passed, _ = gate(key, lg["full"], ctx)
        g.update(n_features=feats["full"].shape[1], median_nonzero=float(np.median([i["n_nonzero"] for i in info])))
        gate_rows.append(g)
        probe_rows += [dict(model=key, **i) for i in info]
        if passed:
            a_, c_, _, _, _ = lesion(key, lg, counts, ctx)
            auc_rows += a_
            conf += c_
    pd.DataFrame(gate_rows).to_csv(os.path.join(OUT, "gate.csv"), index=False)
    pd.DataFrame(probe_rows).to_csv(os.path.join(OUT, "probe.csv"), index=False)
    pd.DataFrame(auc_rows).to_csv(os.path.join(OUT, "condition_auc.csv"), index=False)
    df = pd.DataFrame(conf)
    if len(df):
        df["p_holm"] = holm(df["p"])
        df["supported"] = (df["p_holm"] < 0.05) & (df["mean_d"] < 0)
        df.to_csv(os.path.join(OUT, "confirmatory.csv"), index=False)
        print(df.round(4).to_string(index=False))
    print(pd.DataFrame(gate_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
