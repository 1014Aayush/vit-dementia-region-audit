"""Shared pieces fixed by DECISION_RULES.md: cohort (§2), folds and probe (§5-6), bootstrap (§6, §8).
Used by the baseline here and by the ViT notebook on Kaggle, so both use identical folds and settings."""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "oasis1_original", "oasis_cross-sectional-5708aa0a98d82080.xlsx")
N_REPEATS, N_FOLDS, C = 20, 5, 0.5
B = 10_000


def load_cohort(demo_path=DEMO):
    """First sessions, age >= 60, CDR known. y = 1 if CDR >= 0.5."""
    d = pd.read_excel(demo_path) if demo_path.endswith(".xlsx") else pd.read_csv(demo_path)
    d = d[d["ID"].str.endswith("MR1") & (d["Age"] >= 60) & d["CDR"].notna()].copy()
    d = d.rename(columns={"ID": "sid", "Age": "age", "CDR": "cdr", "nWBV": "nwbv", "M/F": "sex"})
    d["y"] = (d["cdr"] >= 0.5).astype(int)
    return d.sort_values("sid").reset_index(drop=True)


def make_folds(cohort):
    """Long table: sid, repeat, fold. Stratified by y, seeds 0..19."""
    rows = []
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=r)
        for f, (_, te) in enumerate(skf.split(cohort["sid"], cohort["y"])):
            rows += [dict(sid=cohort["sid"].iloc[i], repeat=r, fold=f) for i in te]
    return pd.DataFrame(rows)


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="lbfgs", max_iter=5000))


def cv_scores(X, y, fold_of):
    """Subject-level features. X (n, p), y (n,), fold_of (n_repeats, n) fold index.
    Returns out-of-fold decision scores (n_repeats, n)."""
    out = np.zeros(fold_of.shape)
    for r in range(fold_of.shape[0]):
        for f in range(N_FOLDS):
            te = fold_of[r] == f
            m = probe().fit(X[~te], y[~te])
            out[r, te] = m.decision_function(X[te])
    return out


def summarize(y, scores):
    """scores (n_repeats, n). Mean per-repeat AUC / balanced accuracy, plus bootstrap CI of the
    AUC of the repeat-averaged score."""
    aucs = [roc_auc_score(y, s) for s in scores]
    baccs = [balanced_accuracy_score(y, s > 0) for s in scores]
    avg = scores.mean(0)
    lo, hi = boot_auc_ci(y, avg)
    return dict(auc_mean=np.mean(aucs), auc_sd_repeats=np.std(aucs), auc_avgscore=roc_auc_score(y, avg),
                auc_ci_lo=lo, auc_ci_hi=hi, bacc_mean=np.mean(baccs))


def boot_auc_ci(y, s, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    while len(vals) < B:
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def boot_auc_diff(y, s1, s2, seed=0):
    """Paired bootstrap over subjects of AUC(s1) - AUC(s2). Returns diff, lo, hi, two-sided p."""
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    while len(vals) < B:
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    vals = np.array(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return roc_auc_score(y, s1) - roc_auc_score(y, s2), *np.percentile(vals, [2.5, 97.5]), min(1.0, max(p, 1 / B))


def boot_mean(d, seed=0):
    """Paired bootstrap of a per-subject difference (§8). Returns mean, lo, hi, two-sided p."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.mean(), *np.percentile(means, [2.5, 97.5]), min(1.0, max(p, 1 / B))


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
