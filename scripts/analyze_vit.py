"""ViT analysis fixed by DECISION_RULES.md §5-8. Reads the Kaggle feature files, fits the probes with the
fixed folds (Protocol B: fit on uncut features, apply unchanged to every condition), applies the gate,
and runs the lesion tests only for models that pass it. The gate / lesion / write-out functions are
reused by analyze_ft.py (Experiment 2, §11).

    python scripts/analyze_vit.py --features results/vit_features
Outputs in results/vit/: gate.csv, vs_baseline.csv, condition_auc.csv, region_patch_counts.csv,
confirmatory.csv, exploratory.csv
"""
import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import N_FOLDS, ROOT, boot_auc_ci, boot_auc_diff, boot_mean, holm, probe

GATE_AUC = 0.70
SEEDS = (0, 1, 2)
OUT = os.path.join(ROOT, "results", "vit")


def load(path):
    z = np.load(path)
    feats = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("n_")}
    counts = {k: z[k] for k in z.files if k.startswith("n_")}
    return feats, counts


def context(n_repeats=None):
    """Cohort, labels, slice->subject map and fold table (first n_repeats repeats)."""
    cohort = pd.read_csv(os.path.join(ROOT, "results", "cohort.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "results", "folds.csv"))
    meta = pd.read_csv(os.path.join(ROOT, "data", "vit_input", "meta.csv"))
    base = pd.read_csv(os.path.join(ROOT, "results", "baseline", "scores.csv"))
    sids = cohort["sid"].tolist()
    assert base["sid"].tolist() == sids
    y = cohort["y"].to_numpy()
    slice_subj = meta["sid"].map({s: i for i, s in enumerate(sids)}).to_numpy()
    fold_of = folds.pivot(index="repeat", columns="sid", values="fold")[sids].to_numpy()[:n_repeats]
    return SimpleNamespace(cohort=cohort, y=y, sign=2 * y - 1, n=len(sids), slice_subj=slice_subj,
                           y_slice=y[slice_subj], fold_of=fold_of, base=base["hippo_nwbv_age"].to_numpy())


def slice_logits(feats, y_slice, slice_subj, fold_of, conds):
    """Protocol B. Returns {cond: (n_repeats, n_slices)} out-of-fold slice logits."""
    n_rep = fold_of.shape[0]
    out = {c: np.zeros((n_rep, len(y_slice))) for c in conds}
    for r in range(n_rep):
        fold_slice = fold_of[r][slice_subj]
        for f in range(N_FOLDS):
            te = fold_slice == f
            m = probe().fit(feats["full"][~te], y_slice[~te])
            for c in conds:
                out[c][r, te] = m.decision_function(feats[c][te])
    return out


def subject_mean(v, slice_subj, n_subj, use=None):
    """Mean over a subject's slices (optionally only slices where use is True). v (..., n_slices)."""
    use = np.ones(v.shape[-1], bool) if use is None else use
    w = np.zeros((v.shape[-1], n_subj))
    w[np.where(use)[0], slice_subj[use]] = 1
    cnt = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v @ w) / cnt, cnt


def gate(key, lg_full, ctx):
    """§6. lg_full (n_rep, n_slices). Returns gate row, vs-baseline row, passed, repeat-averaged subject score."""
    y = ctx.y
    subj_full, _ = subject_mean(lg_full, ctx.slice_subj, ctx.n)          # (n_rep, n)
    aucs = [roc_auc_score(y, s) for s in subj_full]
    avg = subj_full.mean(0)
    lo, hi = boot_auc_ci(y, avg)
    passed = np.mean(aucs) >= GATE_AUC
    g = dict(model=key, auc_mean=np.mean(aucs), auc_sd_repeats=np.std(aucs),
             auc_avgscore=roc_auc_score(y, avg), auc_ci_lo=lo, auc_ci_hi=hi,
             bacc_mean=np.mean([balanced_accuracy_score(y, s > 0) for s in subj_full]),
             passed_gate=passed)
    d, dlo, dhi, p = boot_auc_diff(y, avg, ctx.base)
    v = dict(model=key, auc_vit_minus_baseline=d, ci_lo=dlo, ci_hi=dhi, p=p)
    print("  gate AUC %.3f -> %s" % (np.mean(aucs), "PASS" if passed else "fail: no lesion analysis"), flush=True)
    return g, v, passed, avg


def test_list():
    """(family, label, tested, controls, control name, subset, count region), order fixed (§8)."""
    tests = []
    for r in ("mtl", "ventricles"):
        for ctrl in ("random_links", "random_patches"):
            tests.append(("confirmatory", r, f"remove_{r}", [f"{ctrl}_{r}#{s}" for s in SEEDS], ctrl, None, r))
    for ctrl in ("random_links", "random_patches"):
        tests.append(("exploratory", "hippocampus", "remove_hippocampus",
                      [f"{ctrl}_hippocampus#{s}" for s in SEEDS], ctrl, None, "hippocampus"))
        for r in ("mtl", "ventricles"):
            tests.append(("exploratory", f"{r}@50", f"remove_{r}@50",
                          [f"{ctrl}_{r}@50#{s}" for s in SEEDS], ctrl, None, f"{r}@50"))
            tests.append(("exploratory", f"{r} (CDR 0 vs >=1)", f"remove_{r}",
                          [f"{ctrl}_{r}#{s}" for s in SEEDS], ctrl, "cdr1", r))
    for r in ("mtl", "ventricles", "hippocampus"):
        for part in ("early", "late"):
            tests.append(("exploratory", f"{r} {part} blocks", f"remove_{r}_{part}",
                          [f"random_links_{r}_{part}"], "random_links", None, r))
    return tests


def d_vs(lg, counts, ctx, region, tested, controls, subset=None):
    """Per-subject d = delta(tested) - mean delta(controls), slices with >=1 region patch only (§8).
    Returns d (n,) with NaN for subjects not used, n used, n with no region patch."""
    n, slice_subj, sign = ctx.n, ctx.slice_subj, ctx.sign
    use = counts[f"n_{region}"] > 0
    m_full, k = subject_mean(lg["full"], slice_subj, n, use)
    delta = lambda c: (subject_mean(lg[c], slice_subj, n, use)[0] - m_full) * sign   # noqa: E731
    dt = delta(tested).mean(0)
    dc = np.mean([delta(c).mean(0) for c in controls], axis=0)
    ok = k > 0 if subset is None else (k > 0) & subset
    return np.where(ok, dt - dc, np.nan), int(ok.sum()), int((k == 0).sum())


def lesion(key, lg, counts, ctx):
    """§7-8 for one model that passed the gate. lg {cond: (n_rep, n_slices)}.
    Returns condition AUC rows, confirmatory rows, exploratory rows, patch-count rows, and the
    per-subject d vectors of the confirmatory tests {(region, control): (n,) with NaN where unused}."""
    y, n, slice_subj = ctx.y, ctx.n, ctx.slice_subj
    auc_rows, conf, expl, cnt_rows, dvec = [], [], [], [], {}
    for c in lg:
        s, _ = subject_mean(lg[c], slice_subj, n)
        auc_rows.append(dict(model=key, condition=c, auc_mean=np.mean([roc_auc_score(y, v) for v in s])))

    cdr1 = ((ctx.cohort["cdr"] == 0) | (ctx.cohort["cdr"] >= 1)).to_numpy()
    for fam, label, tested, controls, ctrl, subset, cnt_region in test_list():
        dd, n_used, n_noregion = d_vs(lg, counts, ctx, cnt_region, tested, controls, cdr1 if subset else None)
        mean, lo, hi, p = boot_mean(dd[~np.isnan(dd)])
        row = dict(model=key, region=label, tested=tested, control=ctrl, n_subjects=n_used,
                   n_subjects_no_region_patch=n_noregion, mean_d=mean, ci_lo=lo, ci_hi=hi, p=p)
        (conf if fam == "confirmatory" else expl).append(row)
        if fam == "confirmatory":
            dvec[(label, ctrl)] = dd

    # region size by group (§4: reported, not tested)
    for kname, v in counts.items():
        for g in (0, 1):
            sel = ctx.y_slice == g
            cnt_rows.append(dict(model=key, count=kname, group=["non-demented", "dementia"][g],
                                 median=float(np.median(v[sel])), mean=float(v[sel].mean())))
    return auc_rows, conf, expl, cnt_rows, dvec


def write(out, gate_rows, vsb_rows, conf, expl, auc_rows, cnt_rows):
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(gate_rows).to_csv(os.path.join(out, "gate.csv"), index=False)
    pd.DataFrame(vsb_rows).to_csv(os.path.join(out, "vs_baseline.csv"), index=False)
    print(pd.DataFrame(gate_rows).round(3).to_string(index=False))
    print(pd.DataFrame(vsb_rows).round(3).to_string(index=False))
    if not conf:
        print("\nNo model passed the gate (AUC >= %.2f). Stop: no lesion analysis (DECISION_RULES.md §6)." % GATE_AUC)
        return
    for rows, name in ((conf, "confirmatory"), (expl, "exploratory")):
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p"])
        df["supported"] = (df["p_holm"] < 0.05) & (df["mean_d"] < 0)
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        print(f"\n{name} (Holm within family, n tests = {len(df)})")
        print(df.round(4).to_string(index=False))
    pd.DataFrame(auc_rows).to_csv(os.path.join(out, "condition_auc.csv"), index=False)
    pd.DataFrame(cnt_rows).to_csv(os.path.join(out, "region_patch_counts.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "results", "vit_features"))
    a = ap.parse_args()
    ctx = context()

    gate_rows, vsb_rows, auc_rows, cnt_rows, conf, expl = [], [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(a.features, "*.npz"))):
        key = os.path.basename(path)[:-4]
        feats, counts = load(path)
        print(key, "conditions:", len(feats), flush=True)
        lg = slice_logits(feats, ctx.y_slice, ctx.slice_subj, ctx.fold_of, ["full"])
        g, v, passed, _ = gate(key, lg["full"], ctx)
        gate_rows.append(g)
        vsb_rows.append(v)
        if not passed:
            continue
        lg = slice_logits(feats, ctx.y_slice, ctx.slice_subj, ctx.fold_of, list(feats))
        a_, c_, e_, n_, _ = lesion(key, lg, counts, ctx)
        auc_rows += a_
        conf += c_
        expl += e_
        cnt_rows += n_
    write(OUT, gate_rows, vsb_rows, conf, expl, auc_rows, cnt_rows)


if __name__ == "__main__":
    main()
