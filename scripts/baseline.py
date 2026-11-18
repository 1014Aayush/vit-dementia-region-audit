"""Baseline (DECISION_RULES.md §6): logistic regression on SynthSeg hippocampal volume + nWBV + age,
and age alone. Same folds as the ViT. Also writes the fold table the ViT notebook uses.

    python scripts/baseline.py
Outputs: results/folds.csv, results/cohort.csv, results/baseline/{summary,scores}.csv
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, load_cohort, make_folds, cv_scores, summarize

OUT = os.path.join(ROOT, "results", "baseline")
os.makedirs(OUT, exist_ok=True)

cohort = load_cohort()
vols = pd.read_csv(os.path.join(ROOT, "data", "masks", "roi_volumes.csv"))
cohort = cohort.merge(vols[["sid", "hippocampus", "mtl", "ventricles"]], on="sid", how="left")
assert cohort["hippocampus"].notna().all() and cohort["nwbv"].notna().all()
cohort.to_csv(os.path.join(ROOT, "results", "cohort.csv"), index=False)

folds = make_folds(cohort)
folds.to_csv(os.path.join(ROOT, "results", "folds.csv"), index=False)
fold_of = folds.pivot(index="repeat", columns="sid", values="fold")[cohort["sid"]].to_numpy()

y = cohort["y"].to_numpy()
print("cohort: %d subjects, %d non-demented, %d dementia" % (len(y), (y == 0).sum(), (y == 1).sum()))

models = {"age_only": ["age"], "hippo_nwbv_age": ["hippocampus", "nwbv", "age"]}
summ, scores = [], {"sid": cohort["sid"], "y": y}
for name, cols in models.items():
    s = cv_scores(cohort[cols].to_numpy(float), y, fold_of)
    summ.append(dict(model=name, features="+".join(cols), **summarize(y, s)))
    scores[name] = s.mean(0)
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
pd.DataFrame(scores).to_csv(os.path.join(OUT, "scores.csv"), index=False)
print(summ.round(3).to_string(index=False))
