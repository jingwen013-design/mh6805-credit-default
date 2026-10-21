"""
export_folds.py — write the five training CV folds used by every Python model to data/processed/cv_folds.csv,
so that the R cross-check (code/r_crosscheck/) fits its models on exactly the same folds.

Run from Project/code:  python3 export_folds.py

It reads data/processed/train.csv ONLY (the test file is never opened) and applies common.CV
(StratifiedKFold(n_splits=5, shuffle=True, random_state=2607526)) to the 20,975 training rows in file order — the same call
GridSearchCV / cross_val_score / cross_val_predict make inside the model scripts. StratifiedKFold uses only the number of
rows and the labels, so the assignment is identical to the one the Python models were tuned and scored on.
Output columns: row (1-based row number in train.csv, R convention), row_index (0-based, pandas/numpy), fold (1..5),
DEFAULT (label, so the R side can check it is reading the same rows).
"""
import os
import numpy as np
import pandas as pd
from common import CV, PROC, TARGET, SEED

tr = pd.read_csv(os.path.join(PROC, "train.csv"))              # training rows only
y = tr[TARGET].astype(int).values
assert len(y) == 20975, f"expected 20,975 training rows, found {len(y):,}"

fold = np.zeros(len(y), dtype=int)
for k, (_, va) in enumerate(CV.split(np.zeros((len(y), 1)), y), start=1):
    fold[va] = k
assert (fold > 0).all()

out = pd.DataFrame(dict(row=np.arange(1, len(y) + 1), row_index=np.arange(len(y)), fold=fold, DEFAULT=y))
path = os.path.join(PROC, "cv_folds.csv")
out.to_csv(path, index=False)

summary = out.groupby("fold")["DEFAULT"].agg(n="size", defaults="sum", default_rate="mean")
print(f"StratifiedKFold(5, shuffle=True, random_state={SEED}) on {len(y):,} training rows -> {path}")
print(summary.round(4).to_string())
