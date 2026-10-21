"""
rf_drop_column.py — what the forest actually loses when a column is REMOVED.   Owner: LU Xinyu.

Permutation importance answers "how much does this fitted forest lean on this column as it
stands".  It does NOT answer "what would we lose without this column", and on correlated
predictors the two differ by an order of magnitude.  This script answers the second question
the only way it can be answered: refit the identical forest with the column dropped and
compare Breiman's out-of-bag AUC on the training rows.

Nothing here touches the test set.  Until 2026-10-19 the script also scored all seven forests on
the test rows; those columns were never quoted, but a diagnostic has no business looking at the
test set, so they were removed (the out-of-bag numbers are unchanged).

Writes results/rf_refit_drop_column.csv — one row per column tested, plus the full model.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from common import CAT_COLS, NUM_COLS, RESULTS, SEED, load_split

T0 = time.time()
X_train, _, y_train, _, feature_names = load_split()          # test rows deliberately unused
bp = json.load(open(os.path.join(RESULTS, "random_forest.json")))["best_params"]
print(f"forest: {bp}")


def pre_for(cols):
    """make_preprocessor() hardcodes the full column lists, so a dropped column has to be
    removed from them too; everything else (no scaling for trees, drop-first one-hot) is the
    same transformer the rest of the project uses."""
    num = [c for c in NUM_COLS if c in cols]
    cat = [c for c in CAT_COLS if c in cols]
    return ColumnTransformer([("num", "passthrough", num),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)],
                             remainder="drop")


def fit(cols):
    pipe = Pipeline([("pre", pre_for(cols)),
                     ("clf", RandomForestClassifier(
                         n_estimators=bp.get("n_estimators", 500),
                         max_features=bp.get("max_features", "sqrt"),
                         min_samples_leaf=bp.get("min_samples_leaf", 20),
                         criterion=bp.get("criterion", "gini"),
                         bootstrap=True, oob_score=False, n_jobs=2, random_state=SEED))])
    pipe.fit(X_train[cols], y_train)
    clf = pipe.named_steps["clf"]
    # out-of-bag score, recomputed rather than trusted: sklearn's oob_decision_function_
    # needs oob_score=True, which we avoid so the two paths cannot silently diverge.
    Xt = pipe.named_steps["pre"].transform(X_train[cols])
    Xt = np.asarray(Xt.todense()) if hasattr(Xt, "todense") else np.asarray(Xt)
    n = Xt.shape[0]
    votes, count = np.zeros(n), np.zeros(n)
    for est, samp in zip(clf.estimators_, clf.estimators_samples_):
        oob = np.ones(n, bool); oob[np.unique(samp)] = False
        if oob.sum():
            votes[oob] += est.predict_proba(Xt[oob])[:, 1]; count[oob] += 1
    m = count > 0
    return float(roc_auc_score(y_train[m], votes[m] / count[m]))


full_oob = fit(list(feature_names))
print(f"full forest: OOB {full_oob:.6f}  [{time.time()-T0:.0f}s]")

imp = pd.read_csv(os.path.join(RESULTS, "rf_importance.csv"))
targets = imp.sort_values("oob_perm_auc_drop", ascending=False).head(6)["feature"].tolist()

rows = [dict(feature="(none: full model)", oob_auc=full_oob, oob_loss=0.0, oob_perm_drop=np.nan)]
for f in targets:
    cols = [c for c in feature_names if c != f]
    o = fit(cols)
    pd_drop = float(imp.loc[imp.feature == f, "oob_perm_auc_drop"].iloc[0])
    rows.append(dict(feature=f, oob_auc=o, oob_loss=full_oob - o, oob_perm_drop=pd_drop))
    print(f"  without {f:18s} OOB {o:.6f} (loss {full_oob-o:+.5f})   permutation said {pd_drop:.5f}  "
          f"[{time.time()-T0:.0f}s]")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "rf_refit_drop_column.csv"), index=False)
print(f"\nresults/rf_refit_drop_column.csv written in {time.time()-T0:.0f}s")
print(out.round(5).to_string(index=False))
