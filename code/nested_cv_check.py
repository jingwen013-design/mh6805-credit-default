"""
nested_cv_check.py — RUAN Hangjing's v2 second pass (nested CV + pre-registered selection rules), TRAINING SET ONLY.
Owner of the method: RUAN Hangjing (model_knn_v2.py / model_logistic_v2.py, 30 Sep 2026).  Assembled 2026-10-21 by
LI Jingwen from those two scripts, unchanged in logic, with the final test-set evaluation removed.

Why this file exists.  The v2 scripts end by scoring their selected configuration on the test set.  Both selections
coincide with configurations that were already frozen on 16 Sep (results/knn.json: K = 101, uniform weights;
results/logistic.json: plain, unpenalised), so re-running them would only re-score frozen models under new names —
exactly the "test set becomes a second validation set" drift the professor warned about.  What the second pass
contributes is the TRAINING-side evidence: does nested cross-validation (hyper-parameters chosen inside each outer
fold) agree with the plain 5-fold CV figure, and do the rules fixed in advance pick the frozen configuration?  That
evidence needs no test rows, so only that part runs here.  RUAN's own 30 Sep outputs (results/*_v2.*, received 21 Oct) are the
record; this script writes *_repro files so that they can be compared, never overwritten.  The test metrics of the frozen models are read from the
frozen result files; nothing new is scored.

Writes
  results/nested_cv_knn_repro.json              outer-fold choices and AUCs, final grid choice, agreement with the frozen model
  results/logistic_nested_cv_repro.csv       one row per logistic variant (nested AUC ± se, AP, calibration, 5:1 cut-off,
                                          cross-fitted cost) — the table behind the proposal's "second pass" paragraph
  results/logistic_threshold_vs_theory_repro.csv   cost-minimising cut-off on training OOF vs 1/(1+r) for r = 1 … 20
  results/nested_cv_summary_repro.json          the selection-rule outcomes and the comparison with the frozen choices
  figures/knn_cv_auc_vs_k_repro.png, figures/logistic_reliability_repro.png
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from common import CV, RESULTS, SCORING, SEED, load_split, make_preprocessor, plt, roc_auc_score, average_precision_score, save_extra_figure
from mine_v2_utils import (COST_RATIOS, best_threshold, calibration_summary, cross_fitted_cost, plot_reliability)

T0 = time.time()
X_train, _, y_train, _, feature_names = load_split()          # the test rows are not used anywhere in this script
ytr = y_train.values
folds = list(CV.split(X_train, ytr))
INNER = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)   # inner folds differ from the outer ones (as in v2)
CAL_CV = StratifiedKFold(5, shuffle=True, random_state=SEED + 2)

# ============================================================ 1. KNN: nested CV over K x weights (model_knn_v2.py, part 1-2)
K_GRID = [5, 11, 21, 31, 51, 75, 101, 151]
W_GRID = ["uniform", "distance"]
pipe = Pipeline([("pre", make_preprocessor()), ("clf", KNeighborsClassifier())])
grid = {"clf__n_neighbors": K_GRID, "clf__weights": W_GRID}
p_nested, outer_auc, outer_choice = np.zeros(len(ytr)), [], []
for k, (tr, va) in enumerate(folds):
    gs = GridSearchCV(pipe, grid, cv=INNER, scoring=SCORING, n_jobs=2)
    gs.fit(X_train.iloc[tr], ytr[tr])
    p_nested[va] = gs.predict_proba(X_train.iloc[va])[:, 1]
    outer_auc.append(float(roc_auc_score(ytr[va], p_nested[va]))); outer_choice.append(dict(gs.best_params_))
    print(f"KNN outer fold {k}: chose {gs.best_params_}  AUC {outer_auc[-1]:.4f}  [{time.time() - T0:.0f}s]")
gs_full = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, n_jobs=2).fit(X_train, ytr)
cvres = pd.DataFrame(gs_full.cv_results_)
best_k, best_w = int(gs_full.best_params_["clf__n_neighbors"]), gs_full.best_params_["clf__weights"]
frozen_knn = json.load(open(os.path.join(RESULTS, "knn.json")))
knn_same = (best_k == int(frozen_knn["best_params"]["n_neighbors"]) and best_w == frozen_knn["best_params"]["weights"])
print(f"KNN nested CV AUC {np.mean(outer_auc):.4f} ± {np.std(outer_auc):.4f}; full-training grid chose K={best_k}, {best_w} "
      f"(CV AUC {gs_full.best_score_:.4f}); frozen model is K={frozen_knn['best_params']['n_neighbors']}, "
      f"{frozen_knn['best_params']['weights']} -> {'SAME' if knn_same else 'DIFFERENT'}")

fig, ax = plt.subplots(figsize=(6, 4))
for w in W_GRID:
    sub = cvres[cvres["param_clf__weights"] == w]
    ax.errorbar(sub["param_clf__n_neighbors"].astype(int), sub["mean_test_score"], yerr=sub["std_test_score"],
                marker="o", capsize=3, label=w)
ax.set_xlabel("K (neighbours)"); ax.set_ylabel("5-fold CV AUC (training)"); ax.set_title("KNN: CV AUC vs K"); ax.legend(title="weights")
save_extra_figure(fig, "knn_cv_auc_vs_k_repro.png")

knn_out = dict(method="nested CV: outer = project 5-fold, inner = 5-fold (seed+1) grid over K x weights; training rows only",
               grid=dict(K=K_GRID, weights=W_GRID), outer_auc=outer_auc, outer_choices=outer_choice,
               nested_cv_auc=float(np.mean(outer_auc)), nested_cv_auc_std=float(np.std(outer_auc)),
               full_training_choice=dict(n_neighbors=best_k, weights=best_w, cv_auc=float(gs_full.best_score_),
                                         cv_auc_std=float(cvres.loc[gs_full.best_index_, "std_test_score"])),
               frozen_choice=dict(n_neighbors=int(frozen_knn["best_params"]["n_neighbors"]), weights=frozen_knn["best_params"]["weights"],
                                  cv_auc=frozen_knn["cv_auc"], test_auc=frozen_knn["test_auc"], first_scored="2026-09-16"),
               selection_agrees_with_frozen=bool(knn_same),
               oof_calibration=calibration_summary(ytr, p_nested))
json.dump(knn_out, open(os.path.join(RESULTS, "nested_cv_knn_repro.json"), "w"), indent=1)

# ============================================================ 2. logistic family (model_logistic_v2.py, parts 1-3)
C_GRID = list(np.round(np.logspace(-3, 2, 11), 5))


def lr_pipe(**kw):
    return Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=5000, **kw))])


def tuned(**kw):
    return GridSearchCV(lr_pipe(**kw), {"clf__C": C_GRID}, cv=INNER, scoring=SCORING, n_jobs=2)


VARIANTS = {
    "plain":        lr_pipe(C=np.inf),
    "ridge":        tuned(l1_ratio=0.0),
    "lasso":        tuned(l1_ratio=1.0, solver="liblinear"),
    "weighted":     tuned(l1_ratio=0.0, class_weight="balanced"),
    "weighted_cal": CalibratedClassifierCV(tuned(l1_ratio=0.0, class_weight="balanced"), method="sigmoid", cv=CAL_CV),
}
oof, rows = {}, []
for name, est in VARIANTS.items():
    t0 = time.time()
    p = cross_val_predict(est, X_train, ytr, cv=CV, method="predict_proba")[:, 1]
    oof[name] = p
    fold_auc = [roc_auc_score(ytr[va], p[va]) for _, va in folds]
    cal = calibration_summary(ytr, p)
    rows.append(dict(variant=name, nested_auc=float(np.mean(fold_auc)), nested_auc_se=float(np.std(fold_auc, ddof=1) / np.sqrt(len(folds))),
                     nested_ap=float(average_precision_score(ytr, p)), **cal,
                     t_cost5_oof=best_threshold(ytr, p, 5), t_theory5=round(1 / 6, 3),
                     crossfit_cost5=float(cross_fitted_cost(ytr, p, folds, 5)), seconds=round(time.time() - t0, 1)))
    print(f"logistic {name:<13} nested AUC {rows[-1]['nested_auc']:.4f}  AP {rows[-1]['nested_ap']:.4f}  Brier {cal['brier']:.4f}  "
          f"mean p {cal['mean_pred']:.3f}  t*(5:1) {rows[-1]['t_cost5_oof']:.2f}  cross-fitted cost {rows[-1]['crossfit_cost5']:.0f}  [{time.time() - T0:.0f}s]")
nested = pd.DataFrame(rows).set_index("variant")
nested.round(4).to_csv(os.path.join(RESULTS, "logistic_nested_cv_repro.csv"))

# the rules fixed in advance (v2): 1-SE on nested AUC among plain > ridge > lasso; weighting only if the Platt-calibrated
# weighted model's cross-fitted 5:1 cost is more than 1 % lower
rank_cands = ["plain", "ridge", "lasso"]
best_auc = nested.loc[rank_cands, "nested_auc"].max()
se = nested.loc[nested.loc[rank_cands, "nested_auc"].idxmax(), "nested_auc_se"]
step1 = next(v for v in rank_cands if nested.loc[v, "nested_auc"] >= best_auc - se)
gain = 1 - nested.loc["weighted_cal", "crossfit_cost5"] / nested.loc[step1, "crossfit_cost5"]
selected = "weighted_cal" if gain > 0.01 else step1
frozen_lr = json.load(open(os.path.join(RESULTS, "logistic.json")))
lr_same = (selected == "plain")
print(f"\nrule step 1 -> {step1}; weighted_cal cross-fitted cost change {gain:+.2%} -> selected {selected}; "
      f"frozen baseline is the plain model -> {'SAME' if lr_same else 'DIFFERENT'}")

plot_reliability({"plain": (ytr, oof["plain"]), "class-weighted": (ytr, oof["weighted"]),
                  "class-weighted + sigmoid calibration": (ytr, oof["weighted_cal"])},
                 "logistic_reliability_repro.png", "Logistic family: reliability (nested out-of-fold, training)")
thr = pd.DataFrame([dict(ratio=r, theory=round(1 / (1 + r), 3), **{v: best_threshold(ytr, oof[v], r) for v in
                   ["plain", "weighted", "weighted_cal"]}) for r in COST_RATIOS])
thr.to_csv(os.path.join(RESULTS, "logistic_threshold_vs_theory_repro.csv"), index=False)
print("\ncost-minimising cut-off on training OOF vs theory 1/(1+r):\n" + thr.to_string(index=False))

summary = dict(
    generated_by="code/nested_cv_check.py", method_owner="RUAN Hangjing (v2, 30 Sep 2026)", assembled_by="LI Jingwen, 2026-10-21",
    uses_test_set=False,
    knn=dict(nested_cv_auc=knn_out["nested_cv_auc"], nested_cv_auc_std=knn_out["nested_cv_auc_std"],
             full_training_choice=knn_out["full_training_choice"], frozen_choice=knn_out["frozen_choice"],
             agrees_with_frozen=knn_out["selection_agrees_with_frozen"]),
    logistic=dict(rule="1-SE on nested AUC among plain > ridge > lasso; adopt Platt-calibrated class weighting only if its "
                       "cross-fitted 5:1 cost is > 1% lower", step1=step1, weighted_cal_cost_change=float(gain), selected=selected,
                  agrees_with_frozen=bool(lr_same),
                  frozen_baseline=dict(cv_auc=frozen_lr["cv_auc"], oof_auc=frozen_lr.get("oof_auc"), test_auc=frozen_lr["test_auc"],
                                       cost_threshold=frozen_lr["at_tuned_cost"]["threshold"], first_scored="2026-09-16"),
                  nested=nested.round(5).to_dict(orient="index")),
    reading=("Nested CV reproduces the plain 5-fold figures to within one standard error and the pre-registered rules pick "
             "the configurations frozen on 16 September, so the first-round tuning did not flatter the reported CV AUCs; "
             "the frozen test metrics stand and nothing was re-scored."),
    seconds=round(time.time() - T0, 1))
json.dump(summary, open(os.path.join(RESULTS, "nested_cv_summary_repro.json"), "w"), indent=1)
print(f"\nwrote results/nested_cv_knn_repro.json, results/logistic_nested_cv_repro.csv, results/logistic_threshold_vs_theory_repro.csv, "
      f"results/nested_cv_summary_repro.json  [{time.time() - T0:.0f}s]")
