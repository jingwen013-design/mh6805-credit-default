"""
model_discriminant.py — generative classifiers from lecture B6 (self-study part).
Owner: LI Jingwen.   Run from Project/code:  python3 model_discriminant.py

Four final models, every one a sklearn Pipeline(make_preprocessor(), classifier) tuned with GridSearchCV(cv=CV, scoring=SCORING):
  lda             Linear Discriminant Analysis          (shrinkage tuned: None / 'auto' / 0.05 / 0.2 / 0.5, solver='lsqr')
  qda             Quadratic Discriminant Analysis       (reg_param tuned over {0, 0.01, 0.1, 0.5})
  naive_bayes     Gaussian naive Bayes                  (var_smoothing tuned over np.logspace(-9, -1, 9))
  naive_bayes_log Gaussian naive Bayes on signed-log1p amount columns (same var_smoothing grid) — does the transform help?

Rules from common.py / README honoured: load_split() only, preprocessor inside the Pipeline (refitted in every fold),
seed 2607526 everywhere (CV object), n_jobs=2, evaluate() called exactly once per final model, nothing fitted or tuned on the test set.
A plain logistic regression is fitted ONCE as a diagnostic reference for the LDA-vs-logistic comparison (it is another member's
model, so evaluate() is NOT called on it; only its coefficient direction and test AUC are recorded in the LDA extra block).
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from common import *

MEMBER = "LI Jingwen"
N_JOBS = 2
AMT_COLS = ["LIMIT_BAL"] + BILL_COLS + PAYAMT_COLS          # the 13 money columns (NT$), heavily right-skewed


# ----------------------------------------------------------------------------- helpers
def signed_log1p_amounts(X):
    """sign(x)*log1p(|x|) on the 13 amount columns (BILL_AMT can be negative = credit balance). Returns a DataFrame."""
    X = X.copy()
    X[AMT_COLS] = np.sign(X[AMT_COLS]) * np.log1p(np.abs(X[AMT_COLS]))
    return X


def cv_table(gs, param):
    """Tidy CV table for one hyper-parameter from a fitted GridSearchCV."""
    cvr = gs.cv_results_
    rows = []
    for p, m, s, ft in zip(cvr["params"], cvr["mean_test_score"], cvr["std_test_score"], cvr["mean_fit_time"]):
        rows.append(dict(param=param, value=str(p[param]), cv_auc=round(float(m), 5), cv_auc_std=round(float(s), 5),
                         mean_fit_time=round(float(ft), 3)))
    return rows


def tune(pipe, grid, X, y):
    gs = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, n_jobs=N_JOBS, refit=True)
    t0 = time.time(); gs.fit(X, y); secs = round(time.time() - t0, 2)
    return gs, secs


def oof_proba(model, X, y):
    return cross_val_predict(clone(model), X, y, cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]


# ----------------------------------------------------------------------------- data
X_train, X_test, y_train, y_test, feature_names = load_split()
print(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")

skew_raw = X_train[AMT_COLS].skew().round(3).to_dict()
skew_log = signed_log1p_amounts(X_train)[AMT_COLS].skew().round(3).to_dict()
corr_bill = X_train[BILL_COLS].corr().values; mean_corr_bill = float((corr_bill.sum() - 6) / 30)
corr_pay = X_train[PAY_COLS].corr().values; mean_corr_pay = float((corr_pay.sum() - 6) / 30)
print(f"mean pairwise corr BILL_AMT1..6 = {mean_corr_bill:.3f}, PAY_1..6 = {mean_corr_pay:.3f}")

results = {}

# ============================================================================= 1. LDA
print("\n=== LDA ===")
pipe_lda = Pipeline([("pre", make_preprocessor()), ("clf", LinearDiscriminantAnalysis(solver="lsqr"))])
grid_lda = {"clf__shrinkage": [None, "auto", 0.05, 0.2, 0.5]}
gs_lda, secs_lda = tune(pipe_lda, grid_lda, X_train, y_train)
print("LDA grid:", pd.DataFrame(cv_table(gs_lda, "clf__shrinkage")).to_string(index=False))
lda = gs_lda.best_estimator_
p_oof_lda = oof_proba(lda, X_train, y_train)

# --- diagnostic reference: plain logistic regression in the SAME transformed feature space (not evaluated, not a final model here)
pipe_ref = Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=5000, random_state=SEED))])
pipe_ref.fit(X_train, y_train)
w_lda = lda.named_steps["clf"].coef_.ravel(); w_log = pipe_ref.named_steps["clf"].coef_.ravel()
cos_sim = float(w_lda @ w_log / (np.linalg.norm(w_lda) * np.linalg.norm(w_log)))
p_test_lda = lda.predict_proba(X_test)[:, 1]; p_test_ref = pipe_ref.predict_proba(X_test)[:, 1]
rho = float(spearmanr(p_test_lda, p_test_ref).correlation)
ref_auc = float(roc_auc_score(y_test, p_test_ref)); ref_ap = float(average_precision_score(y_test, p_test_ref))
names_out = feature_names_out(lda.named_steps["pre"])
coef_tab = pd.DataFrame({"feature": names_out, "lda_coef": w_lda, "logit_coef": w_log})
coef_tab["abs_lda"] = coef_tab["lda_coef"].abs(); coef_tab = coef_tab.sort_values("abs_lda", ascending=False).drop(columns="abs_lda")
print(f"LDA vs logistic reference: cosine(coef) = {cos_sim:.3f}, Spearman(test scores) = {rho:.3f}, ref test AUC = {ref_auc:.4f}, AP = {ref_ap:.4f}")
print(coef_tab.head(10).round(3).to_string(index=False))

results["lda"] = evaluate(
    "LDA (linear discriminant analysis)", lda, X_train, y_train, X_test, y_test, member=MEMBER, slug="lda",
    best_params={"solver": "lsqr", "shrinkage": gs_lda.best_params_["clf__shrinkage"]},
    cv_auc=float(gs_lda.best_score_), cv_auc_std=float(gs_lda.cv_results_["std_test_score"][gs_lda.best_index_]),
    fit_seconds=secs_lda, p_oof=p_oof_lda,
    notes="Pipeline(StandardScaler+OneHot, LDA solver=lsqr). Shrinkage tuned by 5-fold CV AUC over {None, auto, 0.05, 0.2, 0.5}; "
          "fit_seconds = whole GridSearchCV wall time (n_jobs=2). Generative twin of logistic regression: same linear "
          "log-odds form, parameters estimated from class-conditional Gaussians with a pooled covariance instead of by maximum "
          "conditional likelihood. A plain logistic regression was fitted once as a diagnostic reference (cosine similarity of "
          "coefficient vectors and Spearman rank correlation of test scores are stored in extra); evaluate() was not called on it.",
    extra=dict(cv_grid=cv_table(gs_lda, "clf__shrinkage"), n_features_transformed=int(len(names_out)),
               logistic_reference=dict(cosine_similarity_coef=round(cos_sim, 4), spearman_test_scores=round(rho, 4),
                                       ref_test_auc=round(ref_auc, 4), ref_test_ap=round(ref_ap, 4)),
               top10_coefficients=coef_tab.head(10).round(4).to_dict(orient="records")))

# ============================================================================= 2. QDA
print("\n=== QDA ===")
pipe_qda = Pipeline([("pre", make_preprocessor()), ("clf", QuadraticDiscriminantAnalysis())])
grid_qda = {"clf__reg_param": [0.0, 0.01, 0.1, 0.5]}
gs_qda, secs_qda = tune(pipe_qda, grid_qda, X_train, y_train)
print("QDA grid:", pd.DataFrame(cv_table(gs_qda, "clf__reg_param")).to_string(index=False))
qda = gs_qda.best_estimator_
p_oof_qda = oof_proba(qda, X_train, y_train)
results["qda"] = evaluate(
    "QDA (quadratic discriminant analysis)", qda, X_train, y_train, X_test, y_test, member=MEMBER, slug="qda",
    best_params={"reg_param": gs_qda.best_params_["clf__reg_param"]},
    cv_auc=float(gs_qda.best_score_), cv_auc_std=float(gs_qda.cv_results_["std_test_score"][gs_qda.best_index_]),
    fit_seconds=secs_qda, p_oof=p_oof_qda,
    notes="Pipeline(StandardScaler+OneHot, QDA). reg_param tuned by 5-fold CV AUC over {0, 0.01, 0.1, 0.5} "
          "(reg_param shrinks each class covariance towards the identity: S_k <- (1-r) S_k + r I). fit_seconds = whole GridSearchCV "
          "wall time. One full covariance per class (32x32) is estimated on 32 transformed columns that include 6 binary dummies and "
          "13 heavily skewed amount columns, so the Gaussian assumption is badly violated.",
    extra=dict(cv_grid=cv_table(gs_qda, "clf__reg_param"), n_features_transformed=int(len(names_out)),
               amount_skewness_train=skew_raw, mean_pairwise_corr_bill_amt=round(mean_corr_bill, 4)))

# ============================================================================= 3. Gaussian naive Bayes
print("\n=== Gaussian naive Bayes ===")
vs_grid = np.logspace(-9, -1, 9)
pipe_nb = Pipeline([("pre", make_preprocessor()), ("clf", GaussianNB())])
grid_nb = {"clf__var_smoothing": vs_grid}
gs_nb, secs_nb = tune(pipe_nb, grid_nb, X_train, y_train)
print("NB grid:", pd.DataFrame(cv_table(gs_nb, "clf__var_smoothing")).to_string(index=False))
nb = gs_nb.best_estimator_
p_oof_nb = oof_proba(nb, X_train, y_train)
results["naive_bayes"] = evaluate(
    "Gaussian naive Bayes", nb, X_train, y_train, X_test, y_test, member=MEMBER, slug="naive_bayes",
    best_params={"var_smoothing": float(gs_nb.best_params_["clf__var_smoothing"])},
    cv_auc=float(gs_nb.best_score_), cv_auc_std=float(gs_nb.cv_results_["std_test_score"][gs_nb.best_index_]),
    fit_seconds=secs_nb, p_oof=p_oof_nb,
    notes="Pipeline(StandardScaler+OneHot, GaussianNB). var_smoothing tuned by 5-fold CV AUC over np.logspace(-9,-1,9) "
          "(adds var_smoothing x max feature variance to every per-class variance). fit_seconds = whole GridSearchCV wall time. "
          "Assumes features are conditionally independent given the class and each is Gaussian within class; here BILL_AMT1..6 have "
          "mean pairwise correlation 0.89 and PAY_AMT skewness 10-33, so both assumptions fail and probabilities are extreme.",
    extra=dict(cv_grid=cv_table(gs_nb, "clf__var_smoothing"), n_features_transformed=int(len(names_out)),
               amount_skewness_train=skew_raw, mean_pairwise_corr_bill_amt=round(mean_corr_bill, 4),
               mean_pairwise_corr_pay_status=round(mean_corr_pay, 4),
               share_p_test_extreme=float(np.mean((nb.predict_proba(X_test)[:, 1] < 0.01) | (nb.predict_proba(X_test)[:, 1] > 0.99)))))

# ============================================================================= 4. Gaussian naive Bayes on log1p amounts
print("\n=== Gaussian naive Bayes, signed log1p amounts ===")
pipe_nbl = Pipeline([("log", FunctionTransformer(signed_log1p_amounts, validate=False)),
                     ("pre", make_preprocessor()), ("clf", GaussianNB())])
gs_nbl, secs_nbl = tune(pipe_nbl, grid_nb, X_train, y_train)
print("NB-log grid:", pd.DataFrame(cv_table(gs_nbl, "clf__var_smoothing")).to_string(index=False))
nbl = gs_nbl.best_estimator_
p_oof_nbl = oof_proba(nbl, X_train, y_train)
results["naive_bayes_log"] = evaluate(
    "Gaussian naive Bayes (log1p amounts)", nbl, X_train, y_train, X_test, y_test, member=MEMBER, slug="naive_bayes_log",
    best_params={"var_smoothing": float(gs_nbl.best_params_["clf__var_smoothing"]), "amount_transform": "sign(x)*log1p(|x|)"},
    cv_auc=float(gs_nbl.best_score_), cv_auc_std=float(gs_nbl.cv_results_["std_test_score"][gs_nbl.best_index_]),
    fit_seconds=secs_nbl, p_oof=p_oof_nbl,
    notes="Same as naive_bayes but with sign(x)*log1p(|x|) applied to the 13 amount columns (LIMIT_BAL, BILL_AMT1..6, PAY_AMT1..6) "
          "before scaling, inside the Pipeline (FunctionTransformer). Signed version because BILL_AMT can be negative. The transform "
          "makes the within-class marginals closer to Gaussian (skewness of PAY_AMT drops from 10-33 to about -1) but leaves the "
          "strong dependence between the six monthly bills untouched.",
    extra=dict(cv_grid=cv_table(gs_nbl, "clf__var_smoothing"), amount_skewness_train_raw=skew_raw,
               amount_skewness_train_log1p=skew_log, transformed_columns=AMT_COLS,
               share_p_test_extreme=float(np.mean((nbl.predict_proba(X_test)[:, 1] < 0.01) | (nbl.predict_proba(X_test)[:, 1] > 0.99)))))

# ============================================================================= extra figures
# (a) ROC + PR overlay of the four generative models with the logistic reference dashed
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
cols = {"lda": "C3", "qda": "C1", "naive_bayes": "C2", "naive_bayes_log": "C0"}
for slug, r in results.items():
    ax[0].plot(r["roc"]["fpr"], r["roc"]["tpr"], color=cols[slug], lw=1.8, label=f"{slug}  AUC={r['test_auc']:.3f}")
    ax[1].plot(r["pr"]["recall"], r["pr"]["precision"], color=cols[slug], lw=1.8, label=f"{slug}  AP={r['test_ap']:.3f}")
fpr_r, tpr_r, _ = roc_curve(y_test, p_test_ref); pr_r, rc_r, _ = precision_recall_curve(y_test, p_test_ref)
ax[0].plot(fpr_r, tpr_r, "k--", lw=1, label=f"logistic ref  AUC={ref_auc:.3f}"); ax[0].plot([0, 1], [0, 1], color="grey", lw=0.6)
ax[1].plot(rc_r, pr_r, "k--", lw=1, label=f"logistic ref  AP={ref_ap:.3f}"); ax[1].axhline(y_test.mean(), color="grey", lw=0.6, ls=":")
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC (test) — generative classifiers"); ax[0].legend(loc="lower right", fontsize=8)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision–recall (test)"); ax[1].legend(loc="upper right", fontsize=8)
save_extra_figure(fig, "discriminant_roc_pr.png")

# (b) CV AUC curves for the tuned hyper-parameters
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
t = pd.DataFrame(cv_table(gs_lda, "clf__shrinkage")); ax[0].bar(t["value"], t["cv_auc"], yerr=t["cv_auc_std"], color="C3", alpha=.8)
ax[0].set_ylim(t["cv_auc"].min() - 0.01, t["cv_auc"].max() + 0.01); ax[0].set_title("LDA: CV AUC vs shrinkage"); ax[0].set_xlabel("shrinkage")
t = pd.DataFrame(cv_table(gs_qda, "clf__reg_param")); ax[1].errorbar(t["value"].astype(float), t["cv_auc"], yerr=t["cv_auc_std"], marker="o", color="C1")
ax[1].set_xscale("symlog", linthresh=0.01); ax[1].set_title("QDA: CV AUC vs reg_param"); ax[1].set_xlabel("reg_param")
for gs_, lab, c in ((gs_nb, "raw amounts", "C2"), (gs_nbl, "signed log1p amounts", "C0")):
    t = pd.DataFrame(cv_table(gs_, "clf__var_smoothing")); ax[2].errorbar(t["value"].astype(float), t["cv_auc"], yerr=t["cv_auc_std"], marker="o", color=c, label=lab)
ax[2].set_xscale("log"); ax[2].set_title("Gaussian NB: CV AUC vs var_smoothing"); ax[2].set_xlabel("var_smoothing"); ax[2].legend(fontsize=8)
for a in ax: a.set_ylabel("CV AUC (5-fold)"); a.grid(alpha=.3)
save_extra_figure(fig, "discriminant_cv_curves.png")

# (c) why the Gaussian assumption fails: BILL_AMT1 / PAY_AMT1 by class, raw vs signed log1p (training set only)
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
Xl = signed_log1p_amounts(X_train)
for j, col in enumerate(["BILL_AMT1", "PAY_AMT1"]):
    for i, (df, lab) in enumerate(((X_train, "raw"), (Xl, "signed log1p"))):
        for k, c in ((0, "C0"), (1, "C3")):
            v = df.loc[y_train == k, col]
            ax[i, j].hist(v, bins=80, density=True, alpha=.45, color=c, label=f"DEFAULT={k} (skew {v.skew():.1f})")
        ax[i, j].set_title(f"{col} — {lab} (train)"); ax[i, j].legend(fontsize=8)
        if lab == "raw" and col == "PAY_AMT1": ax[i, j].set_xlim(0, 60000)
        if lab == "raw" and col == "BILL_AMT1": ax[i, j].set_xlim(-20000, 400000)
save_extra_figure(fig, "discriminant_amount_skew.png")

# ============================================================================= summary
print("\n=== summary (test set, thresholds tuned on training OOF) ===")
rows = []
for slug, r in results.items():
    rows.append(dict(model=slug, cv_auc=round(r["cv_auc"], 4), test_auc=round(r["test_auc"], 4), test_ap=round(r["test_ap"], 4),
                     brier=round(r["test_brier"], 4), acc_0_5=round(r["at_0_5"]["accuracy"], 4), rec_0_5=round(r["at_0_5"]["recall_1"], 4),
                     thr_f1=r["at_tuned_f1"]["threshold"], rec_f1=round(r["at_tuned_f1"]["recall_1"], 4),
                     prec_f1=round(r["at_tuned_f1"]["precision_1"], 4), f1=round(r["at_tuned_f1"]["f1_1"], 4),
                     fit_s=r["fit_seconds"], best=r["best_params"]))
print(pd.DataFrame(rows).to_string(index=False))
print(f"logistic reference (diagnostic only): test AUC {ref_auc:.4f}, AP {ref_ap:.4f}; README baseline 0.764 / 0.524")
print("done.")
