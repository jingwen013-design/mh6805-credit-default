"""
model_logistic.py — Logistic-regression family for the MH6805 credit-default project.  Owner: RUAN Hangjing.

Four final models, each evaluated exactly once with common.evaluate():
  (a) logistic           plain (unpenalised, C=inf) logistic regression on all 29 features
                         + coefficient table (standardised coef, odds ratio, statsmodels p-value) -> results/logistic_coefficients.csv
                         + figures/logistic_coefficients.png  (15 largest |coef|)
  (b) logistic_l2        ridge logistic, C tuned on np.logspace(-3, 2, 11) with GridSearchCV (5-fold CV AUC)
  (c) logistic_l1        lasso logistic (solver='saga', max_iter=5000), same C grid; number of exactly-zero coefficients
                         + figures/logistic_l1_path.png  (regularisation path + CV AUC vs C)
  (d) logistic_balanced  plain logistic with class_weight='balanced'

Contract (code/README.md): load_split() only, make_preprocessor() INSIDE a Pipeline (refitted in every CV fold),
cv=CV, scoring=SCORING, seed 2607526 everywhere, n_jobs<=2, nothing fitted or tuned on the test set.

sklearn 1.9 note: `penalty=` is deprecated (removed in 1.10).  Warning-free equivalents used here:
    penalty=None -> C=np.inf        penalty='l2' -> l1_ratio=0.0        penalty='l1' -> l1_ratio=1.0 (solver='saga')
Run:  python3 model_logistic.py     (from Project/code; ~2-4 min on a laptop)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")                       # keep BLAS threads modest: several scripts run concurrently
import time
import numpy as np
import pandas as pd
from common import *                                     # load_split, make_preprocessor, CV, SCORING, SEED, evaluate, ...
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.patches import Patch
import statsmodels.api as sm

MEMBER = "RUAN Hangjing"
N_JOBS = 2
MAX_ITER = 5000
LBFGS_TOL = 1e-8                                         # tight tolerance so the unpenalised lbfgs fit equals the statsmodels MLE
                                                         # (the default 1e-4 stops ~0.02 short on the collinear utilisation/bill block)
C_GRID = np.logspace(-3, 2, 11)                          # tuning grid for (b) and (c)
PATH_GRID = np.logspace(-3, 2, 21)                       # denser grid for the L1 path figure (contains every C_GRID value)

# colours: validated reference palette (diverging pair red/blue for sign, first 6 categorical slots for the path)
RED, BLUE, GREY, INK, INK2, GRID = "#e34948", "#2a78d6", "#c9c8c3", "#0b0b0b", "#52514e", "#e6e5e1"
CAT6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# readable labels for the one-hot columns (OneHotEncoder(drop='first') -> reference level = first category)
LABELS = {"SEX_2": "SEX = female (vs male)",
          "EDUCATION_2": "EDUCATION = university (vs grad school)", "EDUCATION_3": "EDUCATION = high school (vs grad school)",
          "EDUCATION_4": "EDUCATION = others (vs grad school)",
          "MARRIAGE_2": "MARRIAGE = single (vs married)", "MARRIAGE_3": "MARRIAGE = others (vs married)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pipe(clf):
    """Preprocessor INSIDE the pipeline so scaling / one-hot are refitted in every CV fold."""
    return Pipeline([("pre", make_preprocessor()), ("clf", clf)])


def oof_probs(model, X, y):
    """Training-set out-of-fold probabilities (threshold tuning in evaluate() uses these, never the test set)."""
    return cross_val_predict(clone(model), X, y, cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]


def fold_auc(model, X, y):
    s = cross_val_score(clone(model), X, y, cv=CV, scoring=SCORING, n_jobs=N_JOBS)
    return float(s.mean()), float(s.std())


def clean_names(names):
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def coef_of(pipeline):
    return pipeline.named_steps["clf"].coef_[0]


def main():
    t_all = time.time()
    X_train, X_test, y_train, y_test, feature_names = load_split()
    log(f"train {X_train.shape}, test {X_test.shape}; default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")
    summary = []

    # ============================================================== (a) plain logistic regression
    log("(a) plain logistic regression: unpenalised MLE (C=inf), all 29 features")
    plain = pipe(LogisticRegression(C=np.inf, solver="lbfgs", max_iter=MAX_ITER, tol=LBFGS_TOL, random_state=SEED))
    t0 = time.time(); plain.fit(X_train, y_train); fit_a = time.time() - t0
    cv_a, cvsd_a = fold_auc(plain, X_train, y_train)
    p_oof_a = oof_probs(plain, X_train, y_train)
    pre, clf = plain.named_steps["pre"], plain.named_steps["clf"]
    names = clean_names(feature_names_out(pre))
    Xt = dense(pre.transform(X_train))
    coef = coef_of(plain)
    log(f"    fitted in {fit_a:.1f}s, lbfgs iterations = {int(np.ravel(clf.n_iter_)[0])}, {Xt.shape[1]} transformed columns, CV AUC {cv_a:.4f} ± {cvsd_a:.4f}")

    # --- statsmodels Logit on the SAME transformed training matrix -> standard errors and p-values
    sm_X = sm.add_constant(Xt, has_constant="add")
    try:
        sm_res = sm.Logit(y_train.values, sm_X).fit(method="newton", maxiter=200, disp=0)
    except Exception as e:                                # singular Hessian etc. -> quasi-Newton fallback
        log(f"    statsmodels newton failed ({e}); retrying with bfgs")
        sm_res = sm.Logit(y_train.values, sm_X).fit(method="bfgs", maxiter=5000, disp=0)
    sm_params, sm_bse, sm_p = (np.asarray(v) for v in (sm_res.params, sm_res.bse, sm_res.pvalues))
    max_diff = float(np.max(np.abs(np.r_[clf.intercept_, coef] - sm_params)))
    sd = dict(zip(NUM_COLS, pre.named_transformers_["num"].scale_))       # training SD of each numeric feature

    rows = [dict(feature="(intercept)", label="(intercept)", type="intercept", coef=float(clf.intercept_[0]),
                 odds_ratio=float(np.exp(clf.intercept_[0])), sm_coef=float(sm_params[0]), std_err=float(sm_bse[0]),
                 z=float(sm_params[0] / sm_bse[0]), p_value=float(sm_p[0]), sd_train=np.nan, coef_per_raw_unit=np.nan)]
    for j, n in enumerate(names):
        is_num = n in sd
        rows.append(dict(feature=n, label=LABELS.get(n, n), type="numeric (per +1 SD)" if is_num else "one-hot (vs reference level)",
                         coef=float(coef[j]), odds_ratio=float(np.exp(coef[j])), sm_coef=float(sm_params[j + 1]),
                         std_err=float(sm_bse[j + 1]), z=float(sm_params[j + 1] / sm_bse[j + 1]), p_value=float(sm_p[j + 1]),
                         sd_train=float(sd[n]) if is_num else np.nan,
                         coef_per_raw_unit=float(coef[j] / sd[n]) if is_num else np.nan))
    tab = pd.DataFrame(rows)
    tab["abs_coef"] = tab["coef"].abs(); tab["significant_5pct"] = tab["p_value"] < 0.05
    body = tab[tab["type"] != "intercept"].sort_values("abs_coef", ascending=False)
    tab_out = pd.concat([tab[tab["type"] == "intercept"], body]).reset_index(drop=True)
    csv_path = os.path.join(RESULTS, "logistic_coefficients.csv"); tab_out.to_csv(csv_path, index=False)
    log(f"    coefficient table -> {csv_path}; max |sklearn - statsmodels| = {max_diff:.2e}; "
        f"McFadden pseudo-R2 = {sm_res.prsquared:.4f}; {int(body['significant_5pct'].sum())}/{len(body)} columns significant at 5%")

    # --- figure: 15 largest |coef| (horizontal bars, red = raises default odds, blue = lowers; faded = p >= 0.05)
    top = body.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    bars = ax.barh(top["label"], top["coef"], color=[RED if c > 0 else BLUE for c in top["coef"]], height=0.62, edgecolor="none")
    span = float(top["coef"].max() - min(top["coef"].min(), 0.0))
    for b, c, p in zip(bars, top["coef"], top["p_value"]):
        if p >= 0.05:
            b.set_alpha(0.4)
        ax.text(c + (0.012 if c > 0 else -0.012) * span, b.get_y() + b.get_height() / 2,
                f"{c:+.3f}" + ("" if p < 0.05 else "  (n.s.)"), va="center", ha="left" if c > 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo, hi = min(top["coef"].min(), 0.0), max(top["coef"].max(), 0.0)
    ax.set_xlim(lo - 0.28 * span, hi + 0.28 * span)
    ax.tick_params(colors=INK2, labelsize=9); ax.set_xlabel("Coefficient = change in log-odds of default per +1 SD (numeric) or vs reference level (one-hot)", fontsize=9, color=INK2)
    ax.set_title("Plain logistic regression: 15 largest standardised coefficients\n(training set, n = 20,975; 32 columns after one-hot encoding)",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(handles=[Patch(color=RED, label="raises default odds"), Patch(color=BLUE, label="lowers default odds"),
                       Patch(color=GREY, label="faded = not significant (p ≥ 0.05)")], loc="lower right", frameon=False, fontsize=8.5)
    save_extra_figure(fig, "logistic_coefficients.png")

    res_a = evaluate("Logistic regression (plain)", plain, X_train, y_train, X_test, y_test, member=MEMBER,
                     best_params={"C": "inf (unpenalised)", "solver": "lbfgs", "max_iter": MAX_ITER},
                     cv_auc=cv_a, cv_auc_std=cvsd_a, fit_seconds=round(fit_a, 2), p_oof=p_oof_a, slug="logistic",
                     notes="Unpenalised MLE (C=inf) on all 29 features -> 32 columns (26 standardised numeric + 6 one-hot). "
                           "Coefficients coincide with statsmodels Logit on the same matrix; p-values in results/logistic_coefficients.csv.",
                     extra=dict(n_columns_transformed=int(Xt.shape[1]), n_iter_lbfgs=int(np.ravel(clf.n_iter_)[0]),
                                mcfadden_pseudo_r2=float(sm_res.prsquared), llr_pvalue=float(sm_res.llr_pvalue),
                                statsmodels_converged=bool(sm_res.mle_retvals.get("converged", True)),
                                max_abs_diff_sklearn_vs_statsmodels=max_diff, n_significant_5pct=int(body["significant_5pct"].sum()),
                                top15=body.head(15)[["feature", "coef", "odds_ratio", "p_value"]].round(4).to_dict(orient="records"),
                                coefficients_csv="results/logistic_coefficients.csv"))
    summary.append(res_a)

    # ============================================================== (b) L2 (ridge) logistic, C tuned
    log("(b) ridge logistic: GridSearchCV over C in logspace(-3, 2, 11), 5-fold CV AUC")
    gs_l2 = GridSearchCV(pipe(LogisticRegression(l1_ratio=0.0, solver="lbfgs", max_iter=MAX_ITER, tol=LBFGS_TOL, random_state=SEED)),
                         {"clf__C": C_GRID}, cv=CV, scoring=SCORING, n_jobs=N_JOBS, refit=True)
    t0 = time.time(); gs_l2.fit(X_train, y_train); fit_b = time.time() - t0
    r2, i2 = gs_l2.cv_results_, gs_l2.best_index_
    C_l2 = float(gs_l2.best_params_["clf__C"])
    log(f"    best C = {C_l2:g}, CV AUC {r2['mean_test_score'][i2]:.4f} ± {r2['std_test_score'][i2]:.4f} ({fit_b:.0f}s)")
    p_oof_b = oof_probs(gs_l2.best_estimator_, X_train, y_train)
    coef_b = coef_of(gs_l2.best_estimator_)
    res_b = evaluate("Logistic regression (L2 ridge, C tuned)", gs_l2.best_estimator_, X_train, y_train, X_test, y_test, member=MEMBER,
                     best_params={"C": C_l2, "penalty": "l2 (l1_ratio=0)", "solver": "lbfgs"},
                     cv_auc=float(r2["mean_test_score"][i2]), cv_auc_std=float(r2["std_test_score"][i2]),
                     fit_seconds=round(fit_b, 1), p_oof=p_oof_b, slug="logistic_l2",
                     notes="GridSearchCV over C = logspace(-3, 2, 11) (11 values x 5 folds), refit on the full training set.",
                     extra=dict(grid_C=[float(c) for c in C_GRID], grid_mean_cv_auc=[float(v) for v in r2["mean_test_score"]],
                                grid_std_cv_auc=[float(v) for v in r2["std_test_score"]],
                                coef_l2_norm=float(np.linalg.norm(coef_b)), coef_plain_l2_norm=float(np.linalg.norm(coef)),
                                n_zero_coef=int((coef_b == 0).sum())))
    summary.append(res_b)

    # ============================================================== (c) L1 (lasso) logistic, C tuned
    log("(c) lasso logistic (saga, l1): GridSearchCV over the same C grid")
    gs_l1 = GridSearchCV(pipe(LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=MAX_ITER, random_state=SEED)),
                         {"clf__C": C_GRID}, cv=CV, scoring=SCORING, n_jobs=N_JOBS, refit=True)
    t0 = time.time(); gs_l1.fit(X_train, y_train); fit_c = time.time() - t0
    r1, i1 = gs_l1.cv_results_, gs_l1.best_index_
    C_l1 = float(gs_l1.best_params_["clf__C"])
    coef_c = coef_of(gs_l1.best_estimator_)
    zero_feats = [n for n, c in zip(names, coef_c) if c == 0.0]
    n_zero = len(zero_feats)
    log(f"    best C = {C_l1:g}, CV AUC {r1['mean_test_score'][i1]:.4f} ± {r1['std_test_score'][i1]:.4f} ({fit_c:.0f}s); "
        f"{n_zero}/{len(coef_c)} coefficients exactly zero: {zero_feats}")
    p_oof_c = oof_probs(gs_l1.best_estimator_, X_train, y_train)

    # --- regularisation path on the full training set (preprocessor fitted on training data only), warm-started saga
    log("    L1 regularisation path over logspace(-3, 2, 21)")
    Xt_c = dense(gs_l1.best_estimator_.named_steps["pre"].transform(X_train))
    path_clf = LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=MAX_ITER, random_state=SEED, warm_start=True)
    coefs, nnz = [], []
    for C in PATH_GRID:
        path_clf.set_params(C=C); path_clf.fit(Xt_c, y_train)
        coefs.append(path_clf.coef_[0].copy()); nnz.append(int((path_clf.coef_[0] != 0).sum()))
    coefs = np.array(coefs); logC = np.log10(PATH_GRID)
    idx_grid = [int(np.argmin(np.abs(logC - np.log10(C)))) for C in C_GRID]     # PATH_GRID contains every C_GRID value
    nnz_at_grid = [nnz[i] for i in idx_grid]
    nonzero_at_grid = {f"{C:g}": [names[j] for j in range(coefs.shape[1]) if coefs[i, j] != 0.0] for C, i in zip(C_GRID, idx_grid)}
    entry_C = {names[j]: (float(PATH_GRID[np.flatnonzero(coefs[:, j])[0]]) if np.any(coefs[:, j] != 0.0) else None)
               for j in range(coefs.shape[1])}                                   # smallest C at which each column becomes non-zero
    log("    entry order on the path (first 8): " + ", ".join(f"{k} (C={v:g})" for k, v in sorted(entry_C.items(), key=lambda kv: (kv[1] is None, kv[1]))[:8]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    j_best = int(np.argmin(np.abs(logC - np.log10(C_l1))))
    highlight = list(np.argsort(-np.abs(coefs[j_best]))[:6])
    for j in range(coefs.shape[1]):
        if j not in highlight:
            ax1.plot(logC, coefs[:, j], color=GREY, lw=1.0, zorder=1)
    ends = []
    for k, j in enumerate(highlight):
        ax1.plot(logC, coefs[:, j], color=CAT6[k], lw=2.0, zorder=3, label=LABELS.get(names[j], names[j]))
        ends.append((float(coefs[-1, j]), k, j))
    ends.sort()                                            # nudge the direct labels apart at the right edge
    ymin, ymax = coefs.min(), coefs.max(); gap = 0.05 * (ymax - ymin)
    y_prev = -np.inf
    for y_end, k, j in ends:
        y_lab = max(y_end, y_prev + gap); y_prev = y_lab
        ax1.text(logC[-1] + 0.06, y_lab, names[j], color=CAT6[k], fontsize=8.5, va="center")
    ax1.axvline(np.log10(C_l1), color=INK2, ls="--", lw=1)
    ax1.text(np.log10(C_l1) - 0.06, ymin + 0.10 * (ymax - ymin), f"chosen C = {C_l1:g}\n{len(coef_c) - n_zero} non-zero",
             fontsize=8.5, color=INK2, va="bottom", ha="right")
    ax1.axhline(0, color=INK2, lw=0.6)
    ax1.set_xlim(logC[0], logC[-1] + 0.9); ax1.set_xlabel("log10(C)   (weaker penalty ->)", color=INK2); ax1.set_ylabel("standardised coefficient", color=INK2)
    ax1.set_title("L1 regularisation path (training set; grey = other 26 columns)", fontsize=11, loc="left", color=INK)
    ax1.legend(loc="upper left", frameon=False, fontsize=8); ax1.grid(color=GRID, lw=0.6); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    gC = np.log10(C_GRID)
    ax2.errorbar(gC, r1["mean_test_score"], yerr=r1["std_test_score"], color=CAT6[1], lw=2, marker="o", ms=5, capsize=3, label="L1 (saga)")
    ax2.errorbar(gC + 0.04, r2["mean_test_score"], yerr=r2["std_test_score"], color=CAT6[0], lw=2, marker="s", ms=5, capsize=3, label="L2 (lbfgs)")
    lo = min(np.min(r1["mean_test_score"] - r1["std_test_score"]), np.min(r2["mean_test_score"] - r2["std_test_score"])) - 0.004
    hi = max(np.max(r1["mean_test_score"] + r1["std_test_score"]), np.max(r2["mean_test_score"] + r2["std_test_score"])) + 0.002
    ax2.set_ylim(lo, hi); ax2.set_xlim(gC[0] - 1.0, gC[-1] + 0.3)
    for x, n in zip(gC, nnz_at_grid):                     # non-zero counts as a row along the bottom (no second axis)
        ax2.text(x, lo + 0.0008, f"{n}", ha="center", va="bottom", fontsize=8, color=CAT6[1])
    ax2.text(gC[0] - 0.2, lo + 0.0008, "non-zero\ncoefs (L1):", ha="right", va="bottom", fontsize=7.5, color=CAT6[1])
    ax2.axhline(cv_a, color=INK2, ls=":", lw=1, label=f"plain logistic (C=inf): {cv_a:.4f}")
    ax2.set_xlabel("log10(C)", color=INK2); ax2.set_ylabel("5-fold CV AUC (mean ± sd)", color=INK2)
    ax2.set_title("CV AUC vs C, L1 vs L2 (bottom row = non-zero L1 coefficients)", fontsize=11, loc="left", color=INK)
    ax2.legend(loc="center right", frameon=False, fontsize=8.5); ax2.grid(color=GRID, lw=0.6); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    save_extra_figure(fig, "logistic_l1_path.png")

    res_c = evaluate("Logistic regression (L1 lasso, C tuned)", gs_l1.best_estimator_, X_train, y_train, X_test, y_test, member=MEMBER,
                     best_params={"C": C_l1, "penalty": "l1 (l1_ratio=1)", "solver": "saga", "max_iter": MAX_ITER},
                     cv_auc=float(r1["mean_test_score"][i1]), cv_auc_std=float(r1["std_test_score"][i1]),
                     fit_seconds=round(fit_c, 1), p_oof=p_oof_c, slug="logistic_l1",
                     notes=f"GridSearchCV over C = logspace(-3, 2, 11) with saga/L1; at the chosen C {n_zero} of {len(coef_c)} "
                           f"coefficients are exactly zero ({', '.join(zero_feats) if zero_feats else 'none'}). Path figure: figures/logistic_l1_path.png.",
                     extra=dict(grid_C=[float(c) for c in C_GRID], grid_mean_cv_auc=[float(v) for v in r1["mean_test_score"]],
                                grid_std_cv_auc=[float(v) for v in r1["std_test_score"]], n_nonzero_at_grid=nnz_at_grid,
                                n_zero_coef=n_zero, n_nonzero_coef=int(len(coef_c) - n_zero), zero_features=zero_feats,
                                saga_n_iter=int(np.ravel(gs_l1.best_estimator_.named_steps["clf"].n_iter_)[0]),
                                path_logC=[float(v) for v in logC], path_n_nonzero=nnz,
                                nonzero_features_at_grid=nonzero_at_grid, entry_C_on_path=entry_C,
                                coef_at_best={n: float(c) for n, c in zip(names, coef_c)}))
    summary.append(res_c)

    # ============================================================== (d) plain logistic, class_weight='balanced'
    log("(d) plain logistic with class_weight='balanced'")
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.values)
    bal = pipe(LogisticRegression(C=np.inf, class_weight="balanced", solver="lbfgs", max_iter=MAX_ITER, tol=LBFGS_TOL, random_state=SEED))
    t0 = time.time(); bal.fit(X_train, y_train); fit_d = time.time() - t0
    cv_d, cvsd_d = fold_auc(bal, X_train, y_train)
    p_oof_d = oof_probs(bal, X_train, y_train)
    log(f"    class weights 0: {w[0]:.3f}, 1: {w[1]:.3f}; CV AUC {cv_d:.4f} ± {cvsd_d:.4f}")
    res_d = evaluate("Logistic regression (plain, class_weight=balanced)", bal, X_train, y_train, X_test, y_test, member=MEMBER,
                     best_params={"C": "inf (unpenalised)", "class_weight": "balanced", "solver": "lbfgs"},
                     cv_auc=cv_d, cv_auc_std=cvsd_d, fit_seconds=round(fit_d, 2), p_oof=p_oof_d, slug="logistic_balanced",
                     notes=f"Same specification as 'logistic' but sample weights n/(2*n_class): 0 -> {w[0]:.3f}, 1 -> {w[1]:.3f}. "
                           "Re-weighting mainly shifts the intercept / probability scale, so ranking metrics are ~unchanged while "
                           "threshold 0.5 behaves like a tuned threshold (higher recall, lower precision) and probabilities are no longer calibrated.",
                     extra=dict(class_weight_0=float(w[0]), class_weight_1=float(w[1]),
                                intercept_plain=float(clf.intercept_[0]), intercept_balanced=float(bal.named_steps["clf"].intercept_[0]),
                                max_abs_coef_diff_vs_plain=float(np.max(np.abs(coef_of(bal) - coef)))))
    summary.append(res_d)

    # ============================================================== summary
    rows = []
    for r in summary:
        rows.append(dict(slug=r["slug"], cv_auc=r["cv_auc"], test_auc=r["test_auc"], test_ap=r["test_ap"], test_brier=r["test_brier"],
                         acc_0_5=r["at_0_5"]["accuracy"], recall_0_5=r["at_0_5"]["recall_1"], precision_0_5=r["at_0_5"]["precision_1"],
                         thr_f1=r["at_tuned_f1"]["threshold"], recall_f1=r["at_tuned_f1"]["recall_1"], precision_f1=r["at_tuned_f1"]["precision_1"],
                         f1_f1=r["at_tuned_f1"]["f1_1"], thr_cost=r["at_tuned_cost"]["threshold"], cost=r["at_tuned_cost"]["cost"],
                         fit_seconds=r["fit_seconds"], best_params=str(r["best_params"])))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "logistic_family_summary.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n" + df.drop(columns=["best_params"]).round(4).to_string(index=False))
    log(f"done in {time.time() - t_all:.0f}s -> results/logistic*.json, results/logistic_coefficients.csv, "
        f"figures/logistic_coefficients.png, figures/logistic_l1_path.png")


if __name__ == "__main__":
    main()
