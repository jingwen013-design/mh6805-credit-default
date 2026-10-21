"""
model_svm.py — Support vector machines (lecture B8) for credit-card default prediction.
Owner: LI Jingwen.   Run from Project/code:  python3 model_svm.py
Runtime: 8.7 min wall-clock for one clean run of this version (16 Sep 2026, 2 workers, 8-core laptop with other project scripts
running concurrently, load average ~8): tuning grids + boundary probes ~2.4 min, linear-kernel Platt fit + nested OOF ~1.9 min,
RBF Platt fit + nested OOF ~3.8 min, LinearSVC section ~0.7 min.  Timings move by +/-50 % with machine load (the libsvm fits alone
ranged 25-151 s across runs); every metric is seeded and reproduces bit-for-bit.

What this script does
  1. load_split() -> the shared stratified 70/30 split (seed 2607526).  Nothing touches the test set before evaluate(), which is
     called once per final model; the only test-set work after it (prediction timing, distinct-value counts, and the solver-tolerance
     check for the linear kernel) is computed from the p_test vector that evaluate() itself saved and is appended to the JSON's
     `extra` afterwards (keys marked *_after_evaluate).  Nothing is ever fitted or selected on test rows.
  2. libsvm's SMO solver is O(n^2)-O(n^3) in the number of rows and its linear kernel needs tens of millions of iterations at
     large C, so hyper-parameters are tuned on a STRATIFIED SUBSAMPLE of 6,000 training rows
     (train_test_split(stratify=y_train, random_state=SEED)) and the chosen setting is then REFITTED on the full 20,975-row
     training set.  Every CV fit is Pipeline(make_preprocessor(), SVC) with cv=CV and scoring=SCORING (roc_auc), n_jobs=2, so the
     StandardScaler / one-hot encoder are refitted inside every fold.  Because a subsample CV AUC is not comparable with the other
     members' full-training-set CV AUCs, the plain logistic baseline is also cross-validated on the SAME 6,000 rows and that value
     (not the README's 0.760) is the reference line drawn on the subsample figures.
  3. Linear kernel : C in {0.01, 0.1, 1, 10} x class_weight in {None, 'balanced'}                      (8 settings x 5 folds).
     RBF kernel    : C in {0.1, 1, 10} x gamma in {'scale', 0.01, 0.1} x class_weight in {None, 'balanced'} (18 settings x 5 folds).
     class_weight='balanced' is kept only where it improves CV AUC (both numbers are reported in extra / the notes).
     Selection rule (both kernels): one-standard-error rule on CV AUC — among converged settings whose mean CV AUC is within one
     standard error (sd/sqrt(5)) of the best, take the most regularised one (smallest C, then smallest gamma, then no class
     weighting).  The plain arg-max setting and its CV AUC are reported alongside.  The rule matters for the linear kernel: its
     CV curve is flat in C while libsvm's cost grows ~10x per decade of C (C=10 needs 25M SMO iterations on a 4,800-row fold).
  4. Platt-scaled probabilities for the final models.  scikit-learn 1.9 deprecates SVC(probability=True) (removed in 1.11) and
     recommends the equivalent CalibratedClassifierCV(SVC(), method='sigmoid', ensemble=False): fit the SVC on the 5 CV folds,
     fit ONE sigmoid on the out-of-fold decision values, refit the SVC on all training rows and apply the sigmoid to its decision
     function — exactly libsvm's internal Platt scaling, but seeded through CV, not deprecated, and predict_proba is an exact
     logistic function of decision_function (checked and recorded in extra).
  5. Out-of-fold probabilities for the threshold tuning inside evaluate(): the standard NESTED cross_val_predict(predict_proba) of
     the Platt-scaled pipeline (30 SVC fits per model) — inside every outer fold the SVC is refitted on 5 inner folds, the sigmoid on
     the inner out-of-fold decision values, and the row being scored is seen by neither.  The same procedure is used for all three
     models (it is also what evaluate() does by default).  oof_auc in the JSON is the AUC of those probabilities POOLED across the
     5 outer folds; extra.oof.per_fold records the AUC inside each outer fold as well, because pooling 5 different (SVC, sigmoid)
     pairs can interleave near-tied score clusters — which matters for the degenerate linear-kernel score (see 6).
  6. Fast alternative: LinearSVC(C=best) (liblinear, default squared-hinge loss, primal solver) + CalibratedClassifierCV(sigmoid)
     on the full training set — slug svm_linear_calibrated.  A loss='hinge' LinearSVC (the libsvm objective) is fitted as a
     diagnostic only: it converges to the exact hinge optimum (a MAX_DELAY-only ranker, 9 distinct scores) and agrees with the libsvm
     linear-kernel model to ~1e-4 in probability (recorded in extra, compared on training rows only).  A solver-tolerance diagnostic
     in extra.solver_tolerance_diagnostic of svm_linear shows how much of its AUC/AP comes from the sub-2e-4 residual coefficients
     libsvm leaves at tol=1e-3 (scores rounded to 3 dp; MAX_DELAY alone) — training/OOF side before evaluate(), test side after it.
     Grid-edge boundary checks (C below 0.01 for the linear kernel, gamma below 0.01 for RBF) are recorded, never used for selection.
  7. evaluate() once per final model: slugs svm_linear, svm_rbf, svm_linear_calibrated.
     Figures: figures/svm_rbf_cv_heatmap.png (C x gamma, both class weights), figures/svm_linear_cv_auc_vs_C.png.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # libsvm/liblinear are single-threaded; keep BLAS at 1
    os.environ.setdefault(_v, "1")
import sys, time, json
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sklearn
from common import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score

MEMBER = "LI Jingwen"              # PLACEHOLDER — replace with the member's real name before submission (also in run_all.sh)
N_SUB = 6000                                  # stratified tuning subsample (rows)
LIN_C = [0.01, 0.1, 1, 10]
RBF_C = [0.1, 1, 10]
RBF_GAMMA = ["scale", 0.01, 0.1]
CLASS_WEIGHTS = [None, "balanced"]
MAX_ITER_TUNE = 5_000_000        # SMO iteration cap on the tuning subsample: binds only for the linear kernel at C=10 (recorded per fold)
MAX_ITER_FULL = 50_000_000       # safety net for the full-data refits (expected not to bind; recorded)
CACHE_SUB, CACHE_FULL = 400, 1000  # libsvm kernel cache (MB): a 4,800-row fold matrix is 92 MB, a 16,780-row fold 1.1 GB (float32)
N_JOBS = 2
LOGIT_CV_AUC = 0.760             # logistic-regression reference (README)

T0 = time.time()
def log(msg): print(f"[svm] {msg}   [t={time.time() - T0:,.0f}s]", flush=True)
def fmt_cw(cw): return "none" if cw is None else str(cw)
def cw_arg(s): return None if s == "none" else s
def jsonable(o):
    """Recursively convert numpy / pandas scalars so json.dump inside evaluate() never fails after a 10-minute run."""
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, pd.DataFrame): return jsonable(o.to_dict(orient="records"))
    return o

# --------------------------------------------------------------------------- 1. data + tuning subsample
X_train, X_test, y_train, y_test, feature_names = load_split()
X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=N_SUB, stratify=y_train, random_state=SEED)
log(f"train {X_train.shape}, test {X_test.shape}; tuning subsample {X_sub.shape}, default rate {y_sub.mean():.4f} (train {y_train.mean():.4f})")
log(f"scikit-learn {sklearn.__version__}: SVC(probability=True) is deprecated -> CalibratedClassifierCV(SVC, sigmoid, ensemble=False)")

# Reference for the subsample figures: the plain logistic baseline (same spec as model_logistic.py) cross-validated on the SAME 6,000
# rows.  The README's 0.760 is a full-training-set CV AUC and is not comparable with subsample CV AUCs (the subsample is slightly easier).
_logit_cv = cross_validate(Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-8, random_state=SEED))]),
                           X_sub, y_sub, cv=CV, scoring=SCORING, n_jobs=N_JOBS)
LOGIT_SUB_AUC, LOGIT_SUB_STD = float(_logit_cv["test_score"].mean()), float(_logit_cv["test_score"].std())
LOGIT_REF = dict(full_training_set_cv_auc=LOGIT_CV_AUC, source_full="README / results/logistic.json",
                 same_subsample_cv_auc=LOGIT_SUB_AUC, same_subsample_cv_auc_std=LOGIT_SUB_STD,
                 note="compare subsample CV AUCs of the SVM grids with same_subsample_cv_auc, not with the full-training-set value")
log(f"logistic reference: CV AUC {LOGIT_SUB_AUC:.4f} +/- {LOGIT_SUB_STD:.4f} on the 6,000-row subsample (vs {LOGIT_CV_AUC:.3f} on the full training set)")

# --------------------------------------------------------------------------- 2. one 5-fold CV run per grid setting (keeps per-fold diagnostics)
def cv_setting(svc, X, y):
    pipe = Pipeline([("pre", make_preprocessor()), ("svc", svc)])
    cvr = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=N_JOBS, return_estimator=True)
    ests = [e.named_steps["svc"] for e in cvr["estimator"]]
    return dict(cv_auc_mean=float(cvr["test_score"].mean()), cv_auc_std=float(cvr["test_score"].std()),
                fold_auc=[round(float(v), 5) for v in cvr["test_score"]], mean_fit_seconds=round(float(cvr["fit_time"].mean()), 2),
                n_iter_mean=float(np.mean([int(np.ravel(e.n_iter_)[0]) for e in ests])),
                n_folds_hit_iter_cap=int(sum(int(e.fit_status_ != 0) for e in ests)),
                n_sv_mean=float(np.mean([int(e.n_support_.sum()) for e in ests])),
                gamma_value=float(np.mean([getattr(e, "_gamma", np.nan) for e in ests])))

def one_se_pick(tab, order_cols):
    """One-standard-error rule: among converged settings within sd/sqrt(K) of the best mean CV AUC, the most regularised one."""
    ok = tab[tab.n_folds_hit_iter_cap == 0]
    top = ok.loc[ok.cv_auc_mean.idxmax()]
    se = float(top.cv_auc_std / np.sqrt(CV.get_n_splits()))
    within = ok[ok.cv_auc_mean >= top.cv_auc_mean - se].copy()
    within["cw_rank"] = (within.class_weight != "none").astype(int)
    if "gamma_num" not in within: within["gamma_num"] = 0.0
    chosen = within.sort_values(order_cols).iloc[0]
    return chosen, top, se, within

# ---- RBF grid
log("RBF grid: 3 C x 3 gamma x 2 class_weight = 18 settings x 5 folds on the 6,000-row subsample")
rbf_rows = []
for cw in CLASS_WEIGHTS:
    for C in RBF_C:
        for g in RBF_GAMMA:
            r = cv_setting(SVC(kernel="rbf", C=C, gamma=g, class_weight=cw, cache_size=CACHE_SUB, max_iter=MAX_ITER_TUNE), X_sub, y_sub)
            rbf_rows.append(dict(kernel="rbf", C=float(C), gamma=str(g), gamma_num=r["gamma_value"] if g == "scale" else float(g),
                                 class_weight=fmt_cw(cw), **r))
            log(f"  rbf  C={C:<5} gamma={str(g):<6} cw={fmt_cw(cw):<9} CV AUC {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
                f"({r['mean_fit_seconds']:.1f}s/fit, {r['n_sv_mean']:.0f} SVs, {r['n_iter_mean']:,.0f} it)")
rbf_tab = pd.DataFrame(rbf_rows)
gamma_scale_value = float(rbf_tab.loc[rbf_tab.gamma == "scale", "gamma_num"].iloc[0])
rbf_best, rbf_argmax, rbf_se, rbf_within = one_se_pick(rbf_tab, ["C", "gamma_num", "cw_rank"])
rbf_best_by_cw = {cw: rbf_tab[rbf_tab.class_weight == cw].sort_values("cv_auc_mean", ascending=False).iloc[0] for cw in ("none", "balanced")}
log(f"RBF gamma='scale' = {gamma_scale_value:.4f}; arg-max: C={rbf_argmax.C}, gamma={rbf_argmax.gamma}, cw={rbf_argmax.class_weight} "
    f"(CV AUC {rbf_argmax.cv_auc_mean:.4f}); 1-SE pick (SE {rbf_se:.4f}): C={rbf_best.C}, gamma={rbf_best.gamma}, cw={rbf_best.class_weight} "
    f"(CV AUC {rbf_best.cv_auc_mean:.4f}); best none {rbf_best_by_cw['none'].cv_auc_mean:.4f} vs best balanced {rbf_best_by_cw['balanced'].cv_auc_mean:.4f}")

# ---- linear grid
log("linear grid: 4 C x 2 class_weight = 8 settings x 5 folds on the subsample (SMO iteration cap 5M per fit)")
lin_rows = []
for cw in CLASS_WEIGHTS:
    for C in LIN_C:
        r = cv_setting(SVC(kernel="linear", C=C, class_weight=cw, cache_size=CACHE_SUB, max_iter=MAX_ITER_TUNE), X_sub, y_sub)
        lin_rows.append(dict(kernel="linear", C=float(C), class_weight=fmt_cw(cw), **r))
        log(f"  lin  C={C:<5} cw={fmt_cw(cw):<9} CV AUC {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} ({r['mean_fit_seconds']:.1f}s/fit, "
            f"{r['n_sv_mean']:.0f} SVs, {r['n_iter_mean']:,.0f} it, folds at cap: {r['n_folds_hit_iter_cap']})")
lin_tab = pd.DataFrame(lin_rows).drop(columns=["gamma_value"])
lin_best, lin_argmax_ok, lin_se, lin_within = one_se_pick(lin_tab, ["C", "cw_rank"])
lin_argmax_all = lin_tab.loc[lin_tab.cv_auc_mean.idxmax()]            # may be a capped (non-converged) setting; reported only
lin_best_by_cw = {cw: lin_tab[lin_tab.class_weight == cw].sort_values("cv_auc_mean", ascending=False).iloc[0] for cw in ("none", "balanced")}
log(f"linear arg-max (converged): C={lin_argmax_ok.C}, cw={lin_argmax_ok.class_weight} (CV AUC {lin_argmax_ok.cv_auc_mean:.4f}); "
    f"1-SE pick (SE {lin_se:.4f}): C={lin_best.C}, cw={lin_best.class_weight} (CV AUC {lin_best.cv_auc_mean:.4f}); "
    f"best none {lin_best_by_cw['none'].cv_auc_mean:.4f} vs best balanced {lin_best_by_cw['balanced'].cv_auc_mean:.4f}")

# --------------------------------------------------------------------------- 3. helpers for the final (full-training-set) models
def fit_platt(svc_unfitted):
    """Base pipeline (preprocessor + SVC) wrapped in Platt scaling, fitted on the FULL training set."""
    base = Pipeline([("pre", make_preprocessor()), ("svc", svc_unfitted)])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=CV, ensemble=False, n_jobs=N_JOBS)
    t0 = time.time(); model.fit(X_train, y_train); fit_s = time.time() - t0
    pipe = model.calibrated_classifiers_[0].estimator            # the base pipeline refitted on all 20,975 training rows
    cal = model.calibrated_classifiers_[0].calibrators[0]        # Platt sigmoid: p = expit(-(a*s + b)), s = decision_function
    return model, pipe, cal, fit_s, base

def platt_check(model, pipe, cal, n=3000):
    Xc = X_train.iloc[:n]
    return float(np.abs(expit(-(cal.a_ * pipe.decision_function(Xc) + cal.b_)) - model.predict_proba(Xc)[:, 1]).max())

def nested_oof(model):
    """Standard out-of-fold probabilities for threshold tuning: cross_val_predict(clone(model), method='predict_proba') where model is
    the Platt-scaled pipeline.  Inside every outer fold the base estimator is refitted on 5 inner folds, the sigmoid is fitted on the
    inner out-of-fold decision values and the base estimator is refitted on the outer-fold training rows, so neither has seen the row
    it scores (30 base fits).  Outer loop sequential (no n_jobs), inner CalibratedClassifierCV uses n_jobs=2 -> at most 2 workers."""
    t0 = time.time()
    p = cross_val_predict(clone(model), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    return p, time.time() - t0

OUTER_FOLDS = list(CV.split(X_train, y_train))   # the same deterministic folds cross_val_predict(cv=CV) uses

def oof_summary(p_oof, seconds, n_base_fits):
    """oof_auc (pooled over the 5 outer folds, = what evaluate() reports) vs the AUC inside each outer fold.  Each outer fold is scored
    by its own (base estimator, sigmoid) pair; pooling can interleave near-tied score clusters across folds, so the pooled AUC can sit
    below the within-fold AUCs for a score with very few distinct values.  Costs nothing: it re-uses the nested probabilities."""
    per_fold = [float(roc_auc_score(y_train.values[va], p_oof[va])) for _, va in OUTER_FOLDS]
    return dict(method=f"nested cross_val_predict(predict_proba, cv=CV) of the Platt-scaled pipeline on the full training set ({n_base_fits} "
                       f"base fits; the sigmoid never sees the row it scores) — the same procedure as evaluate()'s default",
                seconds=round(seconds, 1), pooled_oof_auc=float(roc_auc_score(y_train, p_oof)),
                per_fold=dict(auc=[round(v, 4) for v in per_fold], mean=float(np.mean(per_fold)), std=float(np.std(per_fold))),
                pooled_minus_mean_per_fold=float(roc_auc_score(y_train, p_oof) - np.mean(per_fold)))

def append_after_evaluate(res, **updates):
    """evaluate() is called exactly once per model; the test-side diagnostics below (computed from the p_test vector evaluate() saved)
    are appended to `extra` afterwards and the JSON is re-written unchanged otherwise."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(res["extra"].get(k), dict): res["extra"][k].update(jsonable(v))
        else: res["extra"][k] = jsonable(v)
    json.dump(res, open(os.path.join(RESULTS, f"{res['slug']}.json"), "w"), indent=1)
    return res

def load_p_test(slug):
    return np.load(os.path.join(RESULTS, f"{slug}_p_test.npy"))

def svc_info(svc):
    ns = svc.n_support_
    d = dict(n_support_total=int(ns.sum()), n_support_class0=int(ns[0]), n_support_class1=int(ns[1]),
             support_fraction_of_train=round(float(ns.sum() / len(y_train)), 4), n_iter=int(np.ravel(svc.n_iter_)[0]),
             fit_status=int(svc.fit_status_), hit_iter_cap=bool(svc.fit_status_ != 0))
    if hasattr(svc, "_gamma"): d["gamma_value"] = float(svc._gamma)
    return d

def time_decision(pipe, X):
    t0 = time.time(); pipe.decision_function(X); return round(time.time() - t0, 3)

def tie_info(p):
    """How many distinct probabilities a model emits (ties cost AUC/AP because tied pairs count 1/2)."""
    vals, cnt = np.unique(np.round(np.asarray(p), 6), return_counts=True)
    return dict(n=int(len(p)), n_unique_6dp=int(len(vals)), modal_value=float(vals[cnt.argmax()]), modal_share=round(float(cnt.max() / len(p)), 4))

def boundary_check(settings):
    """CV AUC on the tuning subsample for settings just OUTSIDE the prescribed grid (reported only, never used for selection)."""
    out = {}
    for label, svc in settings:
        r = cv_setting(svc, X_sub, y_sub)
        out[label] = dict(cv_auc_mean=r["cv_auc_mean"], cv_auc_std=r["cv_auc_std"], n_folds_hit_iter_cap=r["n_folds_hit_iter_cap"], n_sv_mean=r["n_sv_mean"])
        log(f"  boundary check {label}: CV AUC {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} ({r['n_sv_mean']:.0f} SVs)")
    return out

PLATT_NOTE = ("Platt scaling via CalibratedClassifierCV(method='sigmoid', ensemble=False, cv=CV) — the scikit-learn 1.9 "
              "recommended replacement for the deprecated SVC(probability=True); same internal 5-fold Platt procedure.")
SUB_NOTE = (f"Hyper-parameters tuned by 5-fold CV (roc_auc) on a stratified subsample of {N_SUB:,} training rows "
            f"(train_test_split, stratify, random_state=SEED) because libsvm is O(n^2)-O(n^3); the chosen setting was then refitted "
            f"on the full {len(y_train):,}-row training set. cv_auc = subsample CV AUC of the chosen setting; oof_auc = 5-fold "
            f"out-of-fold AUC of the same setting on the full training set — use oof_auc, not cv_auc, in any cross-model CV table (a "
            f"subsample CV AUC is not comparable with the other models' full-training-set CV AUCs). Selection: one-standard-error rule (see extra).")

# ---- grid-edge checks: if the chosen C (linear) / gamma (RBF) sits on the lower edge of the prescribed grid, probe two steps beyond it
lin_boundary = boundary_check([(f"linear C={c}, cw={lin_best.class_weight}",
                                SVC(kernel="linear", C=c, class_weight=cw_arg(lin_best.class_weight), cache_size=CACHE_SUB, max_iter=MAX_ITER_TUNE))
                               for c in (0.001, 0.0001)]) if float(lin_best.C) == min(LIN_C) else {}
rbf_boundary = boundary_check([(f"rbf C={float(rbf_best.C):g}, gamma={g}, cw={rbf_best.class_weight}",
                                SVC(kernel="rbf", C=float(rbf_best.C), gamma=g, class_weight=cw_arg(rbf_best.class_weight), cache_size=CACHE_SUB, max_iter=MAX_ITER_TUNE))
                               for g in (0.003, 0.001)]) if float(rbf_best.gamma_num) == float(rbf_tab.gamma_num.min()) else {}

# --------------------------------------------------------------------------- 4. final model A: SVM with linear kernel
log(f"final linear-kernel SVM on the full training set: C={lin_best.C}, class_weight={lin_best.class_weight} (Platt via 5-fold CV + refit)")
model_lin, pipe_lin, cal_lin, fit_lin_s, base_lin = fit_platt(
    SVC(kernel="linear", C=float(lin_best.C), class_weight=cw_arg(lin_best.class_weight), cache_size=CACHE_FULL, max_iter=MAX_ITER_FULL))
svc_lin = pipe_lin.named_steps["svc"]; info_lin = svc_info(svc_lin)
err_lin = platt_check(model_lin, pipe_lin, cal_lin)
p_oof_lin, nest_lin_s = nested_oof(model_lin)                                 # used for threshold tuning (30 fits, nested)
oof_lin = oof_summary(p_oof_lin, nest_lin_s, 30)
log(f"  fitted in {fit_lin_s:.1f}s: {info_lin['n_support_total']:,} SVs ({info_lin['support_fraction_of_train']:.1%} of train), "
    f"{info_lin['n_iter']:,} SMO iterations, cap hit: {info_lin['hit_iter_cap']}; Platt a={cal_lin.a_:.4f}, b={cal_lin.b_:.4f}, "
    f"max|expit(-(a*s+b)) - predict_proba| = {err_lin:.1e}; nested OOF in {nest_lin_s:.1f}s: pooled AUC {oof_lin['pooled_oof_auc']:.4f}, "
    f"within-fold AUC {oof_lin['per_fold']['mean']:.4f} +/- {oof_lin['per_fold']['std']:.4f}")

w_lin = svc_lin.coef_.ravel(); names_lin = feature_names_out(pipe_lin.named_steps["pre"])      # coef_ exists for the linear kernel
coef_lin = dict(l2_norm=float(np.linalg.norm(w_lin)), share_of_norm_top1=round(float(np.abs(w_lin).max() / np.linalg.norm(w_lin)), 4),
                n_coef_abs_below_1e3=int((np.abs(w_lin) < 1e-3).sum()), n_coef=int(len(w_lin)),
                top10_abs={n: round(float(v), 4) for n, v in sorted(zip(names_lin, w_lin), key=lambda t: -abs(t[1]))[:10]})

# Solver-tolerance diagnostic, TRAINING / OOF side (the test side is added after evaluate()).  The exact hinge optimum at this C is
# w = c * MAX_DELAY alone (the liblinear hinge twin in section 6 converges to it: 9 distinct scores).  libsvm's SMO stops at tol=1e-3 and
# leaves residual coefficients below 2e-4 that happen to point along informative features; rounding the probabilities to 3 dp removes them.
s_train_lin = pipe_lin.decision_function(X_train)
w_abs = np.abs(w_lin); w_sorted = np.sort(w_abs)
md_train = X_train["MAX_DELAY"].values.astype(float)
p_oof_lin_3dp = np.round(p_oof_lin, 3)
tol_diag = dict(
    libsvm_tol=float(svc_lin.tol),
    n_train_rows_on_margin=int((np.abs(np.abs(s_train_lin) - 1.0) < 1e-3).sum()), n_train=int(len(s_train_lin)),
    coefficient_residue=dict(top1=names_lin[int(w_abs.argmax())], top1_value=float(w_lin[int(w_abs.argmax())]),
                             n_coef_abs_below_2e4=int((w_abs < 2e-4).sum()), largest_other_abs_coef=float(w_sorted[-2])),
    oof_scores_rounded_to_3dp=dict(oof_n_unique=int(len(np.unique(p_oof_lin_3dp))), oof_auc=float(roc_auc_score(y_train, p_oof_lin_3dp))),
    max_delay_alone_train=dict(train_auc=float(roc_auc_score(y_train, md_train)), n_unique_train=int(len(np.unique(md_train)))),
    note="The test_auc / test_ap of svm_linear depend on sub-2e-4 residual coefficients left by libsvm's stopping tolerance (tol=1e-3); "
         "the exact hinge optimum ranks by MAX_DELAY alone (compare test_scores_rounded_to_3dp_after_evaluate and max_delay_alone_test_after_evaluate, "
         "both computed from the p_test vector evaluate() saved). They should not be quoted as a robust property of the linear-kernel SVM.")
log(f"  solver-tolerance diagnostic (train side): {tol_diag['n_train_rows_on_margin']:,} of {tol_diag['n_train']:,} training rows on the margin; "
    f"{tol_diag['coefficient_residue']['n_coef_abs_below_2e4']} coefficients |w|<2e-4; OOF rounded to 3 dp: "
    f"{tol_diag['oof_scores_rounded_to_3dp']['oof_n_unique']} distinct values, AUC {tol_diag['oof_scores_rounded_to_3dp']['oof_auc']:.4f}; "
    f"MAX_DELAY alone: train AUC {tol_diag['max_delay_alone_train']['train_auc']:.4f}")
log(f"  linear-kernel coef_: ||w||={coef_lin['l2_norm']:.4f}, top-1 share {coef_lin['share_of_norm_top1']:.3f}, {coef_lin['n_coef_abs_below_1e3']} of {coef_lin['n_coef']} |coef|<1e-3")
extra_lin = jsonable(dict(
    tuning=dict(subsample_rows=N_SUB, subsample_default_rate=float(y_sub.mean()), grid_C=LIN_C, grid_class_weight=[fmt_cw(c) for c in CLASS_WEIGHTS],
                cv="StratifiedKFold(5, shuffle, seed 2607526) on the subsample", scoring=SCORING, smo_iteration_cap_per_fit=MAX_ITER_TUNE,
                probability_during_tuning=False, note="AUC is invariant to the monotone Platt sigmoid, so tuning used decision values",
                logistic_reference=LOGIT_REF),
    cv_table=lin_tab, selection_rule="one-standard-error rule: among converged settings within sd/sqrt(5) of the best mean CV AUC, "
                                     "the smallest C, then class_weight=None before 'balanced'",
    one_se=lin_se, chosen=dict(C=float(lin_best.C), class_weight=lin_best.class_weight, cv_auc=float(lin_best.cv_auc_mean), cv_auc_std=float(lin_best.cv_auc_std)),
    argmax_converged=dict(C=float(lin_argmax_ok.C), class_weight=lin_argmax_ok.class_weight, cv_auc=float(lin_argmax_ok.cv_auc_mean)),
    argmax_including_capped=dict(C=float(lin_argmax_all.C), class_weight=lin_argmax_all.class_weight, cv_auc=float(lin_argmax_all.cv_auc_mean),
                                 n_folds_hit_iter_cap=int(lin_argmax_all.n_folds_hit_iter_cap)),
    best_cv_auc_by_class_weight={cw: dict(C=float(r.C), cv_auc=float(r.cv_auc_mean), cv_auc_std=float(r.cv_auc_std)) for cw, r in lin_best_by_cw.items()},
    final_model=dict(estimator="CalibratedClassifierCV(Pipeline(make_preprocessor(), SVC(kernel='linear')), sigmoid, ensemble=False, cv=CV)",
                     fit_seconds_total=round(fit_lin_s, 1), note="fit_seconds = 5 CV fits (2 workers) + 1 refit on all rows + sigmoid",
                     smo_iteration_cap=MAX_ITER_FULL, cache_size_mb=CACHE_FULL, **info_lin),
    platt=dict(a=float(cal_lin.a_), b=float(cal_lin.b_), formula="p = 1/(1+exp(a*s+b)), s = decision_function", max_abs_reconstruction_error=err_lin),
    oof=oof_lin, solver_tolerance_diagnostic=tol_diag, coefficients=coef_lin, ties=dict(oof_train=tie_info(p_oof_lin)),
    boundary_check_beyond_grid=lin_boundary, sklearn_version=sklearn.__version__))
res_lin = evaluate("SVM (linear kernel)", model_lin, X_train, y_train, X_test, y_test, member=MEMBER,
                   best_params=dict(kernel="linear", C=float(lin_best.C), class_weight=lin_best.class_weight, probability=PLATT_NOTE),
                   cv_auc=float(lin_best.cv_auc_mean), cv_auc_std=float(lin_best.cv_auc_std),
                   notes=SUB_NOTE + " Linear kernel: SVC(kernel='linear') via libsvm; C grid {0.01, 0.1, 1, 10} x class_weight {None, balanced}; "
                         "C=10 hit the 5M-iteration SMO cap on the tuning subsample (see extra.cv_table). Thresholds tuned on nested "
                         "cross_val_predict(predict_proba) OOF probabilities; oof_auc is their AUC pooled over the 5 outer folds, extra.oof.per_fold "
                         "the AUC inside each fold (pooling 5 different sigmoids interleaves this model's near-tied score clusters, so the pooled "
                         "value is the lower of the two). CAUTION: the exact hinge optimum at C=0.01 ranks by MAX_DELAY alone "
                         "(9 distinct scores); whatever test_auc / test_ap show above that single-feature ranker comes from sub-2e-4 residual "
                         "coefficients left by libsvm's tol=1e-3 (extra.solver_tolerance_diagnostic) and is not a robust property of the model.",
                   fit_seconds=round(fit_lin_s, 1), p_oof=p_oof_lin, extra=extra_lin, slug="svm_linear")
log(f"svm_linear: test AUC {res_lin['test_auc']:.4f}, AP {res_lin['test_ap']:.4f}, OOF AUC {res_lin['oof_auc']:.4f}")

# ---- after evaluate(): test-side diagnostics from the p_test vector evaluate() saved (prediction only; appended to extra)
p_test_lin = load_p_test("svm_linear"); p_test_lin_3dp = np.round(p_test_lin, 3); md_test = X_test["MAX_DELAY"].values.astype(float)
pred_lin_s = time_decision(pipe_lin, X_test)
append_after_evaluate(res_lin,
    predict_seconds_test_decision_function_after_evaluate=pred_lin_s,
    ties=dict(test_after_evaluate=tie_info(p_test_lin)),
    solver_tolerance_diagnostic=dict(
        test_scores_rounded_to_3dp_after_evaluate=dict(test_n_unique=int(len(np.unique(p_test_lin_3dp))), test_auc=float(roc_auc_score(y_test, p_test_lin_3dp)),
                                                       test_ap=float(average_precision_score(y_test, p_test_lin_3dp))),
        max_delay_alone_test_after_evaluate=dict(test_auc=float(roc_auc_score(y_test, md_test)), test_ap=float(average_precision_score(y_test, md_test)),
                                                 n_unique_test=int(len(np.unique(md_test))))))
td = res_lin["extra"]["solver_tolerance_diagnostic"]
log(f"  after evaluate(): decision_function on {len(X_test):,} test rows {pred_lin_s:.2f}s; distinct test probabilities "
    f"{res_lin['extra']['ties']['test_after_evaluate']['n_unique_6dp']} (modal share {res_lin['extra']['ties']['test_after_evaluate']['modal_share']:.1%}); "
    f"rounded to 3 dp: {td['test_scores_rounded_to_3dp_after_evaluate']['test_n_unique']} distinct, AUC {td['test_scores_rounded_to_3dp_after_evaluate']['test_auc']:.4f}, "
    f"AP {td['test_scores_rounded_to_3dp_after_evaluate']['test_ap']:.4f}; MAX_DELAY alone: test AUC {td['max_delay_alone_test_after_evaluate']['test_auc']:.4f}, "
    f"AP {td['max_delay_alone_test_after_evaluate']['test_ap']:.4f}")

# --------------------------------------------------------------------------- 5. final model B: SVM with RBF kernel
log(f"final RBF SVM on the full training set: C={rbf_best.C}, gamma={rbf_best.gamma}, class_weight={rbf_best.class_weight}")
model_rbf, pipe_rbf, cal_rbf, fit_rbf_s, base_rbf = fit_platt(
    SVC(kernel="rbf", C=float(rbf_best.C), gamma=("scale" if rbf_best.gamma == "scale" else float(rbf_best.gamma)),
        class_weight=cw_arg(rbf_best.class_weight), cache_size=CACHE_FULL, max_iter=MAX_ITER_FULL))
svc_rbf = pipe_rbf.named_steps["svc"]; info_rbf = svc_info(svc_rbf)
err_rbf = platt_check(model_rbf, pipe_rbf, cal_rbf)
p_oof_rbf, nest_rbf_s = nested_oof(model_rbf)                                 # used for threshold tuning (nested)
oof_rbf = oof_summary(p_oof_rbf, nest_rbf_s, 30)
log(f"  fitted in {fit_rbf_s:.1f}s: {info_rbf['n_support_total']:,} SVs ({info_rbf['support_fraction_of_train']:.1%} of train), "
    f"{info_rbf['n_iter']:,} SMO iterations, gamma={info_rbf['gamma_value']:.4f}; Platt a={cal_rbf.a_:.4f}, b={cal_rbf.b_:.4f}, "
    f"max reconstruction err {err_rbf:.1e}; nested OOF in {nest_rbf_s:.1f}s: pooled AUC {oof_rbf['pooled_oof_auc']:.4f}, "
    f"within-fold AUC {oof_rbf['per_fold']['mean']:.4f} +/- {oof_rbf['per_fold']['std']:.4f}")

extra_rbf = jsonable(dict(
    tuning=dict(subsample_rows=N_SUB, subsample_default_rate=float(y_sub.mean()), grid_C=RBF_C, grid_gamma=[str(g) for g in RBF_GAMMA],
                gamma_scale_value=gamma_scale_value, grid_class_weight=[fmt_cw(c) for c in CLASS_WEIGHTS],
                cv="StratifiedKFold(5, shuffle, seed 2607526) on the subsample", scoring=SCORING, smo_iteration_cap_per_fit=MAX_ITER_TUNE,
                probability_during_tuning=False, logistic_reference=LOGIT_REF),
    cv_table=rbf_tab, selection_rule="one-standard-error rule: among settings within sd/sqrt(5) of the best mean CV AUC, the smallest C, "
                                     "then smallest gamma, then class_weight=None before 'balanced'",
    one_se=rbf_se, chosen=dict(C=float(rbf_best.C), gamma=rbf_best.gamma, gamma_value=float(rbf_best.gamma_num), class_weight=rbf_best.class_weight,
                              cv_auc=float(rbf_best.cv_auc_mean), cv_auc_std=float(rbf_best.cv_auc_std)),
    argmax=dict(C=float(rbf_argmax.C), gamma=rbf_argmax.gamma, class_weight=rbf_argmax.class_weight, cv_auc=float(rbf_argmax.cv_auc_mean),
                cv_auc_std=float(rbf_argmax.cv_auc_std)),
    best_cv_auc_by_class_weight={cw: dict(C=float(r.C), gamma=r.gamma, cv_auc=float(r.cv_auc_mean), cv_auc_std=float(r.cv_auc_std)) for cw, r in rbf_best_by_cw.items()},
    heatmap_figure="figures/svm_rbf_cv_heatmap.png",
    final_model=dict(estimator="CalibratedClassifierCV(Pipeline(make_preprocessor(), SVC(kernel='rbf')), sigmoid, ensemble=False, cv=CV)",
                     fit_seconds_total=round(fit_rbf_s, 1), note="fit_seconds = 5 CV fits (2 workers) + 1 refit on all rows + sigmoid",
                     smo_iteration_cap=MAX_ITER_FULL, cache_size_mb=CACHE_FULL, **info_rbf),
    platt=dict(a=float(cal_rbf.a_), b=float(cal_rbf.b_), formula="p = 1/(1+exp(a*s+b)), s = decision_function", max_abs_reconstruction_error=err_rbf),
    oof=oof_rbf, ties=dict(oof_train=tie_info(p_oof_rbf)),
    boundary_check_beyond_grid=rbf_boundary, sklearn_version=sklearn.__version__))
res_rbf = evaluate("SVM (RBF kernel)", model_rbf, X_train, y_train, X_test, y_test, member=MEMBER,
                   best_params=dict(kernel="rbf", C=float(rbf_best.C), gamma=rbf_best.gamma, gamma_value=round(float(rbf_best.gamma_num), 5),
                                    class_weight=rbf_best.class_weight, probability=PLATT_NOTE),
                   cv_auc=float(rbf_best.cv_auc_mean), cv_auc_std=float(rbf_best.cv_auc_std),
                   notes=SUB_NOTE + " RBF kernel: C grid {0.1, 1, 10} x gamma {'scale', 0.01, 0.1} x class_weight {None, balanced}; "
                         "CV heat-map in figures/svm_rbf_cv_heatmap.png. Thresholds tuned on nested cross_val_predict(predict_proba) OOF "
                         "probabilities (oof_auc = pooled over the 5 outer folds; extra.oof.per_fold = within-fold AUCs).",
                   fit_seconds=round(fit_rbf_s, 1), p_oof=p_oof_rbf, extra=extra_rbf, slug="svm_rbf")
log(f"svm_rbf: test AUC {res_rbf['test_auc']:.4f}, AP {res_rbf['test_ap']:.4f}, OOF AUC {res_rbf['oof_auc']:.4f}")

# ---- after evaluate(): prediction timing and distinct-value count on the p_test vector evaluate() saved
p_test_rbf = load_p_test("svm_rbf"); pred_rbf_s = time_decision(pipe_rbf, X_test)
append_after_evaluate(res_rbf, predict_seconds_test_decision_function_after_evaluate=pred_rbf_s, ties=dict(test_after_evaluate=tie_info(p_test_rbf)))
log(f"  after evaluate(): decision_function on {len(X_test):,} test rows {pred_rbf_s:.2f}s; distinct test probabilities {tie_info(p_test_rbf)['n_unique_6dp']}")

# --------------------------------------------------------------------------- 6. final model C: LinearSVC (liblinear) + Platt — the fast alternative
best_C, best_cw = float(lin_best.C), cw_arg(lin_best.class_weight)
def make_lsvc(C, loss="squared_hinge", max_iter=5000):
    # squared_hinge = sklearn's default LinearSVC loss, solved in the primal (dual='auto' -> False here); hinge only has a dual solver
    return LinearSVC(C=C, loss=loss, dual=(True if loss == "hinge" else "auto"), class_weight=best_cw, max_iter=max_iter, random_state=SEED)

log("LinearSVC (liblinear) diagnostic: CV AUC over the same C grid on the FULL training set, squared-hinge (default) and hinge loss (5,000-iteration cap)")
lsvc_rows = []
for loss in ("squared_hinge", "hinge"):
    for C in LIN_C:
        pipe = Pipeline([("pre", make_preprocessor()), ("svc", make_lsvc(C, loss))])
        cvr = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=N_JOBS, return_estimator=True)
        its = [int(e.named_steps["svc"].n_iter_) for e in cvr["estimator"]]
        lsvc_rows.append(dict(loss=loss, C=float(C), cv_auc_mean=float(cvr["test_score"].mean()), cv_auc_std=float(cvr["test_score"].std()),
                              mean_fit_seconds=round(float(cvr["fit_time"].mean()), 2), n_iter_mean=float(np.mean(its)),
                              n_folds_hit_iter_cap=int(sum(i >= 5000 for i in its))))
        log(f"  LinearSVC {loss:<13} C={C:<5} CV AUC {lsvc_rows[-1]['cv_auc_mean']:.4f} +/- {lsvc_rows[-1]['cv_auc_std']:.4f} "
            f"({lsvc_rows[-1]['mean_fit_seconds']:.1f}s/fit, folds at cap: {lsvc_rows[-1]['n_folds_hit_iter_cap']})")
lsvc_tab = pd.DataFrame(lsvc_rows)
lsvc_at_best = lsvc_tab[(lsvc_tab.loss == "squared_hinge") & (lsvc_tab.C == best_C)].iloc[0]
hinge_at_best = lsvc_tab[(lsvc_tab.loss == "hinge") & (lsvc_tab.C == best_C)].iloc[0]

log(f"final LinearSVC(C={best_C}, squared_hinge, class_weight={fmt_cw(best_cw)}) + Platt on the full training set")
model_cal, pipe_cal, cal_cal, fit_cal_s, base_cal = fit_platt(make_lsvc(best_C))
lsvc_final = pipe_cal.named_steps["svc"]
err_cal = platt_check(model_cal, pipe_cal, cal_cal)
p_oof_cal, oof_cal_s = nested_oof(model_cal)            # same nested procedure as the libsvm models (outer loop sequential, inner n_jobs=2)
oof_cal = oof_summary(p_oof_cal, oof_cal_s, 30)
w = lsvc_final.coef_.ravel(); names_cal = feature_names_out(pipe_cal.named_steps["pre"])

# diagnostic only: the hinge-loss LinearSVC (= the libsvm objective) should reproduce the libsvm linear-kernel model; compared on TRAINING rows
model_h, pipe_h, cal_h, fit_h_s, _ = fit_platt(make_lsvc(best_C, "hinge", max_iter=20000))
w_h = pipe_h.named_steps["svc"].coef_.ravel()
p_train_h = model_h.predict_proba(X_train)[:, 1]
hinge_vs_libsvm = float(np.abs(p_train_h - model_lin.predict_proba(X_train)[:, 1]).max())
hinge_ties_train = tie_info(p_train_h); hinge_train_auc = float(roc_auc_score(y_train, p_train_h))
log(f"  fitted in {fit_cal_s:.1f}s ({int(lsvc_final.n_iter_):,} liblinear iterations); nested OOF in {oof_cal_s:.1f}s (pooled AUC "
    f"{oof_cal['pooled_oof_auc']:.4f}, within-fold {oof_cal['per_fold']['mean']:.4f}); ||w||={np.linalg.norm(w):.3f}; "
    f"hinge-loss LinearSVC vs libsvm linear kernel: max|dp| on training rows = {hinge_vs_libsvm:.2e}; the hinge twin emits "
    f"{hinge_ties_train['n_unique_6dp']} distinct training probabilities (train AUC {hinge_train_auc:.4f})")

extra_cal = jsonable(dict(
    inherited_from_svm_linear=dict(C=best_C, class_weight=fmt_cw(best_cw)),
    solver=dict(estimator="LinearSVC(loss='squared_hinge', dual='auto' -> primal, max_iter=5000)", library="liblinear (trust-region Newton, primal)",
                n_iter=int(lsvc_final.n_iter_), note="squared hinge is sklearn's default LinearSVC loss; liblinear penalises the intercept (intercept_scaling=1)"),
    full_train_cv_over_C_and_loss=lsvc_tab,
    cv_auc_at_inherited_C_full_train=dict(loss="squared_hinge", cv_auc=float(lsvc_at_best.cv_auc_mean), cv_auc_std=float(lsvc_at_best.cv_auc_std)),
    hinge_loss_variant=dict(note="LinearSVC(loss='hinge', dual=True) at the same C / class_weight = the libsvm linear-kernel objective; diagnostic only, "
                                 "compared on training rows; not evaluated on the test set in this version of the script (an earlier iteration did "
                                 "test-score it — see report/notes_svm.md). It converges to the exact hinge optimum (MAX_DELAY-only ranker, 9 distinct scores).",
                            cv_auc_full_train=float(hinge_at_best.cv_auc_mean), cv_auc_std=float(hinge_at_best.cv_auc_std), fit_seconds_with_platt=round(fit_h_s, 2),
                            max_abs_diff_predict_proba_vs_svm_linear_on_train=hinge_vs_libsvm, n_unique_train_proba_6dp=hinge_ties_train["n_unique_6dp"],
                            train_auc=hinge_train_auc, coef_l2_norm=float(np.linalg.norm(w_h)),
                            share_of_norm_top1=round(float(np.abs(w_h).max() / np.linalg.norm(w_h)), 4),
                            top5_abs_coef={n: round(float(v), 4) for n, v in sorted(zip(names_cal, w_h), key=lambda t: -abs(t[1]))[:5]}),
    final_model=dict(estimator="CalibratedClassifierCV(Pipeline(make_preprocessor(), LinearSVC), sigmoid, ensemble=False, cv=CV)", fit_seconds_total=round(fit_cal_s, 2)),
    platt=dict(a=float(cal_cal.a_), b=float(cal_cal.b_), max_abs_reconstruction_error=err_cal),
    oof=oof_cal, speedup_vs_libsvm_linear_fit=round(fit_lin_s / max(fit_cal_s, 1e-6), 1),
    coefficients=dict(l2_norm=float(np.linalg.norm(w)), share_of_norm_top1=round(float(np.abs(w).max() / np.linalg.norm(w)), 4), n_coef=int(len(w)),
                      top10_abs={n: round(float(v), 4) for n, v in sorted(zip(names_cal, w), key=lambda t: -abs(t[1]))[:10]}),
    ties=dict(oof_train=tie_info(p_oof_cal))))
res_cal = evaluate("LinearSVC + Platt calibration", model_cal, X_train, y_train, X_test, y_test, member=MEMBER,
                   best_params=dict(estimator="LinearSVC", loss="squared_hinge (sklearn default)", C=best_C, class_weight=fmt_cw(best_cw),
                                    calibration="sigmoid (Platt), CalibratedClassifierCV ensemble=False, cv=CV"),
                   cv_auc=float(lsvc_at_best.cv_auc_mean), cv_auc_std=float(lsvc_at_best.cv_auc_std),
                   notes="Fast liblinear alternative (default squared-hinge loss, primal solver) with Platt scaling on the full training set. C and class_weight "
                         "inherited from the svm_linear tuning (6,000-row subsample); cv_auc here is the 5-fold CV AUC of LinearSVC(C=best) on the FULL "
                         "training set. A hinge-loss LinearSVC (the libsvm objective) is recorded in extra.hinge_loss_variant as a diagnostic: it reproduces "
                         "the libsvm linear-kernel probabilities to ~1e-4 and converges to the exact hinge optimum (MAX_DELAY-only, 9 distinct scores). "
                         "Squared hinge (the task spec and sklearn's default) is preferred on training evidence alone: full-training-set CV AUC 0.761 vs "
                         "0.727 for hinge at the same C (extra.full_train_cv_over_C_and_loss). Thresholds tuned on nested cross_val_predict(predict_proba) "
                         "OOF probabilities (oof_auc = pooled over the 5 outer folds; extra.oof.per_fold = within-fold AUCs).",
                   fit_seconds=round(fit_cal_s, 2), p_oof=p_oof_cal, extra=extra_cal, slug="svm_linear_calibrated")
log(f"svm_linear_calibrated: test AUC {res_cal['test_auc']:.4f}, AP {res_cal['test_ap']:.4f}, OOF AUC {res_cal['oof_auc']:.4f}")

# ---- after evaluate(): prediction timing, speed-up over the libsvm linear kernel, distinct-value count on the saved p_test
p_test_cal = load_p_test("svm_linear_calibrated"); pred_cal_s = time_decision(pipe_cal, X_test)
append_after_evaluate(res_cal, predict_seconds_test_decision_function_after_evaluate=pred_cal_s,
                      speedup_vs_libsvm_linear_predict=round(pred_lin_s / max(pred_cal_s, 1e-6), 1), ties=dict(test_after_evaluate=tie_info(p_test_cal)))
log(f"  after evaluate(): decision_function on {len(X_test):,} test rows {pred_cal_s:.3f}s -> {pred_lin_s / max(pred_cal_s, 1e-6):,.0f}x faster than the "
    f"libsvm linear kernel ({fit_lin_s / max(fit_cal_s, 1e-6):,.0f}x faster to fit); distinct test probabilities {tie_info(p_test_cal)['n_unique_6dp']}")

# --------------------------------------------------------------------------- 7. figures
INK, INK2, MUTED, GRID, ACCENT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#eb6834"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", BLUES)

# 7a. RBF heat-map: rows C, columns gamma (numeric order), one panel per class_weight, shared colour scale; every cell is annotated,
#     so no colour bar is needed (it only restated the numbers).
gam_order = sorted(RBF_GAMMA, key=lambda g: gamma_scale_value if g == "scale" else float(g))
gam_labels = [f"scale\n(= {gamma_scale_value:.3f})" if g == "scale" else str(g) for g in gam_order]
def heat_matrix(cw):
    sub = rbf_tab[rbf_tab.class_weight == cw]
    M = np.zeros((len(RBF_C), len(gam_order))); S = np.zeros_like(M)
    for i, C in enumerate(RBF_C):
        for j, g in enumerate(gam_order):
            row = sub[(sub.C == C) & (sub.gamma == str(g))].iloc[0]; M[i, j] = row.cv_auc_mean; S[i, j] = row.cv_auc_std
    return M, S
vmin, vmax = float(rbf_tab.cv_auc_mean.min()), float(rbf_tab.cv_auc_mean.max())
same_cell = (float(rbf_argmax.C) == float(rbf_best.C)) and (rbf_argmax.gamma == rbf_best.gamma) and (rbf_argmax.class_weight == rbf_best.class_weight)
chosen_lab = (f"chosen (1-SE rule{' = arg-max' if same_cell else ''}): C={float(rbf_best.C):g}, gamma={rbf_best.gamma}, "
              f"class_weight={rbf_best.class_weight} -> {rbf_best.cv_auc_mean:.4f}")
marks = [(rbf_best, "-", 2.6, chosen_lab)] + ([] if same_cell else [(rbf_argmax, "--", 1.6, f"arg-max -> {rbf_argmax.cv_auc_mean:.4f}")])
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
for ax, cw in zip(axes, ["none", "balanced"]):
    M, S = heat_matrix(cw)
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            frac = (M[i, j] - vmin) / max(vmax - vmin, 1e-9); col = "white" if frac > 0.55 else INK
            ax.text(j, i + 0.10, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=11, color=col)
            ax.text(j, i - 0.20, f"± {S[i, j]:.4f}", ha="center", va="center", fontsize=8, color=col)
    for row, ls, lw, lab in marks:
        if row.class_weight == cw:
            i = RBF_C.index(float(row.C)); j = [str(g) for g in gam_order].index(row.gamma)
            ax.add_patch(mpatches.Rectangle((j - 0.5 + 0.04, i - 0.5 + 0.04), 0.92, 0.92, fill=False, ec=ACCENT, lw=lw, ls=ls, label=lab))
    ax.set_xticks(range(len(gam_order))); ax.set_xticklabels(gam_labels, fontsize=9)
    ax.set_yticks(range(len(RBF_C))); ax.set_yticklabels([str(c) for c in RBF_C], fontsize=9)
    ax.set_xlabel("gamma (RBF kernel width; larger = more local)", color=INK2); ax.set_title(f"class_weight = {cw}", fontsize=11, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    if ax.get_legend_handles_labels()[0]: ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5, ncol=1)
axes[0].set_ylabel("C (larger = less regularisation)", color=INK2)
fig.suptitle(f"RBF-kernel SVM: mean 5-fold CV ROC AUC on the 6,000-row tuning subsample (cell = mean ± sd; darker = higher)\n"
             f"reference: plain logistic regression on the same 6,000 rows = {LOGIT_SUB_AUC:.4f} ± {LOGIT_SUB_STD:.4f} "
             f"(its full-training-set CV AUC, {LOGIT_CV_AUC:.3f}, is not comparable with subsample values)",
             fontsize=9.5, color=INK)
heat_path = save_extra_figure(fig, "svm_rbf_cv_heatmap.png"); log(f"figure -> {heat_path}")

# 7b. linear SVMs: CV AUC vs C — libsvm linear kernel on the subsample (both class weights) and liblinear on the full training set (both losses)
fig, ax = plt.subplots(figsize=(7.8, 5.0))
for cw, col in (("none", "#2a78d6"), ("balanced", "#eb6834")):
    sub = lin_tab[lin_tab.class_weight == cw].sort_values("C")
    ax.fill_between(sub.C, sub.cv_auc_mean - sub.cv_auc_std, sub.cv_auc_mean + sub.cv_auc_std, color=col, alpha=0.12, lw=0)
    ax.plot(sub.C, sub.cv_auc_mean, color=col, lw=2, marker="o", ms=6.5, label=f"SVC linear kernel, class_weight={cw} (6,000-row subsample)")
    capped = sub[sub.n_folds_hit_iter_cap > 0]
    ax.plot(capped.C, capped.cv_auc_mean, ls="none", marker="o", ms=6.5, mfc="white", mec=col, mew=1.8)
for loss, col, mk, ls in (("squared_hinge", "#1baf7a", "s", "--"), ("hinge", "#eda100", "D", ":")):
    sub = lsvc_tab[lsvc_tab.loss == loss].sort_values("C")
    ax.plot(sub.C, sub.cv_auc_mean, color=col, lw=2, ls=ls, marker=mk, ms=5.5, label=f"LinearSVC {loss.replace('_', ' ')} (liblinear), full training set")
    capped = sub[sub.n_folds_hit_iter_cap > 0]
    ax.plot(capped.C, capped.cv_auc_mean, ls="none", marker=mk, ms=5.5, mfc="white", mec=col, mew=1.6)
lo = min(float((lin_tab.cv_auc_mean - lin_tab.cv_auc_std).min()), float(lsvc_tab.cv_auc_mean.min())) - 0.05
hi = max(float((lin_tab.cv_auc_mean + lin_tab.cv_auc_std).max()), float(lsvc_tab.cv_auc_mean.max())) + 0.012
ax.set_ylim(lo, hi)
ax.axhline(LOGIT_SUB_AUC, color="#eb6834", ls=":", lw=1.3, alpha=0.8)
ax.text(LIN_C[-1], LOGIT_SUB_AUC + 0.0025, f"logistic regression, same 6,000-row subsample ({LOGIT_SUB_AUC:.4f}) — reference for the SVC curves",
        ha="right", va="bottom", fontsize=7.8, color="#b8461c")
ax.axhline(LOGIT_CV_AUC, color=MUTED, ls=":", lw=1.3)
ax.text(LIN_C[-1], LOGIT_CV_AUC - 0.0025, f"logistic regression, full training set ({LOGIT_CV_AUC:.3f}) — reference for the LinearSVC curves",
        ha="right", va="top", fontsize=7.8, color=INK2)
star_col = "#2a78d6" if lin_best.class_weight == "none" else "#eb6834"
ax.plot([float(lin_best.C)], [float(lin_best.cv_auc_mean)], marker="*", ms=16, color=star_col, mec="white", mew=1.2, zorder=5)
y_gap = (float(lin_best.cv_auc_mean) + float(lin_tab[lin_tab.class_weight != lin_best.class_weight].cv_auc_mean.max())) / 2   # empty band between the two curves
ax.annotate(f"chosen: C = {float(lin_best.C):g}, class_weight={lin_best.class_weight}\nCV AUC = {lin_best.cv_auc_mean:.4f} (1-SE rule)",
            (float(lin_best.C), float(lin_best.cv_auc_mean)), xytext=(LIN_C[1], y_gap), textcoords="data", ha="center", va="center", fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=9))
ax.text(0.99, 0.02, "hollow marker = iteration cap reached in >= 1 fold", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
ax.set_xscale("log"); ax.set_xticks(LIN_C); ax.set_xticklabels([str(c) for c in LIN_C]); ax.minorticks_off()
ax.set_xlabel("C (log scale)"); ax.set_ylabel("5-fold CV ROC AUC"); ax.set_title("Linear SVMs: cross-validated AUC vs C (band = +/- 1 sd across folds)", fontsize=11)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=8); ax.margins(x=0.08)
lin_fig = save_extra_figure(fig, "svm_linear_cv_auc_vs_C.png"); log(f"figure -> {lin_fig}")

# --------------------------------------------------------------------------- 8. summary
summary = pd.DataFrame([dict(model=r["name"], slug=r["slug"], cv_auc=r["cv_auc"], oof_auc=r["oof_auc"], oof_auc_within_fold=r["extra"]["oof"]["per_fold"]["mean"],
                             test_auc=r["test_auc"], test_ap=r["test_ap"], brier=r["test_brier"], acc_0_5=r["at_0_5"]["accuracy"], recall_0_5=r["at_0_5"]["recall_1"],
                             thr_f1=r["at_tuned_f1"]["threshold"], recall_f1=r["at_tuned_f1"]["recall_1"], precision_f1=r["at_tuned_f1"]["precision_1"],
                             f1=r["at_tuned_f1"]["f1_1"], fit_s=r["fit_seconds"], predict_s=r["extra"]["predict_seconds_test_decision_function_after_evaluate"])
                        for r in (res_lin, res_rbf, res_cal)])
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log(f"done; total wall-clock {(time.time() - T0) / 60:.1f} min. Results: results/svm_linear.json, results/svm_rbf.json, results/svm_linear_calibrated.json")
