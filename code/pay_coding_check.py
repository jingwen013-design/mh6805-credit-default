"""
pay_coding_check.py - how should the repayment-status codes PAY_1..PAY_6 enter the linear and distance models?
Post-freeze sensitivity analysis on the TRAINING ROWS ONLY (audit findings pay-codes-ordinal-in-linear-models,
pay-code-1-only-in-pay1, o4-pay-code-semantics).

Run from Project/code:  /opt/miniconda3/bin/python3 pay_coding_check.py        (about 1-2 minutes on 2 threads; 66-112 s measured)

PURPOSE
  The frozen models feed PAY_k into every linear and distance model as one standardised number, which imposes the
  evenly spaced order -2 (no consumption) < -1 (paid in full) < 0 (revolving credit) < 1 < 2 < ... (months late).
  This script asks how much of the logit's cross-validated gap is that coding choice, by comparing three codings of
  PAY_1..PAY_6 with everything else held identical:
    numeric      PAY_k as one standardised number, exactly as frozen (common.make_preprocessor()).
    delay_flags  PAY_k_DEL = max(PAY_k, 0) (months late, 0 for the three non-delinquent codes) plus two 0/1 state flags
                 PAY_k_NC = 1[PAY_k = -2] and PAY_k_PD = 1[PAY_k = -1]; all 18 columns sit in the standardised numeric
                 block, like the other engineered row-wise features.
    onehot       PAY_k one-hot per month, OneHotEncoder(drop="first", handle_unknown="ignore"), unscaled, i.e. treated
                 exactly like SEX / EDUCATION / MARRIAGE in the frozen preprocessor.
  The other 23 predictors (LIMIT_BAL, AGE, bills, payments, the six engineered features incl. N_MONTHS_DELAYED and
  MAX_DELAY, and the three one-hot demographics) enter exactly as in the frozen preprocessor.
  Models, with the frozen settings read from results/<slug>.json:
    logit_plain  LogisticRegression(C=inf, lbfgs, max_iter 5000, tol 1e-8)          (results/logistic.json)
    logit_l2     LogisticRegression(l1_ratio=0, C = frozen C 31.62, lbfgs, same tol)  (results/logistic_l2.json)
    lda          LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None)           (results/lda.json)
    knn          KNeighborsClassifier(K=101, uniform, brute force)                   (results/knn.json)
    mlp          MLPClassifier hidden (64, 32), alpha 1e-3, lr 1e-3, Adam, batch 200, early stopping
                 (validation_fraction 0.15, n_iter_no_change 15, max_iter 300), random_state 2607526  (results/mlp.json)
  For the numeric coding the out-of-fold probabilities are checked against the frozen results/<slug>_p_oof.npy.

TRAINING-ONLY
  Reads data/processed/train.csv directly and asserts 20,975 rows. It never calls common.load_split() (guarded below),
  never calls evaluate(), never opens data/processed/test.csv, writes no *_p_test file, no frozen slug and does not touch
  results/frozen_manifest.json. Every model is fitted inside the five common.CV training folds (StratifiedKFold(5,
  shuffle=True, random_state=2607526)); the fold assignment is checked against data/processed/cv_folds.csv. The one
  full-training-set fit is the plain logit used only for the coefficient table.

PAIRED COMPARISONS
  Per-fold AUC difference (coding minus numeric, same folds), mean difference, number of folds with a positive
  difference, and the Nadeau-Bengio corrected t interval (mine_v2_utils.corrected_t_ci; for the five folds of one
  5-fold CV the correction uses n_test / n_train = 1 / 4).

INPUTS
  data/processed/train.csv, data/processed/cv_folds.csv, results/{logistic,logistic_l2,lda,knn,mlp,xgboost}.json,
  results/{logistic,logistic_l2,lda,knn,mlp,xgboost}_p_oof.npy (frozen training out-of-fold probabilities).
OUTPUTS (all new files)
  results/pay_coding_cv.csv                    model x coding: fold 1-5 AUC, mean AUC, sd, AP, Brier, paired diff vs numeric
  results/pay_code_default_rates.csv           training default rate and n by code for each PAY month (+ code-1 diagnostics)
  results/pay_coding_logit_coefficients.csv    recoded plain logit (delay_flags; one-hot as a second block): coef, OR
  results/pay_coding_summary.json              settings, reproduction checks, cross-model comparisons, headline numbers
Seed 2607526 everywhere; n_jobs <= 2 (BLAS threads capped at 2).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from common import (ROOT, RESULTS, SEED, CV, TARGET, NUM_COLS, CAT_COLS, PAY_COLS, BILL_COLS, make_preprocessor)
from mine_v2_utils import corrected_t_ci


def _forbidden(*a, **k):
    raise SystemExit("pay_coding_check.py is training-only: common.load_split() would read the held-out file")


common.load_split = _forbidden          # guard: nothing in this script may load the held-out partition

T0 = time.time()
N_TRAIN_ROWS = 20975
NB_RATIO = (4, 1)                        # corrected_t_ci(diffs, n_train, n_held_out): 1/4 for one 5-fold CV
CODINGS = ["numeric", "delay_flags", "onehot"]
MODELS = ["logit_plain", "logit_l2", "lda", "knn", "mlp"]
FROZEN_SLUG = dict(logit_plain="logistic", logit_l2="logistic_l2", lda="lda", knn="knn", mlp="mlp")
MONTH = dict(PAY_1="Sep 2005", PAY_2="Aug 2005", PAY_3="Jul 2005", PAY_4="Jun 2005", PAY_5="May 2005", PAY_6="Apr 2005")
MEANING = {-2: "no consumption (-2 is not defined in the UCI codebook)", -1: "paid in full (pay duly)",
           0: "revolving credit (0 is not defined in the UCI codebook)"}

DEL_COLS = [f"{c}_DEL" for c in PAY_COLS]
NC_COLS = [f"{c}_NC" for c in PAY_COLS]
PD_COLS = [f"{c}_PD" for c in PAY_COLS]
NONPAY_NUM = [c for c in NUM_COLS if c not in PAY_COLS]               # 20 numeric columns that never change
NUM_DELAY_FLAGS = NONPAY_NUM[:2] + DEL_COLS + NC_COLS + PD_COLS + NONPAY_NUM[2:]


def log(msg):
    print(f"[pay] {msg}   [t={time.time() - T0:,.0f}s]", flush=True)


# ----------------------------------------------------------------------------- data and codings
def load_train():
    tr = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
    assert len(tr) == N_TRAIN_ROWS, f"train.csv has {len(tr)} rows, expected {N_TRAIN_ROWS}"
    X = tr.drop(columns=[TARGET]); y = tr[TARGET].astype(int)
    return X, y


def rebuild_pay_aggregates(X):
    """Recompute N_MONTHS_DELAYED and MAX_DELAY from PAY_1..PAY_6 exactly as common.clean() builds them."""
    X = X.copy()
    X["N_MONTHS_DELAYED"] = (X[PAY_COLS] > 0).sum(axis=1)
    X["MAX_DELAY"] = X[PAY_COLS].max(axis=1).clip(lower=0)
    return X


def add_delay_flags(X):
    """Row-wise, fixed recoding (nothing estimated): PAY_k -> PAY_k_DEL, PAY_k_NC, PAY_k_PD. PAY_k itself is kept in the
    frame (so the onehot coding and the aggregates can still be built) but is not selected by the delay_flags preprocessor."""
    X = X.copy()
    for c in PAY_COLS:
        X[f"{c}_DEL"] = X[c].clip(lower=0).astype(float)
        X[f"{c}_NC"] = (X[c] == -2).astype(float)
        X[f"{c}_PD"] = (X[c] == -1).astype(float)
    return X


def make_pre(coding):
    if coding == "numeric":
        return make_preprocessor()                                    # the frozen preprocessor, untouched
    cat = OneHotEncoder(drop="first", handle_unknown="ignore")
    if coding == "delay_flags":
        return ColumnTransformer([("num", StandardScaler(), NUM_DELAY_FLAGS), ("cat", cat, CAT_COLS)],
                                 remainder="drop", sparse_threshold=0.0)
    if coding == "onehot":
        return ColumnTransformer([("num", StandardScaler(), NONPAY_NUM), ("cat", cat, CAT_COLS),
                                  ("pay", OneHotEncoder(drop="first", handle_unknown="ignore"), PAY_COLS)],
                                 remainder="drop", sparse_threshold=0.0)
    raise ValueError(coding)


def frozen_settings():
    js = {m: json.load(open(os.path.join(RESULTS, f"{s}.json"))) for m, s in FROZEN_SLUG.items()}
    bp = {m: j["best_params"] for m, j in js.items()}
    assert str(bp["logit_plain"]["C"]).startswith("inf") and bp["logit_plain"]["solver"] == "lbfgs"
    assert bp["lda"]["solver"] == "lsqr" and bp["lda"]["shrinkage"] is None
    assert bp["knn"]["weights"] == "uniform" and bp["knn"]["algorithm"] == "brute"
    assert bp["mlp"]["early_stopping"] is True
    s = dict(
        logit_plain=dict(C=np.inf, solver="lbfgs", max_iter=int(bp["logit_plain"]["max_iter"]), tol=1e-8, random_state=SEED),
        logit_l2=dict(l1_ratio=0.0, C=float(bp["logit_l2"]["C"]), solver="lbfgs", max_iter=5000, tol=1e-8, random_state=SEED),
        lda=dict(solver="lsqr", shrinkage=None),
        knn=dict(n_neighbors=int(bp["knn"]["n_neighbors"]), weights="uniform", algorithm="brute"),
        mlp=dict(hidden_layer_sizes=tuple(bp["mlp"]["hidden_layer_sizes"]), alpha=float(bp["mlp"]["alpha"]),
                 learning_rate_init=float(bp["mlp"]["learning_rate_init"]), activation=bp["mlp"]["activation"],
                 solver=bp["mlp"]["solver"], batch_size=int(bp["mlp"]["batch_size"]), early_stopping=True,
                 validation_fraction=float(bp["mlp"]["validation_fraction"]),
                 n_iter_no_change=int(bp["mlp"]["n_iter_no_change"]), max_iter=int(bp["mlp"]["max_iter"]),
                 random_state=SEED))
    return s


def make_clf(model, s):
    return dict(logit_plain=lambda: LogisticRegression(**s["logit_plain"]),
                logit_l2=lambda: LogisticRegression(**s["logit_l2"]),
                lda=lambda: LinearDiscriminantAnalysis(**s["lda"]),
                knn=lambda: KNeighborsClassifier(**s["knn"]),
                mlp=lambda: MLPClassifier(**s["mlp"]))[model]()


def fold_list(y):
    folds = list(CV.split(np.zeros(len(y)), y))
    cvf = pd.read_csv(os.path.join(ROOT, "data", "processed", "cv_folds.csv"))
    assert len(cvf) == len(y) and (cvf["DEFAULT"].values == y.values).all()
    for i, (_, va) in enumerate(folds):
        assert set(va) == set(np.where(cvf["fold"].values == i + 1)[0]), f"fold {i + 1} differs from cv_folds.csv"
    return folds


def nb(d):
    return corrected_t_ci(d, *NB_RATIO)


def paired(a, b):
    """Fold-wise a - b with mean, folds positive and the Nadeau-Bengio corrected 95% interval."""
    d = np.asarray(a) - np.asarray(b)
    ci = nb(d)
    return dict(diffs=[float(v) for v in d], mean=float(d.mean()), folds_positive=int((d > 0).sum()),
                nb_lo=ci["lo"], nb_hi=ci["hi"])


def fold_metric(y, p, folds, fn):
    return np.array([fn(y.values[va], p[va]) for _, va in folds])


# ----------------------------------------------------------------------------- 1. default rates by code
def default_rate_table(X, y):
    rows = []
    zero6 = (X[BILL_COLS] == 0).all(axis=1)
    for k, c in enumerate(PAY_COLS, start=1):
        prev = PAY_COLS[k] if k < 6 else None                        # PAY_{k+1} is the month before PAY_k
        for code in sorted(X[c].unique()):
            m = X[c] == code
            r = dict(variable=c, month=MONTH[c], code=int(code),
                     code_meaning=MEANING.get(int(code), f"{int(code)} month(s) late"),
                     n=int(m.sum()), n_default=int(y[m].sum()), default_rate=float(y[m].mean()),
                     share_of_rows=float(m.mean()),
                     share_same_month_bill_nonpositive=float((X.loc[m, f"BILL_AMT{k}"] <= 0).mean()),
                     share_all_six_bills_zero=float(zero6[m].mean()),
                     default_rate_all_six_bills_zero=(float(y[m & zero6].mean()) if (m & zero6).any() else np.nan),
                     share_month_before_code_2=(float((X.loc[m, prev] == 2).mean()) if prev else np.nan),
                     share_months_before_all_minus2=(float((X.loc[m, PAY_COLS[k:]] == -2).all(axis=1).mean())
                                                     if k < 6 else np.nan))
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. coefficient table
def coefficient_table(Xc, y, coding, s, fold_coefs, model="logit_plain"):
    import statsmodels.api as sm
    pipe = Pipeline([("pre", make_pre(coding)), ("clf", make_clf(model, s))]).fit(Xc, y)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    Xt = np.asarray(pre.transform(Xc), float)
    coef = clf.coef_.ravel()
    sd = dict(zip(pre.named_transformers_["num"].feature_names_in_, pre.named_transformers_["num"].scale_))
    try:
        if model != "logit_plain":
            raise RuntimeError("penalised fit: no likelihood-based standard errors")
        smr = sm.Logit(y.values, sm.add_constant(Xt)).fit(disp=0, maxiter=500, method="newton")
        se, z, pv, conv = smr.bse[1:], smr.tvalues[1:], smr.pvalues[1:], bool(smr.mle_retvals.get("converged", True))
        sm_coef = smr.params[1:]
    except Exception as e:                                           # penalised fit, or quasi-separation on a rare level
        se = z = pv = sm_coef = np.full(len(coef), np.nan)
        conv = "not applicable (penalised fit)" if model != "logit_plain" else f"failed: {type(e).__name__}"
    fc = np.array([[fcd.get(n, np.nan) for n in names] for fcd in fold_coefs])   # 5 x p, the five CV fold fits
    rows = []
    for j, n in enumerate(names):
        scaled = n in sd
        per_unit = coef[j] / sd[n] if scaled else coef[j]
        lvl_n = None
        if n.endswith(("_NC", "_PD")):
            base = n.rsplit("_", 1)[0]; lvl_n = int((Xc[base] == (-2 if n.endswith("_NC") else -1)).sum())
        elif not scaled:
            base, lev = n.rsplit("_", 1); lvl_n = int((Xc[base].astype(str) == lev).sum()) if base in Xc else None
        rows.append(dict(coding=coding, model=model, feature=n, scaled_per_sd=scaled, sd_train=(float(sd[n]) if scaled else np.nan),
                         coef=float(coef[j]), odds_ratio=float(np.exp(coef[j])),
                         coef_per_unit=float(per_unit), odds_ratio_per_unit=float(np.exp(per_unit)),
                         sm_coef=float(sm_coef[j]), std_err=float(se[j]), z=float(z[j]), p_value=float(pv[j]),
                         fold_coef_mean=float(np.nanmean(fc[:, j])), fold_coef_sd=float(np.nanstd(fc[:, j], ddof=1)),
                         fold_sign_agreement=int(max((fc[:, j] > 0).sum(), (fc[:, j] < 0).sum())),
                         n_folds_with_column=int(np.isfinite(fc[:, j]).sum()),
                         n_train_rows_with_level=lvl_n))
    out = pd.DataFrame(rows)
    b0 = float(clf.intercept_[0])
    out = pd.concat([pd.DataFrame([dict(coding=coding, model=model, feature="(intercept)", scaled_per_sd=False, coef=b0,
                                        odds_ratio=float(np.exp(b0)), coef_per_unit=b0, odds_ratio_per_unit=float(np.exp(b0)))]),
                     out], ignore_index=True)
    if coding == "onehot":                                           # log-odds of each code against revolving (0), per month
        out["logodds_vs_code0"] = np.nan
        for c in PAY_COLS:
            lv = out[out["feature"].str.match(rf"^{c}_-?\d+$")]
            levels = {int(f.rsplit("_", 1)[1]): b for f, b in zip(lv["feature"], lv["coef"])}
            levels.setdefault(int(Xc[c].min()), 0.0)                 # dropped first level = reference
            ref0 = levels.get(0, np.nan)
            for f in lv["feature"]:
                out.loc[out["feature"] == f, "logodds_vs_code0"] = levels[int(f.rsplit("_", 1)[1])] - ref0
    return out, dict(n_iter_lbfgs=int(np.ravel(clf.n_iter_)[0]), statsmodels_converged=conv,
                     max_abs_diff_sklearn_vs_statsmodels=float(np.nanmax(np.abs(np.asarray(sm_coef) - coef)))
                     if np.isfinite(np.asarray(sm_coef)).any() else None)


# ----------------------------------------------------------------------------- main
def main():
    X, y = load_train()
    X = add_delay_flags(X)
    assert np.allclose(rebuild_pay_aggregates(X)[["N_MONTHS_DELAYED", "MAX_DELAY"]].values,
                       X[["N_MONTHS_DELAYED", "MAX_DELAY"]].values), "aggregates in train.csv differ from common.clean()"
    folds = fold_list(y)
    s = frozen_settings()
    log(f"train {X.shape}, default rate {y.mean():.4f}; folds match data/processed/cv_folds.csv")

    # ---- 1. default rates by code (training rows)
    dr = default_rate_table(X.drop(columns=DEL_COLS + NC_COLS + PD_COLS), y)
    dr.to_csv(os.path.join(RESULTS, "pay_code_default_rates.csv"), index=False)
    p1 = dr[(dr.variable == "PAY_1") & (dr.code == 1)].iloc[0]
    zero6 = (X[BILL_COLS] == 0).all(axis=1); m1 = X["PAY_1"] == 1
    code1 = {c: int((X[c] == 1).sum()) for c in PAY_COLS}
    log(f"PAY_1 = 1: n {p1.n}, default {p1.default_rate:.4f}, Sep bill <= 0 {p1.share_same_month_bill_nonpositive:.4f}, "
        f"PAY_2 = 2 {p1.share_month_before_code_2:.4f}; code 1 by month {code1}")

    # ---- 2. CV of 5 models x 3 codings on the same folds
    oof = {(m, c): np.zeros(len(y)) for m in MODELS for c in CODINGS}
    fold_coefs = {(m, c): [] for m in ("logit_plain", "logit_l2") for c in CODINGS}
    diag = {}
    for m in MODELS:
        for c in CODINGS:
            t0 = time.time(); iters = []
            for tr, va in folds:
                pipe = Pipeline([("pre", make_pre(c)), ("clf", make_clf(m, s))])
                pipe.fit(X.iloc[tr], y.iloc[tr])
                oof[(m, c)][va] = pipe.predict_proba(X.iloc[va])[:, 1]
                clf = pipe.named_steps["clf"]
                if m.startswith("logit"):                               # keyed by name: a rare one-hot level can be absent in a fold
                    fnames = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
                    fold_coefs[(m, c)].append(dict(zip(fnames, clf.coef_.ravel().copy())))
                if m.startswith("logit"):
                    iters.append(int(np.ravel(clf.n_iter_)[0]))
                elif m == "mlp":
                    iters.append(int(clf.n_iter_))
            n_in = int(pipe.named_steps["pre"].transform(X.iloc[:2]).shape[1])
            diag[(m, c)] = dict(n_inputs=n_in, iterations_per_fold=iters, seconds=round(time.time() - t0, 1))
            log(f"{m:11s} {c:11s} CV AUC {fold_metric(y, oof[(m, c)], folds, roc_auc_score).mean():.4f} "
                f"({n_in} inputs, {time.time() - t0:.0f}s, iterations {iters})")

    # ---- reproduction check of the numeric coding against the frozen out-of-fold vectors
    repro = {}
    for m in MODELS:
        fro = np.load(os.path.join(RESULTS, f"{FROZEN_SLUG[m]}_p_oof.npy"))
        mine = oof[(m, "numeric")]
        repro[m] = dict(frozen_oof=f"results/{FROZEN_SLUG[m]}_p_oof.npy", max_abs_diff=float(np.abs(fro - mine).max()),
                        frozen_cv_auc=float(fold_metric(y, fro, folds, roc_auc_score).mean()),
                        this_run_cv_auc=float(fold_metric(y, mine, folds, roc_auc_score).mean()))
        log(f"reproduction {m}: max |p - frozen p_oof| = {repro[m]['max_abs_diff']:.2e}, CV AUC "
            f"{repro[m]['this_run_cv_auc']:.5f} vs frozen {repro[m]['frozen_cv_auc']:.5f}")

    # ---- table
    rows, auc = [], {}
    for m in MODELS:
        for c in CODINGS:
            p = oof[(m, c)]
            a = fold_metric(y, p, folds, roc_auc_score); ap = fold_metric(y, p, folds, average_precision_score)
            br = fold_metric(y, p, folds, brier_score_loss); auc[(m, c)] = a
            r = dict(model=m, coding=c, n_inputs=diag[(m, c)]["n_inputs"])
            r.update({f"auc_fold{i + 1}": float(v) for i, v in enumerate(a)})
            r.update(mean_auc=float(a.mean()), sd_auc=float(a.std()), sd_auc_ddof1=float(a.std(ddof=1)),
                     oof_auc_pooled=float(roc_auc_score(y, p)), mean_ap=float(ap.mean()),
                     oof_ap_pooled=float(average_precision_score(y, p)), mean_brier=float(br.mean()),
                     oof_brier_pooled=float(brier_score_loss(y, p)))
            if c == "numeric":
                r.update(diff_auc_vs_numeric_mean=0.0)
            else:
                d = paired(a, auc[(m, "numeric")])
                dap = paired(ap, fold_metric(y, oof[(m, "numeric")], folds, average_precision_score))
                r.update({f"diff_auc_fold{i + 1}": v for i, v in enumerate(d["diffs"])})
                r.update(diff_auc_vs_numeric_mean=d["mean"], diff_auc_folds_positive=d["folds_positive"],
                         diff_auc_nb_lo=d["nb_lo"], diff_auc_nb_hi=d["nb_hi"],
                         diff_ap_vs_numeric_mean=dap["mean"], diff_ap_folds_positive=dap["folds_positive"],
                         diff_ap_nb_lo=dap["nb_lo"], diff_ap_nb_hi=dap["nb_hi"])
            its = diag[(m, c)]["iterations_per_fold"]
            cap = s[m].get("max_iter")
            r.update(iterations_per_fold=" ".join(map(str, its)),
                     converged_all_folds=(bool(max(its) < cap) if its else True))
            if m == "logit_plain" and c == "onehot" and its and max(its) >= cap:
                r["note"] = ("lbfgs stopped at max_iter in every fold: with one dummy per rare code (e.g. PAY_3 = 8 or "
                             "PAY_4 = 1, n = 1) the unpenalised MLE is not finite; the fold AUCs are still well defined; the "
                             "logit_l2 / onehot row is the converged fit of the same coding")
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS, "pay_coding_cv.csv"), index=False)
    print(tab[["model", "coding", "mean_auc", "sd_auc", "mean_ap", "oof_brier_pooled", "diff_auc_vs_numeric_mean",
               "diff_auc_folds_positive", "diff_auc_nb_lo", "diff_auc_nb_hi"]].round(4).to_string(index=False))

    # ---- cross-model comparisons on the same folds (all training out-of-fold)
    xgb = np.load(os.path.join(RESULTS, "xgboost_p_oof.npy"))
    a_xgb = fold_metric(y, xgb, folds, roc_auc_score)
    knn_num = auc[("knn", "numeric")]; logit_num = auc[("logit_plain", "numeric")]
    gap = float(knn_num.mean() - logit_num.mean())
    cross = {}
    for c in CODINGS:
        cross[f"logit_plain_{c}_minus_knn_numeric"] = paired(auc[("logit_plain", c)], knn_num)
        cross[f"logit_plain_{c}_minus_mlp_numeric"] = paired(auc[("logit_plain", c)], auc[("mlp", "numeric")])
        cross[f"xgboost_frozen_minus_logit_plain_{c}"] = paired(a_xgb, auc[("logit_plain", c)])
    for c in ("delay_flags", "onehot"):
        cross[f"share_of_knn_logit_gap_closed_by_{c}"] = float((auc[("logit_plain", c)].mean() - logit_num.mean()) / gap)

    # ---- 3. coefficient tables of the recoded plain logit (all training rows) + fold stability
    # delay_flags: the plain (unpenalised) logit, which converges and matches statsmodels. onehot: the unpenalised fit has
    # no finite MLE (lbfgs stops at max_iter in every fold: rare levels such as PAY_3 = 8 or PAY_4 = 1, n = 1, and the
    # chained 6/7/8-month codes are quasi-separated or near-collinear), so its block comes from the L2 logit at the frozen
    # C = 31.62, which converges; its rare-level coefficients are still not interpretable (see n_train_rows_with_level).
    coef_tabs, coef_diag = [], {}
    for c, m in (("delay_flags", "logit_plain"), ("onehot", "logit_l2")):
        t, dgn = coefficient_table(X, y, c, s, fold_coefs[(m, c)], model=m)
        coef_tabs.append(t); coef_diag[f"{c}|{m}"] = dgn
    coefs = pd.concat(coef_tabs, ignore_index=True)
    coefs.to_csv(os.path.join(RESULTS, "pay_coding_logit_coefficients.csv"), index=False)
    show = coefs[(coefs.coding == "delay_flags") & coefs.feature.str.startswith("PAY_")]
    print(show[["feature", "coef", "coef_per_unit", "odds_ratio_per_unit", "p_value", "fold_sign_agreement"]].round(4)
          .to_string(index=False))

    # ---- summary JSON
    summary = dict(
        script="code/pay_coding_check.py", status="post-freeze sensitivity analysis, training 5-fold CV only; "
        "nothing scored on the held-out partition, no slug frozen", seed=SEED, n_train=int(len(y)),
        folds="common.CV = StratifiedKFold(5, shuffle=True, random_state=2607526); checked against data/processed/cv_folds.csv",
        nadeau_bengio="mine_v2_utils.corrected_t_ci(diffs, 4, 1): variance x (1/5 + 1/4), t with 4 df, 95%",
        sd_note="sd_auc is np.std over the 5 fold AUCs (ddof=0, as in the frozen cv_auc_std); sd_auc_ddof1 also given",
        versions=dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        codings=dict(numeric="PAY_k as one standardised number (frozen make_preprocessor())",
                     delay_flags="PAY_k_DEL = max(PAY_k,0), PAY_k_NC = 1[PAY_k=-2], PAY_k_PD = 1[PAY_k=-1]; all standardised",
                     onehot="PAY_k one-hot per month, OneHotEncoder(drop='first', handle_unknown='ignore'), unscaled"),
        model_settings={m: {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v) for k, v in s[m].items()}
                        for m in MODELS},
        reproduction_of_frozen_numeric_oof=repro,
        cv_auc={f"{m}|{c}": float(auc[(m, c)].mean()) for m in MODELS for c in CODINGS},
        knn_minus_logit_numeric_cv_gap=gap, cross_model=cross,
        xgboost_frozen_cv_auc=float(a_xgb.mean()),
        coefficient_fit=coef_diag,
        seconds_per_model_coding={f"{m}|{c}": diag[(m, c)]["seconds"] for m in MODELS for c in CODINGS},
        pay1_code1=dict(n=int(p1.n), default_rate=float(p1.default_rate),
                        share_sep_bill_nonpositive=float(p1.share_same_month_bill_nonpositive),
                        n_sep_bill_nonpositive=int(((X["BILL_AMT1"] <= 0) & m1).sum()),
                        share_pay2_eq_2=float(p1.share_month_before_code_2), n_pay2_eq_2=int(((X["PAY_2"] == 2) & m1).sum()),
                        n_all_six_bills_zero=int((m1 & zero6).sum()), default_rate_all_six_bills_zero=float(y[m1 & zero6].mean()),
                        n_pay2_to_6_all_minus2=int((m1 & (X[PAY_COLS[1:]] == -2).all(axis=1)).sum()),
                        code1_count_by_month=code1),
        files=["results/pay_coding_cv.csv", "results/pay_code_default_rates.csv",
               "results/pay_coding_logit_coefficients.csv", "results/pay_coding_summary.json"],
        seconds=round(time.time() - T0, 1))
    json.dump(summary, open(os.path.join(RESULTS, "pay_coding_summary.json"), "w"), indent=1, default=float)
    log("done: results/pay_coding_cv.csv, pay_code_default_rates.csv, pay_coding_logit_coefficients.csv, pay_coding_summary.json")


if __name__ == "__main__":
    main()
