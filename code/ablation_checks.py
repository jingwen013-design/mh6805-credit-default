"""
ablation_checks.py - where the nonlinear gain comes from.   Training rows only.

PURPOSE
    Three ablations on the shared five-fold CV, all fitted with the FROZEN hyper-parameters of
    two models: XGBoost (results/xgboost.json: learning rate 0.03, 300 rounds, depth 3,
    min_child_weight 5, subsample and colsample_bytree 0.8, hist, n_jobs 2, seed 2607526) and the
    plain unpenalised logit (results/logistic.json: C = inf, lbfgs, tol 1e-8, max_iter 5000,
    StandardScaler + drop-first one-hot inside the pipeline).  Only the column set, or for (a) the
    tree depth, changes.  No frozen model is refitted in place and no new model is scored on
    held-out data: every number is a training out-of-fold (OOF) number.

    (a) Interaction depth.  XGBoost max_depth 1 (a sum of one-variable step functions, so purely
        additive in its inputs), 2, 3 (frozen) and 4, on the frozen 29 predictors and on the 23
        RAW predictors only (no engineered utilisation / delay aggregates).  UTIL_1 = BILL_AMT1 /
        LIMIT_BAL, UTIL_MEAN, PAY_RATIO_MEAN, BILL_TREND, MAX_DELAY and N_MONTHS_DELAYED are
        themselves interactions of raw columns, so depth 1 on the engineered set is additive in
        the ENGINEERED features only; the raw-column pair shows how much interaction the
        engineering already encodes (difference-in-differences of the depth gain).
        Sensitivity: the number of rounds re-chosen per depth by inner 3-fold CV inside each outer
        training fold from {300, 600, 1200, 2400} (a shallow booster may need more rounds).
    (b) Information sets.  H0 = LIMIT_BAL + demographics (SEX, EDUCATION, MARRIAGE, AGE);
        H1 = H0 + PAY_1, BILL_AMT1, PAY_AMT1 (September only); H3 = H0 + months 1-3;
        H6 = H0 + months 1-6 (= the 23 raw predictors).  Each also with the six engineered
        features that summarise all six months (H6 + summaries = the frozen 29-predictor set),
        and, as a third variant, with the same six summaries recomputed row-wise over months
        1..k only ("window summaries"; row-wise algebra, so no fold leakage).
    (c) Demographics.  The frozen set minus SEX, EDUCATION, MARRIAGE and AGE (25 predictors)
        against the full 29, identical folds, XGBoost and logit; AUC, AP, Brier, log loss, plus
        recall and 5:1 cost at a cross-fitted cut-off (each fold's cut-off chosen on the other
        four folds' OOF probabilities, as in mine_v2_utils.cross_fitted_cost), and a paired
        bootstrap (2,000 resamples of clients, seed 2607526) on the pooled OOF rows.

    Paired comparisons report the five per-fold values, the mean difference, the number of folds
    where the difference is positive (and where it favours the first-named specification), and a
    Nadeau-Bengio corrected t interval (mine_v2_utils.corrected_t_ci; one 5-fold CV, so the
    correction uses n_val / n_fit = 1/4).

TRAINING-ONLY
    Reads data/processed/train.csv directly and asserts 20,975 rows.  It never opens the held-out
    file, never calls common.load_split() or common.evaluate(), and writes only new files.
    Sanity check: the depth-3 / 29-predictor XGBoost and the 29-predictor logit reproduce the
    frozen OOF vectors results/xgboost_p_oof.npy and results/logistic_p_oof.npy exactly (asserted).

CAVEATS FOR READING (a) (added by the independent verifier, 21 Oct 2026; no number changed)
    - At the frozen 300 rounds the depth-1 booster is under-fitted (inner CV picks 1,200-2,400 rounds), so the
      300-round depth-3 minus depth-1 gain (+0.0107 AUC) overstates the interaction share about threefold; quote the
      rounds-tuned pair (+0.0036 [+0.0022, +0.0050], 5/5 folds).
    - The depth-3 arm is the frozen configuration, which was ranked first by CV AUC on these same five folds
      (results/xgb_grid.csv), and the other frozen settings (rate, min_child_weight, sampling) were tuned at depth 3;
      the depth-1 arm's rounds are chosen by nested inner CV.  Any bias therefore favours depth 3, so the tuned
      interaction gain is, if anything, an upper estimate.
    - With rounds tuned in every arm, the raw-minus-engineered difference-in-differences is +0.0017 [-0.0014, +0.0048],
      positive in 5/5 folds (the 300-round value, +0.0006, is 3/5): the engineered features absorb a little
      interaction (about a third of the raw-column depth gain of +0.0053), not none.  Say "little", not "no".

INPUTS
    data/processed/train.csv, data/processed/cv_folds.csv (fold labels, checked against common.CV),
    results/xgboost.json, results/logistic.json, results/xgboost_p_oof.npy, results/logistic_p_oof.npy.

OUTPUTS
    results/ablation_depth.csv, results/ablation_history.csv, results/ablation_demographics.csv
        (long format: one row per specification x metric, or per comparison x metric; columns
        fold_1..fold_5 follow cv_folds.csv folds 1..5)
    results/ablation_checks_summary.json   key numbers, set definitions, sanity checks, run time
    results/ablation_oof_xgb_nodemo.npy, results/ablation_oof_logit_nodemo.npy
        training OOF probabilities of the demographics-free XGBoost and logit (train.csv row order)
    figures/ablation_summary.png           one panel per question

RUN TIME
    About 6-8 minutes on a laptop (XGBoost n_jobs = 2, no outer parallelism; 356 s and 496 s on two runs of
    21 Oct 2026): the rounds-tuned sensitivity of (a) takes 4-6 minutes and the paired bootstrap of (c) about
    1.5 minutes.  Output is deterministic: both runs gave identical numbers.
    Run from Project/code:  /opt/miniconda3/bin/python3 ablation_checks.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAT_COLS, NUM_COLS, PAY_COLS, BILL_COLS, PAYAMT_COLS, ENG_COLS, CV, SEED, ROOT, RESULTS, FIGURES,
                    TARGET, COST_FN, COST_FP)
from mine_v2_utils import corrected_t_ci, best_threshold, cost_at, delta_metric_ci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
import xgboost
from xgboost import XGBClassifier

T0 = time.time()
np.random.seed(SEED)
N_ROWS = 20975
RATIO = COST_FN / COST_FP                    # 5:1, the project's assumed cost of a missed defaulter vs a false alarm
ROUNDS_GRID = [300, 600, 1200, 2400]         # sensitivity of (a): rounds re-chosen per depth by inner CV
INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
N_BOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data (training only)
TRAIN_CSV = os.path.join(ROOT, "data", "processed", "train.csv")
tr = pd.read_csv(TRAIN_CSV)
assert len(tr) == N_ROWS, f"expected {N_ROWS:,} training rows, got {len(tr):,}"
y = tr[TARGET].astype(int).values
X = tr.drop(columns=[TARGET]).copy()
FOLDS = list(CV.split(np.zeros(len(y)), y))
fold_lab = np.empty(len(y), int)
for k, (_, va) in enumerate(FOLDS):
    fold_lab[va] = k + 1
cvf = pd.read_csv(os.path.join(ROOT, "data", "processed", "cv_folds.csv"))
assert (cvf["fold"].values == fold_lab).all(), "common.CV no longer matches data/processed/cv_folds.csv"

# frozen settings, read from the frozen result files rather than retyped
XB = json.load(open(os.path.join(RESULTS, "xgboost.json")))["best_params"]
XGB_FROZEN = dict(objective=XB["objective"], eval_metric="auc", tree_method=XB["tree_method"], n_jobs=2,
                  random_state=XB["random_state"], subsample=XB["subsample"], colsample_bytree=XB["colsample_bytree"],
                  learning_rate=XB["learning_rate"], min_child_weight=XB["min_child_weight"],
                  scale_pos_weight=XB["scale_pos_weight"])
FROZEN_DEPTH, FROZEN_ROUNDS = int(XB["max_depth"]), int(XB["n_estimators"])
assert (FROZEN_DEPTH, FROZEN_ROUNDS, XB["learning_rate"]) == (3, 300, 0.03)
LB = json.load(open(os.path.join(RESULTS, "logistic.json")))["best_params"]
assert str(LB["C"]).startswith("inf") and LB["solver"] == "lbfgs" and int(LB["max_iter"]) == 5000
LOGIT_KW = dict(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-8, random_state=SEED)   # tol as in model_logistic.py

# ------------------------------------------------------------------------------------------------ feature sets
DEMO_NUM, DEMO_CAT = ["AGE"], list(CAT_COLS)
RAW_NUM = [c for c in NUM_COLS if c not in ENG_COLS]                 # 20 numeric raw + 3 categorical = 23 raw predictors
assert len(RAW_NUM) + len(CAT_COLS) == 23 and len(NUM_COLS) + len(CAT_COLS) == 29


def months(k):
    """Raw repayment-history columns for months 1..k (1 = September 2005, the latest)."""
    return PAY_COLS[:k] + BILL_COLS[:k] + PAYAMT_COLS[:k]


def window_summaries(df, k):
    """The six engineered features recomputed row-wise over months 1..k only (same formulas as common.clean)."""
    lim = df["LIMIT_BAL"].astype(float)
    b, p, s = df[BILL_COLS[:k]].values, df[PAYAMT_COLS[:k]].values, df[PAY_COLS[:k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nanmean(np.where(b > 0, p / np.maximum(b, 1.0), np.nan), axis=1)
    out = pd.DataFrame(index=df.index)
    out[f"UTIL_MEAN_W{k}"] = df[BILL_COLS[:k]].mean(axis=1) / lim
    out[f"PAY_RATIO_MEAN_W{k}"] = np.nan_to_num(np.clip(ratio, 0, 5), nan=0.0)
    out[f"N_MONTHS_DELAYED_W{k}"] = (s > 0).sum(axis=1)
    out[f"MAX_DELAY_W{k}"] = s.max(axis=1).clip(lower=0)
    out[f"BILL_TREND_W{k}"] = (df["BILL_AMT1"] - df[BILL_COLS[k - 1]]) / lim
    return out


import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)                  # nanmean of all-NaN rows -> NaN -> 0, as in common.clean
    W6 = window_summaries(X, 6)
for c in ["UTIL_MEAN", "PAY_RATIO_MEAN", "N_MONTHS_DELAYED", "MAX_DELAY", "BILL_TREND"]:   # formulas match common.clean
    assert np.allclose(W6[f"{c}_W6"].values, X[c].values, rtol=0, atol=1e-9), c
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    W1, W3 = window_summaries(X, 1), window_summaries(X, 3)
# with k = 1 the mean utilisation equals UTIL_1 and the bill trend is identically 0, so both are dropped
W1 = W1.drop(columns=["UTIL_MEAN_W1", "BILL_TREND_W1"])
X = pd.concat([X, W1, W3], axis=1)
WIN = {1: ["UTIL_1"] + list(W1.columns), 3: ["UTIL_1"] + list(W3.columns), 6: list(ENG_COLS)}

H_RAW = {0: ["LIMIT_BAL", "AGE"], 1: ["LIMIT_BAL", "AGE"] + months(1), 3: ["LIMIT_BAL", "AGE"] + months(3),
         6: ["LIMIT_BAL", "AGE"] + months(6)}
# order the numeric columns as in NUM_COLS (XGBoost's column sampling depends on column order)
H_RAW = {k: [c for c in NUM_COLS if c in v] for k, v in H_RAW.items()}
assert H_RAW[6] == RAW_NUM

SPECS = {   # name -> (numeric columns, categorical columns, short description)
    "full29": (list(NUM_COLS), list(CAT_COLS), "frozen 29 predictors (23 raw + 6 engineered)"),
    "raw23": (list(RAW_NUM), list(CAT_COLS), "23 raw predictors, no engineered features"),
    "nodemo25": ([c for c in NUM_COLS if c not in DEMO_NUM], [], "frozen set minus SEX, EDUCATION, MARRIAGE, AGE"),
}
for k in (0, 1, 3, 6):
    SPECS[f"H{k}"] = (H_RAW[k], list(CAT_COLS), f"H{k}: LIMIT_BAL + demographics" + ("" if k == 0 else f" + raw months 1-{k}"))
    SPECS[f"H{k}+S6"] = (H_RAW[k] + list(ENG_COLS), list(CAT_COLS), f"H{k} + the six six-month summaries")
for k in (1, 3, 6):
    SPECS[f"H{k}+W{k}"] = (H_RAW[k] + WIN[k], list(CAT_COLS), f"H{k} + summaries recomputed over months 1-{k}")
assert SPECS["H6+S6"][0] == SPECS["full29"][0] and SPECS["H6+W6"][0] == SPECS["full29"][0]
assert SPECS["H6"][0] == SPECS["raw23"][0]


def n_pred(spec):
    num, cat, _ = SPECS[spec]
    return len(num) + len(cat)


def make_pre(num, cat, scale):
    """Same transformer as common.make_preprocessor(), restricted to the given columns."""
    parts = [("num", StandardScaler() if scale else "passthrough", num)]
    if cat:
        parts.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat))
    return ColumnTransformer(parts, remainder="drop")


# ------------------------------------------------------------------------------------------------ fitting
CACHE = {}


def oof_xgb(spec, depth, rounds=FROZEN_ROUNDS):
    key = ("xgb", spec, depth, rounds)
    if key not in CACHE:
        num, cat, _ = SPECS[spec]; p = np.zeros(len(y)); t = time.time()
        for fit_idx, va in FOLDS:
            m = Pipeline([("pre", make_pre(num, cat, scale=False)),
                          ("clf", XGBClassifier(**XGB_FROZEN, max_depth=depth, n_estimators=rounds))])
            m.fit(X.iloc[fit_idx], y[fit_idx])
            p[va] = m.predict_proba(X.iloc[va])[:, 1]
        CACHE[key] = p
        log(f"XGBoost {spec:9s} depth {depth} rounds {rounds:4d}: CV AUC {fold_metric(p, roc_auc_score).mean():.4f} ({time.time() - t:.1f}s)")
    return CACHE[key]


def oof_xgb_tuned(spec, depth):
    """Rounds chosen per outer fold by inner 3-fold CV on that outer training fold only (AUC, grid ROUNDS_GRID).
    Boosting is sequential and xgboost's row sampling is seeded per round, so the first m rounds of a 2,400-round
    fit equal an m-round fit (checked at start-up)."""
    key = ("xgb_tuned", spec, depth)
    if key not in CACHE:
        num, cat, _ = SPECS[spec]; p = np.zeros(len(y)); chosen = []; inner_tab = []; t = time.time()
        top = max(ROUNDS_GRID)
        for fit_idx, va in FOLDS:
            Xf, yf = X.iloc[fit_idx], y[fit_idx]
            scores = np.zeros(len(ROUNDS_GRID))
            for a, b in INNER.split(np.zeros(len(yf)), yf):
                pre = make_pre(num, cat, scale=False).fit(Xf.iloc[a])
                clf = XGBClassifier(**XGB_FROZEN, max_depth=depth, n_estimators=top).fit(pre.transform(Xf.iloc[a]), yf[a])
                Xb = pre.transform(Xf.iloc[b])
                scores += [roc_auc_score(yf[b], clf.predict_proba(Xb, iteration_range=(0, m))[:, 1]) for m in ROUNDS_GRID]
            m_best = ROUNDS_GRID[int(np.argmax(scores))]
            chosen.append(m_best); inner_tab.append((scores / INNER.get_n_splits()).round(5).tolist())
            pre = make_pre(num, cat, scale=False).fit(Xf)
            clf = XGBClassifier(**XGB_FROZEN, max_depth=depth, n_estimators=m_best).fit(pre.transform(Xf), yf)
            p[va] = clf.predict_proba(pre.transform(X.iloc[va]))[:, 1]
        CACHE[key] = (p, chosen, inner_tab)
        log(f"XGBoost {spec:9s} depth {depth} rounds tuned {chosen}: CV AUC {fold_metric(p, roc_auc_score).mean():.4f} "
            f"({time.time() - t:.1f}s)")
    return CACHE[key]


LOGIT_ITERS = {}


def oof_logit(spec):
    key = ("logit", spec)
    if key not in CACHE:
        num, cat, _ = SPECS[spec]; p = np.zeros(len(y)); its = []
        for fit_idx, va in FOLDS:
            m = Pipeline([("pre", make_pre(num, cat, scale=True)), ("clf", LogisticRegression(**LOGIT_KW))])
            m.fit(X.iloc[fit_idx], y[fit_idx])
            p[va] = m.predict_proba(X.iloc[va])[:, 1]
            its.append(int(m.named_steps["clf"].n_iter_[0]))
        assert max(its) < LOGIT_KW["max_iter"], f"logit on {spec} did not converge"
        CACHE[key] = p; LOGIT_ITERS[spec] = its
        log(f"logit   {spec:9s}: CV AUC {fold_metric(p, roc_auc_score).mean():.4f} (lbfgs iterations {its})")
    return CACHE[key]


# ------------------------------------------------------------------------------------------------ metrics
def _ll(yy, pp):
    return log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1])


METRICS = {"auc": roc_auc_score, "ap": average_precision_score, "brier": brier_score_loss, "log_loss": _ll}
HIGHER_BETTER = {"auc": True, "ap": True, "brier": False, "log_loss": False, "recall_5to1": True, "cost_5to1": False}


def fold_metric(p, fn):
    return np.array([fn(y[va], p[va]) for _, va in FOLDS])


def crossfit_cut(p):
    """Per fold: cut-off minimising 5*FN + FP on the other four folds' OOF probabilities; recall and cost on this fold."""
    rec, cost, cut = [], [], []
    for _, va in FOLDS:
        others = np.setdiff1d(np.arange(len(y)), va)
        t = best_threshold(y[others], p[others], RATIO)
        flag = p[va] >= t
        rec.append(float(np.sum(flag & (y[va] == 1)) / np.sum(y[va] == 1)))
        cost.append(float(cost_at(y[va], p[va], t, RATIO))); cut.append(t)
    return np.array(rec), np.array(cost), cut


def per_fold(p, extra=False):
    out = {m: fold_metric(p, fn) for m, fn in METRICS.items()}
    if extra:
        out["recall_5to1"], out["cost_5to1"], _ = crossfit_cut(p)
    return out


FOLD_COLS = [f"fold_{k}" for k in range(1, 6)]


def spec_rows(question, model, spec, label, p, depth=None, rounds=None, extra=False, note=""):
    rows = []
    pf = per_fold(p, extra)
    for m, v in pf.items():
        pooled = (METRICS[m](y, p) if m in METRICS else (float(v.sum()) if m == "cost_5to1" else np.nan))
        rows.append(dict(question=question, kind="cv", model=model, spec=label, spec_ref="", feature_set=spec,
                         n_predictors=n_pred(spec), max_depth=depth if depth is not None else "", n_estimators=rounds if rounds is not None else "",
                         metric=m, **dict(zip(FOLD_COLS, np.round(v, 6))), mean=round(float(v.mean()), 6),
                         sd=round(float(v.std(ddof=1)), 6), pooled_oof=round(float(pooled), 6), n_pos="", n_better="",
                         nb_lo="", nb_hi="", boot_lo="", boot_hi="", note=note))
    return rows, pf


def paired_rows(question, model, label_a, label_b, pf_a, pf_b, p_a=None, p_b=None, boot=False, note=""):
    """Difference A minus B, per fold, for every metric in both dicts."""
    rows = []
    for m in pf_a:
        if m not in pf_b:
            continue
        d = pf_a[m] - pf_b[m]
        ci = corrected_t_ci(d, 4, 1)                          # n_fit : n_val = 4 : 1 for one 5-fold CV
        better = int(np.sum(d > 0)) if HIGHER_BETTER[m] else int(np.sum(d < 0))
        r = dict(question=question, kind="paired_diff", model=model, spec=label_a, spec_ref=label_b, feature_set="",
                 n_predictors="", max_depth="", n_estimators="", metric=m, **dict(zip(FOLD_COLS, np.round(d, 6))),
                 mean=round(ci["mean"], 6), sd=round(float(d.std(ddof=1)), 6), pooled_oof="", n_pos=int(np.sum(d > 0)),
                 n_better=better, nb_lo=round(ci["lo"], 6), nb_hi=round(ci["hi"], 6), boot_lo="", boot_hi="", note=note)
        if boot and m in METRICS and p_a is not None:
            b = delta_metric_ci(y, p_b, p_a, metric=METRICS[m], B=N_BOOT, seed=SEED)      # returns metric(p_a) - metric(p_b)
            r.update(pooled_oof=round(b["estimate"], 6), boot_lo=round(b["lo"], 6), boot_hi=round(b["hi"], 6))
        rows.append(r)
    return rows


def get(rows, model, spec, metric, field="mean"):
    for r in rows:
        if r["model"] == model and r["spec"] == spec and r["metric"] == metric and r["kind"] == "cv":
            return r[field]
    raise KeyError((model, spec, metric))


def getd(rows, model, a, b, metric):
    for r in rows:
        if r["kind"] == "paired_diff" and r["model"] == model and r["spec"] == a and r["spec_ref"] == b and r["metric"] == metric:
            return {k: r[k] for k in ["mean", "nb_lo", "nb_hi", "n_pos", "n_better"] + FOLD_COLS + ["boot_lo", "boot_hi", "pooled_oof"]}
    raise KeyError((model, a, b, metric))


# ------------------------------------------------------------------------------------------------ sanity checks
log(f"train.csv: {len(y):,} rows, default rate {y.mean():.4f}; folds match cv_folds.csv")
p_frozen_xgb = oof_xgb("full29", FROZEN_DEPTH)
p_frozen_logit = oof_logit("full29")
dx = float(np.abs(p_frozen_xgb - np.load(os.path.join(RESULTS, "xgboost_p_oof.npy"))).max())
dl = float(np.abs(p_frozen_logit - np.load(os.path.join(RESULTS, "logistic_p_oof.npy"))).max())
assert dx < 1e-9 and dl < 1e-9, (dx, dl)
log(f"frozen reproduction: max |OOF - frozen OOF| XGBoost {dx:.1e}, logit {dl:.1e}")
_pre = make_pre(*SPECS["full29"][:2], scale=False).fit(X.iloc[FOLDS[0][0]])
_a = XGBClassifier(**XGB_FROZEN, max_depth=1, n_estimators=600).fit(_pre.transform(X.iloc[FOLDS[0][0]]), y[FOLDS[0][0]])
_b = XGBClassifier(**XGB_FROZEN, max_depth=1, n_estimators=300).fit(_pre.transform(X.iloc[FOLDS[0][0]]), y[FOLDS[0][0]])
_Xv = _pre.transform(X.iloc[FOLDS[0][1]])
d_trunc = float(np.abs(_a.predict_proba(_Xv, iteration_range=(0, 300))[:, 1] - _b.predict_proba(_Xv)[:, 1]).max())
assert d_trunc < 1e-9, d_trunc

# ================================================================================================ (a) interaction depth
log("(a) interaction depth")
A = []; PF = {}
for spec, tag in [("full29", "29 predictors (engineered)"), ("raw23", "23 raw predictors")]:
    r, PF[("logit", spec)] = spec_rows("a_depth", "logit", spec, f"logit | {tag}", oof_logit(spec), note="frozen plain logit settings")
    A += r
    for d in (1, 2, 3, 4):
        note = "frozen XGBoost" if (spec == "full29" and d == FROZEN_DEPTH) else "frozen XGBoost settings, depth changed"
        r, PF[("xgb", spec, d)] = spec_rows("a_depth", "XGBoost", spec, f"depth {d} | {tag}", oof_xgb(spec, d), depth=d,
                                            rounds=FROZEN_ROUNDS, note=note)
        A += r
    for d in (1, 2, 3, 4):
        p, chosen, inner = oof_xgb_tuned(spec, d)
        r, PF[("xgbt", spec, d)] = spec_rows("a_depth_rounds_tuned", "XGBoost", spec, f"depth {d}, rounds tuned | {tag}", p, depth=d,
                                             rounds="/".join(map(str, chosen)),
                                             note="rounds per outer fold chosen by inner 3-fold CV from " + str(ROUNDS_GRID))
        A += r
for spec, tag in [("full29", "29 predictors (engineered)"), ("raw23", "23 raw predictors")]:
    for d in (2, 3, 4):
        A += paired_rows("a_depth", "XGBoost", f"depth {d} | {tag}", f"depth 1 | {tag}", PF[("xgb", spec, d)], PF[("xgb", spec, 1)],
                         note="interaction gain over the additive booster")
    A += paired_rows("a_depth", "XGBoost", f"depth 1 | {tag}", f"logit | {tag}", PF[("xgb", spec, 1)], PF[("logit", spec)],
                     note="additive nonlinearity: step / U-shaped one-variable effects that a linear logit cannot fit")
    A += paired_rows("a_depth", "XGBoost", f"depth 3 | {tag}", f"logit | {tag}", PF[("xgb", spec, 3)], PF[("logit", spec)],
                     note="total gap at the frozen depth")
    for d in (2, 3, 4):
        A += paired_rows("a_depth_rounds_tuned", "XGBoost", f"depth {d}, rounds tuned | {tag}", f"depth 1, rounds tuned | {tag}",
                         PF[("xgbt", spec, d)], PF[("xgbt", spec, 1)], note="interaction gain, both arms with rounds tuned")
    A += paired_rows("a_depth_rounds_tuned", "XGBoost", f"depth 1, rounds tuned | {tag}", f"logit | {tag}", PF[("xgbt", spec, 1)],
                     PF[("logit", spec)], note="additive nonlinearity, rounds tuned")
    A += paired_rows("a_depth_rounds_tuned", "XGBoost", f"depth 1, rounds tuned | {tag}", f"depth 1 | {tag}", PF[("xgbt", spec, 1)],
                     PF[("xgb", spec, 1)], note="what extra rounds buy the additive booster")
E, R = "29 predictors (engineered)", "23 raw predictors"
for d in (1, 3):
    A += paired_rows("a_depth", "XGBoost", f"depth {d} | {E}", f"depth {d} | {R}", PF[("xgb", "full29", d)], PF[("xgb", "raw23", d)],
                     note="value of the six engineered features at this depth")
A += paired_rows("a_depth", "logit", f"logit | {E}", f"logit | {R}", PF[("logit", "full29")], PF[("logit", "raw23")],
                 note="value of the six engineered features to the logit")
did = {m: (PF[("xgb", "raw23", 3)][m] - PF[("xgb", "raw23", 1)][m]) - (PF[("xgb", "full29", 3)][m] - PF[("xgb", "full29", 1)][m])
       for m in METRICS}
A += paired_rows("a_depth", "XGBoost", f"(depth 3 - depth 1) | {R}", f"(depth 3 - depth 1) | {E}", did, {m: np.zeros(5) for m in METRICS},
                 note="difference-in-differences: interaction gain on raw columns minus that on the engineered set "
                      "(the interaction the engineered features already hard-code)")
did_t = {m: (PF[("xgbt", "raw23", 3)][m] - PF[("xgbt", "raw23", 1)][m]) - (PF[("xgbt", "full29", 3)][m] - PF[("xgbt", "full29", 1)][m])
         for m in METRICS}
A += paired_rows("a_depth_rounds_tuned", "XGBoost", f"(depth 3 - depth 1), rounds tuned | {R}", f"(depth 3 - depth 1), rounds tuned | {E}",
                 did_t, {m: np.zeros(5) for m in METRICS},
                 note="difference-in-differences with rounds tuned in every arm")
pd.DataFrame(A).to_csv(os.path.join(RESULTS, "ablation_depth.csv"), index=False)
log("wrote results/ablation_depth.csv")

# ================================================================================================ (b) information sets
log("(b) information sets")
B = []; PB = {}
VARIANTS = [("raw", lambda k: f"H{k}", "raw months only"),
            ("S6", lambda k: f"H{k}+S6", "+ six-month summaries"),
            ("W", lambda k: f"H{k}+W{k}", "+ window summaries (months 1-k)")]
for model in ("XGBoost", "logit"):
    for vtag, fn, vlab in VARIANTS:
        ks = (0, 1, 3, 6) if vtag != "W" else (1, 3, 6)
        for k in ks:
            spec = fn(k)
            p = oof_xgb(spec, FROZEN_DEPTH) if model == "XGBoost" else oof_logit(spec)
            note = SPECS[spec][2] + ("; = frozen 29-predictor model" if SPECS[spec][0] == SPECS["full29"][0] else "")
            r, PB[(model, vtag, k)] = spec_rows("b_history", model, spec, f"H{k} {vlab}", p,
                                                depth=FROZEN_DEPTH if model == "XGBoost" else None,
                                                rounds=FROZEN_ROUNDS if model == "XGBoost" else None, note=note)
            B += r
        steps = [(1, 0), (3, 1), (6, 3), (6, 1)] if vtag != "W" else [(3, 1), (6, 3), (6, 1)]
        for k1, k0 in steps:
            B += paired_rows("b_history", model, f"H{k1} {vlab}", f"H{k0} {vlab}", PB[(model, vtag, k1)], PB[(model, vtag, k0)],
                             note=f"incremental value of months {k0 + 1}-{k1}" if k0 > 0 else "incremental value of September (month 1)")
    for k in (0, 1, 3, 6):
        B += paired_rows("b_history", model, f"H{k} + six-month summaries", f"H{k} raw months only", PB[(model, "S6", k)],
                         PB[(model, "raw", k)], note="value of the six-month summaries given this window")
    for k in (1, 3):
        B += paired_rows("b_history", model, f"H{k} + six-month summaries", f"H{k} + window summaries (months 1-k)",
                         PB[(model, "S6", k)], PB[(model, "W", k)],
                         note="older history entering only through the summaries")
pd.DataFrame(B).to_csv(os.path.join(RESULTS, "ablation_history.csv"), index=False)
log("wrote results/ablation_history.csv")

# ================================================================================================ (c) demographics
log("(c) demographics")
Cr = []; PC = {}; P_ND = {}
for model in ("XGBoost", "logit"):
    for spec, lab in [("full29", "with demographics (29)"), ("nodemo25", "without demographics (25)")]:
        p = oof_xgb(spec, FROZEN_DEPTH) if model == "XGBoost" else oof_logit(spec)
        r, PC[(model, spec)] = spec_rows("c_demographics", model, spec, lab, p, depth=FROZEN_DEPTH if model == "XGBoost" else None,
                                         rounds=FROZEN_ROUNDS if model == "XGBoost" else None, extra=True,
                                         note=SPECS[spec][2] + ("; frozen model" if spec == "full29" else "; never scored on held-out rows"))
        Cr += r
        if spec == "nodemo25":
            P_ND[model] = p
    log(f"   bootstrap ({N_BOOT} resamples) for {model}")
    Cr += paired_rows("c_demographics", model, "with demographics (29)", "without demographics (25)", PC[(model, "full29")],
                      PC[(model, "nodemo25")], p_a=(p_frozen_xgb if model == "XGBoost" else p_frozen_logit), p_b=P_ND[model],
                      boot=True, note="incremental value of SEX, EDUCATION, MARRIAGE, AGE; boot_* = paired bootstrap over pooled OOF rows")
pd.DataFrame(Cr).to_csv(os.path.join(RESULTS, "ablation_demographics.csv"), index=False)
np.save(os.path.join(RESULTS, "ablation_oof_xgb_nodemo.npy"), P_ND["XGBoost"])
np.save(os.path.join(RESULTS, "ablation_oof_logit_nodemo.npy"), P_ND["logit"])
log("wrote results/ablation_demographics.csv and the two demographics-free OOF vectors")

# ================================================================================================ summary json
def dsum(rows, model, a, b, metric):
    g = getd(rows, model, a, b, metric)
    out = dict(mean=g["mean"], nb95=[g["nb_lo"], g["nb_hi"]], folds_positive=g["n_pos"], folds_better=g["n_better"],
               per_fold=[g[c] for c in FOLD_COLS])
    if g["boot_lo"] != "":
        out.update(pooled_oof_diff=g["pooled_oof"], bootstrap95=[g["boot_lo"], g["boot_hi"]])
    return out


S = dict(
    script="code/ablation_checks.py", training_only=True, n_rows=int(len(y)), default_rate=round(float(y.mean()), 6),
    folds="common.CV = StratifiedKFold(5, shuffle=True, random_state=2607526); fold_k = cv_folds.csv fold k",
    nb_correction="corrected_t_ci(diffs, n_train=4, n_test=1): variance factor 1/5 + 1/4, t with 4 df",
    frozen_xgboost=dict(XGB_FROZEN, max_depth=FROZEN_DEPTH, n_estimators=FROZEN_ROUNDS),
    frozen_logit=dict(C="inf", solver="lbfgs", max_iter=5000, tol=1e-8),
    sanity=dict(max_abs_diff_vs_frozen_xgboost_oof=dx, max_abs_diff_vs_frozen_logit_oof=dl,
                truncated_vs_short_fit_max_abs_diff=d_trunc, logit_lbfgs_iterations=LOGIT_ITERS),
    feature_sets={k: dict(numeric=v[0], categorical=v[1], description=v[2], n_predictors=n_pred(k)) for k, v in SPECS.items()},
    a_depth=dict(
        cv_auc={f"{m}|{s}": get(A, mod, f"{m} | {t}", "auc") for s, t in [("full29", E), ("raw23", R)]
                for mod, m in [("logit", "logit")] + [("XGBoost", f"depth {d}") for d in (1, 2, 3, 4)]
                + [("XGBoost", f"depth {d}, rounds tuned") for d in (1, 2, 3, 4)]},
        rounds_chosen={f"depth {d}|{s}": CACHE[("xgb_tuned", s, d)][1] for s in ("full29", "raw23") for d in (1, 2, 3, 4)},
        inner_cv_auc_by_rounds={f"depth {d}|{s}": dict(grid=ROUNDS_GRID, per_outer_fold=CACHE[("xgb_tuned", s, d)][2])
                                for s in ("full29", "raw23") for d in (1, 2, 3, 4)},
        auc_tuned_did_raw_minus_engineered=dsum(A, "XGBoost", f"(depth 3 - depth 1), rounds tuned | {R}",
                                                f"(depth 3 - depth 1), rounds tuned | {E}", "auc"),
        auc_tuned_depth1_minus_logit_raw=dsum(A, "XGBoost", f"depth 1, rounds tuned | {R}", f"logit | {R}", "auc"),
        auc_depth3_minus_logit_raw=dsum(A, "XGBoost", f"depth 3 | {R}", f"logit | {R}", "auc"),
        auc_engineered_minus_raw_depth1=dsum(A, "XGBoost", f"depth 1 | {E}", f"depth 1 | {R}", "auc"),
        auc_engineered_minus_raw_depth3=dsum(A, "XGBoost", f"depth 3 | {E}", f"depth 3 | {R}", "auc"),
        auc_engineered_minus_raw_logit=dsum(A, "logit", f"logit | {E}", f"logit | {R}", "auc"),
        auc_depth3_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 3 | {E}", f"depth 1 | {E}", "auc"),
        auc_depth2_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 2 | {E}", f"depth 1 | {E}", "auc"),
        auc_depth4_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 4 | {E}", f"depth 1 | {E}", "auc"),
        auc_depth1_minus_logit_engineered=dsum(A, "XGBoost", f"depth 1 | {E}", f"logit | {E}", "auc"),
        auc_depth3_minus_logit_engineered=dsum(A, "XGBoost", f"depth 3 | {E}", f"logit | {E}", "auc"),
        auc_depth3_minus_depth1_raw=dsum(A, "XGBoost", f"depth 3 | {R}", f"depth 1 | {R}", "auc"),
        auc_depth1_minus_logit_raw=dsum(A, "XGBoost", f"depth 1 | {R}", f"logit | {R}", "auc"),
        auc_did_raw_minus_engineered=dsum(A, "XGBoost", f"(depth 3 - depth 1) | {R}", f"(depth 3 - depth 1) | {E}", "auc"),
        auc_tuned_depth3_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 3, rounds tuned | {E}", f"depth 1, rounds tuned | {E}", "auc"),
        auc_tuned_depth3_minus_depth1_raw=dsum(A, "XGBoost", f"depth 3, rounds tuned | {R}", f"depth 1, rounds tuned | {R}", "auc"),
        auc_tuned_depth1_minus_logit_engineered=dsum(A, "XGBoost", f"depth 1, rounds tuned | {E}", f"logit | {E}", "auc"),
        ap_depth3_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 3 | {E}", f"depth 1 | {E}", "ap"),
        brier_depth3_minus_depth1_engineered=dsum(A, "XGBoost", f"depth 3 | {E}", f"depth 1 | {E}", "brier"),
    ),
    b_history=dict(
        cv_auc={f"{mod}|H{k} {vlab}": get(B, mod, f"H{k} {vlab}", "auc") for mod in ("XGBoost", "logit") for vt, _, vlab in VARIANTS
                for k in ((0, 1, 3, 6) if vt != "W" else (1, 3, 6))},
        increments_auc={f"{mod}|{vlab}|H{k1}-H{k0}": dsum(B, mod, f"H{k1} {vlab}", f"H{k0} {vlab}", "auc")
                        for mod in ("XGBoost", "logit") for vt, _, vlab in VARIANTS
                        for k1, k0 in ([(1, 0), (3, 1), (6, 3), (6, 1)] if vt != "W" else [(3, 1), (6, 3), (6, 1)])},
        summaries_value_auc={f"{mod}|H{k}": dsum(B, mod, f"H{k} + six-month summaries", f"H{k} raw months only", "auc")
                             for mod in ("XGBoost", "logit") for k in (0, 1, 3, 6)},
        six_month_vs_window_summaries_auc={f"{mod}|H{k}": dsum(B, mod, f"H{k} + six-month summaries",
                                                                  f"H{k} + window summaries (months 1-k)", "auc")
                                           for mod in ("XGBoost", "logit") for k in (1, 3)},
    ),
    c_demographics={mod: {m: dsum(Cr, mod, "with demographics (29)", "without demographics (25)", m)
                          for m in ("auc", "ap", "brier", "log_loss", "recall_5to1", "cost_5to1")} for mod in ("XGBoost", "logit")},
    c_cv_auc={f"{mod}|{lab}": get(Cr, mod, lab, "auc") for mod in ("XGBoost", "logit")
              for lab in ("with demographics (29)", "without demographics (25)")},
    oof_files=dict(xgb_nodemo="results/ablation_oof_xgb_nodemo.npy", logit_nodemo="results/ablation_oof_logit_nodemo.npy",
                   row_order="train.csv order", full_set_equivalents="results/xgboost_p_oof.npy, results/logistic_p_oof.npy"),
    versions=dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                  numpy=np.__version__, pandas=pd.__version__),
)

# ================================================================================================ figure
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C_XGB, C_LOG, C_TUNE = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 of the validated reference palette
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
MONO = dict(family="DejaVu Sans Mono", fontsize=8.2, color=INK)
fig, ax = plt.subplots(1, 3, figsize=(18.5, 7.6))
DM = "with demographics (29)", "without demographics (25)"


def ci_txt(g, nd=4):
    return f"{g['mean']:+.{nd}f} [{g['nb95'][0]:+.{nd}f}, {g['nb95'][1]:+.{nd}f}] {int(g['folds_better'])}/5"


# (a) interaction depth --------------------------------------------------------------------------------------------
a = ax[0]
series_a = [("full29", E, "", "-", True, C_XGB, "o", "XGBoost, 29 predictors, 300 rounds (frozen settings)"),
            ("full29", E, ", rounds tuned", "-", True, C_TUNE, "D", "XGBoost, 29 predictors, rounds re-tuned per depth"),
            ("raw23", R, "", "--", False, C_XGB, "o", "XGBoost, 23 raw predictors, 300 rounds"),
            ("raw23", R, ", rounds tuned", "--", False, C_TUNE, "D", "XGBoost, 23 raw predictors, rounds re-tuned per depth")]
for spec, tag, suf, ls, filled, col, mk, lab in series_a:
    ys = [get(A, "XGBoost", f"depth {d}{suf} | {tag}", "auc") for d in (1, 2, 3, 4)]
    a.plot([1, 2, 3, 4], ys, ls=ls, lw=2, color=col, marker=mk, ms=7.5 if mk == "o" else 6.5, mfc=col if filled else SURF,
           mec=col, mew=2, label=lab, zorder=3 if suf == "" else 2)
    a.annotate(f"{ys[0]:.4f}", (1, ys[0]), textcoords="offset points", xytext=(-11, -3.5), ha="right", fontsize=8.3, color=INK2)
yl = get(A, "logit", f"logit | {E}", "auc"); ylr = get(A, "logit", f"logit | {R}", "auc")
a.plot([0], [yl], ls="none", marker="s", ms=8.5, mfc=C_LOG, mec=C_LOG, mew=2, label="plain logit, 29 predictors (frozen)")
a.annotate(f"{yl:.4f}", (0, yl), textcoords="offset points", xytext=(11, -3.5), ha="left", fontsize=8.3, color=INK2)
y3e, y3r = get(A, "XGBoost", f"depth 3 | {E}", "auc"), get(A, "XGBoost", f"depth 3 | {R}", "auc")
a.annotate(f"{y3e:.4f}", (3, y3e), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.3, color=INK2)
a.annotate(f"{y3r:.4f}", (3, y3r), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8.3, color=INK2)
lo_a = yl - 0.0045
a.set_ylim(lo_a, y3e + 0.0035)
a.annotate(f"plain logit, 23 raw predictors: {ylr:.4f} (below the axis)", (0, lo_a), textcoords="offset points", xytext=(-4, 5),
           ha="left", fontsize=8, color=C_LOG)
a.set_xticks([0, 1, 2, 3, 4]); a.set_xticklabels(["plain\nlogit", "depth 1\n(additive)", "depth 2", "depth 3\n(frozen)", "depth 4"])
a.set_xlim(-0.45, 4.45); a.set_ylabel("CV AUC (mean of the 5 training folds)")
g_e = dsum(A, "XGBoost", f"depth 3 | {E}", f"depth 1 | {E}", "auc")
g_et = dsum(A, "XGBoost", f"depth 3, rounds tuned | {E}", f"depth 1, rounds tuned | {E}", "auc")
g_r = dsum(A, "XGBoost", f"depth 3 | {R}", f"depth 1 | {R}", "auc")
g_rt = dsum(A, "XGBoost", f"depth 3, rounds tuned | {R}", f"depth 1, rounds tuned | {R}", "auc")
g_al = dsum(A, "XGBoost", f"depth 1, rounds tuned | {E}", f"logit | {E}", "auc")
a.text(0.27, 0.035, "AUC gain, depth 3 minus depth 1\n(95% NB interval, folds better)\n"
                    f"29 pred., 300 rounds  {ci_txt(g_e)}\n29 pred., tuned       {ci_txt(g_et)}\n"
                    f"23 raw,   300 rounds  {ci_txt(g_r)}\n23 raw,   tuned       {ci_txt(g_rt)}\n"
                    f"Depth 1 (tuned) minus logit, 29 pred.:\n                      {ci_txt(g_al)}",
       transform=a.transAxes, va="bottom", ha="left", **MONO)
a.set_title("(a) How much of the gain needs interactions?", loc="left", fontweight="bold")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8.3, frameon=False)

# (b) information sets ---------------------------------------------------------------------------------------------
b = ax[1]
for model, col, mk in [("XGBoost", C_XGB, "o"), ("logit", C_LOG, "s")]:
    for vt, ls, filled, vlab in [("S6", "-", True, "+ six-month summaries"), ("raw", "--", False, "raw months only")]:
        ys = [get(B, model, f"H{k} {vlab}", "auc") for k in (0, 1, 3, 6)]
        b.plot([0, 1, 2, 3], ys, ls=ls, lw=2, color=col, marker=mk, ms=7.5, mfc=col if filled else SURF, mec=col, mew=2,
               label=f"{'XGBoost' if model == 'XGBoost' else 'plain logit'}, {vlab}" + (" (H6 = frozen model)" if vt == "S6" else ""))
        b.annotate(f"{ys[-1]:.4f}", (3, ys[-1]), textcoords="offset points", xytext=(9, -3.5), ha="left", fontsize=8.3, color=INK2)
b.set_xticks([0, 1, 2, 3]); b.set_xticklabels(["H0\nlimit +\ndemographics", "H1\n+ September", "H3\n+ Jul-Sep", "H6\n+ Apr-Sep"])
b.set_xlim(-0.3, 3.65); b.set_ylabel("CV AUC (mean of the 5 training folds)")
inc = {(vt, k1): dsum(B, "XGBoost", f"H{k1} {vl}", f"H{k0} {vl}", "auc") for vt, vl in [("raw", "raw months only"),
       ("S6", "+ six-month summaries")] for k1, k0 in [(3, 1), (6, 3)]}
b.text(0.30, 0.035, "XGBoost, AUC increment (95% NB, folds better)\n"
                    f"raw months only\n  H1->H3  {ci_txt(inc[('raw', 3)])}\n  H3->H6  {ci_txt(inc[('raw', 6)])}\n"
                    f"+ six-month summaries\n  H1->H3  {ci_txt(inc[('S6', 3)])}\n  H3->H6  {ci_txt(inc[('S6', 6)])}",
       transform=b.transAxes, va="bottom", ha="left", **MONO)
b.set_title("(b) What do older months add to September?", loc="left", fontweight="bold")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1, fontsize=8.3, frameon=False)

# (c) demographics -------------------------------------------------------------------------------------------------
c = ax[2]
rows_c = [("XGBoost", "auc", "XGBoost\nAUC"), ("XGBoost", "ap", "XGBoost\nAP"), ("logit", "auc", "plain logit\nAUC"),
          ("logit", "ap", "plain logit\nAP")]
for i, (model, m, lab) in enumerate(rows_c):
    g = dsum(Cr, model, *DM, m)
    yy = len(rows_c) - 1 - i; col = C_XGB if model == "XGBoost" else C_LOG
    c.plot(g["nb95"], [yy, yy], color=col, lw=2.2, solid_capstyle="round", zorder=2)
    c.scatter(g["per_fold"], [yy - 0.2] * 5, s=18, color=INK2, alpha=0.6, zorder=3, linewidths=0)
    c.plot([g["mean"]], [yy], marker="o" if model == "XGBoost" else "s", ms=9, color=col, mec=SURF, mew=1.5, zorder=4)
    c.annotate(ci_txt(g) + " folds better", (g["mean"], yy), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.3,
               color=INK2)
c.axvline(0, color=INK2, lw=1, ls=":")
c.set_yticks(range(len(rows_c))); c.set_yticklabels([r[2] for r in rows_c][::-1])
c.set_ylim(-0.6, len(rows_c) - 0.45)
xs = [v for mo, m, _ in rows_c for v in dsum(Cr, mo, *DM, m)["nb95"] + dsum(Cr, mo, *DM, m)["per_fold"]]
c.set_xlim(min(xs) - 0.0025, max(xs) + 0.0025)
c.set_xlabel("with minus without SEX, EDUCATION, MARRIAGE, AGE")
bx, bl = dsum(Cr, "XGBoost", *DM, "brier"), dsum(Cr, "logit", *DM, "brier")
c.text(0.0, -0.17, "Bar = 95% NB interval of the mean; grey dots = the five folds.\n"
                   f"Brier difference (negative = better):\n  XGBoost {ci_txt(bx, 5)}\n  logit   {ci_txt(bl, 5)}",
       transform=c.transAxes, va="top", ha="left", **MONO)
c.set_title("(c) Do demographics add anything?", loc="left", fontweight="bold")

for axx in ax:
    axx.grid(axis="x" if axx is c else "y", color=GRIDC, lw=0.8); axx.set_axisbelow(True)
fig.suptitle("Where XGBoost's gain over the plain logit comes from: ablations on the 20,975 training rows, frozen settings, "
             "the same five CV folds", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.008, "Training out-of-fold numbers only; nothing here is scored on held-out rows. XGBoost: rate 0.03, 300 rounds, "
                      "min_child_weight 5, subsample and colsample 0.8; only the depth or the columns change ('rounds re-tuned': chosen per outer "
                      "fold by inner 3-fold CV from 300-2,400).\nNB = Nadeau-Bengio corrected t interval (n_val/n_fit = 1/4). Six-month "
                      "summaries = UTIL_1, UTIL_MEAN, PAY_RATIO_MEAN, N_MONTHS_DELAYED, MAX_DELAY, BILL_TREND; raw predictors = LIMIT_BAL, "
                      "SEX, EDUCATION, MARRIAGE, AGE and the 18 monthly PAY / BILL_AMT / PAY_AMT columns.",
         fontsize=7.9, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.045, 1, 0.955))
fig.savefig(os.path.join(FIGURES, "ablation_summary.png"), dpi=160)
plt.close(fig)
log("wrote figures/ablation_summary.png")

S["run_seconds"] = round(time.time() - T0, 1)
json.dump(S, open(os.path.join(RESULTS, "ablation_checks_summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log(f"wrote results/ablation_checks_summary.json; done in {S['run_seconds']:.0f}s")
