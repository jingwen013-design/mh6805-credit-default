"""
fairness_check.py — group-level fairness diagnostics for sex, age, marital status and education.   Owner: LI Jingwen.

PURPOSE
    Audit findings d2-fairness-implications / fairness-ethics (professor comment A14: "discuss fairness implications").
    For the frozen XGBoost and the frozen plain logit, at their frozen 5:1 cost cut-offs, the script reports by group
    (SEX; AGE bands <30, 30-39, 40-49, 50+; MARRIAGE; EDUCATION after the project's recoding):
      n, observed default rate, mean predicted PD (calibration-in-the-large), flag rate (share with PD >= cut-off),
      true-positive rate, false-positive rate, precision, within-group AUC, and the ratios of flag rates, FPRs and
      TPRs to a reference group (the largest group of each attribute: female, 30-39, single, university).
    Each model appears in three variants:
      *_frozen       the frozen training out-of-fold probabilities (results/<slug>_p_oof.npy), read as they are;
      *_neutralised  the same fold models (frozen settings, refitted per fold in memory and checked against the frozen
                     OOF vector), but every validation client is given the reference values of all four attributes
                     (female, age 34 = training median, single, university) -> isolates the DIRECT use of the attributes;
      *_nodemo       a demographics-free refit (same settings, same folds) on the 25 non-demographic predictors
                     -> shows what remains once the attributes are removed (base rates and proxies).
    It also measures proxies: how well the 25 non-demographic predictors predict each attribute (out-of-fold AUC).
    Paired comparisons across the five folds report per-fold values, the mean difference, the number of folds with a
    positive difference and a Nadeau-Bengio corrected t interval (n_test / n_train = 1/4).

TRAINING ONLY
    Reads data/processed/train.csv (20,975 rows) and data/processed/cv_folds.csv. The held-out file is never opened,
    common.load_split() and common.evaluate() are never called, no frozen model or frozen result file is modified.
    From results/xgboost.json and results/logistic.json only best_params and the cut-off value are read
    (at_tuned_cost.threshold: chosen by tune_threshold(..., 'cost') on the TRAINING OOF probabilities; the metrics
    stored next to it are held-out metrics and are not read). The cut-off is re-derived from the OOF vector and the
    script stops if the two disagree.

INPUTS
    data/processed/train.csv, data/processed/cv_folds.csv, results/xgboost_p_oof.npy, results/logistic_p_oof.npy,
    results/xgboost.json, results/logistic.json, results/frozen_manifest.json (cut-off cross-check only),
    optional results/ablation_oof_xgb_nodemo.npy and results/ablation_oof_logit_nodemo.npy (written by
    code/ablation_checks.py, if present: compared with this script's own demographics-free refits; a vector that differs
    is reported as an extra variant).

OUTPUTS
    results/fairness_groups.csv        one row per model variant x group (plus an ALL row per variant)
    results/fairness_gap_folds.csv     per-fold group gaps, paired variant-minus-frozen differences, corrected t CIs
    results/fairness_proxies.csv       out-of-fold AUC / R^2 of predicting each attribute from the 25 other predictors
    results/fairness_summary.json      thresholds, overall paired comparisons, key gaps, proxies, plain-English notes
    results/fairness_oof_xgb_nodemo.npy, results/fairness_oof_logistic_nodemo.npy   demographics-free OOF vectors
    figures/fairness_flag_rates.png    flag rates by group, frozen vs demographics-free XGBoost, with default rates

RUN TIME
    About 30-60 seconds on 2 threads, depending on machine load (about 60 small XGBoost fits and 15 logit fits).
    Run from Project/code:  /opt/miniconda3/bin/python3 fairness_check.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (CAT_COLS, CV, FIGURES, NUM_COLS, RESULTS, ROOT, SEED, TARGET, make_preprocessor,  # noqa: E402
                    tune_threshold)
from mine_v2_utils import corrected_t_ci  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

T0 = time.time()
N_JOBS = 2
Z = stats.norm.ppf(0.975)
DEMO_COLS = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
FIN_COLS = [c for c in NUM_COLS if c not in DEMO_COLS]           # the 25 non-demographic predictors
assert len(FIN_COLS) == 25 and set(CAT_COLS) <= set(DEMO_COLS)


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ============================================================================ data (training rows only)
tr = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
assert len(tr) == 20975, f"expected 20,975 training rows, found {len(tr):,}"
y = tr[TARGET].astype(int).values
X = tr.drop(columns=[TARGET])
folds_csv = pd.read_csv(os.path.join(ROOT, "data", "processed", "cv_folds.csv"))
assert len(folds_csv) == len(y) and (folds_csv["DEFAULT"].values == y).all()
fold_id = folds_csv["fold"].values
FOLDS = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(1, 6)]
for (a, b), (c, d) in zip(FOLDS, CV.split(np.zeros(len(y)), y)):       # same assignment as common.CV
    assert np.array_equal(np.sort(a), np.sort(c)) and np.array_equal(np.sort(b), np.sort(d))
N_TRAIN_FOLD, N_TEST_FOLD = len(FOLDS[0][0]), len(FOLDS[0][1])           # 16,780 / 4,195 -> ratio 1/4
log(f"train.csv {len(y):,} rows, default rate {y.mean():.4f}; folds {N_TEST_FOLD:,} x 5")

# ============================================================================ groups
AGE_MEDIAN = float(np.median(X["AGE"]))
age_band = pd.cut(X["AGE"], bins=[0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str).values
GROUPS = {   # attribute -> (array of group labels per row, ordered labels, reference label)
    "SEX": (X["SEX"].map({1: "male", 2: "female"}).values, ["male", "female"], None),
    "AGE": (age_band, ["<30", "30-39", "40-49", "50+"], None),
    "MARRIAGE": (X["MARRIAGE"].map({1: "married", 2: "single", 3: "others"}).values, ["married", "single", "others"], None),
    "EDUCATION": (X["EDUCATION"].map({1: "graduate school", 2: "university", 3: "high school", 4: "others/unknown"}).values,
                  ["graduate school", "university", "high school", "others/unknown"], None),
}
for att, (lab, order, _) in list(GROUPS.items()):
    assert not pd.isna(lab).any() and set(lab) == set(order), att
    counts = pd.Series(lab).value_counts()
    GROUPS[att] = (lab, order, counts.idxmax())                           # reference = largest group
REF_VALUES = {"SEX": 2, "AGE": AGE_MEDIAN, "MARRIAGE": 2, "EDUCATION": 2}  # inputs used by the neutralised variant
assert [GROUPS[a][2] for a in DEMO_COLS] == ["female", "30-39", "single", "university"]
assert 30 <= AGE_MEDIAN <= 39

# ============================================================================ frozen OOF vectors and cut-offs
FROZEN = {}
man = json.load(open(os.path.join(RESULTS, "frozen_manifest.json")))
for key, slug in (("xgb", "xgboost"), ("logit", "logistic")):
    js = json.load(open(os.path.join(RESULTS, f"{slug}.json")))
    p = np.load(os.path.join(RESULTS, f"{slug}_p_oof.npy"))
    assert p.shape == (len(y),)
    t_json = float(js["at_tuned_cost"]["threshold"])                     # cut-off value only (chosen on training OOF)
    t_man = float(man["models"][slug]["thresholds_chosen_on_training_oof"]["cost"])
    t_oof, _ = tune_threshold(y, p, "cost")                              # re-derived from the training OOF vector
    assert abs(t_json - t_oof) < 1e-9 and abs(t_man - t_oof) < 1e-9, (slug, t_json, t_man, t_oof)
    assert abs(roc_auc_score(y, p) - js["oof_auc"]) < 1e-9
    FROZEN[key] = dict(slug=slug, p=p, t=t_oof, params=js["best_params"])
    log(f"{slug}: OOF AUC {roc_auc_score(y, p):.4f}, 5:1 cut-off {t_oof:.2f} (json {t_json}, manifest {t_man})")


def xgb_pipe(cols=None):
    params = {k: v for k, v in FROZEN["xgb"]["params"].items()}
    params.update(eval_metric="auc", n_jobs=N_JOBS)                      # as model_boosting.py XGB_FIXED
    pre = make_preprocessor(scale=False) if cols is None else ColumnTransformer([("num", "passthrough", cols)], remainder="drop")
    return Pipeline([("pre", pre), ("clf", XGBClassifier(**params))])


def logit_pipe(cols=None):
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-8, random_state=SEED)   # as model_logistic.py
    pre = make_preprocessor() if cols is None else ColumnTransformer([("num", StandardScaler(), cols)], remainder="drop")
    return Pipeline([("pre", pre), ("clf", clf)])


def neutralise(Xv):
    Xc = Xv.copy()
    for c, v in REF_VALUES.items():
        Xc[c] = v
    return Xc


# ============================================================================ fold refits (frozen settings, same folds)
P = {"xgb_frozen": FROZEN["xgb"]["p"], "logit_frozen": FROZEN["logit"]["p"]}
REPRO = {}
for key, make in (("xgb", xgb_pipe), ("logit", logit_pipe)):
    p_rep, p_neu, p_nod = (np.full(len(y), np.nan) for _ in range(3))
    for k, (a, b) in enumerate(FOLDS, start=1):
        m = clone(make()).fit(X.iloc[a], y[a])
        p_rep[b] = m.predict_proba(X.iloc[b])[:, 1]
        p_neu[b] = m.predict_proba(neutralise(X.iloc[b]))[:, 1]
        m0 = clone(make(FIN_COLS)).fit(X.iloc[a], y[a])
        p_nod[b] = m0.predict_proba(X.iloc[b])[:, 1]
    diff = float(np.max(np.abs(p_rep - FROZEN[key]["p"])))
    REPRO[key] = dict(max_abs_diff_vs_frozen_oof=diff, reproduced=bool(diff < 1e-4))
    assert diff < 1e-4, f"{key}: fold refits do not reproduce the frozen OOF vector (max diff {diff:.2e})"
    P[f"{key}_neutralised"] = p_neu
    P[f"{key}_nodemo"] = p_nod
    log(f"{key}: frozen OOF reproduced (max |diff| {diff:.1e}); neutralised AUC {roc_auc_score(y, p_neu):.4f}; "
        f"demographics-free AUC {roc_auc_score(y, p_nod):.4f}")
np.save(os.path.join(RESULTS, "fairness_oof_xgb_nodemo.npy"), P["xgb_nodemo"])
np.save(os.path.join(RESULTS, "fairness_oof_logistic_nodemo.npy"), P["logit_nodemo"])

# demographics-free OOF vectors written by code/ablation_checks.py (section c), if they exist: cross-check against our own refit
ABLATION = {}
for key, fname in (("xgb", "ablation_oof_xgb_nodemo.npy"), ("logit", "ablation_oof_logit_nodemo.npy")):
    path = os.path.join(RESULTS, fname)
    info = dict(path=f"results/{fname}", present=os.path.exists(path))
    if info["present"]:
        p_abl = np.load(path)
        assert p_abl.shape == (len(y),)
        d = float(np.max(np.abs(p_abl - P[f"{key}_nodemo"])))
        info.update(max_abs_diff_vs_own_refit=d, identical=bool(d < 1e-6), oof_auc=float(roc_auc_score(y, p_abl)),
                    corr_with_own_refit=float(np.corrcoef(p_abl, P[f"{key}_nodemo"])[0, 1]))
        if d >= 1e-6:                                                    # a different definition: report it as an extra variant
            P[f"{key}_nodemo_ablation"] = p_abl
        log(f"{fname} present: max |diff| vs own refit {d:.2e}")
    else:
        log(f"{fname} absent: the demographics-free {key} is this script's own refit")
    ABLATION[key] = info

# ============================================================================ cut-offs
CUT = {"xgb_frozen": FROZEN["xgb"]["t"], "xgb_neutralised": FROZEN["xgb"]["t"],
       "logit_frozen": FROZEN["logit"]["t"], "logit_neutralised": FROZEN["logit"]["t"]}
for v in [k for k in P if "nodemo" in k]:
    CUT[v] = tune_threshold(y, P[v], "cost")[0]                          # same rule, on its own training OOF
CUT_NOTE = ("frozen / neutralised: the frozen 5:1 cut-off (chosen on the frozen model's training OOF); "
            "nodemo: the same cost rule (tune_threshold(..., 'cost'), grid 0.05-0.95 step 0.01) on its own training OOF")


# ============================================================================ metrics
def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p = k / n; den = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / den; h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n * n)) / den
    return c - h, c + h


def auc_hm(yy, pp):
    """Within-group AUC with a Hanley-McNeil (1982) normal-approximation 95% interval."""
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan
    A = roc_auc_score(yy, pp); q1 = A / (2 - A); q2 = 2 * A * A / (1 + A)
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0))
    return A, A - Z * se, A + Z * se


def rates(yy, pp, t):
    f = pp >= t
    tp = int(np.sum(f & (yy == 1))); fp = int(np.sum(f & (yy == 0)))
    n1 = int(yy.sum()); n0 = len(yy) - n1
    return dict(n=len(yy), n_default=n1, n_flagged=int(f.sum()), tp=tp, fp=fp, n_nondefault=n0,
                default_rate=n1 / len(yy), mean_pd=float(pp.mean()), flag_rate=f.mean(),
                tpr=tp / n1 if n1 else np.nan, fpr=fp / n0 if n0 else np.nan,
                precision=tp / f.sum() if f.sum() else np.nan)


def ratio_ci(k1, n1, k0, n0):
    """Ratio of two independent proportions (group / reference) with the Katz log interval."""
    if min(k1, k0) == 0 or min(n1, n0) == 0:
        return np.nan, np.nan, np.nan
    r = (k1 / n1) / (k0 / n0); se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return r, r * np.exp(-Z * se), r * np.exp(Z * se)


rows = []
for v, p in P.items():
    t = CUT[v]
    r = rates(y, p, t); A, lo, hi = auc_hm(y, p)
    rows.append(dict(variant=v, attribute="ALL", group="all", is_reference=False, threshold=t, **r,
                     calib_gap_pp=100 * (r["mean_pd"] - r["default_rate"]), auc=A, auc_lo=lo, auc_hi=hi,
                     ap=average_precision_score(y, p), brier=brier_score_loss(y, p)))
    for att, (lab, order, ref) in GROUPS.items():
        mr = lab == ref; rr = rates(y[mr], p[mr], t)
        for g in order:
            m = lab == g; rg = rates(y[m], p[m], t); A, lo, hi = auc_hm(y[m], p[m])
            row = dict(variant=v, attribute=att, group=g, is_reference=bool(g == ref), threshold=t, **rg,
                       calib_gap_pp=100 * (rg["mean_pd"] - rg["default_rate"]),
                       observed_over_expected=rg["default_rate"] / rg["mean_pd"],
                       auc=A, auc_lo=lo, auc_hi=hi, ap=average_precision_score(y[m], p[m]), brier=brier_score_loss(y[m], p[m]))
            for name, k, n in (("default_rate", "n_default", "n"), ("flag_rate", "n_flagged", "n"),
                               ("fpr", "fp", "n_nondefault"), ("tpr", "tp", "n_default")):
                row[f"{name}_lo"], row[f"{name}_hi"] = wilson(rg[k], rg[n])
                rat, rlo, rhi = ratio_ci(rg[k], rg[n], rr[k], rr[n])
                row[f"{name}_ratio_to_ref"], row[f"{name}_ratio_lo"], row[f"{name}_ratio_hi"] = rat, rlo, rhi
            row["precision_lo"], row["precision_hi"] = wilson(rg["tp"], rg["n_flagged"])
            row["mean_pd_ratio_to_ref"] = rg["mean_pd"] / rr["mean_pd"]
            row["flag_rate_diff_pp"] = 100 * (rg["flag_rate"] - rr["flag_rate"])
            row["fpr_diff_pp"] = 100 * (rg["fpr"] - rr["fpr"])
            row["tpr_diff_pp"] = 100 * (rg["tpr"] - rr["tpr"])
            row["reference_group"] = ref
            rows.append(row)
G = pd.DataFrame(rows)
lead = ["variant", "attribute", "group", "reference_group", "is_reference", "threshold", "n", "n_default", "default_rate",
        "default_rate_lo", "default_rate_hi", "mean_pd", "calib_gap_pp", "observed_over_expected", "flag_rate", "flag_rate_lo",
        "flag_rate_hi", "tpr", "tpr_lo", "tpr_hi", "fpr", "fpr_lo", "fpr_hi", "precision", "precision_lo", "precision_hi",
        "auc", "auc_lo", "auc_hi", "ap", "brier", "default_rate_ratio_to_ref", "mean_pd_ratio_to_ref",
        "flag_rate_ratio_to_ref", "flag_rate_ratio_lo", "flag_rate_ratio_hi", "fpr_ratio_to_ref", "fpr_ratio_lo",
        "fpr_ratio_hi", "tpr_ratio_to_ref", "tpr_ratio_lo", "tpr_ratio_hi", "flag_rate_diff_pp", "fpr_diff_pp", "tpr_diff_pp"]
G = G[lead + [c for c in G.columns if c not in lead]]
G.to_csv(os.path.join(RESULTS, "fairness_groups.csv"), index=False, float_format="%.6g")
log(f"fairness_groups.csv: {len(G)} rows")

# ============================================================================ per-fold paired comparisons
def paired(vals_a, vals_b):
    d = np.asarray(vals_b, float) - np.asarray(vals_a, float)
    ok = np.isfinite(d)
    out = dict(frozen_per_fold=[float(x) for x in vals_a], variant_per_fold=[float(x) for x in vals_b],
               diff_per_fold=[float(x) for x in d], n_folds_positive=int(np.sum(d[ok] > 0)), n_folds_valid=int(ok.sum()))
    if ok.sum() >= 2:
        ci = corrected_t_ci(d[ok], n_train=N_TRAIN_FOLD, n_test=N_TEST_FOLD)
        out.update(mean_diff=ci["mean"], ci95_lo=ci["lo"], ci95_hi=ci["hi"])
    return out


OVERALL = {}
for fam in ("xgb", "logit"):
    base = f"{fam}_frozen"
    for v in [k for k in P if k.startswith(fam) and k != base]:
        for mname, fn in (("auc", roc_auc_score), ("ap", average_precision_score), ("brier", brier_score_loss)):
            a = [fn(y[b], P[base][b]) for _, b in FOLDS]; bb = [fn(y[b], P[v][b]) for _, b in FOLDS]
            OVERALL[f"{v}_minus_{base}__{mname}"] = paired(a, bb)


def fold_group_stats(p, t, b):
    out = {}
    for att, (lab, order, ref) in GROUPS.items():
        lb = lab[b]; yb = y[b]; pb = p[b]
        rr = rates(yb[lb == ref], pb[lb == ref], t)
        for g in order:
            m = lb == g; rg = rates(yb[m], pb[m], t)
            out[(att, g, "calib_gap_pp")] = 100 * (rg["mean_pd"] - rg["default_rate"])
            out[(att, g, "flag_rate")] = rg["flag_rate"]
            if g != ref:
                out[(att, g, "flag_rate_ratio")] = rg["flag_rate"] / rr["flag_rate"] if rr["flag_rate"] > 0 else np.nan
                out[(att, g, "fpr_ratio")] = rg["fpr"] / rr["fpr"] if rr["fpr"] > 0 else np.nan
                out[(att, g, "tpr_ratio")] = rg["tpr"] / rr["tpr"] if rr["tpr"] > 0 else np.nan
                out[(att, g, "flag_rate_diff_pp")] = 100 * (rg["flag_rate"] - rr["flag_rate"])
    return out


gap_rows = []
for fam in ("xgb", "logit"):
    base = f"{fam}_frozen"
    fs_base = [fold_group_stats(P[base], CUT[base], b) for _, b in FOLDS]
    for v in [k for k in P if k.startswith(fam) and k != base]:
        fs_v = [fold_group_stats(P[v], CUT[v], b) for _, b in FOLDS]
        for key in fs_base[0]:
            pr = paired([f[key] for f in fs_base], [f[key] for f in fs_v])
            gap_rows.append(dict(frozen=base, variant=v, attribute=key[0], group=key[1], metric=key[2],
                                 **{f"frozen_f{i + 1}": x for i, x in enumerate(pr["frozen_per_fold"])},
                                 **{f"variant_f{i + 1}": x for i, x in enumerate(pr["variant_per_fold"])},
                                 frozen_mean=float(np.nanmean(pr["frozen_per_fold"])),
                                 variant_mean=float(np.nanmean(pr["variant_per_fold"])),
                                 mean_diff=pr.get("mean_diff"), n_folds_positive=pr["n_folds_positive"],
                                 n_folds_valid=pr["n_folds_valid"], ci95_lo=pr.get("ci95_lo"), ci95_hi=pr.get("ci95_hi")))
GF = pd.DataFrame(gap_rows)
GF.to_csv(os.path.join(RESULTS, "fairness_gap_folds.csv"), index=False, float_format="%.6g")
log(f"fairness_gap_folds.csv: {len(GF)} rows")

# ============================================================================ proxies: can the other 25 predictors recover the attribute?
PROXY_TARGETS = {
    "female (SEX = 2)": (X["SEX"] == 2).astype(int).values,
    "age < 30": (X["AGE"] < 30).astype(int).values,
    "age 50+": (X["AGE"] >= 50).astype(int).values,
    "single (MARRIAGE = 2)": (X["MARRIAGE"] == 2).astype(int).values,
    "graduate school (EDUCATION = 1)": (X["EDUCATION"] == 1).astype(int).values,
    "high school (EDUCATION = 3)": (X["EDUCATION"] == 3).astype(int).values,
}
prox_rows = []
for name, z in PROXY_TARGETS.items():
    pz = np.full(len(z), np.nan)
    for a, b in FOLDS:
        pz[b] = clone(xgb_pipe(FIN_COLS)).fit(X.iloc[a], z[a]).predict_proba(X.iloc[b])[:, 1]
    full = clone(xgb_pipe(FIN_COLS)).fit(X, z)
    gain = full.named_steps["clf"].get_booster().get_score(importance_type="total_gain")
    gain = pd.Series({FIN_COLS[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
    share = gain / gain.sum()
    prox_rows.append(dict(target=name, kind="binary", prevalence=float(z.mean()), oof_auc=float(roc_auc_score(z, pz)),
                          fold_aucs=";".join(f"{roc_auc_score(z[b], pz[b]):.4f}" for _, b in FOLDS), oof_r2=np.nan,
                          top3_predictors=";".join(f"{c} ({s:.2f})" for c, s in share.head(3).items()),
                          spearman_with_nodemo_xgb_pd=float(stats.spearmanr(z, P["xgb_nodemo"])[0])))
reg_params = {k: v for k, v in FROZEN["xgb"]["params"].items() if k not in ("objective", "scale_pos_weight")}
reg_params.update(objective="reg:squarederror", n_jobs=N_JOBS)
age = X["AGE"].astype(float).values; pa = np.full(len(age), np.nan)
for a, b in FOLDS:
    reg = Pipeline([("pre", ColumnTransformer([("num", "passthrough", FIN_COLS)])), ("clf", XGBRegressor(**reg_params))])
    pa[b] = reg.fit(X.iloc[a], age[a]).predict(X.iloc[b])
prox_rows.append(dict(target="AGE (years, regression)", kind="continuous", prevalence=np.nan, oof_auc=np.nan, fold_aucs="",
                      oof_r2=float(r2_score(age, pa)), top3_predictors="",
                      spearman_with_nodemo_xgb_pd=float(stats.spearmanr(age, P["xgb_nodemo"])[0])))
PX = pd.DataFrame(prox_rows)
PX.to_csv(os.path.join(RESULTS, "fairness_proxies.csv"), index=False, float_format="%.6g")
log("fairness_proxies.csv:\n" + PX[["target", "oof_auc", "oof_r2", "top3_predictors"]].round(4).to_string(index=False))

# ============================================================================ figure: flag rates by group
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"   # validated pair (dataviz skill)
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5})
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1), sharex=True)
titles = {"SEX": "Sex", "AGE": "Age band", "MARRIAGE": "Marital status", "EDUCATION": "Education (recoded)"}
for ax, att in zip(axes, DEMO_COLS):
    lab, order, ref = GROUPS[att]
    sub = G[G["attribute"] == att].set_index(["variant", "group"])
    yy = np.arange(len(order))
    h = 0.30                                                             # same bar thickness in every panel (shared y-range)
    ax.barh(yy - h / 2 - 0.015, [100 * sub.loc[("xgb_frozen", g), "flag_rate"] for g in order], height=h, color=BLUE,
            label="Frozen XGBoost (29 predictors)", zorder=2)
    ax.barh(yy + h / 2 + 0.015, [100 * sub.loc[("xgb_nodemo", g), "flag_rate"] for g in order], height=h, color=ORANGE,
            label="Demographics-free XGBoost (25 predictors)", zorder=2)
    ax.scatter([100 * sub.loc[("xgb_frozen", g), "default_rate"] for g in order], yy, marker="D", s=30, color=INK,
               edgecolor="white", linewidth=1.2, zorder=3, label="Observed default rate")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g}{' (ref)' if g == ref else ''}\nn = {int(sub.loc[('xgb_frozen', g), 'n']):,}" for g in order],
                       fontsize=8.5)
    ax.set_ylim(3.55, -0.55)                                             # four slots in every panel, top-aligned
    ax.set_xlim(0, 72); ax.set_xticks([0, 20, 40, 60]); ax.set_xticklabels(["0%", "20%", "40%", "60%"])
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0); ax.tick_params(length=0)
    ax.set_title(titles[att], fontsize=10.5, loc="left")
axes[0].set_xlabel("Share of group flagged (PD at or above the 5:1 cut-off)", loc="left")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend([h_[1], h_[2], h_[0]], [l_[1], l_[2], l_[0]], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.text(0.01, 0.012, f"Training out-of-fold predictions only (20,975 clients, 5 shared folds). Cut-offs: frozen "
         f"{CUT['xgb_frozen']:.2f}, demographics-free {CUT['xgb_nodemo']:.2f}. Reference = largest group. Education "
         "'others/unknown' pools codes 4, 5, 6 and the undocumented 0.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.045, 1, 0.92))
fig.savefig(os.path.join(FIGURES, "fairness_flag_rates.png"), dpi=150)
plt.close(fig)
log("figures/fairness_flag_rates.png written")

# ============================================================================ summary json
def g_row(v, att, g):
    r = G[(G["variant"] == v) & (G["attribute"] == att) & (G["group"] == g)].iloc[0]
    keep = ["n", "default_rate", "mean_pd", "calib_gap_pp", "flag_rate", "tpr", "fpr", "precision", "auc",
            "default_rate_ratio_to_ref", "flag_rate_ratio_to_ref", "flag_rate_ratio_lo", "flag_rate_ratio_hi",
            "fpr_ratio_to_ref", "fpr_ratio_lo", "fpr_ratio_hi", "tpr_ratio_to_ref"]
    return {k: (None if pd.isna(r[k]) else (int(r[k]) if k == "n" else round(float(r[k]), 4))) for k in keep}


key_gaps = {}
for v in P:
    key_gaps[v] = {att: {g: g_row(v, att, g) for g in GROUPS[att][1]} for att in DEMO_COLS}
overall = {v: {k: round(float(G[(G["variant"] == v) & (G["attribute"] == "ALL")].iloc[0][k]), 4)
               for k in ("auc", "ap", "brier", "flag_rate", "tpr", "fpr", "precision", "mean_pd", "threshold")} for v in P}


def gap_change(fam, v, att, g, metric):
    r = GF[(GF["frozen"] == f"{fam}_frozen") & (GF["variant"] == v) & (GF["attribute"] == att) & (GF["group"] == g)
           & (GF["metric"] == metric)].iloc[0]
    return {k: (None if pd.isna(r[k]) else round(float(r[k]), 4))
            for k in ("frozen_mean", "variant_mean", "mean_diff", "ci95_lo", "ci95_hi", "n_folds_positive")}


headline_gaps = {}
for fam in ("xgb", "logit"):
    for v in (f"{fam}_neutralised", f"{fam}_nodemo"):
        headline_gaps[v] = {f"{att}:{g}": {m: gap_change(fam, v, att, g, m) for m in ("flag_rate_ratio", "fpr_ratio", "calib_gap_pp")}
                            for att in DEMO_COLS for g in GROUPS[att][1] if g != GROUPS[att][2]}

summary = dict(
    script="code/fairness_check.py", run_at=time.strftime("%Y-%m-%d %H:%M"), run_seconds=None, training_only=True,
    held_out_file_read=False, seed=SEED, n_rows=int(len(y)), default_rate=round(float(y.mean()), 4),
    folds="data/processed/cv_folds.csv (= common.CV, StratifiedKFold(5, shuffle=True, random_state=2607526))",
    nadeau_bengio=f"corrected t, J = 5 folds, n_test/n_train = {N_TEST_FOLD}/{N_TRAIN_FOLD} = {N_TEST_FOLD / N_TRAIN_FOLD:.2f}",
    cutoffs={v: CUT[v] for v in P}, cutoff_note=CUT_NOTE,
    cutoff_source=("results/<slug>.json at_tuned_cost.threshold (chosen by tune_threshold(y_train, p_oof, 'cost') on the "
                   "training OOF; equals frozen_manifest.json thresholds_chosen_on_training_oof.cost and is re-derived here)"),
    groups={att: dict(levels=GROUPS[att][1], reference=GROUPS[att][2]) for att in DEMO_COLS},
    recoding="EDUCATION {0,5,6} -> 4 'others/unknown'; MARRIAGE 0 -> 3 'others' (common.clean); AGE bands <30, 30-39, 40-49, 50+",
    variants={
        "xgb_frozen / logit_frozen": "frozen training OOF probabilities, read from results/<slug>_p_oof.npy",
        "xgb_neutralised / logit_neutralised": ("frozen settings refitted per fold in memory (reproduces the frozen OOF vector); "
                                                f"validation clients scored with SEX = female, AGE = {AGE_MEDIAN:g}, MARRIAGE = single, "
                                                "EDUCATION = university -> removes the direct use of the attributes"),
        "xgb_nodemo / logit_nodemo": "same settings refitted per fold on the 25 non-demographic predictors (no SEX, AGE, MARRIAGE, EDUCATION)",
    },
    reproduction_check=REPRO, ablation_file=ABLATION,
    overall=overall, paired_fold_comparisons_overall=OVERALL,
    key_gaps_by_group=key_gaps, gap_change_vs_frozen=headline_gaps,
    proxies=PX.replace({np.nan: None}).to_dict(orient="records"),
)
def gv(v, att, g, k):
    return float(G[(G["variant"] == v) & (G["attribute"] == att) & (G["group"] == g)].iloc[0][k])


def pc(x):
    return f"{100 * x:.1f}%"


def ov(key):
    o = OVERALL[key]
    return f"{o['mean_diff']:+.4f} (95% corrected t CI {o['ci95_lo']:+.4f} to {o['ci95_hi']:+.4f}; {o['n_folds_positive']}/5 folds positive)"


px = PX.set_index("target")

# mechanism checks quoted in the notes (training rows only)
DENS = {k: float(np.mean((P[f"{k}_frozen"] >= CUT[f"{k}_frozen"] - 0.02) & (P[f"{k}_frozen"] < CUT[f"{k}_frozen"] + 0.02)) / 4)
        for k in ("xgb", "logit")}                                       # share of clients per 0.01 of PD around the cut-off
MED = {k: float(np.median(P[f"{k}_frozen"])) for k in ("xgb", "logit")}
limq = pd.qcut(X["LIMIT_BAL"], 5, labels=False, duplicates="drop").values
young = (X["AGE"] < 30).values
LIM_TAB = pd.DataFrame([dict(limit_quintile=int(q) + 1, default_rate_under30=float(y[(limq == q) & young].mean()),
                             default_rate_30plus=float(y[(limq == q) & ~young].mean()), n_under30=int(((limq == q) & young).sum()),
                             n_30plus=int(((limq == q) & ~young).sum())) for q in np.unique(limq)])
MAX_CAL = float(G[(G["variant"] == "xgb_frozen") & (G["attribute"] != "ALL") & (G["n"] >= 1000)]["calib_gap_pp"].abs().max())
summary["mechanism_checks"] = dict(
    pd_density_per_0_01_around_cutoff=DENS, median_oof_pd=MED,
    default_rate_by_limit_quintile_and_age=LIM_TAB.round(4).to_dict(orient="records"),
    median_limit_under30=float(X.loc[young, "LIMIT_BAL"].median()), median_limit_30plus=float(X.loc[~young, "LIMIT_BAL"].median()),
    xgb_frozen_max_abs_calib_gap_pp_groups_n_ge_1000=MAX_CAL)
summary["plain_notes"] = [
    # 1. base rates
    f"Base rates differ before any model is involved: in the training rows men default at {pc(gv('xgb_frozen', 'SEX', 'male', 'default_rate'))} "
    f"and women at {pc(gv('xgb_frozen', 'SEX', 'female', 'default_rate'))} (ratio {gv('xgb_frozen', 'SEX', 'male', 'default_rate_ratio_to_ref'):.2f}); "
    f"married clients {pc(gv('xgb_frozen', 'MARRIAGE', 'married', 'default_rate'))} vs single {pc(gv('xgb_frozen', 'MARRIAGE', 'single', 'default_rate'))}; "
    f"clients aged 50+ {pc(gv('xgb_frozen', 'AGE', '50+', 'default_rate'))} vs 30-39 {pc(gv('xgb_frozen', 'AGE', '30-39', 'default_rate'))}.",
    # 2. frozen XGB: flag gaps track base rates, calibrated
    f"The frozen XGBoost is calibrated in the large within every group of 1,000 or more clients (|mean PD - observed rate| at most "
    f"{MAX_CAL:.2f} percentage points) and flags men {gv('xgb_frozen', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f} times as often as women "
    f"({pc(gv('xgb_frozen', 'SEX', 'male', 'flag_rate'))} vs {pc(gv('xgb_frozen', 'SEX', 'female', 'flag_rate'))}), close to the "
    f"default-rate ratio of {gv('xgb_frozen', 'SEX', 'male', 'default_rate_ratio_to_ref'):.2f}. A score calibrated within groups usually flags "
    f"more of a higher-risk group, and here it also flags more of that group's non-defaulters: FPR {pc(gv('xgb_frozen', 'SEX', 'male', 'fpr'))} "
    f"for men vs {pc(gv('xgb_frozen', 'SEX', 'female', 'fpr'))} for women. A flag ratio that matches the base rates does not mean the "
    f"attributes are unused: scoring every client with the reference values narrows the XGBoost sex ratio to "
    f"{gv('xgb_neutralised', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f} and the married/single ratio to "
    f"{gv('xgb_neutralised', 'MARRIAGE', 'married', 'flag_rate_ratio_to_ref'):.2f}, so part of the XGBoost gap is direct use of SEX and "
    "MARRIAGE and part comes through the other predictors. With different base rates, equal calibration, equal flag rates and equal "
    "error rates cannot all hold at once (Kleinberg, Mullainathan and Raghavan 2017; Chouldechova 2017).",
    # 3. logit amplifies
    f"The plain logit widens the gaps beyond the base rates: it flags men {gv('logit_frozen', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f} "
    f"times as often as women (FPR ratio {gv('logit_frozen', 'SEX', 'male', 'fpr_ratio_to_ref'):.2f}) and married clients "
    f"{gv('logit_frozen', 'MARRIAGE', 'married', 'flag_rate_ratio_to_ref'):.2f} times as often as single ones, although it is calibrated "
    "within these groups. Its SEX and MARRIAGE coefficients shift every member's log-odds by the same amount, and the logit's PDs "
    f"are packed more tightly around the cut-off ({CUT['logit_frozen']:.2f}): {100 * DENS['logit']:.1f}% of clients per 0.01 of PD near it, "
    f"against {100 * DENS['xgb']:.1f}% for XGBoost (both medians are close to the cut-off: logit {MED['logit']:.3f}, XGBoost "
    f"{MED['xgb']:.3f}), so the same shift moves more clients across it. Neutralising the attributes "
    "brings the sex ratio to "
    f"{gv('logit_neutralised', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f} and the marital ratio to "
    f"{gv('logit_neutralised', 'MARRIAGE', 'married', 'flag_rate_ratio_to_ref'):.2f}: most of the logit's excess gap is direct use of the attribute.",
    # 4. removing
    f"Refitting without SEX, AGE, MARRIAGE and EDUCATION costs little ranking power: XGBoost CV AUC {ov('xgb_nodemo_minus_xgb_frozen__auc')}, "
    f"AP {ov('xgb_nodemo_minus_xgb_frozen__ap')}; logit AUC {ov('logit_nodemo_minus_logit_frozen__auc')}. It narrows the sex gap "
    f"(XGBoost flag ratio {gv('xgb_frozen', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f} -> {gv('xgb_nodemo', 'SEX', 'male', 'flag_rate_ratio_to_ref'):.2f}) "
    f"and removes the marital gap ({gv('xgb_frozen', 'MARRIAGE', 'married', 'flag_rate_ratio_to_ref'):.2f} -> "
    f"{gv('xgb_nodemo', 'MARRIAGE', 'married', 'flag_rate_ratio_to_ref'):.2f}), but the price is within-group calibration: the model now "
    f"under-predicts men by {abs(gv('xgb_nodemo', 'SEX', 'male', 'calib_gap_pp')):.1f} and over-predicts women by "
    f"{gv('xgb_nodemo', 'SEX', 'female', 'calib_gap_pp'):.1f} percentage points.",
    # 5. not always narrower
    f"Removing an attribute does not always narrow a gap. Clients under 30 are flagged more often without AGE (ratio to 30-39 "
    f"{gv('xgb_frozen', 'AGE', '<30', 'flag_rate_ratio_to_ref'):.2f} -> {gv('xgb_nodemo', 'AGE', '<30', 'flag_rate_ratio_to_ref'):.2f}; FPR "
    f"{pc(gv('xgb_frozen', 'AGE', '<30', 'fpr'))} -> {pc(gv('xgb_nodemo', 'AGE', '<30', 'fpr'))}) and become over-predicted by "
    f"{gv('xgb_nodemo', 'AGE', '<30', 'calib_gap_pp'):.1f} points. Their credit limits are low (median "
    f"{X.loc[young, 'LIMIT_BAL'].median():,.0f} vs {X.loc[~young, 'LIMIT_BAL'].median():,.0f} NT$), and at the same limit quintile they default "
    f"less often than older clients ({int((LIM_TAB['default_rate_under30'] < LIM_TAB['default_rate_30plus']).sum())} of "
    f"{len(LIM_TAB)} limit quintiles, see mechanism_checks), so once AGE is gone the low limit is read as more risk "
    f"than it carries for them. The small 'others/unknown' education bucket (n = {int(gv('xgb_frozen', 'EDUCATION', 'others/unknown', 'n'))}, "
    f"default rate {pc(gv('xgb_frozen', 'EDUCATION', 'others/unknown', 'default_rate'))}) goes from {pc(gv('xgb_frozen', 'EDUCATION', 'others/unknown', 'flag_rate'))} "
    f"flagged to {pc(gv('xgb_nodemo', 'EDUCATION', 'others/unknown', 'flag_rate'))}; this bucket pools 'others', 'unknown' and the "
    "undocumented code 0, so its numbers say more about how education was recorded than about a real group of people.",
    # 6. proxies
    f"Removing the columns does not remove proxies. From the 25 remaining predictors alone, out-of-fold AUC for recovering the "
    f"attribute is {px.loc['female (SEX = 2)', 'oof_auc']:.2f} for sex, {px.loc['age < 30', 'oof_auc']:.2f} for age under 30, "
    f"{px.loc['single (MARRIAGE = 2)', 'oof_auc']:.2f} for single, {px.loc['graduate school (EDUCATION = 1)', 'oof_auc']:.2f} for "
    f"graduate school (R^2 for age in years {px.loc['AGE (years, regression)', 'oof_r2']:.3f}); LIMIT_BAL is the leading proxy in every "
    "case. The proxies are weak to moderate here, but they are why a demographics-free score still differs by group, and why "
    "'fairness through unawareness' is not by itself a fairness guarantee.",
    # 7. regulation framing
    "What a review in the style of the EU AI Act (Regulation (EU) 2024/1689, which lists creditworthiness assessment and credit "
    "scoring of natural persons as high-risk in Annex III point 5(b)) or of the MAS FEAT principles would ask for: a stated "
    "justification for each personal attribute used (FEAT fairness principle on personal attributes); data-governance evidence that "
    "possible biases were examined and mitigated (AI Act Article 10); a risk-management record and technical documentation that "
    "state the fairness criterion chosen, the reference groups and the trade-offs accepted, with group-level metrics like these "
    "(Articles 9 and 11; the MAS Veritas fairness assessment methodology asks for the same steps); human oversight of declines "
    "(Article 14); monitoring of the group metrics after deployment (Article 72); and, for the deploying lender, a fundamental-rights "
    "impact assessment (Article 27) and an explanation route for declined applicants (Article 86). Article numbers should be checked "
    "against the Official Journal text before citing. This table is diagnostic evidence for such a review, not a legal assessment: "
    "the data are Taiwanese card clients from 2005 and no regulation is claimed to apply to them, nor is any model here claimed "
    "to be compliant or non-compliant.",
]
summary["run_seconds"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(RESULTS, "fairness_summary.json"), "w"), indent=1, default=float)
log(f"fairness_summary.json written; total {summary['run_seconds']}s")
