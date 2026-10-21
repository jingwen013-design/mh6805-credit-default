"""
monotone_respec_check.py - a respecified monotone XGBoost, and a consistent wrong-direction sweep.
Post-freeze sensitivity analysis on the TRAINING ROWS ONLY (audit findings monotone-constraint-on-nondelinquent-codes,
o4-pay-code-semantics, and the two MISSED notes: the sweep changes PAY_1 without updating MAX_DELAY / N_MONTHS_DELAYED,
and the +1 constraint on utilisation contradicts the U-shape).

Run from Project/code:  /opt/miniconda3/bin/python3 monotone_respec_check.py      (about 20-40 seconds on 2 threads)

PURPOSE
  The frozen monotone XGBoost (code/model_monotone.py) puts +1 on PAY_1..PAY_6 read as numbers, which also forces
  p(-2) <= p(-1) <= p(0) on the three non-delinquent codes, and +1 on UTIL_1 / UTIL_MEAN although utilisation is U-shaped.
  (1) Respecified model, 5-fold training CV only. PAY_k is recoded row-wise as PAY_k_DEL = max(PAY_k, 0) plus 0/1 flags
      PAY_k_NC = 1[PAY_k = -2] and PAY_k_PD = 1[PAY_k = -1] (same recoding as code/pay_coding_check.py), and
        +1 only where the business meaning is monotone: PAY_1_DEL..PAY_6_DEL, MAX_DELAY, N_MONTHS_DELAYED;
        -1 on LIMIT_BAL (as in the frozen monotone model);
         0 on everything else - utilisation (the data show a U), the -2 / -1 flags, BILL_TREND, PAY_RATIO_MEAN, amounts,
           AGE and the demographic dummies.
      Hyper-parameters are those of the frozen monotone model, i.e. of the tuned XGBoost (results/xgboost.json:
      learning_rate 0.03, max_depth 3, min_child_weight 5, n_estimators 300, subsample 0.8, colsample_bytree 0.8, hist,
      random_state 2607526, n_jobs 2). A ladder of intermediate variants separates the effect of each change:
        free_numeric          frozen unconstrained XGBoost (reproduced; OOF checked against results/xgboost_p_oof.npy)
        mono_frozen_numeric   frozen monotone XGBoost (reproduced; OOF checked against results/xgboost_monotone_p_oof.npy)
        free_recoded          unconstrained, PAY recoded (coding effect alone)
        mono_frozen_signs_recoded  frozen sign set with PAY_k -> PAY_k_DEL, flags free (frees the -2/-1/0 order only)
        mono_no_util_recoded  the previous one without the UTIL_1 / UTIL_MEAN constraints (frees the U as well)
        mono_respec           the respecified model above (also frees BILL_TREND and PAY_RATIO_MEAN)
      Compared on the same folds with the frozen OOF vectors: per-fold AUC, mean difference, folds positive and the
      Nadeau-Bengio corrected t interval (mine_v2_utils.corrected_t_ci with n_test / n_train = 1 / 4).
  (2) Consistent wrong-direction sweep. Each model is refitted on all 20,975 training rows (deterministic; the
      unconstrained refit is the frozen XGBoost refitted, as in model_monotone.py) and PAY_k is swept over its observed
      codes on the training median profile and on the same 200 random training profiles as model_monotone.py. Mode
      "as_frozen" reproduces the frozen sweep (only PAY_k and, for recoded models, its own DEL / flag columns change);
      mode "consistent" also recomputes MAX_DELAY and N_MONTHS_DELAYED from PAY_1..PAY_6 exactly as common.clean() does.
      Steps are classified as non-delinquent (-2 -> -1 -> 0), entry into arrears (0 -> first delay code) or between
      genuine delay levels (k -> k+1 with k >= 1); only the last class is counted as "wrong direction" in the headline,
      with the training n and default rate of both codes so that thin cells are visible. The frozen 13-feature
      median-profile sweep (the "24 wrong-direction steps") is also reproduced and broken down by feature and step.
  (3) Cross-state check (added by the verifier). The grid steps above only compare adjacent codes, so they never compare
      a delay code k >= 1 with the non-delinquent states -2 / -1. With the -2 / -1 flags left free, the respecified
      constraint guarantees p(0) <= p(1) <= p(2) ... but NOT p(-1) <= p(k): a client who paid in full can score higher
      than the same client k months late. For every fitted model, profile set and month (consistent mode) the JSON key
      cross_state_check counts the pairs (s in {-2, -1}, k >= 1) with p(k) < p(s) and the largest such fall.

TRAINING-ONLY
  Reads data/processed/train.csv directly and asserts 20,975 rows; never calls common.load_split() (guarded) or
  evaluate(), never opens data/processed/test.csv, writes no *_p_test file and no frozen slug, and does not modify any
  existing results file. Nothing is tuned: every hyper-parameter is copied from results/xgboost.json.

INPUTS
  data/processed/train.csv, data/processed/cv_folds.csv, results/xgboost.json, results/xgboost_monotone.json,
  results/monotone_summary.json, results/monotone_check.csv, results/monotone_feature_sweep.csv (reproduction checks),
  results/xgboost_p_oof.npy, results/xgboost_monotone_p_oof.npy.
OUTPUTS (new files)
  results/monotone_respec_cv.json        CV of the six variants, paired comparisons, constraint vectors, data-shape checks,
                                         reproduction checks, the sweep headline counts and the cross-state check
  results/monotone_check_consistent.csv  tidy sweep table: model x profile set x mode x feature x step
Seed 2607526; xgboost n_jobs = 2.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time
import numpy as np
import pandas as pd
import sklearn, xgboost
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from common import RESULTS, SEED, NUM_COLS, CAT_COLS, PAY_COLS, make_preprocessor
from pay_coding_check import (load_train, add_delay_flags, rebuild_pay_aggregates, fold_list, paired, fold_metric,
                              NUM_DELAY_FLAGS, DEL_COLS, NC_COLS, PD_COLS)


def _forbidden(*a, **k):
    raise SystemExit("monotone_respec_check.py is training-only: common.load_split() would read the held-out file")


common.load_split = _forbidden

T0 = time.time()
N_JOBS = 2
N_PROFILES = 200
THIN = 100                                       # a code with fewer training rows than this is a thin cell
ENG_DELAY = ["N_MONTHS_DELAYED", "MAX_DELAY"]

FROZEN_PLUS = PAY_COLS + ["N_MONTHS_DELAYED", "MAX_DELAY", "UTIL_1", "UTIL_MEAN", "BILL_TREND"]
FROZEN_MINUS = ["LIMIT_BAL", "PAY_RATIO_MEAN"]
VARIANTS = {   # name: (coding, +1 features, -1 features)
    "free_numeric": ("numeric", [], []),
    "mono_frozen_numeric": ("numeric", FROZEN_PLUS, FROZEN_MINUS),
    "free_recoded": ("recoded", [], []),
    "mono_frozen_signs_recoded": ("recoded", DEL_COLS + ENG_DELAY + ["UTIL_1", "UTIL_MEAN", "BILL_TREND"], FROZEN_MINUS),
    "mono_no_util_recoded": ("recoded", DEL_COLS + ENG_DELAY + ["BILL_TREND"], FROZEN_MINUS),
    "mono_respec": ("recoded", DEL_COLS + ENG_DELAY, ["LIMIT_BAL"]),
}
PRIMARY = "mono_respec"


def log(msg):
    print(f"[respec] {msg}   [t={time.time() - T0:,.0f}s]", flush=True)


def make_pre(coding):
    if coding == "numeric":
        return make_preprocessor(scale=False)                       # the frozen tree preprocessor
    return ColumnTransformer([("num", "passthrough", NUM_DELAY_FLAGS),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS)],
                             remainder="drop", sparse_threshold=0.0)


def raw_of(tname):
    base = tname.split("__", 1)[1]
    return base.rsplit("_", 1)[0] if tname.startswith("cat__") else base


def constraint_string(tnames, plus, minus):
    signs = [1 if raw_of(n) in plus else (-1 if raw_of(n) in minus else 0) for n in tnames]
    hit = [raw_of(n) for n in tnames]
    for f in plus + minus:
        assert hit.count(f) == 1, f"{f} appears {hit.count(f)} times among the transformed columns"
    return "(" + ",".join(map(str, signs)) + ")", signs


def xgb_params():
    bp = json.load(open(os.path.join(RESULTS, "xgboost.json")))["best_params"]
    mono_bp = json.load(open(os.path.join(RESULTS, "xgboost_monotone.json")))["best_params"]
    keys = ["learning_rate", "max_depth", "min_child_weight", "n_estimators", "subsample", "colsample_bytree",
            "objective", "tree_method", "scale_pos_weight", "random_state"]
    assert all(bp[k] == mono_bp[k] for k in keys), "frozen free and monotone XGBoost differ in a hyper-parameter"
    p = {k: bp[k] for k in keys}
    p.update(eval_metric="auc", n_jobs=N_JOBS)
    assert p["random_state"] == SEED
    return p, mono_bp["monotone_constraints"]


def make_pipe(variant, X, params):
    coding, plus, minus = VARIANTS[variant]
    pre = make_pre(coding).fit(X)
    tnames = list(pre.get_feature_names_out())
    kw = dict(params)
    mono = None
    if plus or minus:
        mono, _ = constraint_string(tnames, plus, minus)
        kw["monotone_constraints"] = mono
    return Pipeline([("pre", make_pre(coding)), ("clf", XGBClassifier(**kw))]), tnames, mono


def booster_constraints(pipe):
    """monotone_constraints read back out of the fitted booster's own config, as a list of ints."""
    cfg = json.loads(pipe.named_steps["clf"].get_booster().save_config())
    s = cfg["learner"]["gradient_booster"]["tree_train_param"].get("monotone_constraints", "()")
    return [int(v) for v in s.strip("()").split(",") if v.strip() != ""]


def parse(mono):
    return [int(v) for v in mono.strip("()").split(",")]


# ----------------------------------------------------------------------------- data-shape checks (training rows)
def shape_table(X, y, col, q=10):
    b = pd.qcut(X[col], q, duplicates="drop")
    g = pd.DataFrame({"x": X[col], "y": y, "b": b}).groupby("b", observed=True)
    t = g.agg(n=("y", "size"), x_mean=("x", "mean"), default_rate=("y", "mean")).reset_index(drop=True)
    d = np.diff(t["default_rate"].values)
    return dict(bins=[{k: (float(v) if k != "n" else int(v)) for k, v in r.items()} for r in t.to_dict(orient="records")],
                n_bins=int(len(t)), n_falls=int((d < 0).sum()), n_rises=int((d > 0).sum()),
                argmin_bin=int(t["default_rate"].values.argmin()),
                min_default_rate=float(t["default_rate"].min()),
                first_bin_default_rate=float(t["default_rate"].iloc[0]), last_bin_default_rate=float(t["default_rate"].iloc[-1]))


# ----------------------------------------------------------------------------- sweeps
def set_status(rows, col, v, consistent):
    """Set PAY status `col` to v; its own DEL / flag columns always follow; aggregates only when consistent."""
    r = rows.copy()
    r[col] = v
    r[f"{col}_DEL"] = float(max(v, 0)); r[f"{col}_NC"] = float(v == -2); r[f"{col}_PD"] = float(v == -1)
    if consistent:
        r = rebuild_pay_aggregates(r)
    return r


def step_class(a, b):
    if b <= 0:
        return "non_delinquent"
    if a <= 0:
        return "entry_into_arrears"
    return "between_delay_levels"


def sweep_status(pipe, profiles, col, grid, consistent):
    big = pd.concat([set_status(profiles, col, v, consistent) for v in grid], ignore_index=True)
    return pipe.predict_proba(big)[:, 1].reshape(len(grid), len(profiles))        # grid x profiles


def median_profile(X0):
    med = X0.median(numeric_only=True)
    prof = X0.iloc[[0]].copy().reset_index(drop=True)
    for c in X0.columns:
        prof.loc[0, c] = med[c]
    for c in CAT_COLS:
        v = int(round(float(med[c]))); assert v in set(X0[c].unique()); prof.loc[0, c] = v
    return prof.astype(X0.dtypes.to_dict())


def main():
    X0, y = load_train()                                              # the 29 frozen predictors
    X = add_delay_flags(X0)                                           # + 18 recoded PAY columns (row-wise, fixed)
    folds = fold_list(y)
    params, frozen_mono_str = xgb_params()
    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__)
    log(f"train {X.shape}; xgboost params {params}")

    # ---- 1. CV of the six variants on the shared folds
    out, oof, fold_auc = {}, {}, {}
    for v in VARIANTS:
        t0 = time.time()
        pipe0, tnames, mono = make_pipe(v, X, params)
        p = np.zeros(len(y)); reads = []
        for tr, va in folds:
            pipe, tn, mono_f = make_pipe(v, X.iloc[tr], params)
            assert tn == tnames, f"{v}: transformed column order differs in a fold"
            pipe.fit(X.iloc[tr], y.iloc[tr])
            if mono:
                reads.append(booster_constraints(pipe) == parse(mono))
            p[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        oof[v] = p
        a = fold_metric(y, p, folds, roc_auc_score); ap = fold_metric(y, p, folds, average_precision_score)
        br = fold_metric(y, p, folds, brier_score_loss); fold_auc[v] = a
        coding, plus, minus = VARIANTS[v]
        out[v] = dict(coding=coding, plus=plus, minus=minus, monotone_constraints=mono, n_columns=len(tnames),
                      constraint_read_back_ok_all_folds=(all(reads) if mono else None),
                      constrained_columns={n: s for n, s in zip(tnames, constraint_string(tnames, plus, minus)[1]) if s != 0},
                      fold_auc=[float(x) for x in a], mean_auc=float(a.mean()), sd_auc=float(a.std()),
                      oof_auc_pooled=float(roc_auc_score(y, p)), fold_ap=[float(x) for x in ap], mean_ap=float(ap.mean()),
                      oof_ap_pooled=float(average_precision_score(y, p)), mean_brier=float(br.mean()),
                      oof_brier_pooled=float(brier_score_loss(y, p)), seconds=round(time.time() - t0, 1))
        log(f"{v:26s} CV AUC {a.mean():.4f} ± {a.std():.4f}, AP {ap.mean():.4f} ({len(tnames)} columns, "
            f"{time.time() - t0:.0f}s)")
    assert out["mono_frozen_numeric"]["monotone_constraints"] == frozen_mono_str, "frozen constraint vector not rebuilt"

    fro = dict(free_numeric=np.load(os.path.join(RESULTS, "xgboost_p_oof.npy")),
               mono_frozen_numeric=np.load(os.path.join(RESULTS, "xgboost_monotone_p_oof.npy")))
    repro = {v: dict(max_abs_diff_vs_frozen_oof=float(np.abs(fro[v] - oof[v]).max()),
                     frozen_cv_auc=float(fold_metric(y, fro[v], folds, roc_auc_score).mean()),
                     this_run_cv_auc=out[v]["mean_auc"]) for v in fro}
    for v, r in repro.items():
        log(f"reproduction {v}: max |p - frozen p_oof| = {r['max_abs_diff_vs_frozen_oof']:.2e}")
    a_free = fold_metric(y, fro["free_numeric"], folds, roc_auc_score)
    a_mono = fold_metric(y, fro["mono_frozen_numeric"], folds, roc_auc_score)
    ap_free = fold_metric(y, fro["free_numeric"], folds, average_precision_score)
    ap_mono = fold_metric(y, fro["mono_frozen_numeric"], folds, average_precision_score)
    comparisons = {}
    for v in VARIANTS:
        ap_v = np.array(out[v]["fold_ap"])
        comparisons[v] = dict(auc_minus_frozen_free=paired(fold_auc[v], a_free), auc_minus_frozen_mono=paired(fold_auc[v], a_mono),
                              ap_minus_frozen_free=paired(ap_v, ap_free), ap_minus_frozen_mono=paired(ap_v, ap_mono))
    ladder = {
        "free_recoded - free_numeric (coding only, no constraint)": ("free_recoded", "free_numeric"),
        "mono_frozen_signs_recoded - mono_frozen_numeric (free the -2/-1/0 order)": ("mono_frozen_signs_recoded", "mono_frozen_numeric"),
        "mono_no_util_recoded - mono_frozen_signs_recoded (free utilisation)": ("mono_no_util_recoded", "mono_frozen_signs_recoded"),
        "mono_respec - mono_no_util_recoded (free BILL_TREND, PAY_RATIO_MEAN)": ("mono_respec", "mono_no_util_recoded"),
        "mono_respec - free_recoded (price of the respecified constraint)": ("mono_respec", "free_recoded"),
        "mono_respec - mono_frozen_numeric (all changes together)": ("mono_respec", "mono_frozen_numeric"),
        "mono_frozen_numeric - free_numeric (price of the frozen constraint)": ("mono_frozen_numeric", "free_numeric"),
    }
    ladder_out = {}
    for k, (a_, b_) in ladder.items():
        ladder_out[k] = dict(auc=paired(fold_auc[a_], fold_auc[b_]),
                             ap=paired(np.array(out[a_]["fold_ap"]), np.array(out[b_]["fold_ap"])))
    for k, r in ladder_out.items():
        log(f"{k}: ΔAUC {r['auc']['mean']:+.4f} ({r['auc']['folds_positive']}/5 folds +, NB [{r['auc']['nb_lo']:+.4f}, "
            f"{r['auc']['nb_hi']:+.4f}]), ΔAP {r['ap']['mean']:+.4f}")

    # ---- data-shape checks behind the sign choices (training rows only)
    shapes = {c: shape_table(X0, y, c) for c in ["UTIL_1", "UTIL_MEAN", "BILL_TREND", "PAY_RATIO_MEAN", "LIMIT_BAL",
                                                 "MAX_DELAY", "N_MONTHS_DELAYED"]}
    for c, s in shapes.items():
        log(f"shape {c:16s}: {s['n_bins']} bins, default rate first {s['first_bin_default_rate']:.3f}, min "
            f"{s['min_default_rate']:.3f} (bin {s['argmin_bin']}), last {s['last_bin_default_rate']:.3f}; falls {s['n_falls']}, rises {s['n_rises']}")

    # ---- 2. sweeps on full-training refits
    fitted = {}
    for v in VARIANTS:
        pipe, _, mono = make_pipe(v, X, params)
        fitted[v] = pipe.fit(X, y)
        if mono:
            assert booster_constraints(fitted[v]) == parse(mono)
    prof_med = add_delay_flags(median_profile(X0))
    rng2 = np.random.default_rng(SEED)                               # the same 200 profiles as model_monotone.py
    prof_rand = X.iloc[rng2.choice(len(X), size=N_PROFILES, replace=False)].reset_index(drop=True)
    profile_sets = {"median_profile": prof_med, "random_200": prof_rand}
    n_code = {c: X0[c].value_counts().to_dict() for c in PAY_COLS}
    dr_code = {c: y.groupby(X0[c]).mean().to_dict() for c in PAY_COLS}

    rows = []
    for v, pipe in fitted.items():
        for ps, prof in profile_sets.items():
            for mode in ("as_frozen", "consistent"):
                for col in PAY_COLS:
                    grid = np.sort(X0[col].unique())
                    pr = sweep_status(pipe, prof, col, grid, mode == "consistent")
                    d = np.diff(pr, axis=0)
                    for i in range(len(grid) - 1):
                        a_, b_ = int(grid[i]), int(grid[i + 1])
                        bad = d[i] < 0
                        rows.append(dict(model=v, profiles=ps, n_profiles=int(pr.shape[1]), mode=mode, feature=col,
                                         step=f"{a_}->{b_}", code_from=a_, code_to=b_, step_class=step_class(a_, b_),
                                         n_violating=int(bad.sum()), largest_fall=float(max(0.0, -d[i].min())),
                                         mean_change=float(d[i].mean()),
                                         n_train_from=int(n_code[col].get(a_, 0)), n_train_to=int(n_code[col].get(b_, 0)),
                                         default_rate_from=float(dr_code[col][a_]), default_rate_to=float(dr_code[col][b_]),
                                         thin_cell=bool(min(n_code[col].get(a_, 0), n_code[col].get(b_, 0)) < THIN),
                                         training_rate_falls=bool(dr_code[col][b_] < dr_code[col][a_])))
    # the frozen 13-feature sweep at the median profile, reproduced for the two frozen models (aggregates held, as frozen)
    for v in ("free_numeric", "mono_frozen_numeric"):
        for f in FROZEN_PLUS + FROZEN_MINUS:
            if f in PAY_COLS:
                continue                                              # already in the PAY sweeps above (mode as_frozen)
            sign = 1 if f in FROZEN_PLUS else -1
            u = np.unique(X0[f]); grid = u if len(u) <= 21 else np.unique(np.quantile(X0[f], np.linspace(0, 1, 21)))
            big = pd.concat([prof_med] * len(grid), ignore_index=True); big[f] = list(grid)
            p = fitted[v].predict_proba(big)[:, 1]; d = np.diff(p) * sign
            for i in range(len(grid) - 1):
                rows.append(dict(model=v, profiles="median_profile", n_profiles=1, mode="as_frozen", feature=f,
                                 step=f"{grid[i]:.6g}->{grid[i + 1]:.6g}", step_class=f"grid_step_required_sign_{sign:+d}",
                                 n_violating=int(d[i] < 0), largest_fall=float(max(0.0, -d[i])), mean_change=float(np.diff(p)[i])))
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(RESULTS, "monotone_check_consistent.csv"), index=False)

    # ---- reproduction of the frozen sweep numbers
    fr_sum = json.load(open(os.path.join(RESULTS, "monotone_summary.json")))
    fr_chk = pd.read_csv(os.path.join(RESULTS, "monotone_check.csv"))
    fr_feat = pd.read_csv(os.path.join(RESULTS, "monotone_feature_sweep.csv"))
    sel = lambda **k: sw[np.logical_and.reduce([sw[a] == b for a, b in k.items()])]
    med_curve = sweep_status(fitted["free_numeric"], prof_med, "PAY_1", np.sort(X0["PAY_1"].unique()), False)[:, 0]
    mono_curve = sweep_status(fitted["mono_frozen_numeric"], prof_med, "PAY_1", np.sort(X0["PAY_1"].unique()), False)[:, 0]
    r576 = int(sel(model="free_numeric", profiles="random_200", mode="as_frozen", feature="PAY_1")["n_violating"].sum())
    med13 = sel(model="free_numeric", profiles="median_profile", mode="as_frozen")
    med13 = med13[med13["feature"].isin(FROZEN_PLUS + FROZEN_MINUS)]
    by_feat = med13.groupby("feature")["n_violating"].sum().reindex(FROZEN_PLUS + FROZEN_MINUS).astype(int)
    sweep_repro = dict(
        median_curve_free_max_abs_diff=float(np.abs(med_curve - fr_chk["p_unconstrained"].values).max()),
        median_curve_mono_max_abs_diff=float(np.abs(mono_curve - fr_chk["p_constrained"].values).max()),
        random_200_free_violating_steps=r576,
        frozen_random_200_free_violating_steps=int(fr_sum["violations"]["pay1_random_profiles"]["unconstrained"]["violating_steps"]),
        median_13_features_total=int(by_feat.sum()),
        frozen_median_13_features_total=int(fr_feat["violations_unconstrained"].sum()),
        median_13_features_by_feature=by_feat.to_dict(),
        frozen_by_feature=dict(zip(fr_feat["feature"], fr_feat["violations_unconstrained"].astype(int))))
    log(f"sweep reproduction: median curve max |Δp| free {sweep_repro['median_curve_free_max_abs_diff']:.2e}, mono "
        f"{sweep_repro['median_curve_mono_max_abs_diff']:.2e}; 200 profiles {r576} (frozen "
        f"{sweep_repro['frozen_random_200_free_violating_steps']}); 13 features {sweep_repro['median_13_features_total']} "
        f"(frozen {sweep_repro['frozen_median_13_features_total']})")

    # ---- headline counts
    def split_thin(g, n):
        th = g["thin_cell"].astype(bool).values
        return dict(violating=int(g["n_violating"].sum()), steps=int(len(g) * n),
                    violating_thin=int(g.loc[th, "n_violating"].sum()), steps_thin=int(th.sum() * n),
                    violating_well_populated=int(g.loc[~th, "n_violating"].sum()), steps_well_populated=int((~th).sum() * n))

    def counts(model, profiles, mode, feature="PAY_1"):
        t = sel(model=model, profiles=profiles, mode=mode, feature=feature)
        n = int(t["n_profiles"].iloc[0])
        res = {c: split_thin(g, n) for c, g in t.groupby("step_class")}
        res["all_steps"] = dict(violating=int(t["n_violating"].sum()), steps=int(len(t) * n))
        res["by_step"] = dict(zip(t["step"], t["n_violating"].astype(int)))
        return res
    headline = {}
    for v in ("free_numeric", "mono_frozen_numeric", "free_recoded", PRIMARY):
        for ps in profile_sets:
            for mode in ("as_frozen", "consistent"):
                headline[f"{v}|{ps}|{mode}|PAY_1"] = counts(v, ps, mode)
            t = sel(model=v, profiles=ps, mode="consistent")
            t = t[t["feature"].isin(PAY_COLS)]
            n = int(t["n_profiles"].iloc[0])
            headline[f"{v}|{ps}|consistent|PAY_1..PAY_6"] = {c: split_thin(g, n) for c, g in t.groupby("step_class")}
    for k in ("free_numeric|random_200|as_frozen|PAY_1", "free_numeric|random_200|consistent|PAY_1",
              f"{PRIMARY}|random_200|consistent|PAY_1", "free_numeric|median_profile|consistent|PAY_1",
              f"{PRIMARY}|median_profile|consistent|PAY_1"):
        h = headline[k]
        log(f"{k}: all {h['all_steps']}, between delay levels {h.get('between_delay_levels')}, entry "
            f"{h.get('entry_into_arrears')}, non-delinquent {h.get('non_delinquent')}; by step {h['by_step']}")

    # ---- 3. cross-state check: delay code k >= 1 against the non-delinquent states -2 / -1 (not adjacent on the grid, so
    # not covered by the step counts above). The respecified model leaves the -2 / -1 flags free, so p(k) >= p(-1) is not
    # guaranteed by its constraint; the frozen monotone model guarantees it through the +1 on the raw code.
    cross_state = {}
    for v, pipe in fitted.items():
        for ps, prof in profile_sets.items():
            by_month, n_pairs, n_bad, worst = {}, 0, 0, 0.0
            for col in PAY_COLS:
                grid = np.sort(X0[col].unique()); g = [int(c) for c in grid]
                pr = sweep_status(pipe, prof, col, grid, True)
                bad_m = {}
                for s0 in (-2, -1):
                    if s0 not in g:
                        continue
                    for k in [c for c in g if c >= 1]:
                        d = pr[g.index(k)] - pr[g.index(s0)]
                        n_pairs += int(d.size); nbad = int((d < 0).sum()); n_bad += nbad
                        worst = max(worst, float(max(0.0, -d.min())))
                        if nbad:
                            bad_m[f"p({k}) < p({s0})"] = nbad
                by_month[col] = bad_m
            cross_state[f"{v}|{ps}|consistent"] = dict(pairs=n_pairs, violating=n_bad, largest_fall=worst, by_month=by_month)
    for k in ("free_numeric|random_200|consistent", "mono_frozen_numeric|random_200|consistent",
              f"{PRIMARY}|random_200|consistent"):
        c = cross_state[k]
        log(f"cross-state {k}: p(k>=1) < p(-2 or -1) on {c['violating']} of {c['pairs']} pairs (largest fall "
            f"{c['largest_fall']:.4f}); PAY_1 {c['by_month']['PAY_1']}")

    res = dict(
        script="code/monotone_respec_check.py",
        status="post-freeze sensitivity analysis, training 5-fold CV and training-row sweeps only; nothing scored on the "
               "held-out partition, no slug frozen, no existing results file modified",
        seed=SEED, n_train=int(len(y)), versions=versions, xgboost_params=params,
        folds="common.CV = StratifiedKFold(5, shuffle=True, random_state=2607526); checked against data/processed/cv_folds.csv",
        nadeau_bengio="mine_v2_utils.corrected_t_ci(diffs, 4, 1): variance x (1/5 + 1/4), t with 4 df, 95%",
        sd_note="sd_auc is np.std over the 5 fold AUCs (ddof=0, as in the frozen cv_auc_std)",
        recoding="PAY_k_DEL = max(PAY_k, 0); PAY_k_NC = 1[PAY_k = -2]; PAY_k_PD = 1[PAY_k = -1] (row-wise, fixed)",
        primary=PRIMARY, variants=out, reproduction_of_frozen_oof=repro,
        paired_vs_frozen_oof=comparisons, ladder=ladder_out, data_shape_training_deciles=shapes,
        sweep_definition=dict(
            profiles="training median profile (categoricals at the rounded median level) and the 200 random training rows "
                     "drawn by np.random.default_rng(2607526).choice(20975, 200, replace=False), as in model_monotone.py",
            grid="the codes of PAY_k observed in train.csv",
            as_frozen="only PAY_k (and its own DEL / flag columns for recoded models) changes; MAX_DELAY and "
                      "N_MONTHS_DELAYED keep their old values, as in model_monotone.py",
            consistent="MAX_DELAY = max(PAY_1..PAY_6) clipped at 0 and N_MONTHS_DELAYED = #(PAY_k > 0) are recomputed, as "
                       "in common.clean()",
            step_classes="non_delinquent: both codes <= 0; entry_into_arrears: from <= 0 to >= 1; between_delay_levels: "
                         "k -> k+1 with k >= 1 (only these are counted as wrong-direction steps in the headline)",
            violation="the predicted probability falls on the step (strictly, tolerance 0, as in model_monotone.py)",
            thin_cell=f"either code has fewer than {THIN} training rows in that month"),
        sweep_reproduction=sweep_repro, sweep_headline=headline,
        cross_state_check=dict(
            definition="consistent mode (MAX_DELAY, N_MONTHS_DELAYED recomputed); for each month, every pair (s, k) with s in "
                       "{-2, -1} and k >= 1 a code observed in train.csv; violating = profiles with p(PAY_k = k) < p(PAY_k = s). "
                       "Not covered by the adjacent grid steps; the respecified constraint does not guarantee it (flags free).",
            counts=cross_state),
        files=["results/monotone_respec_cv.json", "results/monotone_check_consistent.csv"],
        seconds=round(time.time() - T0, 1))
    json.dump(res, open(os.path.join(RESULTS, "monotone_respec_cv.json"), "w"), indent=1, default=float)
    log("done: results/monotone_respec_cv.json, results/monotone_check_consistent.csv")


if __name__ == "__main__":
    main()
