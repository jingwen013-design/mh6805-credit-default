"""
model_monotone.py — Monotonically constrained XGBoost (beyond the syllabus).  Owner: LU Xinyu.

Run from Project/code:  python3 model_monotone.py            (about 2-3 minutes)

WHY.  A bank cannot deploy a scorecard in which a client who falls one more month behind is predicted to be *safer*.
An unconstrained gradient-boosted tree has no such guarantee: it fits whatever wiggles the loss likes, so its response
to PAY_1 can go down in places where the data are thin.  XGBoost can impose the shape as a hard constraint
(monotone_constraints), which is checked at every split: the model is then monotone by construction on the whole input
space, not merely "monotone where we happened to look".  The point of this script is to PRICE that guarantee in AUC / AP.

What it does
  1. Reads the tuned unconstrained XGBoost (results/xgboost.json) and re-uses its hyper-parameters EXACTLY
     (learning_rate, max_depth, min_child_weight, n_estimators, subsample, colsample_bytree, tree_method,
     scale_pos_weight, random_state) so the only difference between the two models is the constraint.
  2. Builds the constraint vector from feature_names_out(pre), i.e. against the TRANSFORMED column order, because
     xgboost reads monotone_constraints positionally - a wrong order silently constrains the wrong columns:
       +1 (more of it can only raise predicted risk): PAY_1..PAY_6, N_MONTHS_DELAYED, MAX_DELAY, UTIL_1, UTIL_MEAN, BILL_TREND
       -1 (more of it can only lower predicted risk):  LIMIT_BAL, PAY_RATIO_MEAN
        0 everything else (AGE, bill and payment amounts, the one-hot dummies of SEX / EDUCATION / MARRIAGE)
     The full index -> column -> sign mapping is printed, the mapping is re-checked in all 5 CV folds (the preprocessor
     is refitted per fold, so the column order must be verified there too), and the constraint string is read back out
     of the fitted booster's own config as proof that xgboost received it.
  3. Fits it inside Pipeline(make_preprocessor(scale=False), XGBClassifier(...)), gets 5-fold out-of-fold probabilities
     on the shared folds, and runs it through evaluate()                              -> evaluate(slug="xgboost_monotone")
     (results/xgboost_monotone.json, figures/xgboost_monotone_eval.png).
  4. Prices the constraint: test AUC / AP against the unconstrained model, plus a PAIRED bootstrap over the 8,990 test
     rows (B = 500, seed 2607526, the same recipe as aggregate.py) so the loss can be compared with sampling noise.
  5. Proves empirically that the constraint binds: a PAY_1 sweep with every other feature held at its training median
     (plus the same sweep on 200 random training profiles, and a sweep of all 13 constrained features), counting the
     steps on which each model's predicted probability moves the wrong way.
     -> results/monotone_check.csv (the two PAY_1 curves), results/monotone_feature_sweep.csv (all constrained features),
        figures/monotone_pay1.png.
  6. Writes results/monotone_summary.json (AUC / AP deltas, the constraint vector actually used, violation counts).

Honest reading of the sweeps: holding the other features at their median makes rows that need not exist in the data
(UTIL_1 is not recomputed when BILL_AMT1 moves, etc.).  That is deliberate - the sweep probes the MODEL's response
surface, which is exactly what the constraint is a claim about; it is not a statement about real clients.
Nothing is fitted, tuned or selected on the test set; the test set is touched only inside evaluate() and for the
bootstrap that attaches uncertainty to a difference which already exists.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
import sklearn, xgboost
from xgboost import XGBClassifier
from common import *

MEMBER = "LU Xinyu"
T0 = time.time()

BASELINE_SLUG = "xgboost"                      # the tuned, unconstrained model this one is compared with
B_BOOT = 500                                   # paired bootstrap replicates (same B and seed as aggregate.py)
N_PROFILES = 200                               # random training profiles for the robustness sweep
N_JOBS = 2

# ---- the economic story, written as signs on raw features (translated to transformed columns below) ----------------
PLUS = PAY_COLS + ["N_MONTHS_DELAYED", "MAX_DELAY", "UTIL_1", "UTIL_MEAN", "BILL_TREND"]   # more -> can only be riskier
MINUS = ["LIMIT_BAL", "PAY_RATIO_MEAN"]                                                    # more -> can only be safer

# validated categorical palette (same slots as model_boosting.py): blue = unconstrained XGBoost, orange = constrained
C_FREE, C_MONO, C_BAD, C_TXT, C_TXT2, C_GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "xtick.color": C_TXT2, "ytick.color": C_TXT2,
                     "text.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def log(msg):
    print(f"[mono] {msg}   [t={time.time() - T0:,.0f}s]", flush=True)


def raw_of(transformed_name):
    """'num__PAY_1' -> 'PAY_1'; 'cat__EDUCATION_2' -> 'EDUCATION' (the one-hot level is stripped)."""
    base = transformed_name.split("__", 1)[1]
    return base.rsplit("_", 1)[0] if transformed_name.startswith("cat__") else base


def constraint_vector(tnames):
    """Sign per TRANSFORMED column, in the order xgboost will read them."""
    signs = []
    for n in tnames:
        r = raw_of(n)
        signs.append(1 if r in PLUS else (-1 if r in MINUS else 0))
    return signs


def make_pipe(params, mono_str=None):
    kw = dict(params)
    if mono_str is not None:
        kw["monotone_constraints"] = mono_str
    return Pipeline([("pre", make_preprocessor(scale=False)), ("clf", XGBClassifier(**kw))])


def fold_aucs(y, p_oof):
    """Per-fold AUC of out-of-fold probabilities on the shared CV folds (deterministic)."""
    return np.array([roc_auc_score(y.iloc[va], p_oof[va]) for _, va in CV.split(np.zeros(len(y)), y)])


def booster_monotone_string(clf):
    """Read monotone_constraints back out of the FITTED booster's own config (proof, not trust)."""
    cfg = json.loads(clf.get_booster().save_config())
    try:
        return cfg["learner"]["gradient_booster"]["tree_train_param"]["monotone_constraints"]
    except KeyError:
        return json.dumps(cfg["learner"])[:200]


def sweep(pipes, profile_row, feature, grid):
    """Predicted probability of each fitted pipeline along a grid of one feature, other features held fixed."""
    Xg = pd.concat([profile_row] * len(grid), ignore_index=True)
    Xg[feature] = list(grid)
    return {k: p.predict_proba(Xg)[:, 1] for k, p in pipes.items()}


def violations(p, sign, tol=0.0):
    """Steps on which the curve moves against the required direction (sign +1 = must not fall, -1 = must not rise)."""
    d = np.diff(p) * sign                       # required: d >= 0 everywhere
    bad = d < -tol
    return int(bad.sum()), float(max(0.0, -d.min())) if len(d) else 0.0


def print_result(r):
    print(f"\n{r['name']}: CV AUC {r['cv_auc']:.4f}±{r['cv_auc_std']:.4f} | OOF AUC {r['oof_auc']:.4f} | train AUC {r['train_auc']:.4f} | "
          f"test AUC {r['test_auc']:.4f} | test AP {r['test_ap']:.4f} | Brier {r['test_brier']:.4f} | log-loss {r['test_logloss']:.4f}")
    for key in ("at_0_5", "at_tuned_f1", "at_tuned_cost"):
        m = r[key]
        print(f"   {key:14s} t={m['threshold']:.2f} acc {m['accuracy']:.4f} bal.acc {m['balanced_accuracy']:.4f} "
              f"P1 {m['precision_1']:.4f} R1 {m['recall_1']:.4f} F1 {m['f1_1']:.4f} cost {m['cost']:.0f}")


def main():
    X_train, X_test, y_train, y_test, feature_names = load_split()
    log(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")
    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__)

    # ================================================================== 1. hyper-parameters of the unconstrained model
    base_path = os.path.join(RESULTS, f"{BASELINE_SLUG}.json")
    if not os.path.exists(base_path):
        raise SystemExit(f"{base_path} not found — run model_boosting.py first; this script must re-use its tuned hyper-parameters.")
    base = json.load(open(base_path))
    bp = base["best_params"]
    params = dict(objective=bp["objective"], eval_metric="auc", tree_method=bp["tree_method"], n_jobs=N_JOBS,
                  random_state=bp["random_state"], learning_rate=bp["learning_rate"], max_depth=bp["max_depth"],
                  min_child_weight=bp["min_child_weight"], n_estimators=bp["n_estimators"], subsample=bp["subsample"],
                  colsample_bytree=bp["colsample_bytree"], scale_pos_weight=bp["scale_pos_weight"])
    assert params["random_state"] == SEED, "the baseline was not fitted with the project seed"
    log(f"unconstrained reference '{base['name']}' (results/{BASELINE_SLUG}.json): test AUC {base['test_auc']:.4f}, "
        f"test AP {base['test_ap']:.4f}, CV AUC {base['cv_auc']:.4f}")
    log(f"hyper-parameters re-used verbatim: {json.dumps({k: v for k, v in params.items() if k not in ('n_jobs', 'eval_metric')})}")

    # ================================================================== 2. constraint vector against the transformed columns
    pre_full = make_preprocessor(scale=False).fit(X_train)
    tnames = feature_names_out(pre_full)
    n_cols = pre_full.transform(X_train.head(5)).shape[1]
    assert tnames is not None and len(tnames) == n_cols, "feature_names_out does not match the transformed width"
    mono = constraint_vector(tnames)
    mono_str = "(" + ",".join(str(s) for s in mono) + ")"

    # every named feature must be found exactly once, otherwise a typo would silently constrain nothing
    hit = [raw_of(n) for n in tnames]
    for f in PLUS + MINUS:
        assert hit.count(f) == 1, f"{f} appears {hit.count(f)} times among the transformed columns"
    n_plus, n_minus, n_zero = mono.count(1), mono.count(-1), mono.count(0)
    print("\nconstraint vector against the TRANSFORMED column order (index -> column -> raw feature -> sign):")
    for i, (n, s) in enumerate(zip(tnames, mono)):
        mark = {1: "+1  risk can only RISE", -1: "-1  risk can only FALL", 0: " 0  unconstrained"}[s]
        print(f"   [{i:2d}] {n:24s} -> {raw_of(n):16s} {mark}")
    print(f"   totals: +1 on {n_plus} columns, -1 on {n_minus}, 0 on {n_zero} (of {len(mono)})")
    assert n_plus == len(PLUS) and n_minus == len(MINUS)

    # the preprocessor is refitted inside every CV fold: verify the column order is identical there as well
    fold_orders_ok = True
    for f, (tr, _) in enumerate(CV.split(X_train, y_train)):
        nm = feature_names_out(make_preprocessor(scale=False).fit(X_train.iloc[tr]))
        if list(nm) != list(tnames):
            fold_orders_ok = False
            print(f"   !! fold {f} column order differs: {nm}")
    log(f"column order identical in all {CV.get_n_splits()} CV folds: {fold_orders_ok} "
        f"(so the positional constraint vector stays aligned when the preprocessor is refitted)")
    assert fold_orders_ok

    # ================================================================== 3. fit the constrained model + CV
    t0 = time.time()
    mono_pipe = make_pipe(params, mono_str).fit(X_train, y_train)
    fit_seconds = time.time() - t0
    read_back = booster_monotone_string(mono_pipe.named_steps["clf"])
    log(f"constrained model fitted in {fit_seconds:.1f}s; booster config reports monotone_constraints = {read_back}")
    assert [int(v) for v in read_back.strip("()").split(",")] == mono, "the booster did NOT receive the intended constraint vector"

    t0 = time.time()
    p_oof = cross_val_predict(clone(mono_pipe), X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    aucs = fold_aucs(y_train, p_oof)
    log(f"5-fold CV AUC of the constrained pipeline {aucs.mean():.4f} ± {aucs.std():.4f} ({time.time() - t0:.0f}s)")

    free_pipe = make_pipe(params).fit(X_train, y_train)      # unconstrained twin, refitted here for the sweeps
    p_test_free_here = free_pipe.predict_proba(X_test)[:, 1]
    p_test_free = np.load(os.path.join(RESULTS, f"{BASELINE_SLUG}_p_test.npy"))
    same = float(np.abs(p_test_free_here - p_test_free).max())
    log(f"unconstrained twin refitted for the sweeps reproduces results/{BASELINE_SLUG}_p_test.npy "
        f"(max |Δp| = {same:.2e}, its test AUC {roc_auc_score(y_test, p_test_free_here):.4f})")

    # ================================================================== 4. evaluate() -> joins the comparison tables
    notes = (f"XGBoost with monotone_constraints and OTHERWISE IDENTICAL hyper-parameters to the tuned unconstrained model "
             f"(results/{BASELINE_SLUG}.json): learning_rate {params['learning_rate']}, max_depth {params['max_depth']}, "
             f"min_child_weight {params['min_child_weight']}, n_estimators {params['n_estimators']}, subsample {params['subsample']}, "
             f"colsample_bytree {params['colsample_bytree']}, tree_method {params['tree_method']}, random_state {SEED}. Nothing is re-tuned, "
             f"so the whole AUC/AP difference is the price of the constraint. Signs are set on the TRANSFORMED columns via "
             f"feature_names_out (xgboost reads monotone_constraints positionally): +1 on {', '.join(PLUS)}; -1 on {', '.join(MINUS)}; "
             f"0 on the remaining {n_zero} columns. The column order was verified to be identical in all 5 CV folds and the vector was "
             f"read back out of the fitted booster's config. Monotonicity then holds by construction over the whole input space, not only "
             f"where the data are dense; see results/monotone_summary.json for the empirical sweeps.")
    res = evaluate("XGBoost (monotone constraints)", mono_pipe, X_train, y_train, X_test, y_test, member=MEMBER,
                   slug="xgboost_monotone", best_params=dict(bp, monotone_constraints=mono_str),
                   cv_auc=float(aucs.mean()), cv_auc_std=float(aucs.std()), fit_seconds=round(fit_seconds, 1), p_oof=p_oof,
                   notes=notes,
                   extra=dict(cv_fold_aucs=np.round(aucs, 5).tolist(), baseline_slug=BASELINE_SLUG,
                              baseline_test_auc=base["test_auc"], baseline_test_ap=base["test_ap"], baseline_cv_auc=base["cv_auc"],
                              monotone_constraints=mono_str, constrained_plus=PLUS, constrained_minus=MINUS,
                              n_features_raw=len(feature_names), n_features_after_onehot=len(tnames), versions=versions))
    p_test_mono = np.load(os.path.join(RESULTS, "xgboost_monotone_p_test.npy"))

    d_auc = res["test_auc"] - base["test_auc"]
    d_ap = res["test_ap"] - base["test_ap"]
    d_cv = res["cv_auc"] - base["cv_auc"]
    res["extra"]["delta_test_auc"] = float(d_auc)
    res["extra"]["delta_test_ap"] = float(d_ap)
    res["extra"]["delta_cv_auc"] = float(d_cv)
    json.dump(res, open(os.path.join(RESULTS, "xgboost_monotone.json"), "w"), indent=1)

    # ---- paired bootstrap of the difference (test rows resampled; nothing is selected) --------------------------------
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    y_te = y_test.values
    idx_all = rng.integers(0, len(y_te), size=(B_BOOT, len(y_te)))
    b_auc = np.empty((B_BOOT, 2)); b_ap = np.empty((B_BOOT, 2))
    for b in range(B_BOOT):
        idx = idx_all[b]; yb = y_te[idx]
        for j, p in enumerate((p_test_free, p_test_mono)):
            pb = p[idx]
            b_auc[b, j] = roc_auc_score(yb, pb); b_ap[b, j] = average_precision_score(yb, pb)
    da, dp = b_auc[:, 1] - b_auc[:, 0], b_ap[:, 1] - b_ap[:, 0]     # constrained minus unconstrained
    boot = dict(B=B_BOOT, seed=SEED,
                auc_diff_mean=float(da.mean()), auc_diff_2_5=float(np.percentile(da, 2.5)), auc_diff_97_5=float(np.percentile(da, 97.5)),
                auc_diff_excludes_zero=bool(np.percentile(da, 2.5) > 0 or np.percentile(da, 97.5) < 0),
                share_replicates_constrained_better=float((da > 0).mean()),
                ap_diff_mean=float(dp.mean()), ap_diff_2_5=float(np.percentile(dp, 2.5)), ap_diff_97_5=float(np.percentile(dp, 97.5)),
                ap_diff_excludes_zero=bool(np.percentile(dp, 2.5) > 0 or np.percentile(dp, 97.5) < 0),
                share_replicates_constrained_better_ap=float((dp > 0).mean()))
    log(f"paired bootstrap B={B_BOOT} done in {time.time() - t0:.0f}s: ΔAUC {boot['auc_diff_mean']:+.5f} "
        f"[{boot['auc_diff_2_5']:+.5f}, {boot['auc_diff_97_5']:+.5f}], excludes zero: {boot['auc_diff_excludes_zero']}")

    # ================================================================== 5. does the constraint bind? PAY_1 sweep
    pipes = {"unconstrained": free_pipe, "constrained": mono_pipe}
    med = X_train.median(numeric_only=True)
    profile = X_train.iloc[[0]].copy().reset_index(drop=True)
    for c in X_train.columns:
        profile.loc[0, c] = med[c]
    for c in CAT_COLS:                                   # median of an integer code, rounded back onto a real level
        v = int(round(float(med[c])))
        assert v in set(X_train[c].unique()), f"median profile has an unseen {c} level {v}"
        profile.loc[0, c] = v
    profile = profile.astype(X_train.dtypes.to_dict())
    print("\nmedian training profile used for the sweeps:")
    print("   " + ", ".join(f"{c}={profile.loc[0, c]:g}" for c in ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE",
                                                                   "BILL_AMT1", "PAY_AMT1", "UTIL_1", "UTIL_MEAN",
                                                                   "PAY_RATIO_MEAN", "N_MONTHS_DELAYED", "MAX_DELAY", "BILL_TREND"]))

    pay1_grid = np.sort(X_train["PAY_1"].unique())
    curves = sweep(pipes, profile, "PAY_1", pay1_grid)
    v_free, drop_free = violations(curves["unconstrained"], +1)
    v_mono, drop_mono = violations(curves["constrained"], +1)
    chk = pd.DataFrame({"PAY_1": pay1_grid,
                        "p_unconstrained": curves["unconstrained"], "p_constrained": curves["constrained"]})
    chk["step_unconstrained"] = chk["p_unconstrained"].diff()
    chk["step_constrained"] = chk["p_constrained"].diff()
    chk["violation_unconstrained"] = chk["step_unconstrained"] < 0
    chk["violation_constrained"] = chk["step_constrained"] < 0
    chk.to_csv(os.path.join(RESULTS, "monotone_check.csv"), index=False)
    print(f"\nPAY_1 sweep at the median profile ({len(pay1_grid)} grid points, {len(pay1_grid) - 1} steps):")
    print(chk.round(5).to_string(index=False))
    print(f"   unconstrained: {v_free} step(s) where predicted risk FALLS as the delay grows (largest fall {drop_free:.5f})")
    print(f"   constrained  : {v_mono} step(s) (largest fall {drop_mono:.5f})")

    # ---- the same sweep on 200 random training profiles (one median row could be a lucky draw) ------------------------
    rng2 = np.random.default_rng(SEED)
    sample = X_train.iloc[rng2.choice(len(X_train), size=N_PROFILES, replace=False)].reset_index(drop=True)
    big = pd.concat([sample.assign(PAY_1=v) for v in pay1_grid], ignore_index=True)   # grid-major: blocks of N_PROFILES rows
    prof_stats = {}
    for k, p in pipes.items():
        pr = p.predict_proba(big)[:, 1].reshape(len(pay1_grid), N_PROFILES)           # grid x profiles
        d = np.diff(pr, axis=0)
        prof_stats[k] = dict(profiles_with_violation=int((d < 0).any(axis=0).sum()), n_profiles=N_PROFILES,
                             violating_steps=int((d < 0).sum()), total_steps=int(d.size),
                             largest_fall=float(max(0.0, -d.min())))
    log(f"PAY_1 sweep on {N_PROFILES} random training profiles: unconstrained violates on "
        f"{prof_stats['unconstrained']['profiles_with_violation']}/{N_PROFILES} profiles "
        f"({prof_stats['unconstrained']['violating_steps']}/{prof_stats['unconstrained']['total_steps']} steps, "
        f"largest fall {prof_stats['unconstrained']['largest_fall']:.4f}); constrained violates on "
        f"{prof_stats['constrained']['profiles_with_violation']}/{N_PROFILES} "
        f"({prof_stats['constrained']['violating_steps']}/{prof_stats['constrained']['total_steps']} steps)")

    # ---- all 13 constrained features, swept at the median profile ----------------------------------------------------
    rows = []
    for f in PLUS + MINUS:
        sign = 1 if f in PLUS else -1
        u = np.unique(X_train[f])
        grid = u if len(u) <= 21 else np.unique(np.quantile(X_train[f], np.linspace(0, 1, 21)))
        cv_ = sweep(pipes, profile, f, grid)
        vf, df_ = violations(cv_["unconstrained"], sign)
        vm, dm = violations(cv_["constrained"], sign)
        rows.append(dict(feature=f, required_sign=sign, n_grid=len(grid), n_steps=len(grid) - 1,
                         violations_unconstrained=vf, largest_wrong_move_unconstrained=df_,
                         violations_constrained=vm, largest_wrong_move_constrained=dm,
                         range_unconstrained=float(cv_["unconstrained"].max() - cv_["unconstrained"].min()),
                         range_constrained=float(cv_["constrained"].max() - cv_["constrained"].min())))
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(RESULTS, "monotone_feature_sweep.csv"), index=False)
    print(f"\nsweep of all {len(sw)} constrained features at the median profile (violations = steps in the wrong direction):")
    print(sw.round(5).to_string(index=False))
    tot_free, tot_mono = int(sw["violations_unconstrained"].sum()), int(sw["violations_constrained"].sum())
    log(f"total wrong-direction steps over all constrained features: unconstrained {tot_free}, constrained {tot_mono}")

    # ================================================================== 6. figure
    fig = plt.figure(figsize=(13.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], width_ratios=[1.05, 1.0], hspace=0.5, wspace=0.42)
    a, a2, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    a.plot(pay1_grid, curves["unconstrained"], "-o", color=C_FREE, lw=2, ms=5, mec="white", mew=0.8,
           label=f"unconstrained XGBoost (test AUC {base['test_auc']:.4f})")
    a.plot(pay1_grid, curves["constrained"], "-s", color=C_MONO, lw=2, ms=5, mec="white", mew=0.8,
           label=f"monotone constraints (test AUC {res['test_auc']:.4f})")
    steps_free, steps_mono = np.diff(curves["unconstrained"]), np.diff(curves["constrained"])
    bad = np.where(steps_free < 0)[0]
    for i in bad:                                # mark the violating points on the curve itself
        a.plot(pay1_grid[i + 1], curves["unconstrained"][i + 1], "o", ms=11, mfc="none", mec=C_BAD, mew=1.8)
    a.set_xlabel("PAY_1 (repayment status last month; -2/-1/0 = paid duly or revolving, k = k months in arrears)", fontsize=9)
    a.set_ylabel("Predicted probability of default")
    a.set_title(f"Response to PAY_1, every other feature at its training median\n"
                f"wrong-direction steps: unconstrained {v_free} of {len(pay1_grid) - 1}, constrained {v_mono}",
                loc="left", fontsize=10.5)
    a.set_xticks(pay1_grid)
    a.grid(color=C_GRID, lw=0.8); a.set_axisbelow(True)
    a.legend(loc="lower right", frameon=False, fontsize=8.5)

    # ---- the violations are a few thousandths on a 0.06-0.22 scale, so the steps get their own panel ----------------
    xs = np.arange(len(steps_free))
    labels = [f"{pay1_grid[i]:g}→{pay1_grid[i + 1]:g}" for i in xs]
    cap = float(max(5 * abs(min(steps_free.min(), steps_mono.min(), 0.0)), 0.004))
    a2.bar(xs - 0.2, np.clip(steps_free, -cap, cap), width=0.38,
           color=[C_BAD if v < 0 else C_FREE for v in steps_free])
    a2.bar(xs + 0.2, np.clip(steps_mono, -cap, cap), width=0.38,
           color=[C_BAD if v < 0 else C_MONO for v in steps_mono])
    for x, v, off in [(x, v, -0.2) for x, v in zip(xs, steps_free)] + [(x, v, 0.2) for x, v in zip(xs, steps_mono)]:
        if abs(v) > cap:                          # honest label for the bars that run off the clipped scale
            a2.text(x + off, cap * 0.92, f"{v:+.3f}", ha="center", va="top", rotation=90, fontsize=7, color="white")
    a2.axhline(0, color=C_TXT2, lw=1)
    a2.set_ylim(-cap * 1.15, cap * 1.15); a2.set_xticks(xs); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel(f"Δ probability\n(clipped at ±{cap:.3f})", fontsize=8.5)
    a2.set_title("Change at each step of the same sweep — red = wrong direction", loc="left", fontsize=9.5)
    a2.grid(axis="y", color=C_GRID, lw=0.8); a2.set_axisbelow(True)

    o = sw.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(o))
    b.barh(yy + 0.19, o["violations_unconstrained"], height=0.36, color=C_FREE, label="unconstrained XGBoost")
    b.barh(yy - 0.19, o["violations_constrained"], height=0.36, color=C_MONO, label="monotone constraints")
    for y_, v, n in zip(yy, o["violations_unconstrained"], o["n_steps"]):
        b.text(v + 0.12, y_ + 0.19, f"{int(v)} / {int(n)}", va="center", fontsize=8, color=C_TXT2)
    for y_, v in zip(yy, o["violations_constrained"]):
        b.text(v + 0.12, y_ - 0.19, f"{int(v)}", va="center", fontsize=8, color=C_TXT2)
    b.set_yticks(yy); b.set_yticklabels([f"{f}  ({'+1' if s > 0 else '-1'})" for f, s in zip(o["feature"], o["required_sign"])],
                                        fontsize=8.5)
    b.set_xlim(0, max(3, float(o["violations_unconstrained"].max()) * 1.35))
    b.set_xlabel("Steps of the sweep that move predicted risk the WRONG way")
    b.set_title(f"Every constrained feature, swept at the median profile\n"
                f"total wrong-direction steps: unconstrained {tot_free}, constrained {tot_mono}", loc="left", fontsize=10.5)
    b.grid(axis="x", color=C_GRID, lw=0.8); b.set_axisbelow(True); b.tick_params(axis="y", length=0)
    b.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle(f"Monotonically constrained XGBoost: the constraint binds, and it costs {abs(d_auc):.4f} AUC "
                 f"({'−' if d_auc < 0 else '+'}{abs(d_auc) / base['test_auc'] * 100:.2f}% of {base['test_auc']:.4f}) "
                 f"and {abs(d_ap):.4f} AP", fontsize=11.5, x=0.01, ha="left")
    save_extra_figure(fig, "monotone_pay1.png")

    # ================================================================== 7. summary JSON
    summary = dict(
        name=res["name"], slug=res["slug"], member=MEMBER, seed=SEED, versions=versions,
        hyper_parameters=dict(params, monotone_constraints=mono_str),
        baseline=dict(slug=BASELINE_SLUG, name=base["name"], test_auc=base["test_auc"], test_ap=base["test_ap"],
                      cv_auc=base["cv_auc"], test_brier=base["test_brier"], test_logloss=base["test_logloss"],
                      f1_at_tuned=base["at_tuned_f1"]["f1_1"], cost_at_tuned=base["at_tuned_cost"]["cost"]),
        constrained=dict(test_auc=res["test_auc"], test_ap=res["test_ap"], cv_auc=res["cv_auc"], cv_auc_std=res["cv_auc_std"],
                         test_brier=res["test_brier"], test_logloss=res["test_logloss"],
                         f1_at_tuned=res["at_tuned_f1"]["f1_1"], cost_at_tuned=res["at_tuned_cost"]["cost"]),
        cost_of_the_constraint=dict(delta_test_auc=float(d_auc), delta_test_ap=float(d_ap), delta_cv_auc=float(d_cv),
                                    delta_test_auc_pct=float(100 * d_auc / base["test_auc"]),
                                    delta_test_ap_pct=float(100 * d_ap / base["test_ap"]),
                                    delta_f1_at_tuned=float(res["at_tuned_f1"]["f1_1"] - base["at_tuned_f1"]["f1_1"]),
                                    delta_cost_at_tuned=float(res["at_tuned_cost"]["cost"] - base["at_tuned_cost"]["cost"]),
                                    constrained_is_worse_on_auc=bool(d_auc < 0), constrained_is_worse_on_ap=bool(d_ap < 0),
                                    paired_bootstrap=boot),
        constraint_vector=dict(order="transformed columns, from feature_names_out(pre)", string=mono_str,
                               n_plus=n_plus, n_minus=n_minus, n_zero=n_zero,
                               plus_features=PLUS, minus_features=MINUS,
                               read_back_from_booster=read_back,
                               column_order_identical_in_all_folds=bool(fold_orders_ok),
                               columns=[dict(index=i, transformed=n, raw=raw_of(n), constraint=s)
                                        for i, (n, s) in enumerate(zip(tnames, mono))]),
        violations=dict(
            method=("one feature is swept while the others are held at their training median (categoricals at the rounded "
                    "median level); a violation is a step of the grid on which the predicted probability moves against the "
                    "required direction. The sweep probes the model's response surface, not real clients."),
            pay1_median_profile=dict(grid=[float(v) for v in pay1_grid], n_steps=int(len(pay1_grid) - 1),
                                     unconstrained=dict(violations=v_free, largest_fall=drop_free,
                                                        p=[float(v) for v in curves["unconstrained"]]),
                                     constrained=dict(violations=v_mono, largest_fall=drop_mono,
                                                      p=[float(v) for v in curves["constrained"]])),
            pay1_random_profiles=prof_stats,
            all_constrained_features=dict(total_violations_unconstrained=tot_free, total_violations_constrained=tot_mono,
                                          per_feature=sw.round(6).to_dict(orient="records"))),
        files=dict(results=["xgboost_monotone.json", "monotone_check.csv", "monotone_feature_sweep.csv", "monotone_summary.json"],
                   figures=["xgboost_monotone_eval.png", "monotone_pay1.png"]),
        seconds=round(time.time() - T0, 1))
    json.dump(summary, open(os.path.join(RESULTS, "monotone_summary.json"), "w"), indent=1)

    # ================================================================== 8. the headline
    print_result(res)
    verdict = ("The constrained model is WORSE" if d_auc < 0 else
               ("The constrained model is BETTER" if d_auc > 0 else "The two models tie"))
    print(f"\n=== price of the monotone constraint ===")
    print(f"   test AUC {base['test_auc']:.4f} (free) -> {res['test_auc']:.4f} (constrained)   Δ = {d_auc:+.4f} "
          f"({100 * d_auc / base['test_auc']:+.2f}%)")
    print(f"   test AP  {base['test_ap']:.4f} (free) -> {res['test_ap']:.4f} (constrained)   Δ = {d_ap:+.4f} "
          f"({100 * d_ap / base['test_ap']:+.2f}%)")
    print(f"   5-fold CV AUC {base['cv_auc']:.4f} -> {res['cv_auc']:.4f}   Δ = {d_cv:+.4f}")
    print(f"   F1 at the OOF-tuned threshold {base['at_tuned_f1']['f1_1']:.4f} -> {res['at_tuned_f1']['f1_1']:.4f}; "
          f"cost at the cost-tuned threshold {base['at_tuned_cost']['cost']:.0f} -> {res['at_tuned_cost']['cost']:.0f}")
    print(f"   {verdict} on test AUC. Paired bootstrap (B={B_BOOT}): ΔAUC {boot['auc_diff_mean']:+.5f} "
          f"95% CI [{boot['auc_diff_2_5']:+.5f}, {boot['auc_diff_97_5']:+.5f}], excludes zero: {boot['auc_diff_excludes_zero']}; "
          f"the constrained model wins in {100 * boot['share_replicates_constrained_better']:.1f}% of replicates.")
    print(f"   ΔAP {boot['ap_diff_mean']:+.5f} 95% CI [{boot['ap_diff_2_5']:+.5f}, {boot['ap_diff_97_5']:+.5f}], "
          f"excludes zero: {boot['ap_diff_excludes_zero']}")
    print(f"   constraint binds: PAY_1 wrong-direction steps at the median profile {v_free} (free) vs {v_mono} (constrained); "
          f"over {N_PROFILES} random profiles {prof_stats['unconstrained']['violating_steps']} vs "
          f"{prof_stats['constrained']['violating_steps']} of {prof_stats['unconstrained']['total_steps']} steps.")
    log(f"done. results/xgboost_monotone.json, results/monotone_check.csv, results/monotone_feature_sweep.csv, "
        f"results/monotone_summary.json, figures/monotone_pay1.png, figures/xgboost_monotone_eval.png")


if __name__ == "__main__":
    main()
