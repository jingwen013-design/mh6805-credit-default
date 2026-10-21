"""
model_boosting.py — Boosting (lecture B7): sklearn GradientBoostingClassifier + XGBoost.  Owner: LU Xinyu.

Run from Project/code:  python3 model_boosting.py            (about 12–15 minutes with 2 cores)

What it does
  1. Gradient boosting (course method, sklearn GradientBoostingClassifier, subsample=0.8, random_state=SEED) inside
     Pipeline(make_preprocessor(scale=False), clf).  Grid: n_estimators {200, 500} x learning_rate {0.03, 0.1} x max_depth {2, 3, 4}
     = 12 settings, 5-fold CV (common.CV), criterion ROC AUC (common.SCORING).
     BUDGET: one 500-tree sklearn GBM takes 35–65 s on the full training set, so the 60 grid fits would need ~45 min.  Two
     documented shortcuts keep the script inside 15 min without changing what is tuned:
       (a) the grid is scored on a stratified 50 % subsample of the TRAINING set (seed 2607526);
       (b) for each (learning_rate, max_depth) one 500-tree model is fitted per fold and the AUC after 200 and after 500 trees is
           read from staged_predict_proba — mathematically identical to fitting the 200- and 500-tree models separately, because
           gradient boosting is sequential (the first 200 trees of a 500-tree model ARE the 200-tree model).
     The winning setting is refitted on the full training set; cv_auc in the JSON is the 5-fold CV AUC of that final pipeline on
     the shared folds (the subsample grid score is kept in extra).                                   -> evaluate(slug="gradient_boosting")
  2. XGBoost (advanced method): XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist', n_jobs=2,
     random_state=SEED, subsample=0.8, colsample_bytree=0.8) inside the same pipeline; full GridSearchCV over
     max_depth {3, 4, 6} x learning_rate {0.03, 0.1} x n_estimators {300, 600} x min_child_weight {1, 5} = 24 settings x 5 folds
     (120 fits, ~2–3 s each, so no subsampling is needed).                                            -> evaluate(slug="xgboost")
  3. Class-weighted variant: same best hyper-parameters, scale_pos_weight = n_neg / n_pos (~3.5).  -> evaluate(slug="xgboost_weighted")
  4. XGBoost gain importance (total gain share and number of splits, one-hot dummies summed back to the raw feature)
     -> results/xgb_importance.csv, figures/xgb_importance.png.
  5. Learning curves: for the best depth / min_child_weight, 5-fold CV AUC (and training-fold AUC) after every boosting round up to
     1,500 rounds for both learning rates -> results/xgb_learning_curve.csv, figures/xgb_learning_curve.png; the staged GBM CV
     curves from step 1 -> results/gbm_cv_curve.csv, figures/gbm_learning_curve.png.
Everything is fitted / tuned on the training rows only; the test set is touched only inside evaluate().
"""
import sys, time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
import sklearn, xgboost
from xgboost import XGBClassifier
from common import *

MEMBER = "LU Xinyu"
assert SCORING == "roc_auc", "the staged GBM loop below computes ROC AUC directly; adapt it if SCORING changes"

# --- gradient boosting (sklearn) ------------------------------------------------------------------------------------
GBM_SUBSAMPLE_FRAC = 0.5                                  # stratified share of the training set used to score the grid (budget)
GBM_LR, GBM_DEPTH, GBM_NTREES = [0.03, 0.1], [2, 3, 4], [200, 500]
GBM_MAX_TREES = max(GBM_NTREES)
GBM_STAGES = list(range(10, GBM_MAX_TREES + 1, 10))     # AUC is read after these many trees (200 and 500 are in the list)
# --- xgboost ----------------------------------------------------------------------------------------------------------
XGB_FIXED = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=2, random_state=SEED,
                 subsample=0.8, colsample_bytree=0.8)
XGB_GRID = {"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.03, 0.1], "clf__n_estimators": [300, 600],
            "clf__min_child_weight": [1, 5]}
CURVE_MAX_ROUNDS = 1500                                   # learning-curve diagnostic (beyond the grid, to show the plateau)
LOGIT_CV_AUC, LOGIT_TEST_AUC = 0.760, 0.764               # reference numbers from README

# validated categorical palette (dataviz skill, light surface): slot 1 blue (XGBoost), slot 2 orange (GBM), slot 3 aqua (weighted)
C_XGB, C_GBM, C_W, C_TXT, C_TXT2, C_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "xtick.color": C_TXT2, "ytick.color": C_TXT2,
                     "text.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def make_gbm_pipe(**clf_kw):
    clf = GradientBoostingClassifier(subsample=0.8, random_state=SEED, **clf_kw)
    return Pipeline([("pre", make_preprocessor(scale=False)), ("clf", clf)])


def make_xgb_pipe(**clf_kw):
    return Pipeline([("pre", make_preprocessor(scale=False)), ("clf", XGBClassifier(**XGB_FIXED, **clf_kw))])


def fold_aucs(y, p_oof):
    """Per-fold AUC of out-of-fold probabilities on the shared CV folds (deterministic)."""
    return np.array([roc_auc_score(y.iloc[va], p_oof[va]) for _, va in CV.split(np.zeros(len(y)), y)])


def staged_auc(clf, Xt, y, stages):
    """AUC after k trees for k in stages, from one fitted GradientBoostingClassifier (staged_predict_proba)."""
    want = set(stages); out = {}
    for k, p in enumerate(clf.staged_predict_proba(Xt), start=1):
        if k in want:
            out[k] = roc_auc_score(y, p[:, 1])
    return np.array([out[k] for k in stages])


def gbm_fold_job(lr, depth, fold, tr, va, X, y):
    """Fit one 500-tree GBM pipeline on a training fold; return validation- and training-fold AUC after every 10 trees."""
    t0 = time.time()
    pipe = make_gbm_pipe(n_estimators=GBM_MAX_TREES, learning_rate=lr, max_depth=depth).fit(X.iloc[tr], y.iloc[tr])
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    return dict(learning_rate=lr, max_depth=depth, fold=fold, seconds=time.time() - t0,
                val=staged_auc(clf, pre.transform(X.iloc[va]), y.iloc[va], GBM_STAGES),
                train=staged_auc(clf, pre.transform(X.iloc[tr]), y.iloc[tr], GBM_STAGES))


def aggregate_to_raw(values, transformed_names, raw_names):
    """Sum a per-transformed-column vector back to the 29 raw features (one-hot dummies of a categorical are summed)."""
    out = dict.fromkeys(raw_names, 0.0)
    for v, n in zip(values, transformed_names):
        base = n.split("__", 1)[1]
        if n.startswith("cat__"):
            base = base.rsplit("_", 1)[0]
        out[base] += float(v)
    return np.array([out[f] for f in raw_names])


def booster_scores(booster, n_cols, importance_type):
    """xgboost get_score -> dense vector aligned with the transformed columns (unused columns get 0)."""
    sc = booster.get_score(importance_type=importance_type)
    names = booster.feature_names or [f"f{i}" for i in range(n_cols)]
    return np.array([float(sc.get(n, 0.0)) for n in names])


def print_result(r, extra_keys=()):
    print(f"\n{r['name']}: CV AUC {r['cv_auc']:.4f}±{r['cv_auc_std']:.4f} | OOF AUC {r['oof_auc']:.4f} | train AUC {r['train_auc']:.4f} | "
          f"test AUC {r['test_auc']:.4f} | test AP {r['test_ap']:.4f} | Brier {r['test_brier']:.4f} | log-loss {r['test_logloss']:.4f}")
    for key in ("at_0_5", "at_tuned_f1", "at_tuned_cost"):
        m = r[key]
        print(f"   {key:14s} t={m['threshold']:.2f} acc {m['accuracy']:.4f} bal.acc {m['balanced_accuracy']:.4f} "
              f"P1 {m['precision_1']:.4f} R1 {m['recall_1']:.4f} F1 {m['f1_1']:.4f} cost {m['cost']:.0f}")


def main():
    t_all = time.time()
    X_train, X_test, y_train, y_test, feature_names = load_split()
    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    print(f"train {X_train.shape}, test {X_test.shape}, default rate {y_train.mean():.4f}  (neg/pos = {n_neg / n_pos:.3f})")
    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__)

    # ================================================================== 1. sklearn gradient boosting: staged grid on a subsample
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=GBM_SUBSAMPLE_FRAC, stratify=y_train, random_state=SEED)
    Xs, ys = Xs.reset_index(drop=True), ys.reset_index(drop=True)
    folds = list(CV.split(Xs, ys))
    print(f"\n[GBM] grid on a stratified {GBM_SUBSAMPLE_FRAC:.0%} subsample of the training set: {len(Xs):,} rows, "
          f"default rate {ys.mean():.4f}; {len(GBM_LR) * len(GBM_DEPTH)} (lr, depth) settings x 5 folds = "
          f"{len(GBM_LR) * len(GBM_DEPTH) * 5} fits of {GBM_MAX_TREES} trees, AUC read at 200 and 500 trees (staged)")
    t0 = time.time()
    jobs = Parallel(n_jobs=2, verbose=0)(delayed(gbm_fold_job)(lr, d, f, tr, va, Xs, ys)
                                         for lr, d in itertools.product(GBM_LR, GBM_DEPTH) for f, (tr, va) in enumerate(folds))
    gbm_grid_seconds = time.time() - t0

    curve_rows, grid_rows = [], []
    for lr, d in itertools.product(GBM_LR, GBM_DEPTH):
        js = [j for j in jobs if j["learning_rate"] == lr and j["max_depth"] == d]
        val = np.stack([j["val"] for j in js]); trn = np.stack([j["train"] for j in js])
        for i, k in enumerate(GBM_STAGES):
            curve_rows.append(dict(learning_rate=lr, max_depth=d, n_estimators=k, cv_auc=val[:, i].mean(), cv_auc_std=val[:, i].std(),
                                   train_auc=trn[:, i].mean()))
        for n in GBM_NTREES:
            i = GBM_STAGES.index(n)
            grid_rows.append(dict(learning_rate=lr, max_depth=d, n_estimators=n, cv_auc=val[:, i].mean(), cv_auc_std=val[:, i].std(),
                                  train_auc=trn[:, i].mean(), fold_aucs=np.round(val[:, i], 5).tolist(),
                                  fit_seconds_500=float(np.mean([j["seconds"] for j in js]))))
    gbm_curve = pd.DataFrame(curve_rows); gbm_curve.to_csv(os.path.join(RESULTS, "gbm_cv_curve.csv"), index=False)
    gbm_grid = pd.DataFrame(grid_rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    gbm_grid["rank"] = np.arange(1, len(gbm_grid) + 1)
    gbm_grid.drop(columns=["fold_aucs"]).to_csv(os.path.join(RESULTS, "gbm_grid.csv"), index=False)
    gb_best = gbm_grid.iloc[0]
    best_lr, best_depth, best_n = float(gb_best["learning_rate"]), int(gb_best["max_depth"]), int(gb_best["n_estimators"])
    staged_opt = gbm_curve.loc[gbm_curve["cv_auc"].idxmax()]
    print(f"[GBM] grid done in {gbm_grid_seconds:.0f}s:\n{gbm_grid.drop(columns=['fold_aucs']).round(4).to_string(index=False)}")
    print(f"[GBM] best: learning_rate={best_lr}, max_depth={best_depth}, n_estimators={best_n}, subsample CV AUC {gb_best['cv_auc']:.4f}")
    print(f"[GBM] diagnostic: the staged CV curve peaks at lr={staged_opt['learning_rate']}, depth={int(staged_opt['max_depth'])}, "
          f"{int(staged_opt['n_estimators'])} trees (CV AUC {staged_opt['cv_auc']:.4f})")

    t0 = time.time()
    gbm = make_gbm_pipe(n_estimators=best_n, learning_rate=best_lr, max_depth=best_depth).fit(X_train, y_train)
    gbm_fit_seconds = time.time() - t0
    t0 = time.time()
    p_oof_gbm = cross_val_predict(clone(gbm), X_train, y_train, cv=CV, method="predict_proba", n_jobs=2)[:, 1]
    aucs_gbm = fold_aucs(y_train, p_oof_gbm)
    print(f"[GBM] final {best_n}-tree model fitted on the full training set in {gbm_fit_seconds:.0f}s; "
          f"5-fold CV AUC {aucs_gbm.mean():.4f} ± {aucs_gbm.std():.4f} ({time.time() - t0:.0f}s)")

    # ================================================================== 2. XGBoost: full GridSearchCV on the full training set
    n_settings = int(np.prod([len(v) for v in XGB_GRID.values()]))
    print(f"\n[XGB] GridSearchCV: {n_settings} settings x 5 folds = {5 * n_settings} fits (n_jobs=1 outer, 2 threads inside xgboost)")
    t0 = time.time()
    gs = GridSearchCV(make_xgb_pipe(), XGB_GRID, cv=CV, scoring=SCORING, n_jobs=1, refit=True, return_train_score=True)
    gs.fit(X_train, y_train)
    xgb_grid_seconds = time.time() - t0
    cvres = pd.DataFrame(gs.cv_results_)
    xgb_grid = cvres[[f"param_{k}" for k in XGB_GRID] + ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score",
                                                        "mean_fit_time"]].copy()
    xgb_grid.columns = [k.replace("clf__", "") for k in XGB_GRID] + ["cv_auc", "cv_auc_std", "train_auc", "rank", "fit_seconds"]
    xgb_grid = xgb_grid.sort_values("rank").reset_index(drop=True)
    xgb_grid.to_csv(os.path.join(RESULTS, "xgb_grid.csv"), index=False)
    xb = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    xgb = gs.best_estimator_                                   # refitted on the full training set by GridSearchCV
    print(f"[XGB] grid done in {xgb_grid_seconds:.0f}s (refit {gs.refit_time_:.1f}s):\n{xgb_grid.round(4).to_string(index=False)}")
    print(f"[XGB] best: {xb}, CV AUC {gs.best_score_:.4f}")
    t0 = time.time()
    p_oof_xgb = cross_val_predict(clone(xgb), X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    aucs_xgb = fold_aucs(y_train, p_oof_xgb)
    print(f"[XGB] 5-fold CV AUC of the final pipeline {aucs_xgb.mean():.4f} ± {aucs_xgb.std():.4f} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ 3. class-weighted variant (same hyper-parameters)
    spw = n_neg / n_pos
    t0 = time.time()
    xgbw = clone(xgb).set_params(clf__scale_pos_weight=spw).fit(X_train, y_train)
    xgbw_fit_seconds = time.time() - t0
    p_oof_xgbw = cross_val_predict(clone(xgbw), X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    aucs_xgbw = fold_aucs(y_train, p_oof_xgbw)
    print(f"[XGB-weighted] scale_pos_weight={spw:.4f}: 5-fold CV AUC {aucs_xgbw.mean():.4f} ± {aucs_xgbw.std():.4f}; "
          f"mean OOF probability {p_oof_xgbw.mean():.4f} vs {p_oof_xgb.mean():.4f} unweighted (base rate {y_train.mean():.4f})")

    # ------------------------------------------------------------------ 4. XGBoost gain importance
    pre, clf = xgb.named_steps["pre"], xgb.named_steps["clf"]
    tnames = feature_names_out(pre); booster = clf.get_booster()
    tg = booster_scores(booster, len(tnames), "total_gain"); wt = booster_scores(booster, len(tnames), "weight")
    imp = pd.DataFrame({"feature": feature_names,
                        "total_gain": aggregate_to_raw(tg, tnames, feature_names), "n_splits": aggregate_to_raw(wt, tnames, feature_names)})
    imp["gain_share"] = imp["total_gain"] / imp["total_gain"].sum()
    imp["split_share"] = imp["n_splits"] / imp["n_splits"].sum()
    imp["gain_per_split"] = imp["total_gain"] / imp["n_splits"].replace(0, np.nan)
    imp["gain_rank"] = imp["gain_share"].rank(ascending=False).astype(int)
    imp = imp.sort_values("gain_share", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(RESULTS, "xgb_importance.csv"), index=False)
    print(f"\n[XGB] importance (gain share, split share), top 10:\n{imp.head(10).round(4).to_string(index=False)}")
    rf_rho = None
    rf_imp_path = os.path.join(RESULTS, "rf_importance.csv")
    if os.path.exists(rf_imp_path):                             # optional cross-check with the random-forest MDI ranking
        from scipy.stats import spearmanr
        rf_imp = pd.read_csv(rf_imp_path).set_index("feature")
        if "mdi" in rf_imp.columns:
            rf_rho = float(spearmanr(imp.set_index("feature").loc[rf_imp.index, "gain_share"], rf_imp["mdi"])[0])
            print(f"[XGB] Spearman(XGB gain share, RF MDI) over the 29 features = {rf_rho:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for a, col, title in [(ax[0], "gain_share", "Gain importance: share of total loss reduction"),
                          (ax[1], "split_share", "Share of the splits that use the feature")]:
        top = imp.sort_values(col, ascending=False).head(15).iloc[::-1]
        a.barh(top["feature"], top[col], color=C_XGB, height=0.62)
        for y_, v in enumerate(top[col]):
            a.text(v + top[col].max() * 0.015, y_, f"{v:.3f}", va="center", fontsize=8, color=C_TXT2)
        a.set_xlim(0, top[col].max() * 1.22); a.set_title(title, fontsize=10.5, loc="left")
        a.grid(axis="x", color=C_GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(axis="y", length=0)
    fig.suptitle(f"XGBoost feature importance, top 15 of {len(feature_names)} (max_depth={xb['max_depth']}, lr={xb['learning_rate']}, "
                 f"{xb['n_estimators']} rounds, min_child_weight={xb['min_child_weight']}); one-hot dummies summed per feature",
                 fontsize=11, x=0.01, ha="left")
    save_extra_figure(fig, "xgb_importance.png")

    # ------------------------------------------------------------------ 5a. XGBoost learning curve (CV AUC vs rounds, best depth/mcw)
    t0 = time.time()
    curve_rows = []
    for lr in XGB_GRID["clf__learning_rate"]:
        val, trn = [], []
        for f, (tr, va) in enumerate(CV.split(X_train, y_train)):
            pre_f = make_preprocessor(scale=False).fit(X_train.iloc[tr])          # fitted on the training fold only
            Xtr, Xva = pre_f.transform(X_train.iloc[tr]), pre_f.transform(X_train.iloc[va])
            c = XGBClassifier(**XGB_FIXED, max_depth=xb["max_depth"], min_child_weight=xb["min_child_weight"], learning_rate=lr,
                              n_estimators=CURVE_MAX_ROUNDS)
            c.fit(Xtr, y_train.iloc[tr], eval_set=[(Xtr, y_train.iloc[tr]), (Xva, y_train.iloc[va])], verbose=False)
            r = c.evals_result(); trn.append(r["validation_0"]["auc"]); val.append(r["validation_1"]["auc"])
        val, trn = np.array(val), np.array(trn)
        for k in range(CURVE_MAX_ROUNDS):
            curve_rows.append(dict(learning_rate=lr, n_rounds=k + 1, cv_auc=val[:, k].mean(), cv_auc_std=val[:, k].std(), train_auc=trn[:, k].mean()))
    xgb_curve = pd.DataFrame(curve_rows); xgb_curve.to_csv(os.path.join(RESULTS, "xgb_learning_curve.csv"), index=False)
    curve_summary = {}
    ks_report = [k for k in (50, 100, 200, 300, 600, 1000, 1500) if k <= CURVE_MAX_ROUNDS] + [CURVE_MAX_ROUNDS]
    for lr in XGB_GRID["clf__learning_rate"]:
        c = xgb_curve[xgb_curve["learning_rate"] == lr].set_index("n_rounds")
        best_k = int(c["cv_auc"].idxmax())
        curve_summary[str(lr)] = dict(best_rounds=best_k, cv_auc_at_best=round(float(c["cv_auc"].max()), 5),
                                      cv_auc_at={str(k): round(float(c.loc[k, "cv_auc"]), 5) for k in ks_report},
                                      train_auc_at={str(k): round(float(c.loc[k, "train_auc"]), 5) for k in ks_report})
    print(f"\n[XGB] learning curves in {time.time() - t0:.0f}s: {json.dumps(curve_summary)}")

    fig, a = plt.subplots(figsize=(8.6, 5.8))
    peak_lines = []
    for lr, color in zip(XGB_GRID["clf__learning_rate"], (C_GBM, C_XGB)):
        c = xgb_curve[xgb_curve["learning_rate"] == lr]
        a.plot(c["n_rounds"], c["cv_auc"], color=color, lw=2, label=f"5-fold CV AUC, learning rate {lr}")
        a.fill_between(c["n_rounds"], c["cv_auc"] - c["cv_auc_std"], c["cv_auc"] + c["cv_auc_std"], color=color, alpha=0.12, lw=0)
        a.plot(c["n_rounds"], c["train_auc"], color=color, lw=1.4, ls="--", label=f"training-fold AUC, learning rate {lr}")
        for k in [k for k in XGB_GRID["clf__n_estimators"] if k <= CURVE_MAX_ROUNDS]:
            a.plot(k, c.loc[c["n_rounds"] == k, "cv_auc"].iloc[0], "o", color=color, ms=7, mec="white", mew=1)
        s = curve_summary[str(lr)]
        a.plot(s["best_rounds"], s["cv_auc_at_best"], marker="v", color=color, ms=8, mec="white", mew=1, ls="none")
        peak_lines.append(f"learning rate {lr}: CV AUC peaks at {s['cv_auc_at_best']:.4f} after {s['best_rounds']} rounds")
    a.axhline(LOGIT_CV_AUC, color=C_TXT2, ls=":", lw=1.2, label=f"Logistic regression CV AUC ({LOGIT_CV_AUC:.3f})")
    cv_max, train_max = float(xgb_curve["cv_auc"].max()), float(xgb_curve["train_auc"].max())
    ymin = max(0.70, float(xgb_curve["cv_auc"].min()) - 0.01); ymax = min(train_max + 0.005, cv_max + 0.07)
    a.set_ylim(ymin, ymax); a.set_xlim(0, CURVE_MAX_ROUNDS)
    if train_max + 0.005 > ymax:
        peak_lines.append(f"training-fold curves run off the top (max {train_max:.3f})")
    a.text(0.98, 0.04, "\n".join(peak_lines) + f"\ndots = grid values {' / '.join(map(str, XGB_GRID['clf__n_estimators']))}, "
                                                "triangles = CV peak, band = ±1 sd across folds",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, color=C_TXT2, linespacing=1.5)
    a.set_xlabel("Boosting rounds (trees)"); a.set_ylabel("ROC AUC")
    a.set_title(f"XGBoost learning curve: max_depth={xb['max_depth']}, min_child_weight={xb['min_child_weight']}, "
                f"subsample=0.8, colsample_bytree=0.8", loc="left", fontsize=10.5)
    a.grid(color=C_GRID, lw=0.8); a.set_axisbelow(True)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8.5)
    save_extra_figure(fig, "xgb_learning_curve.png")

    # ------------------------------------------------------------------ 5b. GBM staged CV curves (subsample grid)
    fig, a = plt.subplots(figsize=(8.6, 5.8))
    styles = {2: ":", 3: "--", 4: "-"}
    for lr, color in zip(GBM_LR, (C_GBM, C_XGB)):
        for d in GBM_DEPTH:
            c = gbm_curve[(gbm_curve["learning_rate"] == lr) & (gbm_curve["max_depth"] == d)]
            a.plot(c["n_estimators"], c["cv_auc"], color=color, lw=2.2 if (d == best_depth and lr == best_lr) else 1.4, ls=styles[d],
                   label=f"lr {lr}, depth {d}")
    for n in GBM_NTREES:
        a.axvline(n, color=C_GRID, lw=1.2)
    a.plot(best_n, gb_best["cv_auc"], "o", color=C_GBM if best_lr == GBM_LR[0] else C_XGB, ms=8, mec="white", mew=1)
    a.axhline(LOGIT_CV_AUC, color=C_TXT2, ls=":", lw=1.2, label=f"Logistic regression CV AUC ({LOGIT_CV_AUC:.3f})")
    a.text(0.98, 0.04, f"grid winner (dot): lr {best_lr}, depth {best_depth}, {best_n} trees, CV AUC {gb_best['cv_auc']:.4f}\n"
                       f"staged-curve peak: lr {staged_opt['learning_rate']}, depth {int(staged_opt['max_depth'])}, "
                       f"{int(staged_opt['n_estimators'])} trees, CV AUC {staged_opt['cv_auc']:.4f}\n"
                       f"vertical lines = grid values {' / '.join(map(str, GBM_NTREES))}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, color=C_TXT2, linespacing=1.5)
    a.set_ylim(max(0.70, float(gbm_curve["cv_auc"].min()) - 0.005), float(gbm_curve["cv_auc"].max()) + 0.006); a.set_xlim(0, GBM_MAX_TREES + 10)
    a.set_xlabel("Number of trees"); a.set_ylabel(f"5-fold CV ROC AUC ({GBM_SUBSAMPLE_FRAC:.0%} training subsample)")
    a.set_title("sklearn gradient boosting: CV AUC after k trees (staged predictions of the 500-tree fits)", loc="left", fontsize=10.5)
    a.grid(color=C_GRID, lw=0.8); a.set_axisbelow(True)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=8.5)
    save_extra_figure(fig, "gbm_learning_curve.png")

    # ================================================================== 6. evaluate() once per final model
    gbm_notes = (f"Grid n_estimators {GBM_NTREES} x learning_rate {GBM_LR} x max_depth {GBM_DEPTH} (subsample=0.8, random_state=SEED), "
                 f"5-fold CV (common.CV), criterion roc_auc. BUDGET: the grid was scored on a stratified {GBM_SUBSAMPLE_FRAC:.0%} subsample "
                 f"of the training set ({len(Xs):,} rows, seed {SEED}) and, per (lr, depth), one 500-tree model per fold was fitted with "
                 f"the AUC after 200 and 500 trees read from staged_predict_proba (identical to fitting both sizes; boosting is sequential). "
                 f"Winner refitted on the full training set; cv_auc = 5-fold CV AUC of that final pipeline on the shared folds. "
                 f"sklearn GBM is single-threaded: 2 parallel workers (joblib).")
    res_gbm = evaluate("Gradient boosting (sklearn)", gbm, X_train, y_train, X_test, y_test, member=MEMBER, slug="gradient_boosting",
                       best_params=dict(n_estimators=best_n, learning_rate=best_lr, max_depth=best_depth, subsample=0.8, loss="log_loss",
                                        criterion="friedman_mse", min_samples_leaf=1, random_state=SEED),
                       cv_auc=float(aucs_gbm.mean()), cv_auc_std=float(aucs_gbm.std()), fit_seconds=round(gbm_fit_seconds, 1), p_oof=p_oof_gbm,
                       notes=gbm_notes,
                       extra=dict(cv_fold_aucs=np.round(aucs_gbm, 5).tolist(), grid_subsample_rows=int(len(Xs)), grid_subsample_frac=GBM_SUBSAMPLE_FRAC,
                                  grid_cv_auc_subsample=round(float(gb_best["cv_auc"]), 5), grid_cv_auc_std_subsample=round(float(gb_best["cv_auc_std"]), 5),
                                  grid_seconds=round(gbm_grid_seconds, 1), grid=gbm_grid.round(5).to_dict(orient="records"),
                                  staged_optimum=dict(learning_rate=float(staged_opt["learning_rate"]), max_depth=int(staged_opt["max_depth"]),
                                                      n_estimators=int(staged_opt["n_estimators"]), cv_auc=round(float(staged_opt["cv_auc"]), 5)),
                                  cv_curve_best_setting={str(int(r.n_estimators)): round(float(r.cv_auc), 5) for r in
                                                         gbm_curve[(gbm_curve["learning_rate"] == best_lr) & (gbm_curve["max_depth"] == best_depth)].itertuples()
                                                         if int(r.n_estimators) % 50 == 0},
                                  n_features_raw=len(feature_names), n_features_after_onehot=len(tnames), versions=versions))

    xgb_notes = (f"Full GridSearchCV over max_depth {XGB_GRID['clf__max_depth']} x learning_rate {XGB_GRID['clf__learning_rate']} x "
                 f"n_estimators {XGB_GRID['clf__n_estimators']} x min_child_weight {XGB_GRID['clf__min_child_weight']} ({n_settings} settings x 5 folds, "
                 f"cv=CV, scoring=roc_auc, n_jobs=1; xgboost n_jobs=2, tree_method=hist, subsample=0.8, colsample_bytree=0.8, random_state=SEED), "
                 f"refit on the full training set; no subsampling needed ({xgb_grid_seconds:.0f}s). cv_auc = 5-fold CV AUC of the final pipeline "
                 f"on the shared folds (GridSearchCV best_score_ in extra). Preprocessor: make_preprocessor(scale=False) inside the Pipeline "
                 f"(one-hot for SEX/EDUCATION/MARRIAGE, no scaling — trees are invariant to monotone rescaling).")
    xgb_best_params = dict(**xb, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic", tree_method="hist", scale_pos_weight=1.0,
                           random_state=SEED)
    res_xgb = evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test, member=MEMBER, slug="xgboost", best_params=xgb_best_params,
                       cv_auc=float(aucs_xgb.mean()), cv_auc_std=float(aucs_xgb.std()), fit_seconds=round(float(gs.refit_time_), 1), p_oof=p_oof_xgb,
                       notes=xgb_notes,
                       extra=dict(cv_fold_aucs=np.round(aucs_xgb, 5).tolist(), grid_best_score=round(float(gs.best_score_), 5),
                                  grid_seconds=round(xgb_grid_seconds, 1), grid=xgb_grid.round(5).to_dict(orient="records"),
                                  learning_curve=curve_summary, learning_curve_max_rounds=CURVE_MAX_ROUNDS,
                                  importance_top10_gain=imp.head(10)[["feature", "gain_share", "split_share", "gain_per_split"]].round(5).to_dict(orient="records"),
                                  spearman_gain_vs_rf_mdi=None if rf_rho is None else round(rf_rho, 4),
                                  n_features_raw=len(feature_names), n_features_after_onehot=len(tnames), versions=versions))

    res_xgbw = evaluate("XGBoost (class-weighted)", xgbw, X_train, y_train, X_test, y_test, member=MEMBER, slug="xgboost_weighted",
                        best_params=dict(xgb_best_params, scale_pos_weight=round(spw, 4)),
                        cv_auc=float(aucs_xgbw.mean()), cv_auc_std=float(aucs_xgbw.std()), fit_seconds=round(xgbw_fit_seconds, 1), p_oof=p_oof_xgbw,
                        notes=(f"Not tuned separately: the best unweighted XGBoost hyper-parameters with scale_pos_weight = n_neg/n_pos = "
                               f"{n_neg}/{n_pos} = {spw:.4f} (each defaulter counts {spw:.2f}x in the gradient / hessian), refit on the full training set. "
                               f"Isolates the effect of cost-sensitive learning on ranking (AUC/AP) vs. on the default-threshold operating point."),
                        extra=dict(cv_fold_aucs=np.round(aucs_xgbw, 5).tolist(), scale_pos_weight=round(spw, 5),
                                   mean_oof_probability=round(float(p_oof_xgbw.mean()), 5), mean_oof_probability_unweighted=round(float(p_oof_xgb.mean()), 5),
                                   train_default_rate=round(float(y_train.mean()), 5), versions=versions))

    for r in (res_gbm, res_xgb, res_xgbw):
        print_result(r)
    print(f"\nTotal wall time {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
