"""
model_rf.py — Random forest (bagging + feature subsampling, lecture B7).  Owner: LU Xinyu.

Run from Project/code:  python3 model_rf.py            (13-20 min with n_jobs=2: the run that produced the current results took
                                                        1220 s with the laptop shared with other jobs (its 200-tree grid alone
                                                        took 481 s against 226 s on an idle machine); the out-of-bag permutation
                                                        added 99 s of that. The wall time is printed at the end and stored in
                                                        results/random_forest.json under extra.timings)

What it does
  1. Pipeline(make_preprocessor(scale=False), RandomForestClassifier(500 trees, oob_score=True)).
  2. GridSearchCV over max_features {'sqrt', 0.3, 0.5} x min_samples_leaf {1, 5, 20}, cv=CV, scoring=SCORING.
     To stay inside the 15-minute budget the grid is scored with 200-tree forests; the winning setting is refitted
     with 500 trees on the full training set.  Two leakage-free checks that 200 trees were enough for the ranking:
       (a) the top-2 grid settings are re-scored with 500-tree forests on the same 5 folds (the winner's 500-tree
           CV AUC is the cv_auc reported in the JSON; the runner-up is re-scored here, ~1 min);
       (b) OOB AUC of the final forests as a function of the number of trees k (first k trees, training rows only).
  3. Final model  -> evaluate(slug="random_forest").
     Pure bagging variant (max_features=None, same leaf size, 500 trees) -> evaluate(slug="bagging"),
     to show the effect of de-correlating the trees by feature subsampling.
  4. OOB AUC / OOB error of each forest vs. its 5-fold CV AUC (same CV folds as everyone else).
  5. Feature importance, three measures side by side -> results/rf_importance.csv, figures/rf_importance.png:
       (a) MDI (impurity decrease, summed over the one-hot columns of a categorical);
       (b) permutation importance on the TRAINING set (in-sample AUC drop, n_repeats=5) — kept only so the report can
           show how optimistic an in-sample measurement is (the forest's training AUC is 0.888 against 0.787 on test);
       (c) Breiman's OUT-OF-BAG permutation importance (lecture B7: "we randomly permute the feature j in the OOB data,
           and if the OOB error increases, that means the variable is important"): feature j is permuted inside each
           tree's own out-of-bag rows and the drop in the forest's OOB AUC is measured, n_repeats=5.
     (c) is the column the report quotes; the Spearman correlations between the three rankings are printed and stored
     in results/random_forest.json under extra.spearman.
  6. AUC as a function of the number of trees (first k trees of the fitted 500-tree forests, k = 10..500):
     OOB AUC on the training rows (leakage-free, used to justify the 200-tree grid) and test AUC (purely descriptive,
     nothing is chosen from it) -> results/rf_auc_vs_ntrees.csv, figures/rf_auc_vs_ntrees.png.
Everything is fitted / tuned on the training rows only; the test set is touched only inside evaluate() and for the
descriptive test-AUC-vs-trees curve.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import sklearn
from common import *

MEMBER = "LU Xinyu"
N_TREES = 500            # final forests
N_TREES_GRID = 200       # forests inside the tuning grid (budget), see notes
PARAM_GRID = {"clf__max_features": ["sqrt", 0.3, 0.5], "clf__min_samples_leaf": [1, 5, 20]}
KS = list(range(10, N_TREES + 1, 10))
N_REPEATS = 5            # permutation repeats, the same number for the training-set and the OOB measure

# validated categorical palette (dataviz skill, light surface): slot 1 blue, slot 2 orange, slot 3 aqua; text greys
C_RF, C_BAG, C_OOB, C_TXT, C_TXT2, C_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "xtick.color": C_TXT2, "ytick.color": C_TXT2,
                     "text.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def make_pipe(n_estimators, oob_score, **clf_kw):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=2, oob_score=oob_score, **clf_kw)
    return Pipeline([("pre", make_preprocessor(scale=False)), ("clf", clf)])


def fold_aucs(y, p_oof):
    """Per-fold AUC of out-of-fold probabilities, using the shared CV folds (deterministic)."""
    return np.array([roc_auc_score(y.iloc[va], p_oof[va]) for _, va in CV.split(np.zeros(len(y)), y)])


def oob_stats(pipe, y):
    clf = pipe.named_steps["clf"]
    p_oob = clf.oob_decision_function_[:, 1]
    ok = ~np.isnan(p_oob)                                   # with 500 trees every row is OOB for some tree
    return dict(oob_accuracy=float(clf.oob_score_), oob_error=float(1.0 - clf.oob_score_),
                oob_auc=float(roc_auc_score(y[ok], p_oob[ok])), n_rows_without_oob=int((~ok).sum()))


def aggregate_to_raw(values, transformed_names, raw_names):
    """Sum a per-transformed-column vector back to the 29 raw features (one-hot dummies of a categorical are summed)."""
    out = dict.fromkeys(raw_names, 0.0)
    for v, n in zip(values, transformed_names):
        base = n.split("__", 1)[1]
        if n.startswith("cat__"):
            base = base.rsplit("_", 1)[0]
        out[base] += float(v)
    return np.array([out[f] for f in raw_names])


def per_tree_proba(pipe, X):
    """Matrix (n_trees, n_rows) of each tree's P(default) for the rows of X."""
    Xt = pipe.named_steps["pre"].transform(X)
    return np.stack([t.predict_proba(Xt)[:, 1] for t in pipe.named_steps["clf"].estimators_]).astype(np.float32)


def auc_vs_ntrees(pipe, X_te, y_te, ks):
    """Test AUC of the forest truncated to its first k trees (cumulative mean of per-tree probabilities). Descriptive only."""
    csum = np.cumsum(per_tree_proba(pipe, X_te), axis=0, dtype=np.float64)
    return [float(roc_auc_score(y_te, csum[k - 1] / k)) for k in ks]


def oob_mask(pipe, n_rows):
    """
    Boolean matrix (n_trees, n_rows): True where the row is OUT of that tree's bootstrap sample.
    Single source of truth for "which rows are out-of-bag for which tree" — used by both
    oob_auc_vs_ntrees() and oob_permutation_importance(), so the two cannot drift apart.
    """
    clf = pipe.named_steps["clf"]
    inbag = np.zeros((len(clf.estimators_), n_rows), dtype=bool)
    for b, idx in enumerate(clf.estimators_samples_):
        inbag[b, idx] = True
    return ~inbag


def oob_auc_vs_ntrees(pipe, X_tr, y_tr, ks):
    """
    OOB AUC of the forest truncated to its first k trees, on the TRAINING rows only (leakage-free).
    Row i's OOB probability from the first k trees = mean of p_t(i) over the trees t <= k whose bootstrap sample
    did not contain i (estimators_samples_). Rows that are in-bag for all first k trees are skipped for that k.
    With k = 500 this reproduces sklearn's oob_decision_function_ exactly.
    """
    per_tree = per_tree_proba(pipe, X_tr)
    oob = oob_mask(pipe, per_tree.shape[1])
    csum = np.cumsum(per_tree * oob, axis=0, dtype=np.float64); cnt = np.cumsum(oob, axis=0)
    out, n_skipped = [], []
    y = np.asarray(y_tr)
    for k in ks:
        ok = cnt[k - 1] > 0
        out.append(float(roc_auc_score(y[ok], csum[k - 1][ok] / cnt[k - 1][ok]))); n_skipped.append(int((~ok).sum()))
    return out, n_skipped


def oob_permutation_importance(pipe, X_tr, y_tr, raw_names, n_repeats=N_REPEATS, seed=SEED):
    """
    Breiman's OUT-OF-BAG permutation importance (lecture B7: "we randomly permute the feature j in the
    OOB data, and if the OOB error increases, that means the variable is important").

    For every tree the rows NOT in its bootstrap sample are its out-of-bag rows — the same mask
    oob_mask() gives oob_auc_vs_ntrees(). Feature j is permuted INSIDE each tree's own OOB rows, that
    tree re-predicts those rows, and the permuted probabilities are averaged per row over the trees that
    have the row out of bag (exactly the aggregation behind sklearn's oob_decision_function_). The
    importance of j is the resulting drop in the forest's OOB AUC, averaged over n_repeats permutations.
    Nothing is measured where the tree was fitted, so the numbers are out-of-sample by construction,
    unlike permutation_importance() on the training set.

    The one-hot dummies of a categorical are permuted TOGETHER with one row order, so a permuted row is
    still a valid category (same grouping as aggregate_to_raw). The permutation of tree t, repeat r is
    drawn from default_rng([seed, t, r]), so the result is reproducible and order-independent.
    For speed the n_features permuted copies of a tree's OOB block are predicted in one call.
    Returns (mean drop, std over repeats, baseline OOB AUC, matrix (n_repeats, n_features) of permuted AUCs).
    """
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = np.asarray(pre.transform(X_tr), dtype=np.float32)
    tnames = feature_names_out(pre)
    groups = {f: [] for f in raw_names}
    for c, n in enumerate(tnames):
        base = n.split("__", 1)[1]
        if n.startswith("cat__"):
            base = base.rsplit("_", 1)[0]
        groups[base].append(c)
    cols = [np.asarray(groups[f], dtype=int) for f in raw_names]
    y = np.asarray(y_tr); n_f = len(raw_names)
    oob = oob_mask(pipe, len(y))
    cnt = oob.sum(axis=0); ok = cnt > 0                       # with 500 trees every row is OOB for some tree
    base_sum = np.zeros(len(y))
    perm_sum = [np.zeros((n_f, len(y))) for _ in range(n_repeats)]
    for t_i, tree in enumerate(clf.estimators_):
        rows = np.flatnonzero(oob[t_i]); m = len(rows)
        if m == 0:
            continue
        Xo = Xt[rows]
        base_sum[rows] += tree.predict_proba(Xo)[:, 1]
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, t_i, r])
            block = np.tile(Xo, (n_f, 1))                     # one stacked copy per feature, only that feature shuffled
            for j, cj in enumerate(cols):
                block[j * m:(j + 1) * m, cj] = Xo[np.ix_(rng.permutation(m), cj)]
            perm_sum[r][:, rows] += tree.predict_proba(block)[:, 1].reshape(n_f, m)
    auc_base = float(roc_auc_score(y[ok], base_sum[ok] / cnt[ok]))
    aucs = np.array([[roc_auc_score(y[ok], perm_sum[r][j][ok] / cnt[ok]) for j in range(n_f)] for r in range(n_repeats)])
    drops = auc_base - aucs
    return drops.mean(axis=0), drops.std(axis=0), auc_base, aucs


def plot_importance(imp, best_mf, best_msl, n_top=15):
    """Top-n features by MDI, training-set permutation and OOB permutation, side by side -> figures/rf_importance.png."""
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.8))
    panels = [(ax[0], "mdi", None, C_RF, "MDI\nmean decrease in impurity (Gini), share of total"),
              (ax[1], "perm_auc_drop", "perm_auc_drop_std", C_BAG,
               f"Permutation on the TRAINING set\nin-sample AUC drop ({N_REPEATS} repeats) — optimistic"),
              (ax[2], "oob_perm_auc_drop", "oob_perm_auc_drop_std", C_OOB,
               f"Out-of-bag permutation (Breiman)\nOOB AUC drop ({N_REPEATS} repeats) — the one we quote")]
    for a, col, err, color, title in panels:
        top = imp.sort_values(col, ascending=False).head(n_top).iloc[::-1]
        e = top[err].values if err else np.zeros(len(top))
        a.barh(top["feature"], top[col], xerr=e if err else None, color=color, height=0.62,
               error_kw=dict(ecolor=C_TXT2, lw=1, capsize=2))
        xmax = float((top[col] + e).max())
        dec = min(4, max(3, 1 - int(np.floor(np.log10(max(float(top[col].min()), 1e-9))))))   # digits that keep the smallest bar readable
        for y_, (v, ev) in enumerate(zip(top[col], e)):
            a.text(v + ev + xmax * 0.02, y_, f"{v:.{dec}f}", va="center", fontsize=8, color=C_TXT2)
        a.set_xlim(0, xmax * 1.25); a.set_title(title, fontsize=10, loc="left")
        a.grid(axis="x", color=C_GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(axis="y", length=0)
    fig.suptitle(f"Random forest feature importance, top {n_top} of {len(imp)} "
                 f"(max_features={best_mf}, min_samples_leaf={best_msl}, {N_TREES} trees)", fontsize=12, x=0.01, ha="left")
    return save_extra_figure(fig, "rf_importance.png")


def plot_auc_vs_ntrees(curve, best_mf):
    """
    Two panels -> figures/rf_auc_vs_ntrees.png.
    Left: OOB AUC on the training rows vs k (leakage-free evidence that the 200-tree grid forests rank settings like 500-tree ones).
    Right: test AUC vs k (descriptive; nothing was chosen from it), with the logistic baseline.
    """
    mark = curve["n_trees"] % 50 == 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    panels = [(axes[0], "oob_auc", "OOB AUC (training rows, first k trees)", "OOB AUC vs number of trees — training rows only (leakage-free)"),
              (axes[1], "test_auc", "Test AUC (first k trees)", "Test AUC vs number of trees — descriptive only")]
    for a, stem, ylab, title in panels:
        for col, color, label in [(f"{stem}_rf", C_RF, f"Random forest (max_features={best_mf})"),
                                  (f"{stem}_bagging", C_BAG, "Bagging (max_features=None)")]:
            a.plot(curve["n_trees"], curve[col], color=color, lw=2, label=label)
            a.plot(curve.loc[mark, "n_trees"], curve.loc[mark, col], "o", color=color, ms=6, mec="white", mew=1)
            a.text(N_TREES + 8, curve[col].iloc[-1], f"{curve[col].iloc[-1]:.4f}", color=C_TXT2, va="center", fontsize=9)
        a.axvline(N_TREES_GRID, color=C_TXT2, ls=":", lw=1.2)
        a.text(N_TREES_GRID - 5, a.get_ylim()[0] + 0.55 * np.diff(a.get_ylim())[0], f"{N_TREES_GRID} trees (grid forests)",
               color=C_TXT2, fontsize=8.5, va="center", ha="right", rotation=90)
        a.set_xlabel("Number of trees"); a.set_ylabel(ylab); a.set_xlim(0, N_TREES + 45)
        a.set_title(title, loc="left", fontsize=10.5)
        a.grid(color=C_GRID, lw=0.8); a.set_axisbelow(True)
    axes[1].axhline(0.764, color=C_TXT2, ls="--", lw=1.2, label="Logistic regression baseline (0.764)")
    axes[1].text(12, 0.764 + 0.0004, "logistic regression 0.764", color=C_TXT2, fontsize=8.5, va="bottom")
    axes[0].legend(loc="lower right", frameon=False); axes[1].legend(loc="center right", frameon=False)
    fig.suptitle(f"AUC vs number of trees (first k of the fitted {N_TREES}-tree forests)", fontsize=12, x=0.01, ha="left")
    return save_extra_figure(fig, "rf_auc_vs_ntrees.png")


def plots_only():
    """Redraw the two extra figures from the saved tables (python3 model_rf.py --plots-only)."""
    bp = json.load(open(os.path.join(RESULTS, "random_forest.json")))["best_params"]
    plot_importance(pd.read_csv(os.path.join(RESULTS, "rf_importance.csv")), bp["max_features"], bp["min_samples_leaf"])
    plot_auc_vs_ntrees(pd.read_csv(os.path.join(RESULTS, "rf_auc_vs_ntrees.csv")), bp["max_features"])
    print("figures redrawn from results/rf_importance.csv and results/rf_auc_vs_ntrees.csv")


def main():
    t_all = time.time(); timings = {}
    X_train, X_test, y_train, y_test, feature_names = load_split()
    print(f"train {X_train.shape}, test {X_test.shape}, default rate {y_train.mean():.4f}")

    # ------------------------------------------------------------------ 1. tune with GridSearchCV (200-tree forests)
    t0 = time.time()
    gs = GridSearchCV(make_pipe(N_TREES_GRID, oob_score=False), PARAM_GRID, cv=CV, scoring=SCORING,
                      n_jobs=1, refit=False, return_train_score=True)
    gs.fit(X_train, y_train)
    timings["grid_200trees"] = round(time.time() - t0, 1)
    cvres = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    grid = cvres[["param_clf__max_features", "param_clf__min_samples_leaf", "mean_test_score", "std_test_score",
                  "mean_train_score", "rank_test_score", "mean_fit_time"]].copy()
    grid.columns = ["max_features", "min_samples_leaf", "cv_auc", "cv_auc_std", "train_auc", "rank", "fit_seconds"]
    grid["max_features"] = grid["max_features"].astype(str)
    grid.to_csv(os.path.join(RESULTS, "rf_grid.csv"), index=False)
    best = gs.best_params_
    best_mf, best_msl = best["clf__max_features"], best["clf__min_samples_leaf"]
    print(f"\nGrid ({N_TREES_GRID} trees, {timings['grid_200trees']:.0f}s):\n{grid.round(4).to_string(index=False)}")
    print(f"best: max_features={best_mf}, min_samples_leaf={best_msl}, CV AUC {gs.best_score_:.4f}")

    # ------------------------------------------------------------------ 1b. runner-up re-scored with 500 trees on the same folds
    runner = cvres["params"].iloc[1]
    run_mf, run_msl = runner["clf__max_features"], runner["clf__min_samples_leaf"]
    t0 = time.time()
    aucs_runner = cross_val_score(make_pipe(N_TREES, oob_score=False, max_features=run_mf, min_samples_leaf=run_msl),
                                  X_train, y_train, cv=CV, scoring=SCORING, n_jobs=1)
    timings["runner_up_cv_500trees"] = round(time.time() - t0, 1)
    print(f"runner-up (max_features={run_mf}, min_samples_leaf={run_msl}) re-scored with {N_TREES} trees: "
          f"CV AUC {aucs_runner.mean():.4f} ± {aucs_runner.std():.4f} (200-tree grid: {cvres['mean_test_score'].iloc[1]:.4f}), "
          f"{timings['runner_up_cv_500trees']:.0f}s")

    # ------------------------------------------------------------------ 2. final random forest (500 trees) + bagging variant
    t0 = time.time()
    rf = make_pipe(N_TREES, oob_score=True, max_features=best_mf, min_samples_leaf=best_msl).fit(X_train, y_train)
    rf_fit_seconds = time.time() - t0; timings["rf_fit_500trees"] = round(rf_fit_seconds, 1)
    rf_oob = oob_stats(rf, y_train)
    print(f"\nRF {N_TREES} trees fitted in {rf_fit_seconds:.0f}s | OOB acc {rf_oob['oob_accuracy']:.4f} "
          f"(error {rf_oob['oob_error']:.4f}) | OOB AUC {rf_oob['oob_auc']:.4f}")

    t0 = time.time()
    bag = make_pipe(N_TREES, oob_score=True, max_features=None, min_samples_leaf=best_msl).fit(X_train, y_train)
    bag_fit_seconds = time.time() - t0; timings["bagging_fit_500trees"] = round(bag_fit_seconds, 1)
    bag_oob = oob_stats(bag, y_train)
    print(f"Bagging (max_features=None) fitted in {bag_fit_seconds:.0f}s | OOB acc {bag_oob['oob_accuracy']:.4f} "
          f"(error {bag_oob['oob_error']:.4f}) | OOB AUC {bag_oob['oob_auc']:.4f}")

    # ------------------------------------------------------------------ 3. feature importance: MDI, permutation on the training set, OOB permutation
    pre, clf = rf.named_steps["pre"], rf.named_steps["clf"]
    tnames = feature_names_out(pre)
    per_tree_mdi = np.stack([aggregate_to_raw(t.feature_importances_, tnames, feature_names) for t in clf.estimators_])
    mdi, mdi_std = per_tree_mdi.mean(axis=0), per_tree_mdi.std(axis=0)

    t0 = time.time()
    clf.n_jobs = 1                                            # permutation_importance uses 2 workers -> 2 cores in total
    perm = permutation_importance(rf, X_train, y_train, scoring="roc_auc", n_repeats=N_REPEATS, n_jobs=2, random_state=SEED)
    clf.n_jobs = 2
    perm_seconds = time.time() - t0; timings["permutation_importance_train"] = round(perm_seconds, 1)

    t0 = time.time()                                          # Breiman's OOB permutation: measured off the bootstrap, never where a tree was fitted
    oob_drop, oob_drop_std, oob_perm_base_auc, _ = oob_permutation_importance(rf, X_train, y_train, feature_names)
    oob_perm_seconds = time.time() - t0; timings["permutation_importance_oob"] = round(oob_perm_seconds, 1)
    assert abs(oob_perm_base_auc - rf_oob["oob_auc"]) < 1e-6, "unpermuted OOB AUC must equal sklearn's oob_decision_function_ AUC"

    imp = pd.DataFrame({"feature": feature_names, "mdi": mdi, "mdi_std": mdi_std,
                        "perm_auc_drop": perm.importances_mean, "perm_auc_drop_std": perm.importances_std,
                        "oob_perm_auc_drop": oob_drop, "oob_perm_auc_drop_std": oob_drop_std})
    imp["mdi_rank"] = imp["mdi"].rank(ascending=False).astype(int)
    imp["perm_rank"] = imp["perm_auc_drop"].rank(ascending=False).astype(int)
    imp["oob_perm_rank"] = imp["oob_perm_auc_drop"].rank(ascending=False).astype(int)
    imp = imp.sort_values("oob_perm_auc_drop", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(RESULTS, "rf_importance.csv"), index=False)
    spear = dict(mdi_vs_train_permutation=round(float(spearmanr(imp["mdi"], imp["perm_auc_drop"])[0]), 4),
                 mdi_vs_oob_permutation=round(float(spearmanr(imp["mdi"], imp["oob_perm_auc_drop"])[0]), 4),
                 train_vs_oob_permutation=round(float(spearmanr(imp["perm_auc_drop"], imp["oob_perm_auc_drop"])[0]), 4))
    rho = spear["mdi_vs_train_permutation"]
    # How much bigger the in-sample drop is. Per-feature ratios are only meaningful where the OOB drop is not
    # noise around zero, so the median is taken over the top 10 by OOB drop; the totals are over all 29 features.
    top10 = imp.head(10)
    gap = dict(median_ratio_top10=round(float((top10["perm_auc_drop"] / top10["oob_perm_auc_drop"]).median()), 2),
               total_ratio_all=round(float(imp["perm_auc_drop"].sum() / imp["oob_perm_auc_drop"].sum()), 2),
               sum_train_drop=round(float(imp["perm_auc_drop"].sum()), 4), sum_oob_drop=round(float(imp["oob_perm_auc_drop"].sum()), 4))
    print(f"\nPermutation importance: training set {perm_seconds:.0f}s, out-of-bag {oob_perm_seconds:.0f}s "
          f"({N_REPEATS} repeats each; unpermuted OOB AUC {oob_perm_base_auc:.4f})")
    print(f"Spearman between the three measures: MDI vs training permutation {spear['mdi_vs_train_permutation']:.3f} | "
          f"MDI vs OOB permutation {spear['mdi_vs_oob_permutation']:.3f} | "
          f"training vs OOB permutation {spear['train_vs_oob_permutation']:.3f}")
    print(f"Magnitude gap (training AUC drop / OOB AUC drop): {gap['median_ratio_top10']:.2f}x median over the top 10 by OOB drop, "
          f"{gap['total_ratio_all']:.2f}x on the totals ({gap['sum_train_drop']:.4f} vs {gap['sum_oob_drop']:.4f} over all "
          f"{len(imp)} features) — same ranking, inflated size in sample")
    print(imp.round(4).to_string(index=False))

    plot_importance(imp, best_mf, best_msl)

    # ------------------------------------------------------------------ 4. AUC vs number of trees: OOB (training, leakage-free) + test (descriptive)
    t0 = time.time()
    oob_rf, skip_rf = oob_auc_vs_ntrees(rf, X_train, y_train, KS)
    oob_bag, skip_bag = oob_auc_vs_ntrees(bag, X_train, y_train, KS)
    curve = pd.DataFrame({"n_trees": KS, "oob_auc_rf": oob_rf, "oob_auc_bagging": oob_bag,
                          "n_train_rows_without_oob_rf": skip_rf, "n_train_rows_without_oob_bagging": skip_bag,
                          "test_auc_rf": auc_vs_ntrees(rf, X_test, y_test, KS),
                          "test_auc_bagging": auc_vs_ntrees(bag, X_test, y_test, KS)})
    curve.to_csv(os.path.join(RESULTS, "rf_auc_vs_ntrees.csv"), index=False)
    timings["auc_vs_ntrees_curves"] = round(time.time() - t0, 1)
    plot_auc_vs_ntrees(curve, best_mf)
    steps = curve[curve["n_trees"] % 50 == 0]
    oob_at = {int(k): (round(r, 4), round(b, 4)) for k, r, b in zip(steps["n_trees"], steps["oob_auc_rf"], steps["oob_auc_bagging"])}
    auc_at = {int(k): (round(r, 4), round(b, 4)) for k, r, b in zip(steps["n_trees"], steps["test_auc_rf"], steps["test_auc_bagging"])}
    assert abs(oob_rf[-1] - rf_oob["oob_auc"]) < 1e-6, "OOB-vs-k at k=500 must equal sklearn's OOB AUC"
    print(f"\nOOB AUC (rf, bagging) at 50-tree steps ({timings['auc_vs_ntrees_curves']:.0f}s):", oob_at)
    print("Test AUC (rf, bagging) at 50-tree steps:", auc_at)

    # ------------------------------------------------------------------ 5. out-of-fold probabilities + evaluate() (once per model)
    t0 = time.time()
    p_oof_rf = cross_val_predict(clone(rf), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    aucs_rf = fold_aucs(y_train, p_oof_rf)
    p_oof_bag = cross_val_predict(clone(bag), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    aucs_bag = fold_aucs(y_train, p_oof_bag)
    timings["oof_passes_rf_and_bagging"] = round(time.time() - t0, 1)
    print(f"\n5-fold CV AUC ({N_TREES} trees): RF {aucs_rf.mean():.4f} ± {aucs_rf.std():.4f} | bagging {aucs_bag.mean():.4f} ± {aucs_bag.std():.4f} "
          f"({timings['oof_passes_rf_and_bagging']:.0f}s)")
    n_folds_ahead = int((aucs_rf > aucs_runner).sum())
    print(f"winner vs runner-up at {N_TREES} trees, fold by fold: {np.round(aucs_rf, 4).tolist()} vs {np.round(aucs_runner, 4).tolist()} "
          f"-> winner ahead in {n_folds_ahead}/5 folds")

    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
    grid_records = grid.round(5).to_dict(orient="records")
    top2_500 = [dict(rank=1, max_features=str(best_mf), min_samples_leaf=int(best_msl), cv_auc_200trees=round(float(gs.best_score_), 5),
                     cv_auc_500trees=round(float(aucs_rf.mean()), 5), cv_auc_500trees_std=round(float(aucs_rf.std()), 5),
                     fold_aucs_500trees=np.round(aucs_rf, 5).tolist()),
                dict(rank=2, max_features=str(run_mf), min_samples_leaf=int(run_msl), cv_auc_200trees=round(float(cvres["mean_test_score"].iloc[1]), 5),
                     cv_auc_500trees=round(float(aucs_runner.mean()), 5), cv_auc_500trees_std=round(float(aucs_runner.std()), 5),
                     fold_aucs_500trees=np.round(aucs_runner, 5).tolist())]
    common_notes = (f"Grid tuned with {N_TREES_GRID}-tree forests (budget), winner refitted with {N_TREES} trees; "
                    f"cv_auc is the 5-fold CV AUC of the final {N_TREES}-tree pipeline on the shared folds. "
                    f"OOB statistics come from the bootstrap out-of-bag rows of the final forest. ")
    res_rf = evaluate("Random forest", rf, X_train, y_train, X_test, y_test, member=MEMBER, slug="random_forest",
                      best_params=dict(max_features=best_mf, min_samples_leaf=best_msl, n_estimators=N_TREES, criterion="gini",
                                       bootstrap=True, max_depth=None),
                      cv_auc=float(aucs_rf.mean()), cv_auc_std=float(aucs_rf.std()), fit_seconds=round(rf_fit_seconds, 1), p_oof=p_oof_rf,
                      notes=common_notes + f"Grid: max_features in {PARAM_GRID['clf__max_features']} x min_samples_leaf in "
                                           f"{PARAM_GRID['clf__min_samples_leaf']}, GridSearchCV(cv=CV, scoring={SCORING}, n_jobs=1), forest n_jobs=2. "
                                           f"Leakage-free checks of the {N_TREES_GRID}-tree grid: top-2 settings re-scored with {N_TREES} trees on the same folds "
                                           "(extra.grid_top2_cv_auc_500trees) and OOB AUC vs number of trees on the training rows (extra.oob_auc_vs_ntrees). "
                                           f"Feature importance reported three ways ({N_REPEATS} permutation repeats each): MDI, permutation on the "
                                           "TRAINING set (in-sample, magnitudes optimistic — kept only as the contrast) and Breiman's OUT-OF-BAG "
                                           "permutation, which is the measure the report quotes; extra.spearman holds the rank correlations between "
                                           "them. extra.test_auc_vs_ntrees is descriptive only (nothing chosen from it).",
                      extra=dict(grid_cv_auc_200trees=float(gs.best_score_), grid_seconds=timings["grid_200trees"], grid=grid_records,
                                 grid_top2_cv_auc_500trees=top2_500, winner_ahead_of_runner_up_folds_500trees=n_folds_ahead,
                                 cv_fold_aucs=np.round(aucs_rf, 5).tolist(), **rf_oob,
                                 n_features_raw=len(feature_names), n_features_after_onehot=len(tnames),
                                 importance_top10_oob_permutation=imp.sort_values("oob_perm_auc_drop", ascending=False).head(10)[
                                     ["feature", "oob_perm_auc_drop", "oob_perm_auc_drop_std", "perm_auc_drop", "mdi"]].round(5).to_dict(orient="records"),
                                 importance_top10_permutation=imp.sort_values("perm_auc_drop", ascending=False).head(10)[
                                     ["feature", "perm_auc_drop", "oob_perm_auc_drop", "mdi"]].round(5).to_dict(orient="records"),
                                 importance_top10_mdi=imp.sort_values("mdi", ascending=False).head(10)[
                                     ["feature", "mdi", "oob_perm_auc_drop", "perm_auc_drop"]].round(5).to_dict(orient="records"),
                                 spearman=spear, spearman_mdi_vs_permutation=rho,
                                 oob_permutation_baseline_auc=round(float(oob_perm_base_auc), 5),
                                 train_over_oob_drop_gap=gap,
                                 permutation_repeats=N_REPEATS, permutation_seconds=round(perm_seconds, 1),
                                 oob_permutation_seconds=round(oob_perm_seconds, 1),
                                 oob_auc_vs_ntrees={str(k): v[0] for k, v in oob_at.items()},
                                 test_auc_vs_ntrees={str(k): v[0] for k, v in auc_at.items()},
                                 timings=timings, versions=versions))
    res_bag = evaluate("Bagging (RF, max_features=None)", bag, X_train, y_train, X_test, y_test, member=MEMBER, slug="bagging",
                       best_params=dict(max_features=None, min_samples_leaf=best_msl, n_estimators=N_TREES, criterion="gini",
                                        bootstrap=True, max_depth=None),
                       cv_auc=float(aucs_bag.mean()), cv_auc_std=float(aucs_bag.std()), fit_seconds=round(bag_fit_seconds, 1), p_oof=p_oof_bag,
                       notes=common_notes + "Not tuned separately: same min_samples_leaf as the tuned random forest, max_features=None "
                                            "(every split sees all 32 columns) — i.e. plain bagging of trees, to isolate the effect of feature subsampling.",
                       extra=dict(cv_fold_aucs=np.round(aucs_bag, 5).tolist(), **bag_oob,
                                  oob_auc_vs_ntrees={str(k): v[1] for k, v in oob_at.items()},
                                  test_auc_vs_ntrees={str(k): v[1] for k, v in auc_at.items()}, versions=versions))

    for r in (res_rf, res_bag):
        print(f"\n{r['name']}: CV AUC {r['cv_auc']:.4f}±{r['cv_auc_std']:.4f} | OOF AUC {r['oof_auc']:.4f} | OOB AUC {r['extra']['oob_auc']:.4f} | "
              f"train AUC {r['train_auc']:.4f} | test AUC {r['test_auc']:.4f} | test AP {r['test_ap']:.4f} | Brier {r['test_brier']:.4f}")
        for key in ("at_0_5", "at_tuned_f1", "at_tuned_cost"):
            m = r[key]
            print(f"   {key:14s} t={m['threshold']:.2f} acc {m['accuracy']:.4f} bal.acc {m['balanced_accuracy']:.4f} "
                  f"P1 {m['precision_1']:.4f} R1 {m['recall_1']:.4f} F1 {m['f1_1']:.4f} cost {m['cost']:.0f}")
    timings["total_wall"] = round(time.time() - t_all, 1)
    res_rf["extra"]["timings"] = timings                      # add the total after the fact, then rewrite the JSON
    json.dump(res_rf, open(os.path.join(RESULTS, "random_forest.json"), "w"), indent=1)
    print(f"\nTimings (s): {timings}\nTotal wall time {timings['total_wall']:.0f}s")


if __name__ == "__main__":
    plots_only() if "--plots-only" in sys.argv else main()
