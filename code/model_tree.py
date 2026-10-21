"""
model_tree.py — LU Xinyu: SINGLE DECISION TREE with cost-complexity pruning (lecture B7).

Run from Project/code:   python3 model_tree.py

What it does
  1. Grows a full (unpruned) entropy tree inside Pipeline(make_preprocessor(scale=False), DecisionTreeClassifier).
  2. Reads the cost-complexity pruning path of that tree (Breiman's weakest-link pruning, alpha = penalty per leaf).
  3. Picks ~20 alpha values from the path (log-spaced in "number of pruning steps remaining", i.e. roughly log-spaced
     tree sizes from 2 leaves up to the full tree) and tunes ccp_alpha x max_depth in {None, 4, 6, 8} with
     GridSearchCV(cv=CV, scoring=SCORING, n_jobs=2).
  4. Evaluates the best pruned tree (slug tree_pruned) and the unpruned full tree (slug tree_full) with evaluate().
  5. Figures: figures/tree_cv_auc_vs_alpha.png, figures/tree_pruned.png (top 4 levels); rules: results/tree_rules.txt;
     the whole CV grid: results/tree_cv_grid.csv.

Nothing is ever fitted or tuned on the test set: the pruning path, the alpha grid, the CV and the thresholds all come
from the training split only.
"""
import time, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
from common import *

MEMBER = "LU Xinyu"
N_ALPHAS = 20
DEPTHS = [None, 4, 6, 8]
BASELINE_CV_AUC, BASELINE_TEST_AUC, BASELINE_TEST_AP = 0.760, 0.764, 0.524     # logistic regression reference (README)


def make_pipe(**tree_kwargs):
    """Preprocessor (no scaling: trees are invariant to monotone transforms) + entropy tree, seed fixed."""
    return Pipeline([("pre", make_preprocessor(scale=False)),
                     ("clf", DecisionTreeClassifier(criterion="entropy", random_state=SEED, **tree_kwargs))])


def clean_names(names):
    """'num__LIMIT_BAL' -> 'LIMIT_BAL', 'cat__EDUCATION_2' -> 'EDUCATION_2'."""
    return [n.split("__", 1)[-1] for n in names]


def tree_stats(pipe):
    clf = pipe.named_steps["clf"]
    return dict(n_leaves=int(clf.get_n_leaves()), depth=int(clf.get_depth()), n_nodes=int(clf.tree_.node_count))


def first_splits(clf, names, max_depth=2):
    """Readable description of the top of the tree: feature, threshold, n samples, default rate per node."""
    t = clf.tree_
    out = []

    def rec(node, depth, path):
        n = int(t.n_node_samples[node]); v = np.asarray(t.value[node][0], dtype=float); rate = float(v[1] / max(v.sum(), 1e-12))
        if t.children_left[node] == -1 or depth > max_depth:
            out.append(dict(path=path or "ROOT", leaf=True, n=n, default_rate=round(rate, 3), depth=depth))
            return
        f, thr = names[t.feature[node]], float(t.threshold[node])
        out.append(dict(path=path or "ROOT", leaf=False, split=f"{f} <= {thr:.3f}", n=n, default_rate=round(rate, 3), depth=depth))
        rec(t.children_left[node], depth + 1, f"{path + ' & ' if path else ''}{f} <= {thr:.3f}")
        rec(t.children_right[node], depth + 1, f"{path + ' & ' if path else ''}{f} > {thr:.3f}")

    rec(0, 0, "")
    return out


def main():
    X_train, X_test, y_train, y_test, feature_names = load_split()
    print(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")

    # ------------------------------------------------------------------ 1. full (unpruned) tree on the training data
    t0 = time.time()
    pipe_full = make_pipe().fit(X_train, y_train)
    full_fit_s = time.time() - t0
    names = clean_names(feature_names_out(pipe_full.named_steps["pre"]))
    st_full = tree_stats(pipe_full)
    print(f"[full tree] leaves={st_full['n_leaves']:,} depth={st_full['depth']} nodes={st_full['n_nodes']:,} "
          f"train AUC={roc_auc_score(y_train, pipe_full.predict_proba(X_train)[:, 1]):.4f}  ({full_fit_s:.1f}s)")

    # ------------------------------------------------------------------ 2. cost-complexity pruning path (training data only)
    Xt_train = pipe_full.named_steps["pre"].transform(X_train)
    path = pipe_full.named_steps["clf"].cost_complexity_pruning_path(Xt_train, y_train)
    alphas, impurities = np.asarray(path.ccp_alphas), np.asarray(path.impurities)
    N = len(alphas)
    print(f"[path] {N:,} alphas, min positive alpha={alphas[alphas > 0].min():.2e}, max alpha (root only)={alphas[-1]:.4e}")
    # ~20 alphas: log-spaced in the number of weakest-link pruning steps still remaining (1 .. N-1), which is
    # roughly log-spaced tree size. Index 0 is alpha=0 (unpruned); index N-1 (root-only stump) is excluded.
    steps = np.unique(np.round(np.geomspace(1, N - 1, N_ALPHAS + 2)).astype(int))
    idx = np.unique((N - 1) - steps)
    idx = idx[idx >= 0]
    alpha_grid = sorted(set(float(alphas[i]) for i in idx))
    if 0.0 not in alpha_grid:
        alpha_grid = [0.0] + alpha_grid
    print(f"[grid] {len(alpha_grid)} ccp_alpha values: {['%.2e' % a for a in alpha_grid]}")

    # size of the tree at each alpha (max_depth=None) — cheap refits on the training set, for reporting/plots only
    size_rows = []
    for a in alpha_grid:
        p = make_pipe(ccp_alpha=a).fit(X_train, y_train)
        size_rows.append(dict(ccp_alpha=a, **tree_stats(p)))
    size_df = pd.DataFrame(size_rows)
    print(size_df.to_string(index=False))

    # ------------------------------------------------------------------ 3. GridSearchCV over ccp_alpha x max_depth
    param_grid = {"clf__ccp_alpha": alpha_grid, "clf__max_depth": DEPTHS}
    gs = GridSearchCV(make_pipe(), param_grid, cv=CV, scoring=SCORING, n_jobs=2, return_train_score=True, refit=True)
    t0 = time.time()
    gs.fit(X_train, y_train)
    fit_s = time.time() - t0
    cvr = pd.DataFrame(gs.cv_results_)
    cvr["ccp_alpha"] = cvr["param_clf__ccp_alpha"].astype(float)
    cvr["max_depth"] = cvr["param_clf__max_depth"].apply(lambda d: "None" if d is None else str(int(d)))
    grid_df = cvr[["ccp_alpha", "max_depth", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score", "mean_fit_time"]]
    grid_df = grid_df.merge(size_df[["ccp_alpha", "n_leaves", "depth"]].rename(columns={"n_leaves": "n_leaves_depthNone", "depth": "depth_depthNone"}),
                            on="ccp_alpha", how="left").sort_values(["max_depth", "ccp_alpha"])
    grid_df.to_csv(os.path.join(RESULTS, "tree_cv_grid.csv"), index=False)
    best = gs.best_params_
    best_i = gs.best_index_
    cv_auc, cv_std = float(gs.cv_results_["mean_test_score"][best_i]), float(gs.cv_results_["std_test_score"][best_i])
    print(f"[grid] {len(cvr)} combos x 5 folds in {fit_s:.0f}s; best {best} CV AUC={cv_auc:.4f} +/- {cv_std:.4f}")
    print(grid_df.sort_values("rank_test_score").head(8).to_string(index=False))

    pipe_pruned = gs.best_estimator_
    st_pruned = tree_stats(pipe_pruned)
    clf_pruned = pipe_pruned.named_steps["clf"]
    print(f"[pruned tree] leaves={st_pruned['n_leaves']} depth={st_pruned['depth']} nodes={st_pruned['n_nodes']}")

    # CV AUC of the unpruned tree (alpha=0, max_depth=None) straight from the same grid
    m_full = (cvr["ccp_alpha"] == 0.0) & (cvr["max_depth"] == "None")
    cv_auc_full, cv_std_full = float(cvr.loc[m_full, "mean_test_score"].iloc[0]), float(cvr.loc[m_full, "std_test_score"].iloc[0])

    # ------------------------------------------------------------------ 4. figure: CV AUC vs alpha
    lo = min(a for a in alpha_grid if a > 0)
    def xplot(a): return np.where(np.asarray(a) == 0, lo / 4, np.asarray(a))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"None": "C3", "4": "C0", "6": "C2", "8": "C1"}
    for d in ["None", "4", "6", "8"]:
        sub = grid_df[grid_df["max_depth"] == d].sort_values("ccp_alpha")
        ax[0].errorbar(xplot(sub["ccp_alpha"]), sub["mean_test_score"], yerr=sub["std_test_score"], marker="o", ms=4, lw=1.6,
                       capsize=2, color=colors[d], label=f"CV AUC, max_depth={d}")
    sub = grid_df[grid_df["max_depth"] == "None"].sort_values("ccp_alpha")
    ax[0].plot(xplot(sub["ccp_alpha"]), sub["mean_train_score"], ls="--", color="C3", alpha=0.6, lw=1.2, label="train AUC, max_depth=None")
    ax[0].axhline(BASELINE_CV_AUC, color="k", ls=":", lw=1.2, label=f"logistic baseline (CV AUC ~ {BASELINE_CV_AUC:.3f})")
    ax[0].axvline(best["clf__ccp_alpha"], color="grey", ls="-.", lw=1, label=f"best alpha = {best['clf__ccp_alpha']:.2e}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("ccp_alpha (leftmost point = 0, unpruned)"); ax[0].set_ylabel("ROC AUC (5-fold CV, training set)")
    ax[0].set_title("Cost-complexity pruning: CV AUC vs alpha"); ax[0].legend(fontsize=7.5, loc="lower left"); ax[0].grid(alpha=0.3)
    ax[0].set_ylim(0.55, 1.01)
    ax[1].plot(xplot(size_df["ccp_alpha"]), size_df["n_leaves"], marker="s", ms=4, color="C7")
    ax[1].axvline(best["clf__ccp_alpha"], color="grey", ls="-.", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("ccp_alpha"); ax[1].set_ylabel("number of leaves (max_depth=None)")
    ax[1].set_title(f"Tree size along the path: {st_full['n_leaves']:,} leaves -> {st_pruned['n_leaves']} at best alpha"); ax[1].grid(alpha=0.3, which="both")
    save_extra_figure(fig, "tree_cv_auc_vs_alpha.png")

    # ------------------------------------------------------------------ 5. figure: pruned tree (top 4 levels) + rules
    fig, ax = plt.subplots(figsize=(26, 12))
    plot_tree(clf_pruned, max_depth=4, feature_names=names, class_names=["no default", "default"], filled=True, rounded=True,
              proportion=True, impurity=False, precision=2, fontsize=9, ax=ax)
    ax.set_title(f"Pruned decision tree (entropy, ccp_alpha={best['clf__ccp_alpha']:.2e}, max_depth={best['clf__max_depth']}): "
                 f"top 4 levels of {st_pruned['n_leaves']} leaves / depth {st_pruned['depth']}. "
                 f"Node colour = majority class, 'value' = [P(no default), P(default)], 'samples' = share of training rows", fontsize=11)
    save_extra_figure(fig, "tree_pruned.png")

    rules = export_text(clf_pruned, feature_names=names, class_names=["no_default", "default"], show_weights=True, decimals=3)
    header = (f"Pruned decision tree rules (LU Xinyu) — criterion=entropy, ccp_alpha={best['clf__ccp_alpha']:.6e}, "
              f"max_depth={best['clf__max_depth']}, leaves={st_pruned['n_leaves']}, depth={st_pruned['depth']}\n"
              f"Trained on the 20,975-row training split only. weights: [no_default, default] counts in each leaf.\n"
              f"Full unpruned tree for comparison: leaves={st_full['n_leaves']:,}, depth={st_full['depth']}.\n"
              + "=" * 100 + "\n")
    with open(os.path.join(RESULTS, "tree_rules.txt"), "w") as f:
        f.write(header + rules)

    # feature importances & top splits of the pruned tree
    imp = pd.Series(clf_pruned.feature_importances_, index=names).sort_values(ascending=False)
    top_imp = {k: round(float(v), 4) for k, v in imp.head(12).items()}
    splits = first_splits(clf_pruned, names, max_depth=2)
    print("[pruned tree] top feature importances:", top_imp)
    for s in splits:
        print("   ", s)

    # ------------------------------------------------------------------ 6. evaluate both trees (once each)
    extra_common = dict(n_leaves_full=st_full["n_leaves"], depth_full=st_full["depth"], n_nodes_full=st_full["n_nodes"],
                        n_leaves_pruned=st_pruned["n_leaves"], depth_pruned=st_pruned["depth"], n_nodes_pruned=st_pruned["n_nodes"],
                        n_path_alphas=int(N), alpha_grid=[float(a) for a in alpha_grid], max_depth_grid=[str(d) for d in DEPTHS],
                        leaves_at_alpha=size_df.to_dict(orient="list"), cv_auc_full=cv_auc_full, cv_auc_std_full=cv_std_full)
    cv_curve = grid_df[["ccp_alpha", "max_depth", "mean_test_score", "std_test_score", "mean_train_score"]].round(5).to_dict(orient="list")

    notes_pruned = (f"Single entropy tree, cost-complexity (weakest-link) pruning. Full tree grown on the training split "
                    f"({st_full['n_leaves']:,} leaves, depth {st_full['depth']}); {len(alpha_grid)} ccp_alpha values taken from its pruning path "
                    f"(log-spaced in pruning steps, alpha=0 included) x max_depth {DEPTHS} tuned by 5-fold CV ROC AUC on the training set "
                    f"(no subsampling needed). Best: ccp_alpha={best['clf__ccp_alpha']:.3e}, max_depth={best['clf__max_depth']} -> "
                    f"{st_pruned['n_leaves']} leaves, depth {st_pruned['depth']}. Thresholds tuned on training OOF probabilities.")
    res_pruned = evaluate("Decision tree (pruned)", pipe_pruned, X_train, y_train, X_test, y_test, member=MEMBER,
                          best_params={"criterion": "entropy", "ccp_alpha": float(best["clf__ccp_alpha"]),
                                       "max_depth": None if best["clf__max_depth"] is None else int(best["clf__max_depth"])},
                          cv_auc=cv_auc, cv_auc_std=cv_std, notes=notes_pruned, fit_seconds=round(fit_s, 1),
                          extra=dict(**extra_common, feature_importances_top=top_imp, first_splits=splits, cv_curve=cv_curve),
                          slug="tree_pruned")

    notes_full = (f"Unpruned entropy tree (ccp_alpha=0, max_depth=None) grown to purity on the training split — reported only to show "
                  f"overfitting: {st_full['n_leaves']:,} leaves, depth {st_full['depth']}, train AUC ~1. Its CV AUC comes from the same grid "
                  f"as the pruned tree. Not a candidate model.")
    res_full = evaluate("Decision tree (unpruned full tree)", pipe_full, X_train, y_train, X_test, y_test, member=MEMBER,
                        best_params={"criterion": "entropy", "ccp_alpha": 0.0, "max_depth": None},
                        cv_auc=cv_auc_full, cv_auc_std=cv_std_full, notes=notes_full, fit_seconds=round(full_fit_s, 1),
                        extra=extra_common, slug="tree_full")

    # ------------------------------------------------------------------ 7. console summary
    for r in (res_pruned, res_full):
        print(f"\n== {r['name']} ({r['slug']}) ==")
        print(f"  train AUC {r['train_auc']:.4f} | OOF AUC {r['oof_auc']:.4f} | CV AUC {r['cv_auc']:.4f} | test AUC {r['test_auc']:.4f} | "
              f"test AP {r['test_ap']:.4f} | Brier {r['test_brier']:.4f}")
        for k in ("at_0_5", "at_tuned_f1", "at_tuned_cost"):
            m = r[k]
            print(f"  {k:14s} thr={m['threshold']:.2f} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                  f"P1={m['precision_1']:.4f} R1={m['recall_1']:.4f} F1={m['f1_1']:.4f} cost={m['cost']:.0f}")
    print(f"\nLogistic baseline for reference: test AUC {BASELINE_TEST_AUC}, test AP {BASELINE_TEST_AP}")
    print("Written:", [os.path.join(RESULTS, f) for f in ("tree_pruned.json", "tree_full.json", "tree_rules.txt", "tree_cv_grid.csv")],
          [os.path.join(FIGURES, f) for f in ("tree_cv_auc_vs_alpha.png", "tree_pruned.png", "tree_pruned_eval.png", "tree_full_eval.png")])


if __name__ == "__main__":
    main()
