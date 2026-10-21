"""
model_mlp.py — Feed-forward neural network / multilayer perceptron (lecture B9).  Owner: LI Jingwen.

Run from Project/code:  python3 model_mlp.py            (about 5-6 minutes with n_jobs=2; the wall-clock times of the run that
                                                        produced results/ are stored in the JSON files -- extra.grid.grid_search_seconds,
                                                        fit_seconds, extra.oof_predict_seconds, extra.total_script_seconds -- and the
                                                        metrics are seeded, so nothing in this file needs editing after a run)

What it does
  1. load_split() -> the shared stratified 70/30 split (seed 2607526). The test set is touched only inside evaluate().
  2. Pipeline(make_preprocessor(), MLPClassifier(early_stopping=True, validation_fraction=0.15, max_iter=300,
     n_iter_no_change=15, random_state=SEED)).  StandardScaler + one-hot are refitted inside every CV fold; inside each
     fit sklearn additionally holds out 15 % of that fold's training rows to decide when to stop (early stopping).
  3. GridSearchCV over hidden_layer_sizes {(32,), (64,), (64,32), (128,64)} x alpha (L2) {1e-4, 1e-3, 1e-2}
     x learning_rate_init {1e-3, 3e-3} = 24 settings x 5 folds, cv=CV, scoring=SCORING (roc_auc), n_jobs=2.
     One early-stopped fit takes about 1 s, so the FULL grid is run (no RandomizedSearchCV or subsampling needed)
     -> results/mlp_cv_table.csv (CV AUC of every setting).
  4. evaluate() the tuned network as "MLP (feed-forward NN)"            -> slug mlp.
  4b. The same tuned network refitted with balanced per-row sample weights ("class weights": MLPClassifier has no class_weight
     argument, but fit() accepts sample_weight, passed through the Pipeline as clf__sample_weight), NOT re-tuned,
     evaluate()d as "MLP (tuned, balanced sample weights)"               -> slug mlp_weighted.
  5. A deliberately over-parameterised network (256,256), NO early stopping, max_iter=200, sklearn default alpha=1e-4,
     evaluate()d as "MLP over-parameterised (256,256)"                    -> slug mlp_overfit  (train vs test AUC = overfitting).
  5b. Paired bootstrap on the TEST set (2,000 resamples of the 8,990 test rows, seed 2607526) of the tuned MLP's AUC / AP minus
     that of logistic regression and of XGBoost, computed from the saved results/<slug>_p_test.npy files (same rows for every
     model).  Pure evaluation of already-fitted models -- nothing is fitted or tuned on the test set.  -> extra.paired_bootstrap_test
  6. Figures
       figures/mlp_loss_curve.png     training log-loss per epoch of both networks (full-training-set fits) and the
                                      validation accuracy that drove early stopping in the tuned network;
       figures/mlp_overfit_epochs.png train vs held-out AUC per epoch for both architectures, trained for 200 epochs
                                      WITHOUT early stopping on a training-internal 85/15 split (results/mlp_overfit_epochs.csv).
                                      Diagnostic only: no choice is made from it and the test set is not involved.
  7. results/mlp_comparison.csv  cross-model table (CV AUC / test AUC / AP / Brier / F1 at the tuned threshold) read from whatever
                                 results/<slug>.json files exist when the script runs; the comparison table in report/notes_mlp.md
                                 is copied from this file (also printed as Markdown) so it cannot drift from results/.
Everything is fitted / tuned on the training rows only.  PyTorch is not installed on the laptop, so the network is
sklearn's MLPClassifier (ReLU, Adam, mini-batches of 200, log-loss + L2 penalty alpha).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")        # BLAS threads of this process = 2; joblib workers are limited to 1 thread each below
import sys, time, json
import numpy as np
import pandas as pd
from joblib import parallel_config
from common import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

MEMBER = "LI Jingwen"
N_JOBS = 2
HLS_GRID = [(32,), (64,), (64, 32), (128, 64)]
ALPHA_GRID = [1e-4, 1e-3, 1e-2]
LR_GRID = [1e-3, 3e-3]
ES_KW = dict(early_stopping=True, validation_fraction=0.15, max_iter=300, n_iter_no_change=15)
OVERFIT_HLS, OVERFIT_ITERS, OVERFIT_ALPHA, OVERFIT_LR = (256, 256), 200, 1e-4, 1e-3     # alpha / lr = sklearn defaults
T_START = time.time()

# validated categorical palette (dataviz skill, light surface): slot 1 blue, slot 2 orange; ink / chrome greys
C1, C2 = "#2a78d6", "#eb6834"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS, C_SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def log(msg):
    print(f"[mlp] {time.time() - T_START:6.0f}s  {msg}", flush=True)


def hls_str(h):
    return "x".join(str(int(v)) for v in h)                       # (64, 32) -> "64x32"


def arch_str(h, n_in):
    return "-".join([str(n_in)] + [str(int(v)) for v in h] + ["1"])   # (64, 32) -> "32-64-32-1"


def n_params(clf):
    return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))


def make_mlp(**kw):
    return Pipeline([("pre", make_preprocessor()), ("clf", MLPClassifier(random_state=SEED, **kw))])


def style_axis(ax):
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=8.5, length=3)
    ax.grid(axis="y", color=C_GRID, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(C_INK2); ax.yaxis.label.set_color(C_INK2); ax.title.set_color(C_INK)


# --------------------------------------------------------------------------- 1. data
X_train, X_test, y_train, y_test, feature_names = load_split()
log(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")
n_in = len(feature_names_out(make_preprocessor().fit(X_train)))            # 32 columns after scaling + one-hot


def fold_aucs(p_oof):
    """Per-fold AUC recomputed from out-of-fold probabilities on the shared CV folds (no extra fitting)."""
    return [float(roc_auc_score(y_train.iloc[va], p_oof[va])) for _, va in CV.split(X_train, y_train)]


# --------------------------------------------------------------------------- 2-3. pipeline + grid search (training set only)
pipe = make_mlp(**ES_KW)
param_grid = {"clf__hidden_layer_sizes": HLS_GRID, "clf__alpha": ALPHA_GRID, "clf__learning_rate_init": LR_GRID}
n_settings = len(HLS_GRID) * len(ALPHA_GRID) * len(LR_GRID)
gs = GridSearchCV(pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=N_JOBS, refit=True, return_train_score=True, verbose=0)
t0 = time.time()
with parallel_config(backend="loky", inner_max_num_threads=1):           # 2 workers x 1 BLAS thread = the 2-core budget
    gs.fit(X_train, y_train)
grid_seconds = time.time() - t0
log(f"grid search over {n_settings} settings x {CV.get_n_splits()} folds ({n_settings * CV.get_n_splits()} fits) done in {grid_seconds:.0f}s")

cvr = pd.DataFrame(gs.cv_results_)
cv_table = pd.DataFrame({
    "hidden_layer_sizes": [hls_str(h) for h in cvr["param_clf__hidden_layer_sizes"]],
    "alpha": cvr["param_clf__alpha"].astype(float).values,
    "learning_rate_init": cvr["param_clf__learning_rate_init"].astype(float).values,
    "cv_auc_mean": cvr["mean_test_score"].values, "cv_auc_std": cvr["std_test_score"].values,
    "train_fold_auc_mean": cvr["mean_train_score"].values, "rank": cvr["rank_test_score"].astype(int).values,
    "mean_fit_seconds": cvr["mean_fit_time"].values,
}).sort_values(["rank", "hidden_layer_sizes", "alpha", "learning_rate_init"]).reset_index(drop=True)
cv_table["optimism_gap"] = cv_table["train_fold_auc_mean"] - cv_table["cv_auc_mean"]
cv_csv = os.path.join(RESULTS, "mlp_cv_table.csv")
cv_table.to_csv(cv_csv, index=False, float_format="%.5f")
print(cv_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log(f"CV table -> {cv_csv}")

bp = gs.best_params_
best_hls = tuple(int(v) for v in bp["clf__hidden_layer_sizes"]); best_alpha = float(bp["clf__alpha"]); best_lr = float(bp["clf__learning_rate_init"])
best_row = cv_table[(cv_table.hidden_layer_sizes == hls_str(best_hls)) & (cv_table.alpha == best_alpha) &
                    (cv_table.learning_rate_init == best_lr)].iloc[0]
cv_auc_best, cv_auc_std_best = float(best_row.cv_auc_mean), float(best_row.cv_auc_std)
worst_row = cv_table.iloc[-1]
arch_best, arch_over = arch_str(best_hls, n_in), arch_str(OVERFIT_HLS, n_in)
log(f"best: hidden={best_hls}, alpha={best_alpha:g}, lr={best_lr:g}, CV AUC {cv_auc_best:.4f} +/- {cv_auc_std_best:.4f}; "
    f"worst setting CV AUC {worst_row.cv_auc_mean:.4f} (spread across the grid {cv_auc_best - worst_row.cv_auc_mean:.4f})")

# is the selected value on an edge of its grid?  An edge value means the grid might deserve extending; the marginal means
# below say whether the choice matters at all.  HLS_GRID is listed from the smallest to the largest network.
def edge_of(value, grid):
    return "lower edge" if value == grid[0] else ("upper edge" if value == grid[-1] else "interior")


best_on_grid_edge = dict(hidden_layer_sizes=edge_of(best_hls, HLS_GRID), alpha=edge_of(best_alpha, ALPHA_GRID),
                         learning_rate_init=edge_of(best_lr, LR_GRID))
log("position of the best value in its grid: " + ", ".join(f"{k}: {v}" for k, v in best_on_grid_edge.items()))

# marginal means: average CV AUC over the other two hyper-parameters (which knob matters?)
marginal = {k: {str(kk): round(float(vv), 5) for kk, vv in cv_table.groupby(k)["cv_auc_mean"].mean().items()}
            for k in ("hidden_layer_sizes", "alpha", "learning_rate_init")}
for k, v in marginal.items():
    log(f"marginal CV AUC by {k}: " + ", ".join(f"{kk}: {vv:.4f}" for kk, vv in v.items()))

# one-standard-error view: smallest network whose CV AUC is within 1 sd of the best (reported only; best-by-AUC is what we ship)
thr_1se = cv_auc_best - cv_auc_std_best
within = cv_table[cv_table.cv_auc_mean >= thr_1se].copy()
within["n_hidden"] = within.hidden_layer_sizes.map(lambda s: sum(int(v) for v in s.split("x")))
r1 = within.sort_values(["n_hidden", "alpha"], ascending=[True, False]).iloc[0]
one_se = dict(hidden_layer_sizes=r1.hidden_layer_sizes, alpha=float(r1.alpha), learning_rate_init=float(r1.learning_rate_init),
              cv_auc_mean=float(r1.cv_auc_mean), threshold_cv_auc=round(thr_1se, 5))

# --------------------------------------------------------------------------- 4. tuned network: refit stats, OOF, evaluate
best_model = gs.best_estimator_                                            # refitted by GridSearchCV on the full training set
clf_best = best_model.named_steps["clf"]
refit_seconds = float(gs.refit_time_)
vs_best = [float(v) for v in clf_best.validation_scores_]
best_epoch = int(np.argmax(vs_best)) + 1
log(f"tuned network {arch_best}: {n_params(clf_best):,} weights, ran {clf_best.n_iter_} epochs, best internal-validation "
    f"accuracy {clf_best.best_validation_score_:.4f} at epoch {best_epoch} (weights restored to that epoch), refit {refit_seconds:.1f}s")

t0 = time.time()
with parallel_config(backend="loky", inner_max_num_threads=1):
    p_oof_best = cross_val_predict(clone(best_model), X_train, y_train, cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]
oof_seconds_best = time.time() - t0
aucs_best = fold_aucs(p_oof_best)

extra_best = dict(
    grid=dict(hidden_layer_sizes=[hls_str(h) for h in HLS_GRID], alpha=ALPHA_GRID, learning_rate_init=LR_GRID,
              n_settings=n_settings, n_folds=CV.get_n_splits(), grid_search_seconds=round(grid_seconds, 1),
              search="GridSearchCV (full grid; ~1 s per fit so no RandomizedSearchCV / subsampling needed)"),
    cv_table=cv_table.round(5).to_dict(orient="list"), cv_table_csv=os.path.relpath(cv_csv, ROOT),
    marginal_cv_auc=marginal, one_se_rule=one_se, best_on_grid_edge=best_on_grid_edge,
    worst_setting=dict(hidden_layer_sizes=worst_row.hidden_layer_sizes, alpha=float(worst_row.alpha),
                       learning_rate_init=float(worst_row.learning_rate_init), cv_auc_mean=float(worst_row.cv_auc_mean)),
    architecture=arch_best, n_features_after_preprocessing=n_in, n_weights=n_params(clf_best),
    epochs_run=int(clf_best.n_iter_), best_epoch=best_epoch, best_validation_accuracy=float(clf_best.best_validation_score_),
    majority_class_accuracy_train=round(float(1 - y_train.mean()), 4),
    final_training_loss=round(float(clf_best.loss_curve_[-1]), 5),
    loss_curve=[round(float(v), 5) for v in clf_best.loss_curve_], validation_scores=[round(v, 5) for v in vs_best],
    oof_fold_aucs=[round(a, 5) for a in aucs_best], oof_predict_seconds=round(oof_seconds_best, 1),
    fixed_settings=dict(activation="relu", solver="adam", batch_size="auto = min(200, n)", loss="log-loss + alpha/2 * ||W||^2",
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, tol=1e-4, max_iter=300),
    threads=dict(gridsearch_n_jobs=N_JOBS, worker_blas_threads=1, main_process_blas_threads=os.environ.get("OMP_NUM_THREADS")),
)
notes_best = (f"GridSearchCV over hidden_layer_sizes in {[hls_str(h) for h in HLS_GRID]} x alpha (L2) in {ALPHA_GRID} x "
              f"learning_rate_init in {LR_GRID} ({n_settings} settings, 5-fold stratified CV, scoring=roc_auc, seed {SEED}, n_jobs=2); "
              f"the full grid took {grid_seconds:.0f}s (early-stopped fits take ~1 s) so no RandomizedSearchCV or subsampling was needed. "
              f"Preprocessing (StandardScaler + one-hot, {n_in} inputs) sits inside the Pipeline and is refitted per fold; inside every "
              f"fit MLPClassifier holds out 15% of the rows it is given to decide when to stop (early_stopping=True, n_iter_no_change=15, "
              f"max_iter=300). Best = {arch_best} (ReLU, Adam, {n_params(clf_best):,} weights), alpha={best_alpha:g}, lr={best_lr:g}; "
              f"the refit on the full training set stopped after {clf_best.n_iter_} epochs (best validation epoch {best_epoch}). "
              f"CV AUC spread across the 24 settings is only {cv_auc_best - worst_row.cv_auc_mean:.4f}.")
res_best = evaluate("MLP (feed-forward NN)", best_model, X_train, y_train, X_test, y_test, member=MEMBER, slug="mlp",
                    best_params=dict(hidden_layer_sizes=list(best_hls), alpha=best_alpha, learning_rate_init=best_lr,
                                     activation="relu", solver="adam", batch_size=200, early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=15, max_iter=300),
                    cv_auc=cv_auc_best, cv_auc_std=cv_auc_std_best, notes=notes_best, fit_seconds=round(refit_seconds, 2),
                    p_oof=p_oof_best, extra=extra_best)
log(f"MLP (tuned): train AUC {res_best['train_auc']:.4f}, OOF AUC {res_best['oof_auc']:.4f}, test AUC {res_best['test_auc']:.4f}, "
    f"AP {res_best['test_ap']:.4f}, Brier {res_best['test_brier']:.4f}")

# --------------------------------------------------------------------------- 4b. tuned network + balanced sample weights (slug mlp_weighted)
# MLPClassifier has no class_weight argument, but fit() accepts sample_weight (sklearn >= 1.7), so "class weights" are balanced
# per-row weights (n / (2 * n_class): about 2.26 for defaulters, 0.64 for non-defaulters) passed through the Pipeline as
# clf__sample_weight.  Hyper-parameters are the tuned ones (not re-tuned with weights).  With weights, sklearn's early-stopping
# score on the internal 15 % split is the weighted accuracy, i.e. balanced accuracy, instead of plain accuracy.
w_train = compute_sample_weight("balanced", y_train)
weighted = make_mlp(hidden_layer_sizes=best_hls, alpha=best_alpha, learning_rate_init=best_lr, **ES_KW)
t0 = time.time(); weighted.fit(X_train, y_train, clf__sample_weight=w_train); w_fit_seconds = time.time() - t0
clf_w = weighted.named_steps["clf"]
vs_w = [float(v) for v in clf_w.validation_scores_]; best_epoch_w = int(np.argmax(vs_w)) + 1
log(f"weighted network {arch_best}: ran {clf_w.n_iter_} epochs, best internal-validation balanced accuracy "
    f"{clf_w.best_validation_score_:.4f} at epoch {best_epoch_w}, fit {w_fit_seconds:.1f}s")
t0 = time.time()
with parallel_config(backend="loky", inner_max_num_threads=1):       # params= is sliced per fold (rows of the training fold only)
    p_oof_w = cross_val_predict(clone(weighted), X_train, y_train, cv=CV, method="predict_proba", n_jobs=N_JOBS,
                                params={"clf__sample_weight": w_train})[:, 1]
oof_seconds_w = time.time() - t0
aucs_w = fold_aucs(p_oof_w)
extra_w = dict(
    variant="tuned hyper-parameters + balanced sample weights via fit(clf__sample_weight=...); not re-tuned",
    sample_weight=dict(scheme="balanced", weight_default=round(float(w_train[y_train.values == 1][0]), 4),
                       weight_non_default=round(float(w_train[y_train.values == 0][0]), 4)),
    architecture=arch_best, n_weights=n_params(clf_w), epochs_run=int(clf_w.n_iter_), best_epoch=best_epoch_w,
    best_validation_balanced_accuracy=float(clf_w.best_validation_score_), early_stopping_criterion="weighted (= balanced) accuracy",
    loss_curve=[round(float(v), 5) for v in clf_w.loss_curve_], validation_scores=[round(v, 5) for v in vs_w],
    oof_fold_aucs=[round(a, 5) for a in aucs_w], oof_predict_seconds=round(oof_seconds_w, 1),
    unweighted_reference=dict(cv_auc=cv_auc_best, test_auc=res_best["test_auc"], test_ap=res_best["test_ap"],
                              threshold_f1=res_best["at_tuned_f1"]["threshold"], threshold_cost=res_best["at_tuned_cost"]["threshold"]),
)
notes_w = (f"Tuned network {arch_best} (alpha={best_alpha:g}, lr={best_lr:g}, early stopping as for slug mlp) refitted with balanced "
           f"per-row sample weights passed through the Pipeline as clf__sample_weight (MLPClassifier has no class_weight argument; "
           f"fit() accepts sample_weight). Hyper-parameters were NOT re-tuned with the weights. Weighting changes the loss the network "
           f"minimises and the early-stopping score (weighted = balanced accuracy on the internal 15% split), so it shifts the "
           f"probability scale towards the defaulters; it is expected to move the tuned thresholds towards 0.5 rather than the ranking "
           f"metrics. CV AUC is the per-fold AUC of the out-of-fold probabilities on the shared 5 folds (no separate grid).")
res_w = evaluate("MLP (tuned, balanced sample weights)", weighted, X_train, y_train, X_test, y_test, member=MEMBER, slug="mlp_weighted",
                 best_params=dict(hidden_layer_sizes=list(best_hls), alpha=best_alpha, learning_rate_init=best_lr, activation="relu",
                                  solver="adam", batch_size=200, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                  max_iter=300, sample_weight="balanced"),
                 cv_auc=float(np.mean(aucs_w)), cv_auc_std=float(np.std(aucs_w)), notes=notes_w, fit_seconds=round(w_fit_seconds, 2),
                 p_oof=p_oof_w, extra=extra_w)
log(f"MLP (weighted): train AUC {res_w['train_auc']:.4f}, OOF AUC {res_w['oof_auc']:.4f}, test AUC {res_w['test_auc']:.4f}, "
    f"AP {res_w['test_ap']:.4f}, Brier {res_w['test_brier']:.4f}; thresholds F1 {res_w['at_tuned_f1']['threshold']:.2f} / "
    f"cost {res_w['at_tuned_cost']['threshold']:.2f} (unweighted: {res_best['at_tuned_f1']['threshold']:.2f} / "
    f"{res_best['at_tuned_cost']['threshold']:.2f}); recall@0.5 {res_w['at_0_5']['recall_1']:.4f} vs {res_best['at_0_5']['recall_1']:.4f}")

# --------------------------------------------------------------------------- 5. over-parameterised network, no early stopping
over = make_mlp(hidden_layer_sizes=OVERFIT_HLS, alpha=OVERFIT_ALPHA, learning_rate_init=OVERFIT_LR, early_stopping=False,
                max_iter=OVERFIT_ITERS)
t0 = time.time(); over.fit(X_train, y_train); over_fit_seconds = time.time() - t0
clf_over = over.named_steps["clf"]
log(f"over-parameterised network {arch_over}: {n_params(clf_over):,} weights, {clf_over.n_iter_} epochs, final training loss "
    f"{clf_over.loss_curve_[-1]:.4f}, fit {over_fit_seconds:.0f}s")
t0 = time.time()
with parallel_config(backend="loky", inner_max_num_threads=1):
    p_oof_over = cross_val_predict(clone(over), X_train, y_train, cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]
oof_seconds_over = time.time() - t0
aucs_over = fold_aucs(p_oof_over)
log(f"over-parameterised OOF done in {oof_seconds_over:.0f}s: fold AUCs " + ", ".join(f"{a:.4f}" for a in aucs_over))

extra_over = dict(
    architecture=arch_over, n_features_after_preprocessing=n_in, n_weights=n_params(clf_over),
    weights_per_training_row=round(n_params(clf_over) / len(X_train), 2), epochs_run=int(clf_over.n_iter_),
    final_training_loss=round(float(clf_over.loss_curve_[-1]), 5), min_training_loss=round(float(clf_over.best_loss_), 5),
    loss_curve=[round(float(v), 5) for v in clf_over.loss_curve_],
    oof_fold_aucs=[round(a, 5) for a in aucs_over], oof_predict_seconds=round(oof_seconds_over, 1),
    tuned_reference=dict(architecture=arch_best, n_weights=n_params(clf_best), cv_auc=cv_auc_best, test_auc=res_best["test_auc"],
                         train_auc=res_best["train_auc"]),
    fixed_settings=dict(activation="relu", solver="adam", batch_size="auto = min(200, n)", early_stopping=False, max_iter=OVERFIT_ITERS,
                        alpha=OVERFIT_ALPHA, learning_rate_init=OVERFIT_LR),
)
notes_over = (f"Deliberately over-parameterised network {arch_over} ({n_params(clf_over):,} weights for {len(X_train):,} training rows), "
              f"NO early stopping, max_iter={OVERFIT_ITERS}, sklearn default alpha={OVERFIT_ALPHA:g} and lr={OVERFIT_LR:g}; not tuned, "
              f"shown only to illustrate overfitting (train vs test AUC). CV AUC here is the per-fold AUC of the out-of-fold "
              f"probabilities on the shared 5 folds (no separate grid).")
res_over = evaluate("MLP over-parameterised (256,256)", over, X_train, y_train, X_test, y_test, member=MEMBER, slug="mlp_overfit",
                    best_params=dict(hidden_layer_sizes=list(OVERFIT_HLS), alpha=OVERFIT_ALPHA, learning_rate_init=OVERFIT_LR,
                                     activation="relu", solver="adam", batch_size=200, early_stopping=False, max_iter=OVERFIT_ITERS),
                    cv_auc=float(np.mean(aucs_over)), cv_auc_std=float(np.std(aucs_over)), notes=notes_over,
                    fit_seconds=round(over_fit_seconds, 1), p_oof=p_oof_over, extra=extra_over)
log(f"MLP (overfit): train AUC {res_over['train_auc']:.4f}, OOF AUC {res_over['oof_auc']:.4f}, test AUC {res_over['test_auc']:.4f}, "
    f"AP {res_over['test_ap']:.4f}, Brier {res_over['test_brier']:.4f}")
THR_FLOOR = 0.05                                       # floor of the shared threshold grid in common.tune_threshold (0.05 .. 0.95)
for r in (res_best, res_w, res_over):
    r["extra"]["tuned_threshold_at_grid_floor"] = dict(
        floor=THR_FLOOR, f1=bool(r["at_tuned_f1"]["threshold"] <= THR_FLOOR), cost=bool(r["at_tuned_cost"]["threshold"] <= THR_FLOOR),
        note="True = the tuned threshold is clipped at the floor of the shared 0.05-0.95 grid, not an interior optimum")
    if r["extra"]["tuned_threshold_at_grid_floor"]["f1"] or r["extra"]["tuned_threshold_at_grid_floor"]["cost"]:
        log(f"{r['slug']}: F1 threshold {r['at_tuned_f1']['threshold']:.2f} / cost threshold {r['at_tuned_cost']['threshold']:.2f} "
            f"-> at the {THR_FLOOR} floor of the shared threshold grid (clipped, not an optimum)")

# --------------------------------------------------------------------------- 6a. figure: loss curves of the two full-training-set fits
lc_b, lc_o = [float(v) for v in clf_best.loss_curve_], [float(v) for v in clf_over.loss_curve_]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6)); fig.patch.set_facecolor(C_SURF)
a = ax[0]; style_axis(a)
a.plot(np.arange(1, len(lc_o) + 1), lc_o, color=C2, lw=2, label=f"over-parameterised {arch_over}, no early stopping")
a.plot(np.arange(1, len(lc_b) + 1), lc_b, color=C1, lw=2, label=f"tuned {arch_best}, early stopping")
a.plot([len(lc_b)], [lc_b[-1]], marker="o", ms=8, color=C1, mec="white", mew=1.2, zorder=5)
a.annotate(f"tuned net stops at epoch {len(lc_b)}\n(best validation epoch {best_epoch}, loss {lc_b[-1]:.3f})",
           (len(lc_b), lc_b[-1]), xytext=(0.36 * len(lc_o), lc_b[-1] - 0.02), textcoords="data", ha="left", va="top",
           fontsize=8.5, color=C_INK2, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.8, shrinkB=6))
a.annotate(f"{arch_over}\nloss {lc_o[-1]:.3f}", (len(lc_o), lc_o[-1]), xytext=(6, 0), textcoords="offset points", va="center",
           fontsize=8.5, color=C_INK2)
a.set_xlabel("Epoch (one pass over the training rows, mini-batches of 200)"); a.set_ylabel("Training log-loss")
a.set_title("Training loss per epoch (fits on the full training set)", fontsize=11)
a.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=C_INK2); a.margins(x=0.1)
a = ax[1]; style_axis(a)
ep = np.arange(1, len(vs_best) + 1); maj = float(1 - y_train.mean())
a.plot(ep, vs_best, color=C1, lw=2)
a.plot([best_epoch], [vs_best[best_epoch - 1]], marker="*", ms=14, color=C1, mec="white", mew=1.2, zorder=5)
a.axhline(maj, color=C_MUTED, ls="--", lw=1.1)
a.text(ep[0], maj, f"  majority-class accuracy {maj:.3f}", va="bottom", ha="left", fontsize=8.5, color=C_INK2)
top = max(vs_best) + 0.014; a.set_ylim(min(maj, min(vs_best)) - 0.004, top)      # head-room so the annotation sits above the curve
a.annotate(f"best epoch {best_epoch}: accuracy {vs_best[best_epoch - 1]:.4f}\n(weights restored to this epoch; training\n"
           f"halted {len(vs_best) - best_epoch} epochs later with no improvement)",
           (best_epoch, vs_best[best_epoch - 1]), xytext=(best_epoch + 2.5, top - 0.0015), textcoords="data", ha="left", va="top",
           fontsize=8.5, color=C_INK2, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.8, shrinkB=9))
a.set_xlabel("Epoch"); a.set_ylabel("Accuracy on the internal 15 % validation split")
a.set_title(f"Early-stopping criterion of the tuned network {arch_best}", fontsize=11); a.margins(x=0.06)
fig.suptitle("MLP loss curves (training set only; test set not involved)", fontsize=12.5, color=C_INK)
fig_path = save_extra_figure(fig, "mlp_loss_curve.png")
log(f"figure -> {fig_path}")

# --------------------------------------------------------------------------- 6b. diagnostic: AUC per epoch on a training-internal split
X_a, X_b, y_a, y_b = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
pre_d = make_preprocessor().fit(X_a); Xa, Xb = pre_d.transform(X_a), pre_d.transform(X_b)


def epoch_curve(hls, alpha, lr, n_epochs):
    """Train for n_epochs with partial_fit (no early stopping) and record train / held-out AUC after every epoch."""
    clf = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, learning_rate_init=lr, random_state=SEED, early_stopping=False)
    rows = []
    for e in range(1, n_epochs + 1):
        clf.partial_fit(Xa, y_a, classes=[0, 1])
        rows.append(dict(epoch=e, train_loss=float(clf.loss_), train_auc=float(roc_auc_score(y_a, clf.predict_proba(Xa)[:, 1])),
                         holdout_auc=float(roc_auc_score(y_b, clf.predict_proba(Xb)[:, 1]))))
    return pd.DataFrame(rows)


t0 = time.time()
curves = {"tuned": epoch_curve(best_hls, best_alpha, best_lr, OVERFIT_ITERS),
          "over": epoch_curve(OVERFIT_HLS, OVERFIT_ALPHA, OVERFIT_LR, OVERFIT_ITERS)}
log(f"epoch diagnostic (2 x {OVERFIT_ITERS} epochs on an 85/15 split of the training set) done in {time.time() - t0:.0f}s")
diag = pd.concat([curves["tuned"].assign(network=arch_best), curves["over"].assign(network=arch_over)], ignore_index=True)
diag_csv = os.path.join(RESULTS, "mlp_overfit_epochs.csv"); diag.to_csv(diag_csv, index=False, float_format="%.5f")
peak = {k: dict(epoch=int(d.epoch[d.holdout_auc.idxmax()]), holdout_auc=float(d.holdout_auc.max()),
                holdout_auc_final=float(d.holdout_auc.iloc[-1]), train_auc_final=float(d.train_auc.iloc[-1]))
        for k, d in curves.items()}
for k, v in peak.items():
    log(f"epoch diagnostic {k}: held-out AUC peaks at epoch {v['epoch']} ({v['holdout_auc']:.4f}), ends at {v['holdout_auc_final']:.4f}; "
        f"train AUC ends at {v['train_auc_final']:.4f}")

lo = min(d.holdout_auc.min() for d in curves.values()) - 0.05; hi = 1.005
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True); fig.patch.set_facecolor(C_SURF)
titles = {"tuned": f"Tuned network {arch_best} (alpha = {best_alpha:g}, {n_params(clf_best):,} weights)",
          "over": f"Over-parameterised {arch_over} (alpha = {OVERFIT_ALPHA:g}, {n_params(clf_over):,} weights)"}
ann_pos = {"tuned": (0.42, 0.12), "over": (0.40, 0.50)}       # empty region of each panel (axes fraction), checked visually
for a, key in zip(ax, ("tuned", "over")):
    d = curves[key]; style_axis(a)
    a.plot(d.epoch, d.train_auc, color=C1, lw=2, label="training rows (85 % of the training set)")
    a.plot(d.epoch, d.holdout_auc, color=C2, lw=2, label="held-out rows (15 % of the training set)")
    i = int(d.holdout_auc.idxmax())
    a.plot([d.epoch[i]], [d.holdout_auc[i]], marker="o", ms=8, color=C2, mec="white", mew=1.2, zorder=5)
    a.annotate(f"held-out AUC peaks at epoch {d.epoch[i]}: {d.holdout_auc[i]:.3f}\nafter {OVERFIT_ITERS} epochs: {d.holdout_auc.iloc[-1]:.3f}",
               (d.epoch[i], d.holdout_auc[i]), xytext=ann_pos[key], textcoords="axes fraction", ha="left", va="bottom",
               fontsize=8.5, color=C_INK2, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.8, shrinkB=6))
    a.annotate("train", (d.epoch.iloc[-1], d.train_auc.iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, color=C_INK2)
    a.annotate("held-out", (d.epoch.iloc[-1], d.holdout_auc.iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, color=C_INK2)
    a.set_ylim(lo, hi); a.set_xlabel("Epoch"); a.set_title(titles[key], fontsize=10.5); a.margins(x=0.1)
ax[0].set_ylabel("ROC AUC"); ax[0].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=C_INK2)
fig.suptitle(f"Overfitting with epochs: both networks trained {OVERFIT_ITERS} epochs without early stopping (training set only)",
             fontsize=12.5, color=C_INK)
fig_path = save_extra_figure(fig, "mlp_overfit_epochs.png")
log(f"figure -> {fig_path}")

# --------------------------------------------------------------------------- 5b. paired bootstrap of test AUC / AP differences
# Evaluation only: the saved test-set probabilities of already-fitted models are resampled together (same 8,990 rows for every
# model, from load_split()).  Nothing is fitted, tuned or selected here; this replaces the "about 1.4 fold-SD" heuristic with a
# proper paired interval, as the best-of-grid CV means of different models are not comparable in that way.
N_BOOT = 2000
p_test_mlp = np.load(os.path.join(RESULTS, "mlp_p_test.npy")); y_te = y_test.values
rng = np.random.default_rng(SEED); boot_idx = [rng.integers(0, len(y_te), len(y_te)) for _ in range(N_BOOT)]
paired = {}
t0 = time.time()
for other in ("logistic", "xgboost"):
    f = os.path.join(RESULTS, f"{other}_p_test.npy")
    if not os.path.exists(f):
        paired[other] = "not available (results/%s_p_test.npy missing at run time)" % other; continue
    p_o = np.load(f).astype(float)
    d_auc = np.array([roc_auc_score(y_te[i], p_test_mlp[i]) - roc_auc_score(y_te[i], p_o[i]) for i in boot_idx])
    d_ap = np.array([average_precision_score(y_te[i], p_test_mlp[i]) - average_precision_score(y_te[i], p_o[i]) for i in boot_idx])
    paired[other] = dict(
        delta_auc=round(float(roc_auc_score(y_te, p_test_mlp) - roc_auc_score(y_te, p_o)), 5),
        delta_auc_ci95=[round(float(v), 5) for v in np.percentile(d_auc, [2.5, 97.5])],
        delta_ap=round(float(average_precision_score(y_te, p_test_mlp) - average_precision_score(y_te, p_o)), 5),
        delta_ap_ci95=[round(float(v), 5) for v in np.percentile(d_ap, [2.5, 97.5])],
        share_resamples_delta_auc_le_0=round(float(np.mean(d_auc <= 0)), 4), share_resamples_delta_ap_le_0=round(float(np.mean(d_ap <= 0)), 4))
    log(f"paired bootstrap mlp - {other}: dAUC {paired[other]['delta_auc']:+.4f} CI95 {paired[other]['delta_auc_ci95']}, "
        f"dAP {paired[other]['delta_ap']:+.4f} CI95 {paired[other]['delta_ap_ci95']}")
res_best["extra"]["paired_bootstrap_test"] = dict(method=f"paired percentile bootstrap, {N_BOOT} resamples of the {len(y_te):,} test rows, "
                                                  f"seed {SEED}, delta = MLP (slug mlp) minus the other model, from results/<slug>_p_test.npy",
                                                  seconds=round(time.time() - t0, 1), **paired)

# store the diagnostic summary in the JSONs (re-write the files with the extra keys added above)
for r, key in ((res_over, "over"), (res_best, "tuned")):
    r["extra"]["epoch_diagnostic_train_internal_split"] = dict(split="85/15 stratified split of the training set, seed 2607526",
                                                              epochs=OVERFIT_ITERS, csv=os.path.relpath(diag_csv, ROOT), **peak[key])
    json.dump(r, open(os.path.join(RESULTS, f"{r['slug']}.json"), "w"), indent=1)

# --------------------------------------------------------------------------- summary (+ whichever other results already exist)
rows = []
for r in (res_best, res_w, res_over):
    rows.append(dict(model=r["name"], cv_auc=r["cv_auc"], train_auc=r["train_auc"], test_auc=r["test_auc"], test_ap=r["test_ap"],
                     brier=r["test_brier"], acc_0_5=r["at_0_5"]["accuracy"], recall_0_5=r["at_0_5"]["recall_1"],
                     thr_f1=r["at_tuned_f1"]["threshold"], recall_f1=r["at_tuned_f1"]["recall_1"], precision_f1=r["at_tuned_f1"]["precision_1"],
                     f1=r["at_tuned_f1"]["f1_1"], fit_s=r["fit_seconds"]))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------------------- 7. cross-model comparison from the results/ files present now
COMPARE = ["xgboost", "gradient_boosting", "random_forest", "bagging", "mlp", "mlp_weighted", "knn", "tree_pruned", "logistic", "mlp_overfit"]
allres = {r["slug"]: r for r in load_all_results() if "test_auc" in r}
allres["mlp"], allres["mlp_weighted"], allres["mlp_overfit"] = res_best, res_w, res_over   # this run's numbers (= the files just written)


def _f(v):
    return float("nan") if v is None else float(v)


comp = pd.DataFrame([dict(slug=s, name=allres[s]["name"], member=allres[s].get("member"), cv_auc=_f(allres[s].get("cv_auc")),
                          train_auc=_f(allres[s]["train_auc"]), test_auc=_f(allres[s]["test_auc"]), test_ap=_f(allres[s]["test_ap"]),
                          test_brier=_f(allres[s]["test_brier"]), thr_f1=_f(allres[s]["at_tuned_f1"]["threshold"]),
                          f1_tuned=_f(allres[s]["at_tuned_f1"]["f1_1"]),
                          json_written=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(RESULTS, f"{s}.json")))))
                     for s in COMPARE if s in allres]).sort_values("test_ap", ascending=False).reset_index(drop=True)
comp_csv = os.path.join(RESULTS, "mlp_comparison.csv"); comp.to_csv(comp_csv, index=False, float_format="%.4f")
missing = [s for s in COMPARE if s not in allres]
print(f"\nCross-model comparison (test set, same split, sorted by test AP; thresholds tuned on training OOF probabilities) -> {comp_csv}"
      + (f"   [not present yet: {missing}]" if missing else ""))
print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nMarkdown for report/notes_mlp.md:")
print("| Model (slug) | CV AUC | Test AUC | Test AP | Brier | F1 @tuned thr |")
print("|---|---|---|---|---|---|")
for _, r in comp.iterrows():
    print(f"| {r['name']} (`{r['slug']}`) | {r.cv_auc:.4f} | {r.test_auc:.4f} | {r.test_ap:.4f} | {r.test_brier:.4f} | {r.f1_tuned:.4f} |")
print("AP ranking: " + " > ".join(comp.slug))
total_seconds = round(time.time() - T_START, 1)
for r in (res_best, res_w, res_over):                                       # record this run's wall-clock in every JSON (metrics unchanged)
    r["extra"]["total_script_seconds"] = total_seconds
    json.dump(r, open(os.path.join(RESULTS, f"{r['slug']}.json"), "w"), indent=1)
log(f"done in {total_seconds:.0f}s.")
