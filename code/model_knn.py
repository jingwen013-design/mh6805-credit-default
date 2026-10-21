"""
model_knn.py — K-nearest neighbours (the course benchmark) for credit-card default prediction.
Owner: RUAN Hangjing.   Run from Project/code:  python3 model_knn.py

What this script does
  1. load_split()  -> the shared stratified 70/30 split (seed 2607526); the test set is touched only inside evaluate().
  2. Pipeline(make_preprocessor(), KNeighborsClassifier): StandardScaler + one-hot are refitted inside every CV fold.
  3. GridSearchCV over K in 12 odd values 5..151 x weights in {uniform, distance}, cv=CV, scoring=SCORING (roc_auc), n_jobs=2.
  4. Figure figures/knn_cv_auc_vs_k.png: mean CV AUC (+/- 1 sd across folds) vs K for both weightings.
  5. evaluate() the tuned model as "KNN" (slug knn) and the untuned course default K=5/uniform as "KNN (K=5)" (slug knn_k5).
  6. Two cheap diagnostics on the TRAINING set only (recorded in extra, never used for model selection):
       - the same tuned K/weights WITHOUT StandardScaler (make_preprocessor(scale=False)) -> shows why scaling is necessary;
       - prediction-time benchmark (KNN vs. a logistic regression fitted on the same pipeline) -> shows the lazy-learner cost.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS/OpenMP threads within the n_jobs<=2 budget
    os.environ.setdefault(_v, "2")
import sys, time, json
import numpy as np
import pandas as pd
from common import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.base import clone

MEMBER = "RUAN Hangjing"
K_GRID = [5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101, 151]          # 12 odd values (odd -> no tied votes with uniform weights)
WEIGHTS = ["uniform", "distance"]

def log(msg):
    print(f"[knn] {msg}", flush=True)

# --------------------------------------------------------------------------- 1. data
X_train, X_test, y_train, y_test, feature_names = load_split()
log(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")

# --------------------------------------------------------------------------- 2. pipeline + 3. grid search (CV on training set only)
pipe = Pipeline([("pre", make_preprocessor()),                    # StandardScaler on 24 numeric, one-hot on SEX/EDUCATION/MARRIAGE
                 ("clf", KNeighborsClassifier(algorithm="brute"))])  # 32 features -> sklearn would pick brute anyway; make it explicit
param_grid = {"clf__n_neighbors": K_GRID, "clf__weights": WEIGHTS}
gs = GridSearchCV(pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=2, refit=True, return_train_score=False, verbose=0)
t0 = time.time(); gs.fit(X_train, y_train); grid_seconds = time.time() - t0
log(f"grid search over {len(K_GRID) * len(WEIGHTS)} settings x {CV.get_n_splits()} folds done in {grid_seconds:.1f}s")

cvr = pd.DataFrame(gs.cv_results_)
cv_table = (cvr[["param_clf__n_neighbors", "param_clf__weights", "mean_test_score", "std_test_score", "rank_test_score",
                 "mean_fit_time", "mean_score_time"]]
            .rename(columns={"param_clf__n_neighbors": "K", "param_clf__weights": "weights", "mean_test_score": "cv_auc_mean",
                             "std_test_score": "cv_auc_std", "rank_test_score": "rank"})
            .sort_values(["weights", "K"]).reset_index(drop=True))
cv_table["K"] = cv_table["K"].astype(int)
print(cv_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_K = int(gs.best_params_["clf__n_neighbors"]); best_w = gs.best_params_["clf__weights"]
best_row = cv_table[(cv_table.K == best_K) & (cv_table.weights == best_w)].iloc[0]
cv_auc_best, cv_auc_std_best = float(best_row.cv_auc_mean), float(best_row.cv_auc_std)
log(f"best: K={best_K}, weights={best_w}, CV AUC {cv_auc_best:.4f} +/- {cv_auc_std_best:.4f}")

# One-standard-error view: smallest K whose CV AUC is within 1 sd of the best (reported only, best-by-AUC is what we ship).
thr_1se = cv_auc_best - cv_auc_std_best
within = cv_table[cv_table.cv_auc_mean >= thr_1se].sort_values("K")
k_1se, w_1se = int(within.iloc[0].K), within.iloc[0].weights

# --------------------------------------------------------------------------- 4. figure: CV AUC vs K
COL = {"uniform": "#2a78d6", "distance": "#eb6834"}                # validated categorical slots 1 and 2 (light surface)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for w in WEIGHTS:
    sub = cv_table[cv_table.weights == w]
    ax.fill_between(sub.K, sub.cv_auc_mean - sub.cv_auc_std, sub.cv_auc_mean + sub.cv_auc_std, color=COL[w], alpha=0.12, lw=0)
    ax.plot(sub.K, sub.cv_auc_mean, color=COL[w], lw=2, marker="o", ms=5, label=f"weights = {w}")
    # direct label at the curve end; the two curves end within 0.001 AUC of each other, so stagger them vertically
    ax.annotate(w, (sub.K.iloc[-1], sub.cv_auc_mean.iloc[-1]), xytext=(7, 6 if w == "uniform" else -7),
                textcoords="offset points", va="center", fontsize=9, color="#52514e")
ax.axhline(0.760, color="#8a8984", ls="--", lw=1.2)
ax.text(K_GRID[0], 0.760, "  logistic regression baseline (CV AUC ~ 0.760)", va="bottom", ha="left", fontsize=8.5, color="#52514e")
ax.plot([best_K], [cv_auc_best], marker="*", ms=15, color=COL[best_w], mec="white", mew=1.2, zorder=5)
lo, hi = cv_table.cv_auc_mean.min() - 0.012, cv_table.cv_auc_mean.max() + 0.014
ax.set_ylim(lo, hi)
# annotation text goes in the empty upper-left region (small K, high AUC) with a leader line to the star
ax.annotate(f"best: K = {best_K}, {best_w}\nCV AUC = {cv_auc_best:.4f}", (best_K, cv_auc_best),
            xytext=(K_GRID[3], hi - 0.004), textcoords="data", ha="center", va="top", fontsize=9, color="#0b0b0b",
            arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8, shrinkB=8))
ax.set_xscale("log"); ax.set_xticks(K_GRID); ax.set_xticklabels([str(k) for k in K_GRID], fontsize=8.5)
ax.minorticks_off()
ax.set_xlabel("K (number of neighbours, log scale)"); ax.set_ylabel("5-fold CV ROC AUC (training set)")
ax.set_title("KNN: cross-validated AUC vs K (band = +/- 1 sd across folds)", fontsize=11)
ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.margins(x=0.06)
fig_path = save_extra_figure(fig, "knn_cv_auc_vs_k.png")
log(f"figure -> {fig_path}")

# --------------------------------------------------------------------------- 6a. diagnostic: is scaling necessary? (training CV only)
t0 = time.time()
pipe_unscaled = Pipeline([("pre", make_preprocessor(scale=False)),
                          ("clf", KNeighborsClassifier(n_neighbors=best_K, weights=best_w, algorithm="brute"))])
auc_unscaled = cross_val_score(pipe_unscaled, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=2)
log(f"ablation, same K/weights without StandardScaler: CV AUC {auc_unscaled.mean():.4f} +/- {auc_unscaled.std():.4f} "
    f"({time.time() - t0:.1f}s)")

# --------------------------------------------------------------------------- 5. final models
best_model = gs.best_estimator_                                    # already refitted by GridSearchCV on the full training set
t0 = time.time(); refit_model = clone(best_model).fit(X_train, y_train); refit_seconds = time.time() - t0   # time the fit alone
n_feat = len(feature_names_out(best_model.named_steps["pre"]))

def time_predict(model, X, reps=3):
    """Median wall-clock seconds of predict_proba over `reps` runs (first run warms caches)."""
    ts = []
    for _ in range(reps):
        t0 = time.time(); model.predict_proba(X); ts.append(time.time() - t0)
    return float(np.median(ts))

pred_sec_best = time_predict(best_model, X_test)
log(f"tuned KNN: predict_proba on {len(X_test):,} test rows = {pred_sec_best:.3f}s ({len(X_test) / pred_sec_best:,.0f} rows/s)")

# Reference for the lazy-learner argument: logistic regression on the identical preprocessing (timed only, NOT evaluated here).
logit = Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
t0 = time.time(); logit.fit(X_train, y_train); logit_fit_sec = time.time() - t0
pred_sec_logit = time_predict(logit, X_test)
log(f"reference logistic: fit {logit_fit_sec:.3f}s, predict {pred_sec_logit:.4f}s -> KNN predict is {pred_sec_best / pred_sec_logit:,.0f}x slower")

# Training out-of-fold probabilities for the threshold tuning inside evaluate() (training set only).
t0 = time.time()
p_oof_best = cross_val_predict(clone(best_model), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
oof_seconds = time.time() - t0

# Boundary check: is the AUC curve still rising at the edge of the grid? (diagnostic only, not used for selection)
boundary = {}
if best_K == K_GRID[-1]:
    for k in (201, 301):
        s = cross_val_score(Pipeline([("pre", make_preprocessor()),
                                      ("clf", KNeighborsClassifier(n_neighbors=k, weights=best_w, algorithm="brute"))]),
                            X_train, y_train, cv=CV, scoring=SCORING, n_jobs=2)
        boundary[str(k)] = dict(cv_auc_mean=float(s.mean()), cv_auc_std=float(s.std()))
    log(f"best K sits on the grid boundary; post-hoc CV AUC at K=201/301 ({best_w}): " +
        ", ".join(f"{k}: {v['cv_auc_mean']:.4f}" for k, v in boundary.items()))

extra_best = dict(
    grid=dict(K=K_GRID, weights=WEIGHTS, n_settings=len(K_GRID) * len(WEIGHTS), n_folds=CV.get_n_splits(),
              grid_search_seconds=round(grid_seconds, 1)),
    cv_table=cv_table.round(5).to_dict(orient="list"),
    one_se_rule=dict(K=k_1se, weights=w_1se, threshold_cv_auc=round(thr_1se, 5)),
    n_features_after_preprocessing=n_feat,
    stored_training_matrix=dict(rows=int(len(X_train)), cols=n_feat, megabytes=round(len(X_train) * n_feat * 8 / 1e6, 2)),
    neighbour_search_algorithm=best_model.named_steps["clf"]._fit_method,
    predict_seconds_test=round(pred_sec_best, 4), predict_rows_per_second=round(len(X_test) / pred_sec_best, 0),
    reference_logistic=dict(fit_seconds=round(logit_fit_sec, 4), predict_seconds_test=round(pred_sec_logit, 5),
                            knn_predict_slowdown_factor=round(pred_sec_best / pred_sec_logit, 1)),
    oof_predict_seconds=round(oof_seconds, 1),
    ablation_unscaled_same_K=dict(cv_auc_mean=float(auc_unscaled.mean()), cv_auc_std=float(auc_unscaled.std()),
                                  cv_auc_drop_vs_scaled=round(cv_auc_best - float(auc_unscaled.mean()), 4)),
    boundary_check=boundary,
    threads=dict(gridsearch_n_jobs=2, blas_threads=os.environ.get("OMP_NUM_THREADS")),
)
notes_best = (f"GridSearchCV over K in {K_GRID} x weights in {WEIGHTS} ({len(K_GRID) * len(WEIGHTS)} settings, 5-fold stratified CV, "
              f"scoring=roc_auc, seed {SEED}); grid took {grid_seconds:.0f}s so no subsampling was needed. Preprocessing "
              f"(StandardScaler + one-hot, {n_feat} columns) sits inside the Pipeline and is refitted per fold. Best = K={best_K}, "
              f"weights={best_w}. Same setting without scaling gives CV AUC {auc_unscaled.mean():.4f}. Test-set prediction takes "
              f"{pred_sec_best:.2f}s for {len(X_test):,} rows (brute-force search over {len(X_train):,} stored training rows) vs "
              f"{pred_sec_logit:.4f}s for logistic regression on the same pipeline.")
res_best = evaluate("KNN", best_model, X_train, y_train, X_test, y_test, member=MEMBER, slug="knn",
                    best_params=dict(n_neighbors=best_K, weights=best_w, algorithm="brute", metric="minkowski (p=2, euclidean)"),
                    cv_auc=cv_auc_best, cv_auc_std=cv_auc_std_best, notes=notes_best,
                    fit_seconds=round(refit_seconds, 3), p_oof=p_oof_best, extra=extra_best)
log(f"KNN (tuned): test AUC {res_best['test_auc']:.4f}, AP {res_best['test_ap']:.4f}, OOF AUC {res_best['oof_auc']:.4f}, "
    f"train AUC {res_best['train_auc']:.4f}")

# ---- untuned course default K=5, uniform (shows the effect of tuning). CV AUC taken from the same grid (no extra fitting on test).
k5_row = cv_table[(cv_table.K == 5) & (cv_table.weights == "uniform")].iloc[0]
pipe_k5 = Pipeline([("pre", make_preprocessor()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="brute"))])
t0 = time.time(); pipe_k5.fit(X_train, y_train); k5_fit_seconds = time.time() - t0
pred_sec_k5 = time_predict(pipe_k5, X_test)
p_oof_k5 = cross_val_predict(clone(pipe_k5), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
extra_k5 = dict(predict_seconds_test=round(pred_sec_k5, 4), predict_rows_per_second=round(len(X_test) / pred_sec_k5, 0),
                n_features_after_preprocessing=n_feat, neighbour_search_algorithm=pipe_k5.named_steps["clf"]._fit_method,
                cv_rank_in_grid=int(k5_row["rank"]), tuned_reference=dict(K=best_K, weights=best_w, cv_auc=cv_auc_best),
                note="Probabilities are multiples of 0.2 (5 votes), so the ROC/PR curves have only 6 operating points.")
res_k5 = evaluate("KNN (K=5)", pipe_k5, X_train, y_train, X_test, y_test, member=MEMBER, slug="knn_k5",
                  best_params=dict(n_neighbors=5, weights="uniform", algorithm="brute", metric="minkowski (p=2, euclidean)"),
                  cv_auc=float(k5_row.cv_auc_mean), cv_auc_std=float(k5_row.cv_auc_std),
                  notes=("Untuned course default (K=5, uniform votes) on the same scaled pipeline, shown to quantify the effect of "
                         f"tuning K. Its CV AUC comes from the same grid search (rank {int(k5_row['rank'])} of {len(cv_table)})."),
                  fit_seconds=round(k5_fit_seconds, 3), p_oof=p_oof_k5, extra=extra_k5)
log(f"KNN (K=5): test AUC {res_k5['test_auc']:.4f}, AP {res_k5['test_ap']:.4f}, OOF AUC {res_k5['oof_auc']:.4f}")

# --------------------------------------------------------------------------- summary
summary = pd.DataFrame([
    dict(model=r["name"], cv_auc=r["cv_auc"], test_auc=r["test_auc"], test_ap=r["test_ap"], test_brier=r["test_brier"],
         acc_0_5=r["at_0_5"]["accuracy"], recall_0_5=r["at_0_5"]["recall_1"], precision_0_5=r["at_0_5"]["precision_1"],
         thr_f1=r["at_tuned_f1"]["threshold"], recall_f1=r["at_tuned_f1"]["recall_1"], precision_f1=r["at_tuned_f1"]["precision_1"],
         f1=r["at_tuned_f1"]["f1_1"], predict_s=r["extra"]["predict_seconds_test"])
    for r in (res_best, res_k5)])
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("done.")
