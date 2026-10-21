"""
tuning_checks.py — checks of the tuning PROCEDURE behind the frozen models (audit findings a2-tuning-data-heterogeneity,
a5-svm-selection-statistic, a6-grid-boundaries, h1-knn-dimensionality, and the note that the earlier KNN nested CV used a
smaller grid than the frozen model_knn.py).  Added 2026-10-21 (post-freeze sensitivity analysis).

TRAINING ROWS ONLY.  The script reads data/processed/train.csv directly (asserting 20,975 rows) and the frozen
training out-of-fold vectors results/<slug>_p_oof.npy.  It never opens the held-out file, never calls common.load_split()
or common.evaluate(), never refits or overwrites a frozen model, and scores nothing outside the five training folds.
Every result below is a diagnostic: no frozen configuration is changed and nothing found here is scored on held-out rows.

Folds: common.CV (StratifiedKFold(5, shuffle=True, random_state=2607526)) on the training rows in file order; the
assignment is asserted to equal data/processed/cv_folds.csv.  Seed 2607526 everywhere (inner folds of the nested CV use
seed + 1, as nested_cv_check.py does).  n_jobs <= 2 throughout; BLAS/OpenMP threads capped at 2.

Parts (run all with `python3 tuning_checks.py`, or a subset, e.g. `python3 tuning_checks.py c e`):
  a  Gradient boosting: the frozen 12-setting grid (n_estimators {200, 500} x learning_rate {0.03, 0.1} x max_depth
     {2, 3, 4}, subsample 0.8) re-run on the FULL training set with the five project folds instead of the stratified 50 %
     subsample with its own folds.  Same staged shortcut as model_boosting.py (one 500-tree fit per fold and
     (lr, depth), AUC read after 200 and 500 trees).                       -> results/gbm_grid_fulltrain.csv
  b  Nested CV for KNN with EXACTLY the frozen grid of model_knn.py (12 K x 2 weights = 24 settings, brute-force
     search): outer = project folds, inner = 5-fold StratifiedKFold(seed + 1) on each outer training part.
                                                                            -> results/nested_cv_knn_frozen_grid.json
  c  KNN dimensionality: KNN K = 101 (uniform) on (i) the frozen 32 encoded columns, (ii) the first 5 / 10 / 15
     principal components fitted inside each fold, (iii) the 8 columns with the highest training mutual information,
     selected inside each fold (integer-coded columns scored as discrete; a robustness row repeats the selection with
     scikit-learn's default all-continuous estimator).                      -> results/knn_dimensionality_cv.csv
  d  Grid boundaries: for each headline model, whether the chosen value of each tuned hyper-parameter sits on the edge
     of its grid; for XGBoost a small extension (max_depth {2, 3} x learning_rate {0.01, 0.02, 0.03} x rounds 150-600,
     other settings frozen: min_child_weight 5, subsample 0.8, colsample_bytree 0.8); rows for 650-1,200 rounds at
     learning rates 0.01 / 0.02 are added as a flagged diagnostic beyond the requested range.
                                                                            -> results/grid_edge_check.csv
  e  RBF SVM selection statistic: 6,000-row subsample CV AUC vs pooled OOF AUC vs mean of the per-fold OOF AUCs.
                                                                            -> results/svm_selection_statistics.json
  A run-level summary of all parts is merged into results/tuning_checks_summary.json.

Paired comparisons over the five folds report the per-fold values, the mean difference, the number of folds in which
the difference is positive and a Nadeau-Bengio corrected t interval (mine_v2_utils.corrected_t_ci with the
held-out-to-fitting ratio 1/4 of one 5-fold CV).

Decision rules written before the runs:
  * GBM (a): the frozen GBM (200 trees, lr 0.03, depth 3) is kept whatever the full-set grid picks; a different winner
    is reported with its CV only.
  * KNN (b, c): the frozen K = 101 on 32 columns stays.  A variant is flagged only if its mean paired gain exceeds one
    corrected standard error (t > 1); even then it is not adopted or scored now.
  * XGBoost (d): a probe setting is flagged if it beats the frozen CV AUC by more than one fold-SD of the frozen model
    (0.00682, ddof = 0 as in results/xgboost.json); a flagged setting is a limitation to report, never a new model.

Run time (MacBook, 2 workers, machine shared with other jobs; 21 Oct 2026 run): a about 7 min (431 s), b about 45 s,
c about 17 s, d about 31 s, e under 1 s; about 9 min (523 s) in all.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys, json, re, time, itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from common import CV, SEED, ROOT, RESULTS, TARGET, Pipeline, make_preprocessor, roc_auc_score
from mine_v2_utils import corrected_t_ci

TRAIN_CSV = os.path.join(ROOT, "data", "processed", "train.csv")
FOLDS_CSV = os.path.join(ROOT, "data", "processed", "cv_folds.csv")
N_ROWS = 20975
N_JOBS = 2
NB_RATIO = 1.0 / 4.0          # held-out / fitting rows in one 5-fold CV (Nadeau-Bengio correction)
SUMMARY = os.path.join(RESULTS, "tuning_checks_summary.json")
HEADLINE = ["knn", "logistic", "logistic_l1", "lda", "naive_bayes", "tree_pruned", "random_forest",
            "gradient_boosting", "xgboost", "svm_rbf", "mlp"]           # as in aggregate.py
T0 = time.time()


def log(msg):
    print(f"[tuning_checks {time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------- shared helpers
def load_training():
    tr = pd.read_csv(TRAIN_CSV)
    assert len(tr) == N_ROWS, f"expected {N_ROWS:,} training rows, found {len(tr):,}"
    X, y = tr.drop(columns=[TARGET]), tr[TARGET].astype(int)
    folds = list(CV.split(np.zeros(len(y)), y))
    fid = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(folds, start=1):
        fid[va] = k
    cvf = pd.read_csv(FOLDS_CSV)
    assert len(cvf) == N_ROWS and (cvf["fold"].values == fid).all() and (cvf["DEFAULT"].values == y.values).all(), \
        "common.CV folds differ from data/processed/cv_folds.csv"
    return X, y, folds


def frozen(slug):
    return json.load(open(os.path.join(RESULTS, f"{slug}.json")))


def frozen_oof(slug):
    p = np.load(os.path.join(RESULTS, f"{slug}_p_oof.npy"))
    assert len(p) == N_ROWS
    return p


def fold_aucs(y, p, folds):
    y = np.asarray(y)
    return np.array([roc_auc_score(y[va], p[va]) for _, va in folds])


def paired(a, b, name_a, name_b):
    """Fold-paired comparison of per-fold AUCs, b minus a, with the Nadeau-Bengio corrected t interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ci = corrected_t_ci(d, n_train=4, n_test=1)             # n_test / n_train = 1/4 for one 5-fold CV
    se = float(np.sqrt((1.0 / len(d) + NB_RATIO) * d.var(ddof=1)))
    return dict(comparison=f"{name_b} minus {name_a}", per_fold_a=np.round(a, 5).tolist(), per_fold_b=np.round(b, 5).tolist(),
                per_fold_diff=np.round(d, 5).tolist(), mean_diff=float(d.mean()), n_folds_positive=int((d > 0).sum()),
                n_folds_exactly_zero=int((np.abs(d) < 1e-12).sum()),
                nb_corrected_se=se, nb_t=float(d.mean() / se) if se > 0 else None,
                nb_ci95=[ci["lo"], ci["hi"]], ci_includes_zero=bool(ci["lo"] <= 0 <= ci["hi"]))


def update_summary(part, content):
    s = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    s.setdefault("generated_by", "code/tuning_checks.py")
    s["uses_heldout_rows"] = False
    s["seed"] = SEED
    s[part] = content
    s.setdefault("elapsed_seconds_of_last_run", {})[part] = round(time.time() - T0, 1)
    json.dump(s, open(SUMMARY, "w"), indent=1)


def knn_pipe(steps=(), k=101, weights="uniform"):
    return Pipeline([("pre", make_preprocessor()), *steps,
                     ("clf", KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm="brute"))])


# ======================================================================================== (a) GBM grid on the full set
GBM_LR, GBM_DEPTH, GBM_NTREES = [0.03, 0.1], [2, 3, 4], [200, 500]
GBM_STAGES = list(range(10, 501, 10))
GBM_FROZEN = dict(learning_rate=0.03, max_depth=3, n_estimators=200)


def gbm_job(lr, depth, f, tr, va, X, y):
    t0 = time.time()
    pipe = Pipeline([("pre", make_preprocessor(scale=False)),
                     ("clf", GradientBoostingClassifier(subsample=0.8, random_state=SEED, n_estimators=max(GBM_NTREES),
                                                        learning_rate=lr, max_depth=depth))]).fit(X.iloc[tr], y.iloc[tr])
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xva, Xtr = pre.transform(X.iloc[va]), pre.transform(X.iloc[tr])
    yva, ytr = y.values[va], y.values[tr]
    val, p_keep = {}, {}
    for k, p in enumerate(clf.staged_predict_proba(Xva), start=1):
        if k in GBM_STAGES:
            val[k] = roc_auc_score(yva, p[:, 1])
        if k in GBM_NTREES:
            p_keep[k] = p[:, 1].copy()
    trn = {}
    for k, p in enumerate(clf.staged_predict_proba(Xtr), start=1):
        if k in GBM_NTREES:
            trn[k] = roc_auc_score(ytr, p[:, 1])
    return dict(lr=lr, depth=depth, fold=f, seconds=time.time() - t0, val=val, train=trn, p=p_keep)


def part_a(X, y, folds):
    log("(a) GBM: 12-setting grid on the full training set, project folds (6 (lr, depth) x 5 folds = 30 fits of 500 trees)")
    jobs = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(gbm_job)(lr, d, f, tr, va, X, y)
                                              for lr, d in itertools.product(GBM_LR, GBM_DEPTH) for f, (tr, va) in enumerate(folds))
    sub = pd.read_csv(os.path.join(RESULTS, "gbm_grid.csv"))
    gbj = frozen("gradient_boosting")
    rows, curve = [], []
    for lr, d in itertools.product(GBM_LR, GBM_DEPTH):
        js = sorted([j for j in jobs if j["lr"] == lr and j["depth"] == d], key=lambda j: j["fold"])
        for k in GBM_STAGES:
            v = np.array([j["val"][k] for j in js])
            curve.append(dict(learning_rate=lr, max_depth=d, n_estimators=k, cv_auc=v.mean()))
        for n in GBM_NTREES:
            v = np.array([j["val"][n] for j in js]); t = np.array([j["train"][n] for j in js])
            s = sub[(sub.learning_rate == lr) & (sub.max_depth == d) & (sub.n_estimators == n)].iloc[0]
            rows.append(dict(learning_rate=lr, max_depth=d, n_estimators=n, cv_auc_fulltrain=v.mean(), cv_auc_std_fulltrain=v.std(),
                             cv_auc_sd_ddof1_fulltrain=v.std(ddof=1),
                             **{f"fold{i + 1}_auc": v[i] for i in range(5)}, train_auc_fulltrain=t.mean(),
                             cv_auc_subsample=float(s.cv_auc), cv_auc_std_subsample=float(s.cv_auc_std), rank_subsample=int(s["rank"]),
                             is_frozen_setting=bool(lr == GBM_FROZEN["learning_rate"] and d == GBM_FROZEN["max_depth"]
                                                    and n == GBM_FROZEN["n_estimators"]),
                             fit_seconds_500=float(np.mean([j["seconds"] for j in js]))))
    g = pd.DataFrame(rows).sort_values("cv_auc_fulltrain", ascending=False).reset_index(drop=True)
    g.insert(g.columns.get_loc("cv_auc_subsample"), "rank_fulltrain", np.arange(1, len(g) + 1))
    g.to_csv(os.path.join(RESULTS, "gbm_grid_fulltrain.csv"), index=False)

    # reproduction check: the frozen setting's 200-tree OOF from the staged fits vs results/gradient_boosting_p_oof.npy
    p_fz = np.zeros(N_ROWS)
    for j in jobs:
        if j["lr"] == GBM_FROZEN["learning_rate"] and j["depth"] == GBM_FROZEN["max_depth"]:
            p_fz[folds[j["fold"]][1]] = j["p"][GBM_FROZEN["n_estimators"]]
    p_frozen = frozen_oof("gradient_boosting")
    fz_row = g[g.is_frozen_setting].iloc[0]
    win = g.iloc[0]
    same = bool(win.is_frozen_setting)
    fold_cols = [f"fold{i + 1}_auc" for i in range(5)]
    cmp_ = None if same else paired(fz_row[fold_cols].values, win[fold_cols].values, "frozen GBM setting (200, 0.03, 3)",
                                     f"full-set winner ({int(win.n_estimators)}, {win.learning_rate}, {int(win.max_depth)})")
    xgb_folds = fold_aucs(y, frozen_oof("xgboost"), folds)
    xgb_vs = dict(xgb_minus_frozen_gbm=paired(fz_row[fold_cols].values, xgb_folds, "frozen GBM (200, 0.03, 3)", "frozen XGBoost"),
                  xgb_minus_fulltrain_gbm_winner=paired(win[fold_cols].values, xgb_folds,
                                                        f"full-set GBM winner ({int(win.n_estimators)}, {win.learning_rate}, "
                                                        f"{int(win.max_depth)})", "frozen XGBoost"),
                  note=("With the GBM grid also scored on the project folds, both boosters' CV AUCs are maxima over their own grids "
                        "on the same folds, so this is the like-for-like comparison of the two tuning procedures."))
    cv = pd.DataFrame(curve); opt = cv.loc[cv.cv_auc.idxmax()]
    rho = float(spearmanr(g.cv_auc_fulltrain, g.cv_auc_subsample)[0])
    out = dict(
        grid="n_estimators {200, 500} x learning_rate {0.03, 0.1} x max_depth {2, 3, 4}, subsample 0.8, random_state 2607526",
        tuning_rows=int(N_ROWS), folds="project folds (common.CV)",
        winner_fulltrain=dict(learning_rate=float(win.learning_rate), max_depth=int(win.max_depth), n_estimators=int(win.n_estimators),
                              cv_auc=float(win.cv_auc_fulltrain), cv_auc_std=float(win.cv_auc_std_fulltrain)),
        winner_subsample=dict(GBM_FROZEN, cv_auc_subsample=float(sub.iloc[0].cv_auc)),
        same_setting_wins=same,
        frozen_setting_fulltrain=dict(cv_auc=float(fz_row.cv_auc_fulltrain), cv_auc_std=float(fz_row.cv_auc_std_fulltrain),
                                      rank_fulltrain=int(fz_row.rank_fulltrain), frozen_json_cv_auc=gbj["cv_auc"]),
        reproduction=dict(max_abs_diff_p_oof_vs_frozen=float(np.max(np.abs(p_fz - p_frozen))),
                          frozen_fold_aucs_from_p_oof=np.round(fold_aucs(y, p_frozen, folds), 5).tolist()),
        winner_vs_frozen=cmp_,
        xgboost_comparison=xgb_vs,
        decision="frozen GBM (200 trees, lr 0.03, depth 3) kept by the rule fixed beforehand; the full-set winner is reported with its CV only",
        spearman_fulltrain_vs_subsample_cv=rho,
        staged_optimum_fulltrain=dict(learning_rate=float(opt.learning_rate), max_depth=int(opt.max_depth),
                                      n_estimators=int(opt.n_estimators), cv_auc=float(opt.cv_auc)),
        all_12=g[["learning_rate", "max_depth", "n_estimators", "cv_auc_fulltrain", "cv_auc_std_fulltrain", "rank_fulltrain",
                  "cv_auc_subsample", "rank_subsample"]].round(5).to_dict(orient="records"),
        note=("Diagnostic only: the frozen GBM is kept and nothing is re-scored.  A full-set re-grid on the project folds gives "
              "the GBM the same max-over-grid optimism as XGBoost's reported CV AUC."))
    update_summary("a_gbm_grid_fulltrain", out)
    log(f"(a) done; winner {out['winner_fulltrain']} (same as frozen: {same}); frozen setting full-set CV "
        f"{fz_row.cv_auc_fulltrain:.5f} (JSON {gbj['cv_auc']:.5f}); max |p_oof diff| {out['reproduction']['max_abs_diff_p_oof_vs_frozen']:.2e}")
    print(g[["learning_rate", "max_depth", "n_estimators", "cv_auc_fulltrain", "cv_auc_std_fulltrain", "rank_fulltrain",
             "cv_auc_subsample", "rank_subsample"]].round(5).to_string(index=False))


# ======================================================================================= (b) nested CV, frozen KNN grid
def frozen_knn_grid():
    src = open(os.path.join(ROOT, "code", "model_knn.py")).read()
    k_code = [int(v) for v in re.search(r"^K_GRID\s*=\s*\[([^\]]*)\]", src, re.M).group(1).split(",")]
    w_code = re.findall(r'"(\w+)"', re.search(r"^WEIGHTS\s*=\s*\[([^\]]*)\]", src, re.M).group(1))
    g = frozen("knn")["extra"]["grid"]
    assert k_code == g["K"] and w_code == g["weights"], "model_knn.py grid differs from results/knn.json"
    return k_code, w_code


def part_b(X, y, folds):
    K_GRID, W_GRID = frozen_knn_grid()
    log(f"(b) nested CV for KNN with the frozen grid: K {K_GRID} x weights {W_GRID} = {len(K_GRID) * len(W_GRID)} settings")
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    grid = {"clf__n_neighbors": K_GRID, "clf__weights": W_GRID}
    p_nested = np.zeros(N_ROWS)
    outer = []
    for k, (tr, va) in enumerate(folds):
        gs = GridSearchCV(knn_pipe(), grid, cv=inner, scoring="roc_auc", n_jobs=N_JOBS, refit=True)
        gs.fit(X.iloc[tr], y.iloc[tr])
        p_nested[va] = gs.predict_proba(X.iloc[va])[:, 1]
        r = pd.DataFrame(gs.cv_results_)
        inner_score = r["mean_test_score"]                         # inner validation folds of the outer training part
        i101 = int(r[(r.param_clf__n_neighbors == 101) & (r.param_clf__weights == "uniform")].index[0])
        top3 = r.sort_values("rank_test_score").head(3)
        outer.append(dict(outer_fold=k + 1, chosen_K=int(gs.best_params_["clf__n_neighbors"]), chosen_weights=gs.best_params_["clf__weights"],
                          inner_cv_auc_chosen=float(gs.best_score_), inner_cv_auc_K101_uniform=float(inner_score[i101]),
                          inner_top3=[dict(K=int(a), weights=b, inner_cv_auc=round(float(c), 5)) for a, b, c in
                                      zip(top3.param_clf__n_neighbors, top3.param_clf__weights, top3.mean_test_score)],
                          outer_auc=float(roc_auc_score(y.values[va], p_nested[va]))))
        log(f"(b) outer fold {k + 1}: chose K={outer[-1]['chosen_K']}, {outer[-1]['chosen_weights']}; outer AUC {outer[-1]['outer_auc']:.5f}")
    oa = np.array([o["outer_auc"] for o in outer])
    knn = frozen("knn")
    plain = fold_aucs(y, frozen_oof("knn"), folds)
    old = json.load(open(os.path.join(RESULTS, "nested_cv_knn_repro.json")))
    out = dict(
        method=("nested CV: outer = project 5-fold (common.CV); inner = StratifiedKFold(5, shuffle, seed 2607527) on each outer "
                "training part; GridSearchCV over EXACTLY the frozen model_knn.py grid, scoring roc_auc, argmax rule, "
                "Pipeline(make_preprocessor(), KNeighborsClassifier(algorithm='brute')); training rows only"),
        grid=dict(K=K_GRID, weights=W_GRID, n_settings=len(K_GRID) * len(W_GRID), source="code/model_knn.py K_GRID / WEIGHTS = results/knn.json extra.grid"),
        outer_folds=outer,
        nested_cv_auc=float(oa.mean()), nested_cv_auc_std=float(oa.std()), nested_cv_auc_sd_ddof1=float(oa.std(ddof=1)),
        nested_cv_auc_se=float(oa.std(ddof=1) / np.sqrt(5)),
        nested_pooled_oof_auc=float(roc_auc_score(y, p_nested)),
        chosen_settings_count={f"K={a}, {b}": int(sum(1 for o in outer if o["chosen_K"] == a and o["chosen_weights"] == b))
                               for a, b in sorted({(o["chosen_K"], o["chosen_weights"]) for o in outer})},
        frozen_choice=dict(n_neighbors=int(knn["best_params"]["n_neighbors"]), weights=knn["best_params"]["weights"],
                           plain_cv_auc=knn["cv_auc"], plain_cv_auc_std=knn["cv_auc_std"]),
        nested_minus_plain_frozen=paired(plain, oa, "plain CV of the frozen K=101 (knn_p_oof)", "nested CV, frozen 24-setting grid"),
        earlier_nested_16_settings=dict(source="results/nested_cv_knn_repro.json (nested_cv_check.py)", grid=old["grid"],
                                        nested_cv_auc=old["nested_cv_auc"], outer_auc=old["outer_auc"],
                                        outer_choices=old["outer_choices"]),
        frozen_grid_minus_earlier_grid=paired(old["outer_auc"], oa, "earlier nested CV (16-setting grid)", "nested CV (frozen 24-setting grid)"),
        note="Diagnostic only. The frozen KNN (K = 101, uniform) is unchanged and nothing is re-scored.")
    json.dump(out, open(os.path.join(RESULTS, "nested_cv_knn_frozen_grid.json"), "w"), indent=1)
    update_summary("b_nested_cv_knn_frozen_grid", {k: v for k, v in out.items() if k != "outer_folds"} |
                   dict(choices=[f"K={o['chosen_K']}, {o['chosen_weights']}" for o in outer], outer_auc=oa.round(5).tolist()))
    log(f"(b) nested AUC {oa.mean():.5f} +/- {oa.std():.5f} (plain frozen {knn['cv_auc']:.5f}; earlier 16-setting nested {old['nested_cv_auc']:.5f})")


# ======================================================================================== (c) KNN dimensionality
def mi_scores(X, y):
    """Mutual information of each encoded column with the label, on the fitting rows only.  Columns with <= 20 distinct
    values on those rows (PAY_1..6, N_MONTHS_DELAYED, MAX_DELAY, the dummies) are treated as discrete."""
    X = np.asarray(X)
    disc = np.array([len(np.unique(X[:, j])) <= 20 for j in range(X.shape[1])])
    return mutual_info_classif(X, y, discrete_features=disc, random_state=SEED)


def mi_scores_all_continuous(X, y):
    """Robustness variant: scikit-learn's default estimator (every column treated as continuous)."""
    return mutual_info_classif(np.asarray(X), y, random_state=SEED)


def part_c(X, y, folds):
    log("(c) KNN K=101 on 32 columns vs 5/10/15 in-fold principal components vs 8 in-fold mutual-information columns")
    variants = {
        "frozen_32_columns": (lambda: knn_pipe(), 32),
        "pca_5": (lambda: knn_pipe([("pca", PCA(n_components=5, random_state=SEED))]), 5),
        "pca_10": (lambda: knn_pipe([("pca", PCA(n_components=10, random_state=SEED))]), 10),
        "pca_15": (lambda: knn_pipe([("pca", PCA(n_components=15, random_state=SEED))]), 15),
        "mutual_info_top8": (lambda: knn_pipe([("sel", SelectKBest(mi_scores, k=8))]), 8),
        "mutual_info_top8_all_continuous": (lambda: knn_pipe([("sel", SelectKBest(mi_scores_all_continuous, k=8))]), 8),
    }
    p_frozen = frozen_oof("knn")
    oof, extra = {}, {}
    for name, (make, nd) in variants.items():
        t0 = time.time(); p = np.zeros(N_ROWS); ev, sel, nvec = [], [], []
        for tr, va in folds:
            m = make().fit(X.iloc[tr], y.iloc[tr])
            p[va] = m.predict_proba(X.iloc[va])[:, 1]
            Z = m[:-1].transform(X.iloc[tr])                     # the space KNN measures distance in (fitting rows)
            nvec.append(int(len(np.unique(np.round(np.asarray(Z, float), 9), axis=0))))
            if "pca" in m.named_steps:
                ev.append(float(m.named_steps["pca"].explained_variance_ratio_.sum()))
            if "sel" in m.named_steps:
                names = m.named_steps["pre"].get_feature_names_out()
                s = m.named_steps["sel"]
                order = np.argsort(-s.scores_)[:8]
                sel.append([f"{names[i].split('__', 1)[1]} ({s.scores_[i]:.4f})" for i in order])
        oof[name] = p
        extra[name] = dict(n_dims=nd, seconds=round(time.time() - t0, 1), pca_var_share=ev, mi_selected=sel,
                           n_distinct_vectors=float(np.mean(nvec)))
        log(f"(c) {name}: CV AUC {fold_aucs(y, p, folds).mean():.5f} [{time.time() - t0:.0f}s]")
    base = fold_aucs(y, oof["frozen_32_columns"], folds)
    rows = []
    for name, p in oof.items():
        fa = fold_aucs(y, p, folds)
        row = dict(variant=name, n_dims=extra[name]["n_dims"], K=101, weights="uniform",
                   **{f"fold{i + 1}_auc": fa[i] for i in range(5)}, cv_auc_mean=fa.mean(), cv_auc_std=fa.std(),
                   pooled_oof_auc=roc_auc_score(y, p),
                   distinct_fitting_vectors_mean=extra[name]["n_distinct_vectors"])
        if name != "frozen_32_columns":
            c = paired(base, fa, "frozen 32 columns", name)
            row.update(mean_diff_vs_frozen=c["mean_diff"], n_folds_positive=c["n_folds_positive"], nb_se=c["nb_corrected_se"],
                       nb_t=c["nb_t"], nb_ci95_lo=c["nb_ci95"][0], nb_ci95_hi=c["nb_ci95"][1],
                       flag_gain_above_one_se=bool(c["nb_t"] is not None and c["nb_t"] > 1))
        ev = extra[name]["pca_var_share"]
        row["pca_variance_share_mean"] = float(np.mean(ev)) if ev else np.nan
        sel = extra[name]["mi_selected"]
        row["mi_selected_per_fold"] = " | ".join("; ".join(s) for s in sel) if sel else ""
        row["seconds"] = extra[name]["seconds"]
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS, "knn_dimensionality_cv.csv"), index=False)
    sel_count = {}
    for var in ("mutual_info_top8", "mutual_info_top8_all_continuous"):
        cnt = {}
        for s in extra[var]["mi_selected"]:
            for v in s:
                n = v.split(" (")[0]; cnt[n] = cnt.get(n, 0) + 1
        sel_count[var] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
    out = dict(reproduction_max_abs_diff_vs_knn_p_oof=float(np.max(np.abs(oof["frozen_32_columns"] - p_frozen))),
               table=tab.drop(columns=["mi_selected_per_fold"]).round(5).to_dict(orient="records"),
               mi_selection_frequency=sel_count,
               note=("K fixed at the frozen 101 (uniform) for every space; K was not re-tuned for the reduced spaces. PCA and the "
                     "mutual-information selector are fitted on the fitting folds only.  distinct_fitting_vectors_mean counts the "
                     "distinct points among the ~16,780 fitting rows in the space KNN measures distance in: the mutual-information "
                     "columns are all repayment-status codes, so many clients coincide and the 101 neighbours are drawn from large "
                     "ties.  Diagnostic only."))
    update_summary("c_knn_dimensionality", out)
    print(tab.drop(columns=["mi_selected_per_fold"]).round(5).to_string(index=False))


# ======================================================================================== (d) grid boundaries
def position(values, chosen):
    v = sorted(values)
    if chosen == v[0]:
        return "lower edge"
    if chosen == v[-1]:
        return "upper edge"
    return "interior"


def edge_table():
    J = {s: frozen(s) for s in HEADLINE}
    R = []

    def add(slug, param, grid, chosen, pos, evidence):
        R.append(dict(section="edge_check", model=slug, parameter=param, grid=str(grid), chosen=str(chosen), position=pos, evidence=evidence))

    # KNN
    ct = pd.DataFrame(J["knn"]["extra"]["cv_table"]); ku = ct[ct.weights == "uniform"].set_index("K").cv_auc_mean
    add("knn", "n_neighbors", J["knn"]["extra"]["grid"]["K"], 101, position(J["knn"]["extra"]["grid"]["K"], 101),
        f"bracketed: K=71 {ku[71]:.5f}, K=101 {ku[101]:.5f}, K=151 {ku[151]:.5f} (uniform)")
    add("knn", "weights", ["uniform", "distance"], "uniform", "categorical", "two-level choice; no edge")
    # logistic
    add("logistic", "(none)", "untuned (C = inf)", "C = inf", "not tuned", "plain unpenalised baseline")
    gC, gA = J["logistic_l1"]["extra"]["grid_C"], J["logistic_l1"]["extra"]["grid_mean_cv_auc"]
    add("logistic_l1", "C", [round(c, 5) for c in gC], 100.0, position(gC, 100.0),
        f"upper edge is the unpenalised limit: CV {gA[-2]:.6f} at C=31.6, {gA[-1]:.6f} at C=100, plain logit (C=inf) "
        f"{J['logistic']['cv_auc']:.6f}; nothing to gain beyond it")
    # LDA
    lg = {r["value"]: r["cv_auc"] for r in J["lda"]["extra"]["cv_grid"]}
    add("lda", "shrinkage", list(lg), "None (= 0)", "lower edge (natural limit 0)",
        f"shrinkage cannot go below 0; CV None {lg['None']:.5f}, auto {lg['auto']:.5f}, 0.05 {lg['0.05']:.5f}")
    # NB
    ng = {float(r["value"]): r["cv_auc"] for r in J["naive_bayes"]["extra"]["cv_grid"]}
    add("naive_bayes", "var_smoothing", sorted(ng), 0.01, position(list(ng), 0.01),
        f"bracketed: 1e-3 {ng[1e-3]:.5f}, 1e-2 {ng[1e-2]:.5f}, 1e-1 {ng[1e-1]:.5f}")
    # pruned tree
    tp = J["tree_pruned"]; ag = tp["extra"]["alpha_grid"]; ca = tp["best_params"]["ccp_alpha"]
    cc = pd.DataFrame(tp["extra"]["cv_curve"])
    add("tree_pruned", "ccp_alpha", f"{len(ag)} values on the pruning path, 0 to {max(ag):.4g}", f"{ca:.4g}",
        f"interior (index {ag.index(ca) + 1} of {len(ag)})", "interior on the cost-complexity path")
    add("tree_pruned", "max_depth", ["4", "6", "8", "None"], tp["best_params"]["max_depth"], "interior",
        "bracketed by depth 4 and 8 / unlimited")
    # random forest
    rg = pd.DataFrame(J["random_forest"]["extra"]["grid"]); rs = rg[rg.max_features == "sqrt"].set_index("min_samples_leaf").cv_auc
    rl = rg[rg.min_samples_leaf == 20].set_index("max_features").cv_auc
    add("random_forest", "max_features", ["sqrt (5 of 32 = 0.156)", 0.3, 0.5], "sqrt", "lower edge",
        f"at leaf 20: sqrt {rl['sqrt']:.5f}, 0.3 {rl['0.3']:.5f}, 0.5 {rl['0.5']:.5f} (200-tree grid); smaller shares not tried")
    add("random_forest", "min_samples_leaf", [1, 5, 20], 20, "upper edge",
        f"NOT bracketed, CV still rising: leaf 1 {rs[1]:.5f}, 5 {rs[5]:.5f}, 20 {rs[20]:.5f} (sqrt, 200 trees); no probe beyond 20")
    add("random_forest", "n_estimators", "fixed 500 (grid scored with 200)", 500, "not tuned", "top-2 settings re-scored at 500 trees")
    # GBM
    gb = J["gradient_boosting"]; so = gb["extra"]["staged_optimum"]
    add("gradient_boosting", "learning_rate", [0.03, 0.1], 0.03, "lower edge", "lr below 0.03 not tried")
    add("gradient_boosting", "max_depth", [2, 3, 4], 3, "interior", "bracketed by depth 2 and 4")
    add("gradient_boosting", "n_estimators", [200, 500], 200, "lower edge (bracketed by staged curve)",
        f"staged subsample curve peaks at {so['n_estimators']} trees (CV {so['cv_auc']:.5f}), so 200 is near the optimum; "
        "see results/gbm_grid_fulltrain.csv for the full-set curve")
    # XGBoost
    xb = J["xgboost"]; lc = xb["extra"]["learning_curve"]["0.03"]
    add("xgboost", "max_depth", [3, 4, 6], 3, "lower edge", "probed below in this file (depth 2)")
    add("xgboost", "learning_rate", [0.03, 0.1], 0.03, "lower edge", "probed below in this file (0.01, 0.02)")
    add("xgboost", "n_estimators", [300, 600], 300, "lower edge (bracketed by learning curve)",
        f"learning curve at lr 0.03 peaks at {lc['best_rounds']} rounds (CV {lc['cv_auc_at_best']:.5f}); probed 150-600 below")
    xg = pd.DataFrame(xb["extra"]["grid"]).set_index(["max_depth", "learning_rate", "n_estimators", "min_child_weight"]).cv_auc
    add("xgboost", "min_child_weight", [1, 5], 5, "upper edge",
        f"not probed (held at 5 in the extension); at (3, 0.03, 300) mcw 1 {xg[(3, 0.03, 300, 1)]:.5f} vs mcw 5 {xg[(3, 0.03, 300, 5)]:.5f}")
    # SVM RBF
    sv = J["svm_rbf"]; bc = sv["extra"]["boundary_check_beyond_grid"]
    add("svm_rbf", "C", [0.1, 1, 10], 1.0, "interior", "bracketed (subsample CV)")
    add("svm_rbf", "gamma", ["0.01", "scale (0.0361)", "0.1"], "0.01", "lower edge (probed)",
        "probe beyond the grid on the subsample: " + "; ".join(f"{k} {v['cv_auc_mean']:.5f}" for k, v in bc.items())
        + f"; chosen {sv['extra']['chosen']['cv_auc']:.5f}, so the optimum is bracketed")
    add("svm_rbf", "class_weight", ["none", "balanced"], "balanced", "categorical", "two-level choice")
    # MLP
    me = J["mlp"]["extra"]["best_on_grid_edge"]
    add("mlp", "hidden_layer_sizes", ["32", "64", "64x32", "128x64"], "64x32", me["hidden_layer_sizes"], "ordered by size; interior")
    add("mlp", "alpha", [1e-4, 1e-3, 1e-2], 1e-3, me["alpha"], "bracketed")
    add("mlp", "learning_rate_init", [1e-3, 3e-3], 1e-3, me["learning_rate_init"],
        f"not probed below 0.001; mean CV over the other settings is flat across the two values (0.001: "
        f"{J['mlp']['extra']['marginal_cv_auc']['learning_rate_init']['0.001']:.5f}, 0.003: "
        f"{J['mlp']['extra']['marginal_cv_auc']['learning_rate_init']['0.003']:.5f})")
    return pd.DataFrame(R)


XGB_FIXED = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
XGB_DEPTH, XGB_LR, XGB_ROUNDS = [2, 3], [0.01, 0.02, 0.03], list(range(150, 601, 50))
XGB_ROUNDS_BEYOND = list(range(650, 1201, 50))      # diagnostic only, for lr < 0.03 whose curves may still rise at 600


def xgb_rounds(lr):
    return XGB_ROUNDS + (XGB_ROUNDS_BEYOND if lr < 0.03 else [])


def part_d(X, y, folds):
    log("(d) grid-edge table for the headline models, then the XGBoost extension")
    edges = edge_table()
    xj = frozen("xgboost")
    fz_auc, fz_sd = float(xj["cv_auc"]), float(xj["cv_auc_std"])
    fz_folds = fold_aucs(y, frozen_oof("xgboost"), folds)
    val = {}
    p300 = np.zeros(N_ROWS)
    for d, lr in itertools.product(XGB_DEPTH, XGB_LR):
        for f, (tr, va) in enumerate(folds):
            pre = make_preprocessor(scale=False).fit(X.iloc[tr])
            Xtr, Xva = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
            c = XGBClassifier(**XGB_FIXED, max_depth=d, learning_rate=lr, n_estimators=max(xgb_rounds(lr))).fit(Xtr, y.values[tr])
            for n in xgb_rounds(lr):
                p = c.predict_proba(Xva, iteration_range=(0, n))[:, 1]
                val[(d, lr, n, f)] = roc_auc_score(y.values[va], p)
                if d == 3 and lr == 0.03 and n == 300:
                    p300[va] = p
        log(f"(d) XGB depth {d}, lr {lr}: best {max(np.mean([val[(d, lr, n, f)] for f in range(5)]) for n in XGB_ROUNDS):.5f} "
            f"within 150-600 rounds")
    rows = []
    for d, lr in itertools.product(XGB_DEPTH, XGB_LR):
        for n in xgb_rounds(lr):
            fa = np.array([val[(d, lr, n, f)] for f in range(5)])
            c = paired(fz_folds, fa, "frozen XGBoost (3, 0.03, 300, mcw 5)", f"XGB ({d}, {lr}, {n}, mcw 5)")
            rows.append(dict(section="xgb_extension", model="xgboost", max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=5,
                             **{f"fold{i + 1}_auc": fa[i] for i in range(5)}, cv_auc=fa.mean(), cv_auc_std=fa.std(),
                             diff_vs_frozen=fa.mean() - fz_auc, n_folds_positive=c["n_folds_positive"], nb_se=c["nb_corrected_se"],
                             nb_ci95_lo=c["nb_ci95"][0], nb_ci95_hi=c["nb_ci95"][1],
                             beats_frozen_by_more_than_one_fold_sd=bool(fa.mean() - fz_auc > fz_sd),
                             beats_frozen_by_more_than_one_fold_se=bool(fa.mean() - fz_auc > fz_sd / np.sqrt(5)),
                             in_original_grid=bool(d == 3 and lr == 0.03 and n in (300, 600)),
                             in_requested_range=bool(n <= max(XGB_ROUNDS)),
                             is_frozen_setting=bool(d == 3 and lr == 0.03 and n == 300)))
    ext_all = pd.DataFrame(rows)
    pd.concat([edges, ext_all], ignore_index=True).to_csv(os.path.join(RESULTS, "grid_edge_check.csv"), index=False)
    ext, beyond = ext_all[ext_all.in_requested_range], ext_all[~ext_all.in_requested_range]
    best = ext.loc[ext.cv_auc.idxmax()]
    bb = beyond.loc[beyond.cv_auc.idxmax()]
    bc = paired(fz_folds, best[[f"fold{i + 1}_auc" for i in range(5)]].values.astype(float), "frozen XGBoost (3, 0.03, 300, mcw 5)",
                f"best extension setting ({int(best.max_depth)}, {best.learning_rate}, {int(best.n_estimators)}, mcw 5)")
    per_cell = (ext.sort_values("cv_auc", ascending=False).groupby(["max_depth", "learning_rate"]).head(1)
                .sort_values(["max_depth", "learning_rate"])[["max_depth", "learning_rate", "n_estimators", "cv_auc"]])
    out = dict(
        edge_summary=edges[["model", "parameter", "chosen", "position"]].to_dict(orient="records"),
        xgb_extension=dict(
            grid="max_depth {2, 3} x learning_rate {0.01, 0.02, 0.03} x n_estimators 150-600 step 50; min_child_weight 5, "
                 "subsample 0.8, colsample_bytree 0.8 frozen; one fit per fold and (depth, lr) (600 rounds; 1,200 for lr < 0.03), AUC read with iteration_range",
            n_settings=int(len(ext)), frozen_cv_auc=fz_auc, frozen_cv_auc_std_ddof0=fz_sd, frozen_fold_se=fz_sd / np.sqrt(5),
            reproduction=dict(frozen_setting_cv_auc_here=float(ext[ext.is_frozen_setting].cv_auc.iloc[0]),
                              max_abs_diff_p_oof_vs_frozen=float(np.max(np.abs(p300 - frozen_oof("xgboost")))),
                              setting_600_cv_auc_here=float(ext[(ext.max_depth == 3) & (ext.learning_rate == 0.03) &
                                                                (ext.n_estimators == 600)].cv_auc.iloc[0])),
            best_setting=dict(max_depth=int(best.max_depth), learning_rate=float(best.learning_rate),
                              n_estimators=int(best.n_estimators), cv_auc=float(best.cv_auc), diff_vs_frozen=float(best.diff_vs_frozen)),
            best_vs_frozen=bc,
            best_per_depth_lr=per_cell.round(5).to_dict(orient="records"),
            any_beats_by_more_than_one_fold_sd=bool(ext.beats_frozen_by_more_than_one_fold_sd.any()),
            any_beats_by_more_than_one_fold_se=bool(ext.beats_frozen_by_more_than_one_fold_se.any()),
            beyond_requested_range=dict(
                rounds="650-1200 step 50, learning_rate 0.01 and 0.02 only (diagnostic: their curves were still rising at 600)",
                best_per_depth_lr=(beyond.sort_values("cv_auc", ascending=False).groupby(["max_depth", "learning_rate"]).head(1)
                                   .sort_values(["max_depth", "learning_rate"])[["max_depth", "learning_rate", "n_estimators", "cv_auc"]]
                                   .round(5).to_dict(orient="records")),
                best=dict(max_depth=int(bb.max_depth), learning_rate=float(bb.learning_rate), n_estimators=int(bb.n_estimators),
                          cv_auc=float(bb.cv_auc), diff_vs_frozen=float(bb.diff_vs_frozen)),
                best_vs_frozen=paired(fz_folds, bb[[f"fold{i + 1}_auc" for i in range(5)]].values.astype(float),
                                      "frozen XGBoost (3, 0.03, 300, mcw 5)",
                                      f"best beyond-range setting ({int(bb.max_depth)}, {bb.learning_rate}, {int(bb.n_estimators)}, mcw 5)"),
                any_beats_by_more_than_one_fold_sd=bool(beyond.beats_frozen_by_more_than_one_fold_sd.any()),
                any_beats_by_more_than_one_fold_se=bool(beyond.beats_frozen_by_more_than_one_fold_se.any())),
            note="Diagnostic only; the frozen XGBoost is kept and nothing is re-scored."))
    update_summary("d_grid_edges", out)
    print(edges[["model", "parameter", "chosen", "position"]].to_string(index=False))
    print(per_cell.round(5).to_string(index=False))
    log(f"(d) best extension {out['xgb_extension']['best_setting']}; beats frozen by > 1 fold-SD: "
        f"{out['xgb_extension']['any_beats_by_more_than_one_fold_sd']}")


# ======================================================================================== (e) SVM selection statistic
def part_e(X, y, folds):
    log("(e) RBF SVM: subsample CV AUC vs pooled OOF AUC vs per-fold mean OOF AUC")
    out = {}
    for slug in ("svm_rbf", "svm_linear"):
        j = frozen(slug); p = frozen_oof(slug); fa = fold_aucs(y, p, folds)
        mp = [float(p[va].mean()) for _, va in folds]
        floor = [float(p[va].min()) for _, va in folds]
        near_floor = [float(np.mean(p[va] < p[va].min() + 0.002)) for _, va in folds]
        out[slug] = dict(
            subsample_cv_auc=dict(value=float(j["extra"]["chosen"]["cv_auc"]), sd=float(j["extra"]["chosen"]["cv_auc_std"]),
                                  rows=int(j["extra"]["tuning"]["subsample_rows"]), folds="StratifiedKFold(5, shuffle, 2607526) on the subsample's own rows",
                                  is_max_over_grid=True, logistic_same_subsample=float(j["extra"]["tuning"]["logistic_reference"]["same_subsample_cv_auc"])),
            pooled_oof_auc=float(roc_auc_score(y, p)),
            per_fold_oof_auc=np.round(fa, 5).tolist(), per_fold_mean_oof_auc=float(fa.mean()), per_fold_sd_ddof0=float(fa.std()),
            per_fold_sd_ddof1=float(fa.std(ddof=1)), pooled_minus_mean=float(roc_auc_score(y, p) - fa.mean()),
            per_fold_mean_predicted_probability=np.round(mp, 4).tolist(),
            per_fold_min_probability=np.round(floor, 5).tolist(),
            per_fold_share_within_0002_of_fold_min=np.round(near_floor, 4).tolist(),
            json_values=dict(cv_auc=j["cv_auc"], oof_auc=j["oof_auc"], oof_per_fold_mean=j["extra"]["oof"]["per_fold"]["mean"]),
            used_in_tables="pooled_oof_auc (aggregate.select_auc_of uses oof_auc for subsample-tuned scripts)")
    # the linear SVM's floor block: which clients are in it, and how much ranking a single column gives
    p_lin = frozen_oof("svm_linear")
    near = np.zeros(N_ROWS, dtype=bool)
    for _, va in folds:
        near[va] = p_lin[va] < p_lin[va].min() + 0.002
    md = X["MAX_DELAY"].values
    out["svm_linear"]["floor_block"] = dict(
        share_of_rows=float(near.mean()), share_of_block_with_MAX_DELAY_0=float((md[near] == 0).mean()),
        share_of_MAX_DELAY_0_rows_in_block=float(near[md == 0].mean()), share_MAX_DELAY_0=float((md == 0).mean()),
        MAX_DELAY_alone_auc_pooled=float(roc_auc_score(y, md)),
        MAX_DELAY_alone_auc_per_fold_mean=float(fold_aucs(y, md.astype(float), folds).mean()))
    # context: pooled minus per-fold mean for every other headline model, and where the RBF SVM ranks under each statistic
    ctx = {}
    for s in HEADLINE:
        if s == "svm_rbf":
            continue
        p = frozen_oof(s); fa = fold_aucs(y, p, folds)
        ctx[s] = dict(cv_auc_json=frozen(s)["cv_auc"], per_fold_mean=float(fa.mean()), pooled=float(roc_auc_score(y, p)),
                      pooled_minus_mean=float(roc_auc_score(y, p) - fa.mean()))
    others = {s: v["per_fold_mean"] for s, v in ctx.items()}
    r = out["svm_rbf"]
    ranks = {}
    for lab, v in [("subsample_cv_auc", r["subsample_cv_auc"]["value"]), ("pooled_oof_auc", r["pooled_oof_auc"]),
                   ("per_fold_mean_oof_auc", r["per_fold_mean_oof_auc"])]:
        above = sorted([s for s, a in others.items() if a > v], key=lambda s: others[s])
        below = sorted([s for s, a in others.items() if a < v], key=lambda s: -others[s])
        ranks[lab] = dict(value=v, rank_among_11_headline=1 + len(above), next_above=above[0] if above else None,
                          next_below=below[0] if below else None)
    rbf_folds = np.array(r["per_fold_oof_auc"])
    pairs = [paired(fold_aucs(y, frozen_oof(s), folds), fold_aucs(y, frozen_oof("svm_rbf"), folds), s, "svm_rbf")
             for s in ("logistic", "tree_pruned", "knn")]
    out["svm_rbf_rank_by_statistic"] = ranks
    out["svm_rbf_paired_vs_neighbours"] = pairs
    out["other_headline_models_pooled_vs_mean"] = ctx
    out["max_abs_pooled_minus_mean_other_headline"] = float(max(abs(v["pooled_minus_mean"]) for v in ctx.values()))
    lin = out["svm_linear"]
    out["reading"] = ("The pooled OOF AUC ranks rows from five different Platt-scaled models together, so for the SVMs it falls below "
                      "the mean of the five per-fold AUCs; for every other headline model the two agree to within "
                      f"{out['max_abs_pooled_minus_mean_other_headline']:.4f}.  The linear SVM's large gap has a visible cause: in every "
                      f"fold {min(lin['per_fold_share_within_0002_of_fold_min']):.0%}-{max(lin['per_fold_share_within_0002_of_fold_min']):.0%} "
                      "of its OOF probabilities lie within 0.002 of a fold-specific floor "
                      f"({min(lin['per_fold_min_probability']):.4f}-{max(lin['per_fold_min_probability']):.4f}), and pooling stacks these "
                      "near-tied blocks on top of each other.  The 6,000-row figure is the maximum over 18 settings on the subsample's own "
                      "folds and is not comparable with full-set CV AUCs.  RBF SVM rank among the 11 headline models (others by their "
                      "per-fold mean CV AUC): " + "; ".join(f"{k} {v['rank_among_11_headline']} (between {v['next_above']} and "
                                                            f"{v['next_below']})" for k, v in ranks.items()) + ".")
    json.dump(out, open(os.path.join(RESULTS, "svm_selection_statistics.json"), "w"), indent=1)
    update_summary("e_svm_selection_statistics", dict(svm_rbf={k: r[k] for k in ("pooled_oof_auc", "per_fold_mean_oof_auc", "pooled_minus_mean")}
                                                      | dict(subsample_cv_auc=r["subsample_cv_auc"]["value"]),
                                                      svm_linear={k: out["svm_linear"][k] for k in ("pooled_oof_auc", "per_fold_mean_oof_auc", "pooled_minus_mean")}
                                                      | dict(subsample_cv_auc=out["svm_linear"]["subsample_cv_auc"]["value"]),
                                                      rank_by_statistic=ranks,
                                                      max_abs_pooled_minus_mean_other_headline=out["max_abs_pooled_minus_mean_other_headline"]))
    log(f"(e) RBF: subsample {r['subsample_cv_auc']['value']:.5f}, pooled {r['pooled_oof_auc']:.5f}, per-fold mean "
        f"{r['per_fold_mean_oof_auc']:.5f}; other models max |pooled-mean| {out['max_abs_pooled_minus_mean_other_headline']:.5f}")


if __name__ == "__main__":
    parts = [a.lower() for a in sys.argv[1:]] or ["a", "b", "c", "d", "e"]
    X, y, folds = load_training()
    log(f"train.csv: {len(y):,} rows, default rate {y.mean():.4f}; folds match cv_folds.csv; parts {parts}")
    for part in ("e", "c", "b", "d", "a"):                    # cheapest first; the GBM grid is the slow one
        if part in parts:
            globals()[f"part_{part}"](X, y, folds)
    log("done")
