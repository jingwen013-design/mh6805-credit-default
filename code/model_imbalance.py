"""
model_imbalance.py — Class-imbalance strategy study (lecture B4/B6: cost-sensitive learning and resampling).  Owner: LI Jingwen.

Run from Project/code:  python3 model_imbalance.py            (about 5–6 minutes with n_jobs=2; 334 s last run)

Question: does re-balancing the 22 % / 78 % training classes make a model *rank* defaulters better (AUC / AP), or does it
merely move the decision threshold?  We answer it with two base learners x four strategies, all scored on the SAME five
stratified CV folds (common.CV) and on the same held-out test set:

  learners    logistic regression (C = 1)                                          — linear, probabilistic
              random forest (500 trees, max_features='sqrt', min_samples_leaf=5)  — non-linear, bootstrap ensemble
  strategies  (a) none            plain fit on the imbalanced training folds
              (b) balanced        class_weight='balanced' (cost-sensitive re-weighting, no new rows)
              (c) smote           SMOTE(random_state=SEED) INSIDE the CV folds via imblearn.pipeline.Pipeline
              (d) under           RandomUnderSampler(random_state=SEED) INSIDE the CV folds via imblearn.pipeline.Pipeline

Why the imblearn Pipeline matters (the leak-free way): a sampler placed as a pipeline step is applied in fit() only — so
inside cross-validation each training fold is resampled and the validation fold is left untouched; at predict time the
sampler is skipped.  Out-of-fold probabilities are therefore honest, thresholds tuned on them are honest, and the test
set never contains a synthetic row.  The wrong way — resampling the whole data set BEFORE splitting — is demonstrated at
the end (slug imb_leaky_smote_demo, name prefixed "LEAKAGE DEMO") so that the inflated "test" AUC is documented and
never mistaken for a real result.  To isolate the leak, the demo resamples in the SAME feature space as the honest runs
(the make_preprocessor() output: 26 standardised numerics + 6 one-hot dummies, the preprocessor being fitted once on all
rows) and ends with the same classifiers, so the only material difference is that SMOTE saw the test rows.

Both learners use make_preprocessor() (scaled numerics + one-hot categoricals) so that SMOTE / undersampling see exactly
the same transformed matrix and generate the same resampled rows in a given fold for both learners; trees are invariant
to monotone rescaling, so the forest is unaffected by the scaling.

Threshold-equivalence diagnostic: for every re-balanced model we find, on the TRAINING out-of-fold probabilities, the
threshold t* of the plain model that best reproduces the re-balanced model's decisions at 0.5, and then report how often
the two rules agree on the test set (no threshold is chosen on the test set).  For the logistic model t* is also
predicted analytically from the intercept shift: t* = 1 / (1 + exp(delta_intercept)).

Outputs: results/imb_<learner>_<strategy>.json (8 evaluate() calls), results/imbalance_comparison.csv,
         results/imbalance_threshold_equivalence.csv, results/imbalance_leakage_demo.csv, results/imbalance_lr_coefficients.csv,
         figures/imbalance_comparison.png, figures/imbalance_threshold_shift.png, figures/imbalance_leakage_demo.png
         (+ the standard <slug>_eval.png of every evaluate() call).
         The leakage demo's evaluate() output (imb_leaky_smote_demo.json / _p_test.npy / _p_oof.npy, _eval.png) is MOVED to
         results/leakage_demo/ and figures/leakage_demo/ so that common.load_all_results() and any report aggregation over
         results/*.json cannot pick up a 14,001-row, 50 %-prevalence "test" set by accident.  Report code that does scan
         sub-folders should still filter slug.startswith("imb_leaky").
The test set is only ever predicted, never fitted on or used for any selection: predictions are made once in run_study()
for the rank-correlation / threshold-equivalence diagnostics and once inside evaluate().
"""
import os, time, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from common import *

MEMBER = "LI Jingwen"
LEARNERS = ["logistic", "rf"]
STRATEGIES = ["none", "balanced", "smote", "under"]
LEARNER_LABEL = {"logistic": "Logistic (C=1)", "rf": "Random forest (500 trees)"}
STRATEGY_LABEL = {"none": "none", "balanced": "class_weight='balanced'", "smote": "SMOTE in folds", "under": "under-sampling in folds"}
N_TREES = 500

# validated categorical palette (dataviz skill, light surface): slots 1-4 = blue, orange, aqua, yellow; text greys
C_STRAT = {"none": "#2a78d6", "balanced": "#eb6834", "smote": "#1baf7a", "under": "#eda100"}
C_LEAK, C_TXT, C_TXT2, C_GRID = "#e34948", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "xtick.color": C_TXT2, "ytick.color": C_TXT2,
                     "text.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


# ----------------------------------------------------------------------------- pipelines
def make_clf(learner, class_weight=None):
    if learner == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight, random_state=SEED)
    return RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=5, class_weight=class_weight,
                                  n_jobs=2, random_state=SEED)


def make_pipe(learner, strategy):
    """Preprocessor + (sampler) + classifier.  Samplers are imblearn steps => applied in fit() only (leak-free in CV)."""
    pre = make_preprocessor()                                   # scaled numerics + one-hot; refitted inside every fold
    if strategy == "none":
        return Pipeline([("pre", pre), ("clf", make_clf(learner))])
    if strategy == "balanced":
        return Pipeline([("pre", pre), ("clf", make_clf(learner, class_weight="balanced"))])
    if strategy == "smote":
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)), ("clf", make_clf(learner))])
    if strategy == "under":
        return ImbPipeline([("pre", pre), ("under", RandomUnderSampler(random_state=SEED)), ("clf", make_clf(learner))])
    raise ValueError(strategy)


def fold_aucs(y, p_oof):
    """Per-fold AUC of out-of-fold probabilities on the shared CV folds (CV is seeded => identical folds every call)."""
    return np.array([roc_auc_score(y.iloc[va], p_oof[va]) for _, va in CV.split(np.zeros(len(y)), y)])


def rows_after_resampling(strategy, y):
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"none": n0 + n1, "balanced": n0 + n1, "smote": 2 * n0, "under": 2 * n1}[strategy]


def lr_coefficients(pipe):
    clf = pipe.named_steps["clf"]
    names = feature_names_out(pipe.named_steps["pre"])
    return pd.Series(np.r_[clf.intercept_, clf.coef_.ravel()], index=["intercept"] + names)


THR_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def agreement(p_a, t_a, p_b, t_b):
    """Share of rows on which the rules (p_a >= t_a) and (p_b >= t_b) make the same decision."""
    return float(((p_a >= t_a) == (p_b >= t_b)).mean())


def equivalent_plain_threshold(p_oof_none, p_oof_strat, t_strat=0.5):
    """Threshold of the PLAIN model that best reproduces the re-balanced model's decisions at t_strat.
    Chosen on training out-of-fold probabilities only (never on the test set)."""
    agr = np.array([agreement(p_oof_none, t, p_oof_strat, t_strat) for t in THR_GRID])
    return float(THR_GRID[int(agr.argmax())]), float(agr.max())


# ----------------------------------------------------------------------------- main study
def run_study(X_train, X_test, y_train, y_test):
    rows, equiv_rows, coefs, p_test_store, p_oof_store, thr_store = [], [], {}, {}, {}, {}
    prior_log_ratio = float(np.log((y_train == 0).mean() / (y_train == 1).mean()))    # log(pi0 / pi1) = 1.26
    for learner in LEARNERS:
        for strategy in STRATEGIES:
            slug = f"imb_{learner}_{strategy}"
            pipe = make_pipe(learner, strategy)
            t0 = time.time()
            p_oof = cross_val_predict(clone(pipe), X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
            aucs = fold_aucs(y_train, p_oof)
            cv_seconds = time.time() - t0
            t0 = time.time()
            pipe.fit(X_train, y_train)
            fit_seconds = time.time() - t0
            # Test-set predictions for the rank-correlation / threshold-equivalence diagnostics only (evaluate() predicts
            # again below).  Nothing is fitted, tuned or selected on them.
            p_test = pipe.predict_proba(X_test)[:, 1]
            t_f1, _ = tune_threshold(y_train.values, p_oof, "f1")               # same rule evaluate() applies (training OOF only)
            p_test_store[slug], p_oof_store[slug], thr_store[slug] = p_test, p_oof, t_f1
            rho = float(spearmanr(p_test, p_test_store[f"imb_{learner}_none"]).statistic)
            extra = dict(learner=learner, strategy=strategy, fold_aucs=[round(float(a), 4) for a in aucs],
                         cv_ap=float(np.mean([average_precision_score(y_train.iloc[va], p_oof[va]) for _, va in CV.split(np.zeros(len(y_train)), y_train)])),
                         n_train_rows_seen_by_clf=rows_after_resampling(strategy, y_train), prevalence_train=float(y_train.mean()),
                         mean_p_oof=float(p_oof.mean()), mean_p_test=float(p_test.mean()), prevalence_test=float(y_test.mean()),
                         spearman_p_test_vs_none=rho, cv_seconds=round(cv_seconds, 1))
            if learner == "logistic":
                c = lr_coefficients(pipe); coefs[strategy] = c
                extra.update(n_iter=int(pipe.named_steps["clf"].n_iter_[0]), intercept=float(c["intercept"]),
                             log_prior_ratio=prior_log_ratio, n_slopes=int(len(c) - 1),
                             delta_intercept_vs_none=float(c["intercept"] - coefs["none"]["intercept"]),
                             max_abs_delta_slope_vs_none=float((c.drop("intercept") - coefs["none"].drop("intercept")).abs().max()))
            if strategy != "none":
                # Which plain-model threshold is "the same rule" as this re-balanced model at 0.5?  Chosen on training OOF.
                none_slug = f"imb_{learner}_none"
                t_equiv, agr_oof = equivalent_plain_threshold(p_oof_store[none_slug], p_oof)
                eq = dict(learner=learner, strategy=strategy,
                          plain_threshold_equivalent_to_0_5=t_equiv, agreement_oof=agr_oof,
                          agreement_test=agreement(p_test_store[none_slug], t_equiv, p_test, 0.5),
                          agreement_test_plain_at_0_30=agreement(p_test_store[none_slug], 0.30, p_test, 0.5),
                          recall_test_plain_at_equiv=float(recall_score(y_test, p_test_store[none_slug] >= t_equiv)),
                          recall_test_strat_at_0_5=float(recall_score(y_test, p_test >= 0.5)),
                          n_flagged_test_plain_at_equiv=int((p_test_store[none_slug] >= t_equiv).sum()),
                          n_flagged_test_strat_at_0_5=int((p_test >= 0.5).sum()),
                          tuned_f1_threshold_plain=thr_store[none_slug], tuned_f1_threshold_strat=t_f1,
                          agreement_test_tuned_pair=agreement(p_test_store[none_slug], thr_store[none_slug], p_test, t_f1))
                if learner == "logistic":
                    eq["plain_threshold_from_intercept_shift"] = float(1.0 / (1.0 + np.exp(extra["delta_intercept_vs_none"])))
                equiv_rows.append(eq); extra["threshold_equivalence"] = eq
            notes = (f"{STRATEGY_LABEL[strategy]}; preprocessor refitted in every fold; sampler (if any) is an imblearn pipeline step "
                     f"=> applied to the training folds only, validation folds and test set contain no synthetic rows. "
                     f"cv_auc = mean of 5 fold AUCs on common.CV; thresholds tuned on training OOF probabilities.")
            res = evaluate(f"{LEARNER_LABEL[learner]} - {STRATEGY_LABEL[strategy]}", pipe, X_train, y_train, X_test, y_test,
                           member=MEMBER, best_params=dict(learner=learner, strategy=strategy,
                                                           **({"C": 1.0} if learner == "logistic" else
                                                              {"n_estimators": N_TREES, "max_features": "sqrt", "min_samples_leaf": 5})),
                           cv_auc=float(aucs.mean()), cv_auc_std=float(aucs.std()), notes=notes, fit_seconds=round(fit_seconds, 1),
                           p_oof=p_oof, extra=extra, slug=slug)
            rows.append(dict(learner=learner, strategy=strategy, slug=slug, n_train_rows_seen_by_clf=extra["n_train_rows_seen_by_clf"],
                             cv_auc=res["cv_auc"], cv_auc_std=res["cv_auc_std"], cv_ap=extra["cv_ap"], oof_auc=res["oof_auc"],
                             test_auc=res["test_auc"], test_ap=res["test_ap"], test_brier=res["test_brier"], mean_p_test=extra["mean_p_test"],
                             tuned_threshold_f1=res["at_tuned_f1"]["threshold"],
                             recall_1_at_0_5=res["at_0_5"]["recall_1"], precision_1_at_0_5=res["at_0_5"]["precision_1"],
                             f1_at_0_5=res["at_0_5"]["f1_1"], accuracy_at_0_5=res["at_0_5"]["accuracy"],
                             recall_1_at_tuned=res["at_tuned_f1"]["recall_1"], precision_1_at_tuned=res["at_tuned_f1"]["precision_1"],
                             f1_at_tuned=res["at_tuned_f1"]["f1_1"], accuracy_at_tuned=res["at_tuned_f1"]["accuracy"],
                             tuned_threshold_cost=res["at_tuned_cost"]["threshold"], recall_1_at_cost=res["at_tuned_cost"]["recall_1"],
                             cost_at_cost_threshold=res["at_tuned_cost"]["cost"], spearman_vs_none=rho,
                             fit_seconds=round(fit_seconds, 1), cv_seconds=round(cv_seconds, 1)))
            print(f"{slug:<24s} rows seen {extra['n_train_rows_seen_by_clf']:>6,d} | CV AUC {aucs.mean():.4f}±{aucs.std():.4f} | "
                  f"test AUC {res['test_auc']:.4f} AP {res['test_ap']:.4f} Brier {res['test_brier']:.4f} | mean p {p_test.mean():.3f} | "
                  f"rec@0.5 {res['at_0_5']['recall_1']:.3f} | thr {res['at_tuned_f1']['threshold']:.2f} -> rec {res['at_tuned_f1']['recall_1']:.3f} "
                  f"prec {res['at_tuned_f1']['precision_1']:.3f} | rho vs none {rho:.4f} | {cv_seconds + fit_seconds:.0f}s")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "imbalance_comparison.csv"), index=False)
    equiv = pd.DataFrame(equiv_rows); equiv.to_csv(os.path.join(RESULTS, "imbalance_threshold_equivalence.csv"), index=False)
    coef_df = pd.DataFrame(coefs); coef_df.to_csv(os.path.join(RESULTS, "imbalance_lr_coefficients.csv"))
    return table, equiv, coef_df, p_test_store, p_oof_store


# ----------------------------------------------------------------------------- leakage demonstration (the WRONG way)
LEAK_SLUG = "imb_leaky_smote_demo"
LEAK_RESULTS_DIR = os.path.join(RESULTS, "leakage_demo")
LEAK_FIGURES_DIR = os.path.join(FIGURES, "leakage_demo")


def quarantine_leaky_outputs():
    """Move the demo's evaluate() files out of results/ and figures/ so no aggregation over results/*.json can load them."""
    os.makedirs(LEAK_RESULTS_DIR, exist_ok=True); os.makedirs(LEAK_FIGURES_DIR, exist_ok=True)
    moved = []
    for src_dir, dst_dir, names in [(RESULTS, LEAK_RESULTS_DIR, [f"{LEAK_SLUG}.json", f"{LEAK_SLUG}_p_test.npy", f"{LEAK_SLUG}_p_oof.npy"]),
                                    (FIGURES, LEAK_FIGURES_DIR, [f"{LEAK_SLUG}_eval.png"])]:
        for n in names:
            src = os.path.join(src_dir, n)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dst_dir, n)); moved.append(os.path.relpath(os.path.join(dst_dir, n), ROOT))
    assert not any(f.startswith(LEAK_SLUG) for f in os.listdir(RESULTS)), "leaky demo files must not stay in results/"
    return moved


def leaky_demo(X_train, X_test, y_train, y_test, honest):
    """SMOTE applied to ALL rows BEFORE the split, then the same split recipe (stratified 70/30, SEED) and the same classifiers.
    For demonstration only.  To isolate the leak, SMOTE runs in the SAME feature space as the honest imb_<learner>_smote runs:
    the shared make_preprocessor() (StandardScaler + one-hot, drop='first') is fitted once on all 29,965 rows and its 32-column
    output is resampled, so the only material difference to the honest runs is that SMOTE saw the test rows.  (Fitting the scaler
    on all rows is a second, negligible leak of column means / SDs - part of the same 'resample first, split later' mistake.)
    Because the matrix is already transformed, the classifiers are the bare make_clf() estimators the honest pipelines end with."""
    X_all = pd.concat([X_train, X_test], ignore_index=True); y_all = pd.concat([y_train, y_test], ignore_index=True)
    n_orig = len(X_all)
    pre = make_preprocessor().fit(X_all)                                       # fitted on ALL rows - the wrong way, deliberately
    Z = pre.transform(X_all); Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    Z_all = pd.DataFrame(Z, columns=feature_names_out(pre))
    Z_res, y_res = SMOTE(random_state=SEED).fit_resample(Z_all, y_all)         # whole data set, scaled one-hot space (as in the honest runs)
    assert np.allclose(Z_res.iloc[:n_orig].values, Z_all.values)              # imblearn appends synthetic rows after the originals
    is_synth = pd.Series(np.arange(len(Z_res)) >= n_orig, index=Z_res.index)
    Ztr, Zte, ytr, yte, str_, ste = train_test_split(Z_res, y_res, is_synth, test_size=0.30, stratify=y_res, random_state=SEED)
    print(f"\nLEAKY demo: SMOTE on all {n_orig:,} rows ({Z_all.shape[1]} transformed columns) -> {len(Z_res):,} rows "
          f"({int(is_synth.sum()):,} synthetic); split {len(Ztr):,} / {len(Zte):,}; "
          f"synthetic share of leaky test defaulters {ste[yte == 1].mean():.3f}")
    out = []
    for learner in LEARNERS:
        clf = make_clf(learner)                                                # same classifier as the honest runs; matrix already transformed
        t0 = time.time()
        p_oof = cross_val_predict(clone(clf), Ztr, ytr, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
        aucs = fold_aucs(ytr, p_oof)
        cv_seconds = time.time() - t0
        t0 = time.time()                                                       # reset so fit_seconds = final fit only (as in run_study)
        clf.fit(Ztr, ytr); fit_seconds = time.time() - t0
        p = clf.predict_proba(Zte)[:, 1]
        real, synth = (~ste).values, ste.values
        row = dict(learner=learner, leaky_cv_auc=float(aucs.mean()), leaky_test_auc=float(roc_auc_score(yte, p)),
                   leaky_test_ap=float(average_precision_score(yte, p)),
                   leaky_test_auc_real_rows_only=float(roc_auc_score(yte[real], p[real])),
                   leaky_test_auc_synthetic_defaulters_vs_real_goods=float(roc_auc_score(yte[synth | (yte == 0).values], p[synth | (yte == 0).values])),
                   honest_cv_auc_smote=float(honest.loc[(honest.learner == learner) & (honest.strategy == "smote"), "cv_auc"].iloc[0]),
                   honest_test_auc_smote=float(honest.loc[(honest.learner == learner) & (honest.strategy == "smote"), "test_auc"].iloc[0]),
                   honest_test_auc_none=float(honest.loc[(honest.learner == learner) & (honest.strategy == "none"), "test_auc"].iloc[0]),
                   n_leaky_train=int(len(Ztr)), n_leaky_test=int(len(Zte)), n_synthetic_total=int(is_synth.sum()),
                   share_synthetic_in_leaky_test_defaulters=float(ste[yte == 1].mean()), leaky_test_default_rate=float(yte.mean()),
                   smote_feature_space=f"make_preprocessor() output ({Z_all.shape[1]} columns: 26 standardised numeric + 6 one-hot dummies), "
                                       "preprocessor fitted on all rows - same space as the honest in-fold SMOTE",
                   cv_seconds=round(cv_seconds, 1), fit_seconds=round(fit_seconds, 1))
        row["inflation_vs_honest_smote"] = row["leaky_test_auc"] - row["honest_test_auc_smote"]
        out.append(row)
        print(f"  {learner:<9s} leaky CV AUC {aucs.mean():.4f} | leaky 'test' AUC {row['leaky_test_auc']:.4f} (real rows only {row['leaky_test_auc_real_rows_only']:.4f}) "
              f"| honest SMOTE test AUC {row['honest_test_auc_smote']:.4f} | inflation {row['inflation_vs_honest_smote']:+.4f} | CV {cv_seconds:.0f}s + fit {fit_seconds:.0f}s")
        if learner == "rf":                                                    # the one evaluate() call of the demo (RF = the more dramatic case)
            evaluate("LEAKAGE DEMO - SMOTE before split (RF) - NOT a valid result", clf, Ztr, ytr, Zte, yte, member=MEMBER,
                     best_params=dict(learner="rf", strategy="smote applied to ALL 29,965 rows BEFORE the split", n_estimators=N_TREES,
                                      max_features="sqrt", min_samples_leaf=5),
                     cv_auc=float(aucs.mean()), cv_auc_std=float(aucs.std()), fit_seconds=round(fit_seconds, 1), p_oof=p_oof,
                     notes="LEAKAGE DEMONSTRATION ONLY - do not put in the model comparison table (files are quarantined under "
                           "results/leakage_demo/ and figures/leakage_demo/; filter slug.startswith('imb_leaky') in any aggregation). "
                           "SMOTE was applied to the whole data set (train+test) BEFORE splitting, in the same scaled one-hot feature "
                           "space as the honest runs (make_preprocessor() fitted on all rows), so the 'test' set is 50/50 balanced "
                           "(14,001 rows, not the shared 8,990-row test set) and its defaulters are mostly synthetic interpolations of "
                           "neighbours that sit in the training set. The 'test' AUC is therefore inflated relative to the honest "
                           "imb_rf_smote run (SMOTE inside the CV folds). Same RF, same split recipe/seed - only the leak differs. "
                           "fit_seconds = final fit only; CV time is in extra['cv_seconds'].",
                     extra=row, slug=LEAK_SLUG)
    leak = pd.DataFrame(out); leak.to_csv(os.path.join(RESULTS, "imbalance_leakage_demo.csv"), index=False)
    moved = quarantine_leaky_outputs()
    print("  quarantined leaky-demo files -> " + ", ".join(moved))
    return leak


# ----------------------------------------------------------------------------- figures
def bar_panel(ax, table, metric, title, ylim=(0, 1), fmt="{:.3f}"):
    width, gap = 0.12, 0.09
    for gi, learner in enumerate(LEARNERS):
        for si, strategy in enumerate(STRATEGIES):
            v = float(table.loc[(table.learner == learner) & (table.strategy == strategy), metric].iloc[0])
            x = gi + (si - 1.5) * (width + gap)
            ax.bar(x, v, width=width, color=C_STRAT[strategy], label=STRATEGY_LABEL[strategy] if gi == 0 else None, zorder=3)
            ax.text(x, v + 0.012, fmt.format(v), ha="center", va="bottom", fontsize=6.8, color=C_TXT2, zorder=4)
    ax.set_xticks(range(len(LEARNERS))); ax.set_xticklabels([LEARNER_LABEL[l] for l in LEARNERS])
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="x", length=0)


def fig_comparison(table):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    panels = [("cv_auc", "CV AUC (5 folds, training set)"), ("test_auc", "Test AUC"), ("test_ap", "Test average precision"),
              ("recall_1_at_0_5", "Recall of defaulters at threshold 0.50"), ("recall_1_at_tuned", "Recall at F1-tuned threshold (tuned on training OOF)"),
              ("precision_1_at_tuned", "Precision at F1-tuned threshold")]
    for ax, (m, t) in zip(axes.ravel(), panels):
        bar_panel(ax, table, m, t)
    thr_line = lambda l: " / ".join(f"{table.loc[(table.learner == l) & (table.strategy == s), 'tuned_threshold_f1'].iloc[0]:.2f}" for s in STRATEGIES)
    axes[1, 1].text(0.5, 0.97, "F1-tuned thresholds (none / balanced / SMOTE / under)\n"
                    f"logistic {thr_line('logistic')}   |   forest {thr_line('rf')}",
                    transform=axes[1, 1].transAxes, ha="center", va="top", fontsize=7.5, color=C_TXT2, linespacing=1.4)
    axes[0, 0].legend(loc="upper left", ncol=2, frameon=False, fontsize=8)      # inside the empty top band of the first panel
    fig.suptitle("Class-imbalance strategies: resampling moves the recall at 0.5, not the ranking (AUC / AP) - "
                 "threshold tuning on the plain model recovers the same recall", fontsize=11.5)
    return save_extra_figure(fig, "imbalance_comparison.png")


def fig_threshold_shift(table, p_test_store, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, learner in zip(axes, LEARNERS):
        for strategy in STRATEGIES:
            p = p_test_store[f"imb_{learner}_{strategy}"]
            thr = float(table.loc[(table.learner == learner) & (table.strategy == strategy), "tuned_threshold_f1"].iloc[0])
            ax.hist(p, bins=bins, histtype="step", lw=2, color=C_STRAT[strategy], density=True,
                    label=f"{STRATEGY_LABEL[strategy]}: mean p {p.mean():.2f}, F1 thr {thr:.2f}", zorder=3)
            ax.axvline(thr, color=C_STRAT[strategy], lw=1.2, alpha=0.9, zorder=2)
        ax.axvline(y_test.mean(), color=C_TXT2, lw=1, zorder=1)
        ax.text(y_test.mean() + 0.01, ax.get_ylim()[1] * 0.60, f"test prevalence {y_test.mean():.2f}", fontsize=8, color=C_TXT2)
        ax.set_title(f"{LEARNER_LABEL[learner]}: predicted P(default) on the test set", fontsize=10.5, loc="left")
        ax.set_xlabel("predicted probability of default"); ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
        ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95, edgecolor="none", facecolor="white")
    axes[0].set_ylabel("density")
    fig.suptitle("Re-balancing shifts the whole probability distribution to the right - and the tuned threshold shifts with it (vertical lines)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_extra_figure(fig, "imbalance_threshold_shift.png")


def fig_leakage(leak):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width, gap = 0.18, 0.03
    xlabels = []
    for gi, learner in enumerate(LEARNERS):
        r = leak.loc[leak.learner == learner].iloc[0]
        vals = [("honest: SMOTE inside CV folds, real test rows", r["honest_test_auc_smote"], C_STRAT["smote"], None),
                ("LEAKY: SMOTE on all rows before the split", r["leaky_test_auc"], C_LEAK, "//")]
        for vi, (lab, v, col, hatch) in enumerate(vals):
            x = gi + (vi - 0.5) * (width + gap)
            ax.bar(x, v, width=width, color=col, hatch=hatch, edgecolor="white" if hatch else col, lw=0.8, label=lab if gi == 0 else None, zorder=3)
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TXT2, zorder=4)
        xlabels.append(f"{LEARNER_LABEL[learner]}\nleaky 'test' AUC on its real rows only: {r['leaky_test_auc_real_rows_only']:.3f}")
    ax.set_xticks(range(len(LEARNERS))); ax.set_xticklabels(xlabels, fontsize=9); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("test AUC"); ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Leakage demonstration: SMOTE on ALL rows BEFORE the split inflates the 'test' AUC\n"
                 "same scaled one-hot feature space, same classifiers, same split recipe and seed - only the leak differs",
                 fontsize=10, loc="left")
    fig.tight_layout()
    return save_extra_figure(fig, "imbalance_leakage_demo.png")


def main():
    t_all = time.time()
    X_train, X_test, y_train, y_test, feature_names = load_split()
    print(f"train {X_train.shape}, test {X_test.shape}, default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")
    print(f"log prior ratio log(pi0/pi1) = {np.log((y_train == 0).mean() / (y_train == 1).mean()):.4f}\n")

    table, equiv, coef_df, p_test_store, p_oof_store = run_study(X_train, X_test, y_train, y_test)
    print("\nComparison table:\n" + table[["learner", "strategy", "cv_auc", "test_auc", "test_ap", "test_brier", "mean_p_test", "tuned_threshold_f1",
                                         "recall_1_at_0_5", "recall_1_at_tuned", "precision_1_at_tuned", "spearman_vs_none"]].round(4).to_string(index=False))
    d = coef_df.sub(coef_df["none"], axis=0)
    print(f"\nLogistic coefficients vs 'none' ({len(d) - 1} slopes; intercept shift vs largest slope shift):")
    for s in STRATEGIES[1:]:
        print(f"  {s:<9s} d_intercept {d.loc['intercept', s]:+.4f} | max |d slope| {d.drop('intercept')[s].abs().max():.4f} "
              f"({d.drop('intercept')[s].abs().idxmax()}) | slope Spearman with none {spearmanr(coef_df['none'].drop('intercept'), coef_df[s].drop('intercept')).statistic:.4f}")
    print("\nThreshold equivalence (plain-model threshold chosen on training OOF to reproduce the re-balanced model's decisions at 0.5; "
          "agreement then measured on the test set):\n" + equiv.round(4).to_string(index=False))

    leak = leaky_demo(X_train, X_test, y_train, y_test, table)

    fig_comparison(table); fig_threshold_shift(table, p_test_store, y_test); fig_leakage(leak)
    print(f"\nWrote results/imbalance_comparison.csv, results/imbalance_threshold_equivalence.csv, results/imbalance_leakage_demo.csv, "
          f"results/imbalance_lr_coefficients.csv, figures/imbalance_comparison.png, imbalance_threshold_shift.png, imbalance_leakage_demo.png; "
          f"leaky-demo evaluate() files under results/leakage_demo/ and figures/leakage_demo/. Total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
