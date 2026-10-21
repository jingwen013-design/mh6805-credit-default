# Test-set log: every read of the held-out test rows, 16 Sep – 21 Oct 2026

Written 21 Oct 2026 as a record-hygiene task. This file lists **every time the 8,990 test rows (`data/processed/test.csv`) were used**: their labels, their features, or the frozen test predictions scored against those labels. For each read it says whether it influenced any modelling choice. `results/frozen_manifest.json` lists only the **configurations** that were scored through `common.evaluate()` or registered by `recalibrate.py`. This log is meant to be the complete list, as far as the files, the notes and the two reviews record it; the manifest is a subset of it.

**Sources.**
- `results/frozen_manifest.json`.
- Modification times of the result files.
- `results/mlp_comparison.csv`, column `json_written`.
- `results/frozen_choices_ruan_v2.json`.
- `report/notes_*.md`.
- The 6 Oct review `_audit/2026-10-06-评语审计原始发现.md` (lenses CAL, COST and T).
- The snapshots in `_audit/2026-10-19-*`.
- `report/CHANGELOG_2026-10-21.md`.
- `results/REPRODUCTION_2026-10-21.md`.
- The 21 Oct audit of every professor comment.

`_audit/` is a local folder of backups that `.gitignore` keeps out of the repository. The facts quoted from it are written out here so that this log stands on its own.

**How dates were established.** A date here is a file-modification time, so the read happened at or before it. Days are reliable; minutes are not always reliable. Three cases matter:

- **The 31 round-1 JSONs share 16 Sep 18:54.** They were rewritten together at that time, when the `member` fields were set. Their `*_p_test.npy` vectors were written between 11:09 and 15:42 that day, so those are the scoring times given below.
- **The manifest's `first_scored` is the result-file time.** It is not a scoring timestamp. For the forest and bagging it records the second scoring (18 Sep), not the first (16 Sep).
- **The test file in this folder was regenerated twice,** on 18 Sep 16:45 and 21 Oct 16:36; the fresh clone of 21 Oct wrote its own copy. `common.py` rebuilds the same split from the raw file with the same seed. Regenerating the file uses no label in any computation.

**What "influenced a choice" means here.** A modelling choice is a hyper-parameter, feature set, threshold, imbalance strategy, recalibration method, or the model carried forward. Choices of presentation or wording are listed separately where a test number led the text.

---

## Summary

| | Count |
|---|---|
| Manifest entries | 46 = 45 configurations + 1 leakage negative control |
| Round 1, 16–18 Sep | 34 configurations + the negative control |
| RUAN's second pass, 30 Sep | 2 re-fits of frozen configurations |
| Platt variants, 19 Oct | 9 |
| Extra configurations scored on test but not in the manifest | 8 models: 1 hinge-loss LinearSVC, 6 drop-one-column forests, 1 plain-logistic reference (C = 1) in the discriminant script. There are also exploratory variants: 98 truncated forests on the tree-count curve, and the 6 Oct review's isotonic and prior-corrected recalibrations |
| Manifest configurations scored more than once | Random forest and bagging (16 Sep and 18 Sep; the forest a third time as the full model of the drop-column script on 18 Sep). The plain `logistic`, re-fitted by RUAN and scored by 18 Sep outside the repository, then again on 30 Sep as `logistic_plain_v2`. `knn`, re-fitted and re-scored on 30 Sep as `knn_v2`. The nine Platt maps (applied on 6 Oct in the review, registered on 19 Oct). In addition, the 21 Oct fresh-clone reproduction (section C) re-ran every configuration through `evaluate()` and reproduced its frozen predictions. |
| Reads that influenced a modelling choice | No modelling choice was made on a test number. One choice was made after its test score had been seen, the hinge-to-squared-hinge swap of the linear SVM (B2): its stated reason is training CV, which supports the same choice. Until 19 Oct, test AUC set two presentation choices in `aggregate.py`: the reference model of the bootstrap and the four models drawn in the figures. |

---

## A. The 46 manifest entries

### A1. Round 1, 16–18 Sep 2026: 34 configurations and the negative control

These were scored before any freeze existed and before the professor's 30 Sep comment. The manifest registered them retroactively on 19 Oct 16:07. Hyper-parameters and both thresholds were chosen on training CV / OOF in every case.

| Scored (p_test time) | Entries, with test AUC |
|---|---|
| 16 Sep 11:09 | `lda` 0.7592, `qda` 0.7550, `naive_bayes` 0.7398, `naive_bayes_log` 0.7563 |
| 16 Sep 11:11 | `knn` 0.7706, `knn_k5` 0.7087 |
| 16 Sep 11:15 | `tree_pruned` 0.7755, `tree_full` 0.6125 |
| 16 Sep 11:18–11:23 | `logistic` 0.7636, `logistic_l2` 0.7636, `logistic_l1` 0.7636, `logistic_balanced` 0.7630 |
| 16 Sep 11:38 | `logistic_on_pcs` 0.7617, `logistic_plus_clusters` 0.7632 (K = 4 first pass) |
| 16 Sep 11:39 | `gradient_boosting` 0.7872, `xgboost` 0.7887, `xgboost_weighted` 0.7887 |
| 16 Sep ≤ 11:42 | `random_forest` 0.7866, `bagging` 0.7833: first scoring (`mlp_comparison.csv` json_written 16 Sep 11:42) |
| 16 Sep 15:26–15:30 | `imb_logistic_{none, balanced, smote, under}` 0.7637 / 0.7630 / 0.7624 / 0.7628; `imb_rf_{none, balanced, smote, under}` 0.7820 / 0.7818 / 0.7744 / 0.7814 |
| 16 Sep 15:29–15:32 | `mlp` 0.7731, `mlp_weighted` 0.7710, `mlp_overfit` 0.6776 |
| 16 Sep 15:31 (JSON 17 Sep 16:02) | `imb_leaky_smote_demo` 0.9142: negative control. It pools the training and the **real test rows**, applies SMOTE, then re-splits. So it did read the test rows and labels, although its "test" split is not the real test set. |
| 16 Sep 15:38–15:42 | `svm_linear` 0.7506, `svm_rbf` 0.7616, `svm_linear_calibrated` 0.7627 |
| 18 Sep 18:48 | `random_forest`, `bagging`: **second scoring**, by a re-run of `model_rf.py`. Same configuration, same test AUC / AP (0.7866 / 0.5756 and 0.7833 / 0.5712). The manifest dates them here. |
| 18 Sep 18:54 | `xgboost_monotone` 0.7870: a new configuration, scored after the round-1 numbers were known. The constraint signs were fixed a priori and the hyper-parameters were inherited unchanged from `xgboost.json`. |

**Influence.**
- **Modelling choices:** none. Every hyper-parameter and both cut-offs come from training CV / OOF.
- **The aggregation, until 19 Oct:** every `aggregate.py` run before 19 Oct 16:07 ranked the headline models by **test AUC**: `BEST` and `TOP4` = xgboost, gradient boosting, random forest, pruned tree (snapshot in `_audit/2026-10-19-aggregate-before-cv-selection/aggregate.py:140-148`). This chose the reference model of the paired bootstrap and the four models drawn in `threshold_curves.png`, `cost_curves.png` and `calibration.png`. It also wrote `*_best_strategy_by_test_auc` and `best_test_cost_curve_argmin`; no reader of those keys was found.
- **The aggregation, since 19 Oct:** the ranking is by training CV AUC. The leader is the same (XGBoost); the MLP replaces the pruned tree in the top four.
- **Wording:** the monotone model's price and the imbalance paragraph were first written from test differences (−0.0016 and 0.0076). The training-CV evidence (−0.0019, 0/5 folds; 0.0060) points the same way.

### A2. 30 Sep 2026: RUAN's second pass, 2 entries

| Scored | Entries |
|---|---|
| 30 Sep 21:39 | `knn_v2` 0.7706 (K = 101, uniform) |
| 30 Sep 21:40 | `logistic_plain_v2` 0.7637 (unpenalised) |

- **What these are:** re-fits of the round-1 `knn` and `logistic` configurations, run on her machine after the professor's comment of the same day. They were frozen first in her `frozen_choices_ruan_v2.json` (21:39:46 / 21:40:13) and adopted into the manifest on 21 Oct 14:59.
- **Test reads in the same run (reporting):** test cost sensitivity at nine cost ratios (`knn_cost_sensitivity_v2.csv`, `logistic_cost_sensitivity_v2.csv`); a paired bootstrap of threshold saving against model saving versus the frozen XGBoost vectors (`threshold_vs_model_saving_v2.csv`, B = 1,000); paired AUC / AP intervals against XGBoost (`model_logistic_v2.py:124-125`).
- **Not scored:** the K = 5 cluster model, because her pre-registered repeated-CV rule did not adopt it (`model_unsupervised_v2.py`). The selected logistic is the plain one under the 1-SE rule, so no separate `logistic_selected_v2` was scored.
- **Influence:** none on a model choice; the selections come from nested CV on training rows. These are the proposal-v2 figures (7,065 at 0.5, cut-off 0.17, 5,141).

### A3. 19 Oct 2026, 16:24–16:25: nine Platt-recalibrated variants

`recalibrate.py` registered each of these before reading its labels: `logistic_balanced_platt`, `imb_logistic_{balanced, smote, under}_platt`, `imb_rf_{balanced, smote, under}_platt`, `xgboost_weighted_platt`, `mlp_weighted_platt`. Their test AUC is unchanged by Platt.

- **Why they were scored:** after the professor's comment, for reporting.
- **Platt was pre-committed in writing** in proposal v2 on 30 Sep.
- **This was not the first look:** the same Platt maps, with isotonic and prior-corrected alternatives, had already been applied to the test vectors in the 6 Oct review (B8 below).
- **Influence:** none on a modelling choice. The decision not to adopt weighting is supported on training rows by `nested_cv_check.py`: the weighted and recalibrated logistic changes the cross-fitted cost by −0.43%, which is below the pre-registered 1% rule, so the plain model is kept.

---

## B. Reads outside the manifest: configurations or test-label computations not recorded there

| # | When | What | Where it is recorded | Influenced a choice? |
|---|---|---|---|---|
| B1 | 16 Sep, before 11:09 (`discriminant_roc_pr.png` 11:09); still in the code | `model_discriminant.py:83-90` fits a **plain logistic reference** (L2, C = 1, lbfgs) beside LDA and scores it on test **before** LDA's `evaluate()`: test AUC 0.7637, AP 0.5235, plus ROC / PR curves. This is not the frozen `logistic` (unpenalised) or `logistic_l2` (C = 31.6). | `results/lda.json` extra.logistic_reference; `figures/discriminant_roc_pr.png` | No. A diagnostic (coefficient cosine 0.934, Spearman of test scores 0.990). On 21 Oct the code was changed to compute this reference on training out-of-fold scores only (not re-run; the JSON field and figure keep the old test-based values). |
| B2 | 16 Sep, on or before the 15:38–15:42 SVM scorings | **Hinge-loss LinearSVC**, an earlier iteration of `model_svm.py`, scored on test: AUC 0.7406. It was then replaced by the squared-hinge `svm_linear_calibrated`. | `report/notes_svm.md` bullet 8 (17 Sep 16:02) | The swap was made after the hinge model's test score was seen. The documented reason is training evidence: full-training CV 0.7611 against 0.7266, and the task's `LinearSVC` specification. That evidence supports the same choice, but the test read came first. |
| B3 | 16 Sep 11:42 and 18 Sep 18:48 | **Random forest and bagging scored twice**, same configuration, identical metrics. | `mlp_comparison.csv` json_written; `notes_rf.md` (17 Sep) already quotes 0.7866; result files 18 Sep 18:48 | No. But the round-1 label in the manifest ("scored once … not re-scored since") and "scored once" in the report are inaccurate for these two. |
| B4 | 18 Sep 18:42 (`rf_auc_vs_ntrees.csv`); probably also in the 16 Sep run, whose outputs were overwritten | `model_rf.py`: **test AUC against the number of trees** for both forests (k = 10…500 in steps of 10: 98 truncated ensembles besides the two frozen ones), computed **before** `evaluate()`. | `results/rf_auc_vs_ntrees.csv` columns test_auc_rf / test_auc_bagging (kept as a record); `random_forest.json` / `bagging.json` extra.test_auc_vs_ntrees | No. `N_TREES = 500` was a constant fixed in advance; the 200-tree grid budget is justified by the OOB curve. **Removed from the code on 21 Oct 2026, without a re-run.** The figure was redrawn from the OOB columns only. |
| B5 | 18 Sep 18:54 | `model_monotone.py`: before `evaluate()`, the test default rate at start-up and the **test AUC of the refitted unconstrained twin** (= 0.7887, the frozen `xgboost` value). After `evaluate()`: a paired bootstrap, B = 500, constrained against unconstrained. | `monotone_summary.json`; the run log | No. It was a reproduction check, and the configuration was fixed a priori. **The pre-`evaluate()` reads were removed on 21 Oct 2026, without a re-run.** |
| B6 | 18 Sep 21:58 – 19 Oct 15:59 | `rf_drop_column.py`: the **full forest (a third scoring of the `random_forest` configuration) and six drop-one-column forests** (PAY_1, UTIL_MEAN, BILL_AMT1, LIMIT_BAL, PAY_AMT2, UTIL_1) scored on test. Without PAY_1 the test AUC is 0.7800 against 0.7866. | `_audit/2026-10-19-shap-dropcol-before/results/rf_refit_drop_column.csv` (7 rows with test_auc / test_loss) | No. The six columns were the top six by OOB permutation importance (training), and the report quoted only the OOB losses. The script was moved to OOB-only on 19 Oct 15:59. |
| B7 | ≤ 18 Sep 21:36 (outside this repository) | RUAN's **separately fitted plain logistic** scored at six cost ratios for the first proposal draft ("Project 复现/proposal_1.tex", 18 Sep 21:36: 0.17, 7,065, 5,141). The same configuration was re-run on 30 Sep as `logistic_plain_v2`. | 6 Oct review, finding T2 | No. These are proposal-stage figures only; the report footnotes them. |
| B8 | 6 Oct 2026 (review, read-only) | An AI-assisted audit of the project against the professor's comment recomputed many quantities from the frozen `*_p_test.npy` vectors and the test labels: **(a)** Platt, **isotonic** and analytic **prior-corrected** maps for the nine weighted / resampled models (test Brier, mean p, cost at 1/6 and at searched cut-offs); **(b)** isotonic versions of the ten headline models (cost at 0.5, e.g. XGBoost 6,642 → 6,493) and the RBF SVM's decile calibration; **(c)** paired bootstraps of threshold saving against model saving (B = 4,000); test costs at 0.15–0.19 and at exactly 1/6; **(d)** a winner's-curse bootstrap (B = 1,000) and the Spearman correlation of CV and test AUC (0.95). | `_audit/2026-10-06-评语审计原始发现.md` (CAL-1…6, COST-1…7, T1…T9) | **Recalibration method:** Platt was kept because proposal v2 had committed to it on 30 Sep, and the review saw Platt and isotonic agree within 0.0013 Brier. So the choice did not hinge on the test numbers, but the 19 Oct Platt scoring was not blind. **Weighting not adopted:** that decision is now backed by training-only nested CV (A3). **Otherwise** the review shaped wording and the fix list: freeze, CV ranking, cost range. It changed no model. |
| B9 | 16–17 Sep (summary file 17 Sep 16:02) | `model_unsupervised.py`: **test default rate and lift per K = 4 cluster**. | `unsupervised_summary.json` default_rate_test | No. K = 4 was chosen on training rows, and the K = 5 clustering of record never touched test. |
| B10 | 16 Sep | `model_imbalance.py`: test agreement and recall statistics between strategies at equivalent cut-offs; mean predicted probability on test. | `imbalance_threshold_equivalence.csv`, `imbalance_comparison.csv` | No. Descriptive. The strategy is compared by CV AUC (since 19 Oct the key is `*_best_strategy_by_cv_auc`). |
| B11 | 16 Sep | `model_mlp.py`: after `evaluate()`, a paired bootstrap (B = 2,000) of the MLP against the other saved test vectors. | `mlp.json` extra.paired_bootstrap_test | No. Reporting only. |
| B12 | 16 Sep | `model_svm.py`: after `evaluate()`, the test AUC of the 3-dp-rounded linear-SVM scores and of MAX_DELAY alone (0.7202 / 0.3987). | `svm_linear.json` extra | No. A degeneracy check. |

---

## C. Reporting reads of frozen test predictions (no new configuration)

These read the test labels against already-frozen vectors and select nothing. They are listed so that the count of reads is complete.

| When | Script / activity | Output |
|---|---|---|
| 16–18 Sep, then every run up to 19 Oct 16:07 | `aggregate.py` (ranking by test AUC until 19 Oct, see A1): ROC / PR overlays, AUC bars, paired bootstrap of AUC / AP differences, threshold, cost and calibration curves | `model_comparison*`, `auc_bootstrap_ci.csv`, `aggregate_summary.json`, figures |
| 19 Oct 16:07–16:08 | `aggregate.py` (CV ranking). It still draws a "test-set minimum" marker on the cost curve and stores `best_test_cost_curve_argmin`, as a labelled diagnostic | same files |
| 21 Oct 15:02–15:04 | `threshold_vs_model_saving.py`: threshold saving against model saving at 1:1–20:1, paired bootstrap (B = 1,000) | `threshold_vs_model_saving.*`, figure |
| 21 Oct 15:46 | `cost_sensitivity.py`: test cost at each model's training-OOF cut-off, 1:1–20:1 | `cost_sensitivity.csv`, `cost_anchors.json` (training rows only), figure |
| 21 Oct, late evening | `cost_sensitivity.py` re-run on the shared 0.01 cut-off grid (reporting only; frozen vectors re-read, nothing refitted); one cell changed (logistic at 20:1, 6,976 -> 6,993) | `cost_sensitivity.csv`, figure |
| 21 Oct 15:46 | `eda.py`: test class balance (split check, 22.12%) and the target-distribution figure. The data-quality counts include the test rows' **features** (Section D) | `eda_*` |
| 21 Oct, afternoon | Report verify / fix rounds: at least one recount against the test labels (naive Bayes at 0.15 → 5,473 against 5,337 at 0.25) | `report/CHANGELOG_2026-10-21.md` line 244 |
| 21 Oct 17:13 | `make_slide_figures.py`: ROC / PR and cost curves on the test rows for the deck | `figures/slides/` |
| 21 Oct 16:49–18:05 (fresh clone, not this folder) | Full `run_all.sh` reproduction. Every `evaluate()` and every reporting script above re-read the test set; every frozen configuration came back `reproduced`. | `results/REPRODUCTION_2026-10-21.md` |

---

## D. Test features read without labels, or the file opened and discarded

- **Predictions on test features, compared with frozen vectors and no labels:**
  - `model_pdp.py` (reproduction check of `xgboost_p_test.npy`);
  - `model_monotone.py` (the same check, now predictions only);
  - `model_shap.py` before 19 Oct 16:02, which explained all 8,990 test rows with TreeSHAP and has used training rows since;
  - `model_knn.py` and `model_svm.py` (prediction timing).
- **Feature counts over the full download, test rows included:** the data-quality table `results/eda_data_quality.csv`. Its counts only describe fixed cleaning rules. The wrong claim "test set untouched in this EDA" was corrected on 21 Oct, by a patch to the CSV and to `eda.py`, without re-running `eda.py`.
- **Calls that open `test.csv` and discard it:** `load_split()` in `nested_cv_check.py`, `cluster_stability.py`, `model_shap.py`, `rf_drop_column.py` and `make_proposal_figures.py`. Since 21 Oct the new training-only scripts read `train.csv` directly instead.

---

## E. Corrections this log makes to earlier statements

1. **"Every configuration that has ever been scored on the test set" (README, until 21 Oct).** Not true. B1, B2, B6, B7 and the 6 Oct recalibration variants (B8) are outside the manifest. The manifest records the configurations scored through `evaluate()` or registered by `recalibrate.py`.
2. **"Scored once … not re-scored since" (manifest round-1 label; "scored once" in the report).** Not true for the random forest and bagging (B3; the forest also in B6). It is also not true for the `knn` / `logistic` configurations, which RUAN re-fitted and re-scored (A2, B7), and every configuration was re-run, with its frozen predictions reproduced, in the 21 Oct fresh-clone reproduction (C). The manifest entries are frozen and were not edited. This log is the correction.
3. **`first_scored`.** It is a file time, not a scoring time (see "How dates were established"). Since 21 Oct the manifest summary says so (`first_scored_is`).
4. **The negative control's note "its 'test' rows are not the real test set".** True of its re-split, but the demonstration pools the real test rows and labels before re-splitting (A1).
5. **Platt "scored once on 19 October".** It is the first registered scoring. The same maps had been applied to the test vectors on 6 Oct (B8).

## F. Activity that did not read the test set

- **21 Oct, whole-project audit against the professor's comments, with verifiers:** training rows and OOF vectors only.
- **The R cross-check (`code/r_crosscheck/`, 21 Oct 22:19):** reads `train.csv`, `cv_folds.csv` and the OOF vectors.
- **The training-only diagnostic scripts added on the evening of 21 Oct:** `ablation_checks.py`, `fairness_check.py`, `fold_stability.py`, `monotone_respec_check.py`, `pay_coding_check.py`, `tuning_checks.py`. As of 22:45 that day, by their docstrings and a grep, none calls `load_split()` or `evaluate()` or opens `test.csv`; two of them disable `load_split()` explicitly. They read `train.csv` and the OOF vectors.
- **This hygiene pass:** the code edits were not run against data. `freeze.py` hashes files. The figure redraw reads `results/rf_auc_vs_ntrees.csv` (its OOB columns) and `random_forest.json`. The threshold-grid check used `train.csv` labels and the OOF vectors only.
