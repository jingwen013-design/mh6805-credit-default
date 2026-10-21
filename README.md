# Predicting credit-card default: a comparative study of machine-learning classifiers

Group project for MH6805 Machine Learning in Finance (MSc Financial Technology, Nanyang Technological University), Group B19, Semester 1 2026/27.

A missed defaulter costs a card issuer several times more than a false alarm, so a default model has to be judged at an operating point, not by ranking alone. We compare 34 classifier configurations (KNN, the logistic family, LDA / QDA / naive Bayes, a pruned tree, bagging and random forest, gradient boosting and XGBoost, linear and RBF SVM, a feed-forward network, and a class-imbalance study) on the UCI *Default of Credit Card Clients* data: 29,965 Taiwanese card holders after dropping 35 exact duplicates, 29 predictors (23 raw + 6 engineered), 22.1 % defaulters, one stratified 70/30 split with seed 2607526 (20,975 training rows, 8,990 test rows). Every choice is made on training rows and each configuration is frozen at its first test scoring (every read of the test set is dated in `results/TEST_SET_LOG.md`). XGBoost wins on training CV and also leads on the test set (CV AUC 0.786, test AUC 0.789) against the KNN benchmark (0.770 / 0.771) and the plain logistic baseline (0.760 / 0.764); accuracy alone is uninformative, because predicting "no default" for everyone scores 77.9 %. The practical finding is that the operating point matters more than the model: at a 5:1 cost of a missed default to a false alarm, moving the logistic cut-off from 0.5 to its training-OOF optimum of 0.15 saves 1,937 [1,604, 2,218] cost units on the test rows (7,063 to 5,126), whereas switching from logistic to XGBoost at tuned cut-offs saves 145 [−18, 299] (paired bootstrap, B = 1,000). The 5:1 ratio is one point of a 1:1–20:1 sensitivity: the model-saving interval includes zero at 5:1 and 20:1, and at 20:1 the optimal cut-off is 0.01, i.e. flag everyone.

## Team

| Member | Student ID | Model families and components |
|---|---|---|
| RUAN Hangjing | G2607546F | KNN benchmark; logistic family (plain, ridge, lasso, class-weighted); nested-CV check of the frozen choices; PCA + K-means / Ward segmentation and its stability |
| LU Xinyu | G2603537D | Pruned decision tree; bagging and random forest (with drop-column refits); gradient boosting and XGBoost; monotone-constrained XGBoost; TreeSHAP; partial dependence |
| LI Jingwen | G2607526K | Linear and RBF SVM; feed-forward network; LDA / QDA / naive Bayes; class-imbalance study; Platt recalibration; freeze machinery; aggregation and cost analysis |

EDA, the report and the slides are shared. The contribution statement with percentages per section is in `slides/contribution_statement_draft.md` and in the report.

## Protocol

The professor's written comment on the proposal (30 Sep 2026) asked us to freeze modelling, feature and threshold choices before any further use of the test set, to show the cost ratio as a sensitivity range rather than a single truth, to report the incremental out-of-sample value and the stability of the clusters, to keep resampling inside folds and to recalibrate probabilities after weighting or SMOTE. The repository implements this as follows.

1. **One split, saved to disk.** `code/common.py` cleans the raw file, engineers the six extra features and writes the stratified 70/30 split once (`data/processed/train.csv`, `test.csv`, `split_meta.json`). Every script loads these rows; nothing is fitted on the test set.
2. **Pipelines inside folds.** The preprocessor (standardisation of the numeric columns, one-hot encoding of SEX / EDUCATION / MARRIAGE) and any resampler sit inside an sklearn / imbalanced-learn `Pipeline`, so they are refitted in each of the five stratified CV folds. A deliberate negative control (SMOTE applied before the split; test AUC 0.914) is kept in `results/leakage_demo/` to show what leakage looks like.
3. **Selection on training data only.** Hyper-parameters by `GridSearchCV` on the training folds; the reference model and the top four in `code/aggregate.py` by training CV AUC (OOF AUC for the RBF SVM, which was tuned on a stratified subsample); the imbalance strategy by CV AUC. Both decision thresholds (F1 and the 5:1 cost criterion) are chosen on training out-of-fold probabilities. CV top four: XGBoost, gradient boosting, random forest, MLP.
4. **Freeze before test.** `results/frozen_manifest.json` records every configuration scored through `common.evaluate()` or registered by `recalibrate.py`: 46 entries = 34 configurations scored 16–18 Sep 2026 + 2 entries adopted from RUAN's 30 Sep second pass (`knn_v2`, `logistic_plain_v2`) + 9 Platt-recalibrated variants scored 19 Oct + 1 leakage negative control. It is not the complete list of test-set reads. Outside it are: a superseded hinge-loss linear SVM; six drop-one-column forests (18 Sep, moved off the test set on 19 Oct); a logistic reference inside the discriminant script; the tree-count test curve; the second scoring of the forest and bagging; and the 6 Oct review's exploratory recalibration comparisons. Every read, dated and with whether it influenced any choice, is in `results/TEST_SET_LOG.md`. Each entry stores the configuration hash and the SHA-256 of the frozen test predictions (`results/frozen/*.npy`). The guard in `common.evaluate()` compares a run with the manifest before any test label is read: a frozen slug may be re-run only if it reproduces the frozen predictions; a changed configuration under a frozen name is refused (score it under a new slug, or set `MH6805_ALLOW_RESCORE="<reason>"`, which is logged). `python3 code/freeze.py --check` verifies the whole manifest.
5. **Test set for reporting only.** Bootstrap confidence intervals, overlaid ROC / PR curves, calibration and cost curves and the cost-ratio sensitivity read the test rows but select nothing. Post-hoc explanations (drop-column refits by out-of-bag AUC, TreeSHAP, partial dependence), the nested-CV check and the clustering stability analysis use training rows only.
6. **Recalibration.** The nine class-weighted or resampled models are Platt-recalibrated on training out-of-fold scores with cross-fitted thresholds (`code/recalibrate.py`); each recalibrated variant is its own frozen entry.

## Repository map

| Folder | Contents |
|---|---|
| `code/` | `common.py` (shared contract: cleaning, features, split, preprocessor, `evaluate()` with the freeze guard); `fetch_data.py`; `eda.py`; one `model_<family>.py` per model family; RUAN's 30 Sep scripts `model_knn_v2.py`, `model_logistic_v2.py`, `model_unsupervised_v2.py`, `mine_v2_utils.py`; `nested_cv_check.py`, `cluster_stability.py`; `recalibrate.py`, `threshold_vs_model_saving.py`, `cost_sensitivity.py`; `aggregate.py`, `freeze.py`; `make_report_appendix.py`, `make_slide_figures.py`; `run_all.sh`, `check_repro.sh`. Conventions in `code/README.md`. |
| `data/raw/` | The UCI `.xls` as downloaded and the `.csv` that `fetch_data.py` writes from it. |
| `data/processed/` | `train.csv`, `test.csv`, `split_meta.json`: the one shared split. |
| `results/` | One `<slug>.json` per scored configuration (metrics, ROC / PR curves, threshold tables) with `<slug>_p_oof.npy` and `<slug>_p_test.npy`; `frozen_manifest.json` and `frozen/` (the 46 frozen test-prediction vectors); `recalibrated/` (Platt test and cross-fitted OOF vectors); `leakage_demo/`; `model_comparison*.csv/.md`, `auc_bootstrap_ci.csv`, `aggregate_summary.json`; `threshold_vs_model_saving.*`, `cost_sensitivity.csv`, `cost_anchors.json`; `recalibration.*`; `nested_cv_*.json`, `unsupervised_summary*.json`, `*_v2.*`, `*_repro.*`; EDA tables; grid and curve CSVs per family. |
| `figures/` | `eda_*.png`; `<slug>_eval.png` (confusion matrix + ROC + PR) per configuration; `all_roc.png`, `all_pr.png`, `all_auc_bar*.png`, `calibration.png`, `cost_curves.png`, `threshold_curves.png`, `cost_sensitivity.png`, `threshold_vs_model_saving.png`, `recalibration.png`; importance, SHAP, partial-dependence and monotone figures; `*_repro.png`; `slides/` (eleven 1920 × 960 renders placed in the deck); `leakage_demo/`. |
| `report/` | `report.tex` + `sec_*.tex` in the NeurIPS 2021 style (`neurips_2021.sty` supplied by the instructor), `build_report_tex.sh`, `MH6805_Group_Project_Report.pdf`; `sec_appendix.tex` is generated by `code/make_report_appendix.py`; `notes_<family>.md` are the per-family interpretation notes the report was written from. |
| `proposal/` | `proposal.tex`, `build_proposal_tex.sh`, `MH6805_Group_Project_Proposal.pdf` (4 pages, same style file). |
| `slides/` | `slide_spec_2026-10-21.md` (slide-by-slide specification with speaker notes, English then Chinese), `qa_bank.md`, `contribution_statement_draft.md`. |

### Which result files are of record

- **Headline tables** use the 16–18 Sep scoring of every model: `results/<slug>.json`, aggregated in `model_comparison*.csv/.md` and `aggregate_summary.json`.
- **`*_v2.*` (RUAN, 30 Sep)** are her second pass: nested CV, pre-registered selection rules, cost sensitivity and the K = 5 clustering with stability. Her `knn_v2` and `logistic_plain_v2` refits are the same configurations as the frozen `knn` (K = 101, uniform weights) and `logistic` (plain) and differ only by solver tolerance (test AUC 0.7706 vs 0.7706 and 0.7637 vs 0.7636); the report footnotes her proposal-stage figures (cost 7,065 at 0.5, 5:1 cut-off 0.17).
- **`*_repro.*` (21 Oct)** come from `nested_cv_check.py` and `cluster_stability.py`, which re-run the training-only parts of her analysis on the shared split and reproduce her numbers: KNN nested-CV AUC 0.768 ± 0.007 with K = 101 uniform re-chosen on the full training set; logistic family nested AUC 0.760–0.761 with the 1-SE rule keeping the plain model; PCA with 10 components = 81.8 % of the variance; K = 5, silhouette 0.269; bootstrap ARI 0.957 [0.745, 0.990]; the cluster label as an extra predictor changes AUC by +0.0001 [−0.0004, +0.0007] and AP by −0.0003 [−0.0016, +0.0010] under 5 × 5 repeated CV with Nadeau–Bengio intervals, so it was not adopted and the test set was not consulted. These are the clustering numbers in the report and slides.
- **K = 4 first pass (16 Sep)**: `kmeans_profiles.csv`, `kmeans_selection.csv`, `hclust_agreement.csv`, `pca_loadings.csv`, `pca_variance.csv`, `unsupervised_summary.json` and `figures/kmeans_*.png`, `pca_*.png`, `hclust_dendrogram.png` are written by `model_unsupervised.py`. They are kept only because `run_all.sh` must reproduce the round-1 manifest; the report and slides do not use them.

## How to reproduce

```bash
git clone https://github.com/jingwen013-design/mh6805-credit-default.git
cd mh6805-credit-default
pip install -r requirements.txt   # Python 3.13.5; the versions the results were produced with are pinned
zsh code/check_repro.sh           # about 1 minute: raw-file hash, the shared split, every frozen test scoring
zsh code/run_all.sh               # 40–60 minutes on a laptop: regenerates every result and figure, then freeze.py --check
```

- `fetch_data.py` downloads the UCI zip if the `.xls` is absent and checks its SHA-256 before anything else runs.
- RUAN's `*_v2.py` scripts are gated behind `MH6805_RUN_V2=1` in `run_all.sh`: each ends with a test-set evaluation of a configuration that is already frozen, and a re-fit on another machine could drift past the 1e-6 tolerance, in which case `evaluate()` refuses it. Their training-only parts (`nested_cv_check.py`, `cluster_stability.py`) run by default.
- **Reproduced on a fresh clone on 21 Oct 2026** (`results/REPRODUCTION_2026-10-21.md`): every selection and test metric came back identical, `freeze.py --check` passed on the regenerated results, and the only differences were wall-clock fields and floating-point noise below 1e-15. The full run took 76 minutes at reduced priority while other work ran; allow 40–60 minutes on an idle laptop.
- One model at a time: `cd code && python3 model_rf.py` writes `results/random_forest.json`, `figures/random_forest_eval.png` and the companion files. Scripts use at most `n_jobs=2`.
- Report: `zsh report/build_report_tex.sh` (needs `pdflatex` and the Python package `pymupdf` for the page check); proposal: `zsh proposal/build_proposal_tex.sh`.

## Deliverables

- **Report**: `report/MH6805_Group_Project_Report.pdf`, built by `report/build_report_tex.sh` from the NeurIPS 2021 template; body of at most 7 pages including references, appendix from page 8. Due with code and data on 29 Oct 2026.
- **Proposal**: `proposal/MH6805_Group_Project_Proposal.pdf` (4 pages).
- **Presentation**: 27 Oct 2026, 10 minutes plus Q&A, all three members presenting; slides due 26 Oct noon. `slides/` holds the speaker specification (the running order and cut rules are in `slides/slides_outline.md`), the Q&A bank and the contribution draft. The final 15-slide deck is exported here as `slides/MH6805_B19_Presentation.pdf` and `slides/MH6805_B19_Presentation.pptx` (Canva design DAHYodggd94, view link https://www.canva.com/d/z5qNbWxTPuQSw2t); the slide figures come from `code/make_slide_figures.py`.

## Data and licence

- Data: UCI *Default of Credit Card Clients* (Yeh & Lien, 2009; UCI Machine Learning Repository id 350, https://doi.org/10.24432/C55S3H), licensed CC BY 4.0 and redistributed unchanged under `data/raw/`. SHA-256 of the `.xls`: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
- Code, figures, report and slides: MIT licence (see `LICENSE`).

## Changelog

- **16–18 Sep 2026**: round 1. Shared split and EDA; 34 configurations tuned on training CV and scored on the test set (the leakage negative control on 17 Sep, the monotone XGBoost on 18 Sep; the forest and bagging scored twice with the same configuration, 16 and 18 Sep); 4-page proposal written.
- **30 Sep**: the professor's written comment on the proposal (freeze before further test use, cost ratio as a range, cluster incremental value and stability, resampling inside folds, recalibration after weighting). RUAN's second pass: nested CV, pre-registered selection rules, cost sensitivity, K = 5 clustering with stability (`results/*_v2.*`); revised proposal submitted.
- **19 Oct**: `freeze.py` and the guard in `common.evaluate()`; `results/frozen_manifest.json` created; Platt recalibration of the nine weighted / resampled models; reference model and top four now chosen by training CV AUC; drop-column refits, TreeSHAP and partial dependence moved off the test set.
- **21 Oct**: RUAN's 30 Sep result files adopted into the manifest (46 entries); `nested_cv_check.py` and `cluster_stability.py` (`*_repro.*`); threshold-versus-model cost sweep 1:1–20:1 with paired bootstrap; cost sensitivity of every headline model; report revised (cost range, K = 5 clustering, contribution statement); slide specification and Q&A bank rewritten; this repository created.
- **21 Oct, record hygiene**: `results/TEST_SET_LOG.md` written. `model_rf.py` no longer computes a test curve against the number of trees: the figure was redrawn from the OOB columns and the script was not re-run. `model_monotone.py` no longer reads test labels before `evaluate()`. `cost_sensitivity.py` uses the 0.01 cut-off grid of `threshold_vs_model_saving.py` (not re-run: only the logit's 20:1 cell depends on it). The manifest's stale summary block was refreshed by `freeze.py` after a clean check, with no frozen entry changed. The data-quality table now states its scope, and its imputation row no longer contradicts the body.
