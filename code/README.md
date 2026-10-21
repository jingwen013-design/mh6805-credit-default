# Project code — conventions (read before writing any model script)

Dataset: UCI "Default of Credit Card Clients" (Yeh & Lien 2009), 30,000 Taiwanese credit-card clients, target `DEFAULT`
(1 = defaulted on next month's payment, 22.1%). Raw file: `data/raw/default_of_credit_card_clients.csv`.

`common.py` is the shared contract — import it, do not modify it:
- `load_split()` → `X_train, X_test, y_train, y_test, feature_names`. Stratified 70/30 split, seed 2607526, 35 exact duplicates dropped
  (train 20,975 / test 8,990, default rate 22.1% in both). Everyone uses the same rows. Never fit anything on the test set.
- Features (29): LIMIT_BAL, AGE, PAY_1..PAY_6 (repayment status, ordinal ints −2..8), BILL_AMT1..6, PAY_AMT1..6, SEX/EDUCATION/MARRIAGE
  (categorical; EDUCATION 0/5/6 recoded to 4 = others, MARRIAGE 0 → 3 = others), plus 6 engineered features:
  UTIL_1, UTIL_MEAN (bill / limit), PAY_RATIO_MEAN (payment / bill), N_MONTHS_DELAYED, MAX_DELAY, BILL_TREND.
- `make_preprocessor()` → ColumnTransformer (StandardScaler on numeric, one-hot on the 3 categoricals). Put it INSIDE a sklearn
  `Pipeline` so it is refitted in every CV fold (no leakage). Tree ensembles may use `make_preprocessor(scale=False)`.
- `CV` = StratifiedKFold(5, shuffle=True, random_state=2607526); `SCORING = "roc_auc"` is the primary tuning criterion.
- `evaluate(name, fitted_model, X_train, y_train, X_test, y_test, member=..., best_params=..., cv_auc=..., cv_auc_std=..., notes=...,
  fit_seconds=..., extra={...})` computes AUC/AP/Brier, metrics at threshold 0.5 and at thresholds tuned on TRAINING out-of-fold
  probabilities (F1 and a 5:1 cost criterion), writes `results/<slug>.json`, `results/<slug>_p_test.npy` and `figures/<slug>_eval.png`.
  Call it once per final model. `slug=` lets you fix the file name.
- Freeze (since 2026-10-19): every configuration already scored on the test set is recorded in `results/frozen_manifest.json`
  (`python3 freeze.py` creates / verifies it; `--check` only verifies). `evaluate()` compares a run with that record before any
  test label is read: a frozen slug may be re-run only if it reproduces the frozen test predictions; a changed configuration
  under a frozen slug is refused (score it under a new slug, or set `MH6805_ALLOW_RESCORE="<reason>"`, which is logged).
  Choose models, thresholds and strategies on training CV / OOF only — the test set only reports.
- `recalibrate.py` (LI): Platt recalibration of the nine weighted / resampled models from their saved OOF / test vectors
  (fitted on training OOF only; registered in the manifest as `<slug>_platt`).
- RUAN's v2 scripts of 30 Sep (`mine_v2_utils.py`, `model_knn_v2.py`, `model_logistic_v2.py`, `model_unsupervised_v2.py`,
  merged unchanged on 2026-10-19): `run_all.sh` runs them only after her 30 Sep `results/*_v2.*` and
  `results/frozen_choices_ruan_v2.json` have been copied in with `cp -p` and adopted with `python3 freeze.py --adopt`.
- `save_extra_figure(fig, "name.png")` saves additional figures under `figures/`.
- Seeds: always `random_state=SEED` (2607526). CPU: pass `n_jobs=2` at most (several scripts run concurrently on an 8-core laptop).
- Runtime budget per script: about 15 minutes. If a grid is too slow, shrink the grid or tune on a stratified subsample and refit on
  the full training set — and say so in `notes=`.

Deliverables per model script: `code/model_<slug>.py` (runs end-to-end with `python3 model_<slug>.py` from this directory),
the JSON/figures written by `evaluate()`, and `report/notes_<slug>.md` with 5–10 bullets of interpretation (what was tuned, what the
best setting was, how it compares with the logistic baseline, why, caveats) that the report writer can lift.
Reference numbers (plain logistic regression, all features): CV AUC ≈ 0.760, test AUC 0.764, test AP 0.524; majority-class accuracy 0.779.

## Results of record
- Clustering: the results of record are RUAN's 30 Sep files (`results/*_v2.*`: PCA on the 32 preprocessed columns, 10 components,
  K-means K = 5, bootstrap ARI, per-cluster Jaccard, Ward agreement, incremental value of the cluster label by repeated CV on
  training rows) and their 21 Oct reproduction `results/*_repro.*` / `figures/*_repro.png` from `nested_cv_check.py` and
  `cluster_stability.py` (training rows only; they reproduce her numbers).
- `model_unsupervised.py` (K = 4, 16 Sep) is kept because `run_all.sh` must reproduce the 34-entry round-1 manifest. Its outputs
  (`kmeans_profiles.csv`, `kmeans_selection.csv`, `hclust_agreement.csv`, `pca_loadings.csv`, `pca_variance.csv`,
  `unsupervised_summary.json`, `figures/kmeans_*.png`, `pca_*.png`, `hclust_dendrogram.png`) are not used in the report or slides.
- `check_repro.sh` is the one-minute gate (raw-file hash, shared split, `freeze.py --check`); run it from the repository root after
  a clone and again after `run_all.sh`.
- `../requirements.txt` pins the package versions the results were produced with (Python 3.13.5).
