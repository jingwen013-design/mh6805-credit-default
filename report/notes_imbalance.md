# Notes — Class-imbalance strategy study (LI Jingwen, slug `imbalance`)

Files: `code/model_imbalance.py` · `results/imb_logistic_{none,balanced,smote,under}.json`, `results/imb_rf_{none,balanced,smote,under}.json`,
`results/imbalance_comparison.csv`, `results/imbalance_threshold_equivalence.csv`, `results/imbalance_leakage_demo.csv`,
`results/imbalance_lr_coefficients.csv` · `figures/imbalance_comparison.png`, `figures/imbalance_threshold_shift.png`,
`figures/imbalance_leakage_demo.png` (+ one `imb_<slug>_eval.png` per evaluate() call).
The leakage demonstration's evaluate() output is **quarantined** in `results/leakage_demo/imb_leaky_smote_demo{.json,_p_test.npy,_p_oof.npy}` and
`figures/leakage_demo/imb_leaky_smote_demo_eval.png` — it is NOT a model result, `common.load_all_results()` no longer loads it, and any report
code that scans sub-folders must filter `slug.startswith("imb_leaky")` (its "test" set has 14,001 rows and 50 % prevalence, not the shared 8,990).
All numbers below are copied from the JSON/CSV files (seed 2607526, shared 70/30 split, train n = 20,975 with 22.1 % defaulters, test n = 8,990).
Runtime 334 s with `n_jobs=2` (wall-clock on a shared laptop); no subsampling was needed. Every metric is fully seeded and reproduces to all
printed digits between runs — only the `fit_seconds` / `cv_seconds` bookkeeping moves.

- **What was compared (lecture B4/B6).** Two base learners — logistic regression (C = 1) and a random forest (500 trees,
  `max_features='sqrt'`, `min_samples_leaf=5`) — under four imbalance strategies: (a) none, (b) `class_weight='balanced'`
  (cost-sensitive re-weighting, no new rows), (c) SMOTE and (d) random under-sampling, both to a 1:1 ratio. All eight
  combinations use `make_preprocessor()` inside the pipeline, the same five stratified folds (`common.CV`), `roc_auc` as the
  CV criterion, and thresholds tuned on training out-of-fold probabilities only. **The samplers are steps of an
  `imblearn.pipeline.Pipeline`** — a sampler step is executed only in `fit()`, so inside CV every training fold is resampled
  while the validation fold is left untouched, and at prediction time the sampler is skipped. That is the leak-free way:
  OOF probabilities, tuned thresholds and the test set never contain a synthetic row (SMOTE makes the classifier see 32,668
  rows, under-sampling 9,282, the original 20,975 otherwise). Because both learners share the scaled preprocessor, SMOTE and
  under-sampling generate the *same* resampled rows for both learners in a given fold (trees are invariant to rescaling).

- **Headline: re-balancing does not improve the ranking.** Logistic regression — CV AUC 0.7602 / 0.7609 / 0.7600 / 0.7597 (fold SD
  ≈ 0.010) and **test AUC 0.7637 / 0.7630 / 0.7624 / 0.7628, test AP 0.5235 / 0.5171 / 0.5162 / 0.5153** for none / balanced / SMOTE /
  under. Random forest — CV AUC 0.7800 / 0.7805 / 0.7745 / 0.7790 (fold SD ≈ 0.007), **test AUC 0.7820 / 0.7818 / 0.7744 / 0.7814,
  test AP 0.5694 / 0.5688 / 0.5544 / 0.5579**. No strategy improves on "none" beyond noise: the CV-AUC differences (balanced sits
  +0.0007 / +0.0005 above none) are an order of magnitude inside the fold SD, and on the test set "none" is the highest on both AUC and
  AP for both learners; the only visible move is SMOTE on the forest (−0.008 AUC, −0.015 AP), which is a small *loss*. The Spearman rank correlation between each strategy's test
  probabilities and the plain model's is 0.996 / 0.991 / 0.991 (logistic) and 0.989 / 0.937 / 0.968 (forest): the models order the
  clients almost identically. `imb_logistic_none` reproduces the report's logistic baseline to three decimals (test AUC 0.7637 vs
  0.7636, AP 0.5235 vs 0.5234; this run uses C = 1, the baseline `results/logistic.json` is unpenalised, max |Δp| on the test set 0.019).
  The plain forest beats that baseline by +0.018 AUC / +0.046 AP, and the best imbalance strategy for the forest is still "none".

- **Why: for a logistic model, re-balancing is (almost) a pure intercept shift.** Balancing the classes multiplies the
  prior odds by n0/n1, so theory predicts the intercept to move by log(π0/π1) = log(0.779/0.221) = **1.258** and the slopes to
  stay put. Observed (`imbalance_lr_coefficients.csv`): intercept −1.308 → −0.065 / −0.037 / −0.092, i.e. **+1.243 / +1.271 /
  +1.216** for balanced / SMOTE / under, while the largest change in any of the 32 slope coefficients (26 standardised numeric +
  6 one-hot dummies) is 0.047 (balanced), 0.313 and 0.209 (SMOTE / under — extra noise from synthetic or discarded rows, on the rare
  `EDUCATION_4` dummy). An intercept shift is a monotone transformation of the score: same ROC curve, same ranking, only the
  probability scale moves — mean predicted P(default) on the test set rises from 0.221 (= prevalence) to 0.440 / 0.440 / 0.441
  (`imbalance_threshold_shift.png`).

- **So the recall "gain" at threshold 0.5 is a threshold effect, and threshold tuning gives the same thing for free.**
  At the default 0.5 cut-off recall of defaulters jumps from 0.321 to 0.633 / 0.631 / 0.632 (logistic) and from 0.376 to
  0.587 / 0.528 / 0.665 (forest) — the numbers a naïve comparison would advertise. But precision collapses with it (0.674 → 0.45;
  0.683 → 0.44–0.54) and, once every model is given its own F1-tuned threshold, they all land in the same place:
  logistic recall **0.562 / 0.598 / 0.568 / 0.549** with precision **0.512 / 0.477 / 0.506 / 0.523** (F1 0.536 / 0.531 / 0.535 / 0.535)
  at thresholds 0.30 / 0.56 / 0.60 / 0.62; forest recall **0.562 / 0.574 / 0.557 / 0.616** with precision **0.536 / 0.525 / 0.507 / 0.488**
  (F1 0.549 / 0.548 / 0.530 / 0.545) at 0.31 / 0.52 / 0.47 / 0.55. The tuned threshold of the re-balanced models sits at ≈ 0.5–0.6
  exactly because their probabilities were inflated by ≈ 2×. Made precise (`imbalance_threshold_equivalence.csv`; the equivalent
  plain-model threshold is chosen on training OOF probabilities, agreement is then measured on the test set): **"balanced at 0.5" is
  the plain logistic model at about 0.22 — the prevalence, as the +1.24 intercept shift predicts (1/(1+e^1.243) = 0.224)** — the two
  rules agree on 99.4 % of the 8,990 test rows (recall 0.631 vs 0.633, 2,788 vs 2,794 clients flagged); the same plain threshold 0.22
  reproduces SMOTE@0.5 and under@0.5 on 98.9 % / 99.0 % of rows. (Plain@0.30 is *not* that rule — only 93.2 % agreement.) **The F1-tuned
  pairs — 0.30 (plain) and 0.56 / 0.60 / 0.62 (re-balanced) — are likewise the same rule** (agreement 96.6 % / 98.9 % / 98.3 %). For the
  forest the equivalent plain thresholds are 0.29 / 0.34 / 0.24 for balanced / SMOTE / under at 0.5 (agreement 97.6 % / 94.7 % / 93.2 %;
  lower because a forest's probability shift is not a clean monotone map), and the tuned pairs 0.31 vs 0.52 / 0.47 / 0.55 agree on
  97.6 % / 94.1 % / 94.0 %. Under the 5:1 cost criterion the picture is identical: total cost 5,135 / 5,088 / 5,116 / 5,086 (logistic,
  thresholds 0.17 vs 0.41) and 5,024 / 5,027 / 5,101 / 4,991 (forest, 0.18 vs 0.32–0.44) — differences of 1–3 %, i.e. within noise,
  with recall 0.68–0.74 for every variant.

- **What re-balancing does cost: calibration.** Brier score 0.138 → **0.190 / 0.191 / 0.190** (logistic) and 0.133 → 0.161 / 0.153 /
  0.185 (forest); log-loss 0.440 → 0.571 (logistic). The re-balanced models say "44 % default probability" for a portfolio whose
  true default rate is 22 %, so their outputs can no longer be read as PDs (pricing, expected-loss, capital) without a
  re-calibration step — whereas the plain models are calibrated by construction (mean p 0.221 / 0.224 vs prevalence 0.221).
- **Re-calibration done (2026-10-19, `code/recalibrate.py`).** Platt scaling fitted on each model's training out-of-fold scores
  (cross-fitted on the five folds for the threshold search; no model refitted) brings every re-balanced model back: test Brier
  0.190 → 0.139 (logistic, all three strategies), 0.161 / 0.153 / 0.185 → 0.134 / 0.136 / 0.135 (forest), mean p back to
  0.220–0.222, and the 5:1 cut-off from 0.32–0.44 back to 0.15–0.18, beside the closed form 1/6. Each calibrated version was
  scored on the test set once and is frozen (`results/frozen_manifest.json`); numbers in `results/recalibration.csv`,
  reliability plot `figures/recalibration.png`, appendix table `tab:a-recal`.
  Under-sampling additionally discards 56 % of the training rows (9,282 of 20,975 kept): harmless for the linear model, but the
  forest's training AUC drops from 0.972 to 0.915 and it has the worst Brier (0.185) of the four.

- **SMOTE is the weakest option on this tabular, mixed-type data.** It interpolates in the transformed feature space, which produces
  fractional one-hot dummies and non-integer repayment codes (PAY_x ∈ {−2..8}) — synthetic "clients" that cannot exist — and, for
  the forest, blurs the sharp thresholds (`MAX_DELAY ≥ 2`, `PAY_1 ≥ 2`) that carry most of the signal: CV AUC 0.7745 vs 0.7800, test AP
  0.554 vs 0.569, and the lowest rank correlation with the plain model (0.937). SMOTENC or an ordinal-aware variant would be the
  correct tool if oversampling were needed at all; here it is not. It is also the slowest strategy for the forest: 70 s of CV + 21 s
  final fit, versus 20 s + 4 s for under-sampling and 36 s + 9 s for the plain forest (`cv_seconds` / `fit_seconds` in
  `imbalance_comparison.csv`; wall-clock on a shared laptop, so ±30 % between runs — the metrics themselves are seeded and do not move).

- **Recommendation for the report.** Keep a probabilistic model trained on the natural class prior, and set the operating point
  from the business objective on training OOF probabilities: threshold ≈ 0.30 (F1) or ≈ 0.17–0.18 (5:1 cost, recall 0.68–0.74). Use
  `class_weight='balanced'` only as a convenience when the threshold of a downstream system cannot be changed — it is equivalent to
  the threshold move (plain model at ≈ 0.22) and costs calibration. Do not resample. The underlying argument (cost-sensitive learning /
  resampling material, B4/B6): re-weighting or resampling changes the effective class prior, not the likelihood ratio that determines
  the ranking, so the cut-off should be chosen from the cost matrix rather than baked into the training data.

- **The wrong way, demonstrated (`imb_leaky_smote_demo`, clearly labelled "LEAKAGE DEMO – NOT a valid result").** SMOTE was applied to
  all 29,965 rows *before* splitting, in the **same feature space the honest runs use** — the `make_preprocessor()` output (26 standardised
  numerics + 6 one-hot dummies), with the preprocessor fitted once on all rows — followed by the same classifiers, so the only material
  difference to the honest `imb_<learner>_smote` runs is that SMOTE saw the test rows. 29,965 → 46,670 rows (16,705 synthetic), then the
  same stratified 70/30 split (32,669 / 14,001). Result for the forest: leaky CV AUC **0.9006 ± 0.0039**, leaky "test" AUC **0.9142**, "AP"
  0.912, accuracy 0.839 at 0.5 — versus the honest 0.7744 / 0.5544 of `imb_rf_smote`: **+0.140 AUC out of thin air**. Mechanism: 72.6 % of
  the defaulters in the leaky "test" set are synthetic points, each a convex combination of a real defaulter and one of its 5 nearest
  defaulter neighbours — which now sit in the training set (AUC restricted to synthetic defaulters vs real non-defaulters: 0.940). The
  contamination is not confined to synthetic rows: on the *real* rows of the leaky test set the forest still scores 0.845, because those
  real defaulters were the seeds and neighbours of synthetic training rows. Two further lessons: (i) cross-validation did not protect
  against it (0.9006), since the leak happened before the folds were made, and the 50 % "prevalence" makes the AP and accuracy figures
  meaningless; (ii) the leak inflates a *flexible* learner far more than a linear one — the logistic model moves only from 0.7624 to 0.7697
  (+0.007; 0.770 on the real rows), because a 32-parameter linear model cannot memorise the interpolated neighbours that a 500-tree
  forest with 5-row leaves can — so a leaky pipeline can also reverse a model ranking. The demo's evaluate() files live under
  `results/leakage_demo/` and `figures/leakage_demo/` only as a documented negative example — never put them in the comparison table (its
  `fit_seconds` 20.2 is the final fit only; the 5-fold CV took a further 76.8 s).

- **Caveats.** (a) The 1:1 sampling ratio and SMOTE's k = 5 are library defaults and were not tuned; milder ratios would simply move
  the intercept less. (b) The strategy comparison is *within* one seed and one split; the logistic differences (≤ 0.0013 AUC) are far
  inside the fold SD of 0.010, so they should be described as "identical", not ranked. (c) **`imb_rf_none` is a fixed-setting forest
  (`min_samples_leaf=5`, not re-tuned per strategy), about 0.005 AUC below LU Xinyu's tuned `random_forest` (`min_samples_leaf=20`:
  CV AUC 0.7849, test AUC 0.7866, AP 0.5756, versus 0.7800 / 0.7820 / 0.5694 here).** The difference is the hyper-parameters, not the
  scaled preprocessor (trees are scale-invariant). The deliverable of this script is the *strategy comparison*, not the absolute forest
  number — quote LU Xinyu's forest as "the" project forest. (d) `class_weight='balanced'` in a forest re-weights the impurity criterion
  but not `min_samples_leaf`, which counts unweighted samples. (e) Thresholds come from a 0.01 grid on OOF probabilities; a different
  threshold criterion (e.g. a recall target) would shift all strategies together. (f) The test set is only ever predicted, never fitted
  on or used for any selection: predictions are made once in `run_study()` for the rank-correlation / threshold-equivalence diagnostics
  (the equivalent thresholds themselves are chosen on training OOF probabilities) and once inside `evaluate()`. (g) The leakage demo
  also fits the scaler on all rows — a second, negligible leak of column means / SDs that belongs to the same "resample first, split later"
  mistake. An earlier version of the demo resampled the *raw, unscaled* 29-column matrix instead (forest 0.908 / +0.134, logistic 0.809 /
  +0.046); the larger logistic inflation there came from the raw-space k-NN geometry, not from the leak alone, which is why the demo was
  moved into the shared feature space so that only the leak differs.
