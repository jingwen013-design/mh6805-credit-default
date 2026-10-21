# Notes — aggregation of all model results (slug `aggregate`)

Script `code/aggregate.py` (runs in about 20 s; it fits nothing and tunes nothing — it only reads the `results/*.json` and
`results/<slug>_p_test.npy` files written by the eleven model scripts and re-scores them against the shared `y_test` from
`load_split()`). Outputs: `results/model_comparison.csv` / `.md` (all 34 evaluate() results, ranked by CV AUC — the training-set selection score; test metrics are reported, never used to rank),
`results/model_comparison_main.csv` / `.md` (the eleven headline models with class 0 / class 1 precision, recall and F1 at the
F1-tuned threshold and at 0.5, in the sample-report layout), `results/auc_bootstrap_ci.csv` (paired bootstrap, B = 500, seed
2607526), `results/aggregate_summary.json` (rankings, thresholds, bootstrap verdicts, imbalance and unsupervised summaries);
figures `figures/all_roc.png`, `all_pr.png`, `all_auc_bar.png` (+ `all_auc_bar_full.png` for all 34 results), `threshold_curves.png`,
`cost_curves.png`, `calibration.png`. Excluded from every table: the leakage demonstration (`imb_leaky_*`, quarantined under
`results/leakage_demo/`) and the EDA / unsupervised summary files, which are not model results. All numbers below are copied from
the JSON / CSV files (test set n = 8,990, 1,988 defaulters = 22.1 %; training set n = 20,975; majority-class accuracy 0.779).

| headline model | member | CV AUC (5-fold) | test AUC | test AP | Brier | acc @0.5 | recall / precision @0.5 | thr (F1) | recall / precision / F1 @thr | thr (5:1 cost) | recall @cost thr | cost @cost thr | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | LU Xinyu | 0.7863 ± 0.0068 | **0.7887** | **0.5784** | **0.1311** | 0.8249 | 0.363 / 0.702 | 0.27 | 0.594 / 0.512 / 0.550 | 0.15 | 0.784 | **4,981** | 0.5 |
| Gradient boosting (sklearn) | LU Xinyu | 0.7850 ± 0.0074 | 0.7872 | 0.5759 | 0.1314 | 0.8258 | 0.363 / 0.707 | 0.26 | 0.592 / 0.511 / 0.548 | 0.14 | 0.802 | 5,060 | 16.3 |
| Random forest | LU Xinyu | 0.7849 ± 0.0076 | 0.7866 | 0.5756 | 0.1321 | 0.8245 | 0.369 / 0.694 | 0.31 | 0.564 / 0.539 / **0.551** | 0.17 | 0.757 | 4,998 | 7.9 |
| Decision tree (pruned) | LU Xinyu | 0.7689 ± 0.0076 | 0.7755 | 0.5346 | 0.1332 | 0.8250 | 0.326 / 0.736 | 0.28 | 0.566 / 0.531 / 0.548 | 0.16 | 0.758 | 5,085 | 87.1 |
| MLP (64, 32) | LI Jingwen | 0.7762 ± 0.0069 | 0.7731 | 0.5513 | 0.1353 | 0.8224 | 0.368 / 0.683 | 0.27 | 0.593 / 0.488 / 0.536 | 0.19 | 0.710 | 5,136 | 1.0 |
| KNN (K = 101) | LI Jingwen | 0.7700 ± 0.0087 | 0.7706 | 0.5423 | 0.1361 | 0.8196 | 0.342 / 0.684 | 0.25 | 0.596 / 0.495 / 0.541 | 0.16 | 0.748 | 5,219 | 0.03 |
| Logistic regression (plain) | LI Jingwen | 0.7602 ± 0.0100 | 0.7636 | 0.5234 | 0.1383 | 0.8155 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 / 0.536 | 0.15 | 0.739 | 5,126 | 0.4 |
| Logistic regression (L1, C tuned) | LI Jingwen | 0.7602 ± 0.0100 | 0.7636 | 0.5234 | 0.1383 | 0.8154 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 / 0.536 | 0.15 | 0.739 | 5,125 | 185.5 |
| SVM (RBF, balanced, Platt) | LI Jingwen | 0.7681 ± 0.0055 † | 0.7616 | 0.4944 | 0.1448 | 0.7842 | 0.073 / 0.602 | 0.42 | 0.568 / 0.516 / 0.541 | 0.12 | 0.689 | 5,155 | 47.3 |
| LDA | LI Jingwen | 0.7572 ± 0.0109 | 0.7592 | 0.5251 | 0.1394 | 0.8174 | 0.375 / 0.652 | 0.27 | 0.574 / 0.506 / 0.538 | 0.12 | 0.700 | 5,181 | 2.1 |
| Gaussian naive Bayes | LI Jingwen | 0.7356 ± 0.0109 | 0.7398 | 0.5040 | 0.2034 | 0.7729 | 0.575 / 0.489 | 0.41 | 0.589 / 0.474 / 0.525 | 0.25 | 0.642 | 5,337 | 1.2 |

† the SVM's CV AUC was measured on its 6,000-row tuning subsample and is not comparable; its OOF AUC on the full training set is
0.7606. Cost = 5·FN + FP on the 8,990 test rows (flag nobody = 9,940, flag everybody = 7,002). Thresholds were chosen on training
out-of-fold probabilities by each script; `fit s` is what each script recorded (final refit for XGBoost / MLP, the whole grid for
L1 / LDA / NB / tree) — see the runtime bullet.

- **Which family wins, and by how much.** Tree ensembles occupy the top of every ranking: XGBoost test AUC 0.7887 / AP 0.5784 /
  Brier 0.1311, gradient boosting 0.7872 / 0.5759 / 0.1314, random forest 0.7866 / 0.5756 / 0.1321. Against the logistic baseline
  (0.7636 / 0.5234 / 0.1383 — the README reference 0.764 / 0.524) the best model gains **+0.025 AUC, +0.055 AP, −0.007 Brier and −145
  cost units (4,981 vs 5,126 at the cost-tuned thresholds, −2.8 %)**. The single pruned tree (0.7755) and the MLP (0.7731) sit in
  between, tuned KNN (0.7706) is the best distance-based model, and every linear / generative model — logistic, L1 logistic, LDA
  (0.7592), the RBF SVM (0.7616) — lands within 0.005 of the baseline; naive Bayes (0.7398, Brier 0.203) is last. The whole field
  spans only 0.05 AUC: the target is noisy (the univariate AUC of PAY_1 alone is already ≈ 0.70 in `notes_eda.md`), so no model
  family finds much beyond the repayment-status signal that the linear models already capture. Across all 33 results the
  class-weighted XGBoost ties the plain one on AUC (0.7887 vs 0.7887, +0.00007) but its Brier is 0.177, i.e. its probabilities are
  inflated — the plain XGBoost is the model to quote.
- **Which differences are real: paired bootstrap of the test set (B = 500, seed 2607526).** Resampling the 8,990 test rows and
  re-scoring every model on the same rows gives XGBoost a 95 % AUC interval of [0.7760, 0.7999] — ±0.012, roughly the size of the
  5-fold CV sd — so single-number rankings must be read with that in mind. Paired differences vs XGBoost: **random forest +0.0020
  [−0.0010, +0.0051] — not distinguishable (XGBoost is behind in 11 % of replicates); gradient boosting +0.0015 [+0.0001, +0.0030]
  — borderline (the interval barely clears zero; on AP it does not: +0.0026 [−0.0003, +0.0056])**. Everything else is clearly
  below: pruned tree +0.0132 [0.0082, 0.0184], MLP +0.0156 [0.0103, 0.0199], KNN +0.0181 [0.0125, 0.0237], logistic +0.0249 [0.0183,
  0.0314] (AP +0.0546 [0.0406, 0.0682]), RBF SVM +0.0271 [0.0216, 0.0330], LDA +0.0293 [0.0224, 0.0360], naive Bayes +0.0486
  [0.0388, 0.0566]. Conclusion for the report: "the three ensembles are statistically tied; the ensembles beat every non-ensemble;
  the ordering *within* the linear family (logistic vs L1 vs LDA vs SVM, all within 0.005) is noise".
- **Selection uses CV AUC only (changed 2026-10-19).** The reference model of the bootstrap and the top-4 drawn in
  `threshold_curves.png`, `cost_curves.png` and `calibration.png` are picked on the 5-fold CV AUC of the training set (OOF AUC
  for the RBF SVM, whose CV ran on a 6,000-row subsample); the test set only reports. XGBoost leads on CV (0.7863) and also
  on test, so the reference model is unchanged; the CV top-4 is XGBoost, GBM, forest and MLP (the pruned tree was fourth on
  test AUC but is sixth on CV, 0.7689).
- **CV and test agree — no model was over-selected.** For every headline model the test AUC is within one CV sd of the CV AUC
  (XGBoost 0.7863 → 0.7887, forest 0.7849 → 0.7866, logistic 0.7602 → 0.7636, KNN 0.7700 → 0.7706, MLP 0.7762 → 0.7731, the only
  one that drops), and the pooled OOF AUCs (0.7861 / 0.7847 / 0.7601 …) match to three decimals. This is what the shared contract
  buys: one split, preprocessing inside every fold, thresholds from OOF probabilities, every choice made on training data, and
  each configuration scored on the test set once (by 18 September) and then frozen (`results/frozen_manifest.json`).
  The one CV number that must not be compared is the SVM's (0.7681 on a 6,000-row subsample; its full-training OOF AUC is 0.7606).
- **Accuracy is the wrong headline; thresholds are the real decision.** At the default cut-off every model except NB and the SVM
  scores 0.815–0.826 accuracy — barely above the 0.779 of predicting "no default" for everyone — while catching only 32–37 % of the
  defaulters (SVM 7 %: its Platt probabilities rarely exceed 0.5, max 0.63). Moving to the F1-tuned thresholds (0.25–0.31 for all
  well-calibrated models; 0.41–0.42 for NB and the SVM whose probability scales differ) costs 3–5 points of accuracy (XGBoost
  0.825 → 0.785) and roughly doubles recall (XGBoost 0.363 → 0.594, logistic 0.321 → 0.563), with class-1 precision ≈ 0.49–0.54 and
  F1 ≈ 0.54–0.55 for *every* model from logistic to XGBoost. In the per-class table the models are therefore almost
  indistinguishable at one operating point (class 0: precision 0.87–0.88, recall 0.81–0.86; class 1: 0.47–0.54 / 0.56–0.60) — the
  ensembles' advantage is spread across the whole curve (`all_roc.png`, `all_pr.png`), not concentrated at one threshold.
- **The 5:1 cost view (`cost_curves.png`).** If a missed defaulter costs five false alarms, the optimal cut-off drops to 0.12–0.19
  (XGBoost 0.15, forest 0.17, GBM 0.14, logistic 0.15) and the models flag 40 % of the non-defaulters to catch 74–80 % of the
  defaulters (XGBoost recall 0.784, FPR 0.404, FNR 0.216; GBM recall 0.802 at FPR 0.442). Test cost per 8,990 clients: XGBoost
  6,642 at 0.5 → 5,164 at the F1 threshold → **4,981 at the cost threshold (−25 % vs 0.5, −50 % vs flagging nobody, 9,940)**; random
  forest 4,998, GBM 5,060, tree 5,085, logistic 5,126, NB 5,337. The threshold chosen on training OOF probabilities transfers: for
  XGBoost the test-set cost curve bottoms out at t = 0.19 (545 per 1,000 clients) versus 554 per 1,000 at the OOF-chosen 0.15 — a
  1.6 % gap, and the curve is flat between 0.13 and 0.22. The cost differences between the ensembles (4,981 / 4,998 / 5,060) are
  smaller than the bootstrap noise; the cost gap ensembles-vs-linear (≈ 150 units, 3 %) is real but modest — the threshold choice
  (0.5 vs 0.15) is worth 1,600 units, ten times the model choice.
- **Calibration (`calibration.png`).** The four models with the highest CV AUC (XGBoost, GBM, forest, MLP) are calibrated by
  construction: mean predicted P(default) 0.220 / 0.219 / 0.221 / 0.221 against a prevalence of 0.221, reliability curves on the
  diagonal in all ten deciles (top decile: predicted 0.65–0.69, observed 0.70–0.73), Brier 0.1311 / 0.1314 / 0.1321 / 0.1353
  versus 0.1383 for logistic. Their probabilities can be used as PDs
  (pricing, expected loss) without a re-calibration step. The models that *are* mis-calibrated are the ones whose training changed
  the class prior — class-weighted XGBoost (Brier 0.177, mean p 0.37), balanced / SMOTE / under-sampled logistic (0.190 – 0.191) and
  the balanced forest (0.161) — and naive Bayes (0.203, probabilities piled at 0 and 1). Platt scaling on training out-of-fold
  scores repairs all nine weighted / resampled models (Brier back to 0.131–0.139, `results/recalibration.csv`). The pruned tree (not in the CV top-4, so not
  drawn here) outputs only 17 distinct probabilities, one per leaf, which is why its PR curve in `all_pr.png` is a polygon.
- **What the imbalance study showed (LI Jingwen, `notes_imbalance.md`).** None of `class_weight='balanced'`, SMOTE or random
  under-sampling improves the ranking: logistic test AUC 0.7637 / 0.7630 / 0.7624 / 0.7628 and AP 0.5235 / 0.5171 / 0.5162 / 0.5153
  for none / balanced / SMOTE / under; forest 0.7820 / 0.7818 / 0.7744 / 0.7814 (SMOTE is a small *loss*, −0.008 AUC). The recall
  "gain" at 0.5 (logistic 0.32 → 0.63) is a threshold effect — balanced-at-0.5 agrees with plain-at-0.22 on 99.4 % of test rows and
  the F1-tuned pairs agree on 96.6 % — bought with calibration (Brier 0.138 → 0.190). The resample-before-split demonstration
  quantifies the classic mistake: leaky SMOTE lifts the forest to a fake 0.914 test AUC (+0.140 over the honest 0.774) and the
  logistic only to 0.770 (+0.007), so a leak both inflates and *re-orders* models. Recommendation carried into this comparison:
  natural prior + threshold from the cost matrix, which is what every headline model does.
- **What unsupervised learning added (LI Jingwen, `notes_unsupervised.md`).** As *features*, nothing: the four k-means segment dummies
  leave the logistic at 0.7632 / 0.5219 (vs 0.7636 / 0.5234) and a logistic on the first 15 PCs is worse (0.7617 / 0.5183) — an
  unpenalised logistic is rotation-invariant, so PCA can only discard signal, and the low-variance components (PC12, PC13, PC15) are
  the ones that carry the delay-recency signal. As a *map*, a lot: PC2 (delay vs affluence; AUC 0.701 on its own) explains why every
  model ranks PAY_1 / MAX_DELAY / N_MONTHS_DELAYED first, and the K = 4 segmentation gives management a stable partition —
  transactors (38.7 %, 16.0 % default), affluent revolvers (10.1 %, 17.4 %), mass-market revolvers (39.7 %, 18.2 %) and delinquents
  (11.6 % of clients, **59.7 % default train / 61.9 % test**, 31.4 % of all defaults) — on which limit policy, collections and
  segment-wise monitoring of the XGBoost scores can be organised.
- **Runtime — the two industry defaults are also the cheapest to refit.** XGBoost refits in 0.5 s (24-setting grid 212 s), the
  forest in 7.9 s (grid 226 s), sklearn's exact-split GBM in 16.3 s (grid 301 s, scored on a 50 % subsample of the training set for
  budget), the MLP in 1 s (whole script 333 s), logistic in 0.4 s. The expensive models are the ones that pay at prediction time or
  in the solver: KNN stores the 20,975-row matrix and needs 0.38 s per 8,990 predictions (K = 101 brute force), the RBF SVM keeps
  13,140 support vectors, took 47 s to refit and 7.8 s to score the test set (and could only be tuned on 6,000 rows), the L1
  logistic's saga path took 186 s for a model identical to the plain one (no coefficient reaches zero), and cost-complexity pruning
  of the tree took 87 s. `fit_seconds` are wall-clock on a laptop shared by concurrent scripts — indicative only; the metrics are seeded.
- **Reading the per-class "main table" the way the sample report does.** At the tuned thresholds all eleven models have class-0
  precision 0.87–0.88 and class-1 recall 0.56–0.60; the sample report's 0.95+ accuracies and 0.99 AUCs come from a different, far
  easier data set (and a leaky pipeline would produce exactly such numbers here — see the imbalance bullet). On this data an honest
  AUC of 0.79 is the ceiling that four independent model families converge on (boosting 0.789, bagging 0.787, MLP 0.773, KNN 0.771).
- **Recommendation.** Deploy the plain (unweighted) **XGBoost** — best or tied-best on AUC, AP, Brier and cost, calibrated, 0.5 s to
  refit — with the operating point chosen from the cost matrix on training OOF probabilities (t ≈ 0.15 under 5:1, flagging ≈ 45 % of
  clients to catch 78 % of defaulters; t ≈ 0.27 if precision and recall are weighted equally). Keep the **random forest** as the
  statistically equivalent, simpler fallback and the **logistic regression** (AUC 0.7636, 33 coefficients, p-values in
  `logistic_coefficients.csv`) as the interpretable benchmark that a regulator can read; it gives up 0.025 AUC and about 3 % of cost.
- **Caveats.** (a) One split, one seed: the bootstrap quantifies test-set sampling noise (±0.012 AUC) but not split-to-split
  variation; repeating the whole pipeline over several splits would widen the intervals somewhat. (b) The bootstrap is a percentile
  interval on the paired AUC difference, not a DeLong test; with B = 500 the interval end-points themselves carry Monte-Carlo noise of
  about ±0.0003, which matters only for the borderline GBM verdict. (c) Thresholds come from a 0.01 grid on OOF probabilities and the
  5:1 cost ratio is an assumption, not a measured loss-given-default; a different ratio moves all models' cut-offs together. (d)
  `fit_seconds` semantics differ by script and are wall-clock on a shared machine. (e) The SVM and GBM grids were tuned on
  subsamples (6,000 and 10,487 training rows) for budget; their *choices* are documented as robust in the model notes, but their
  CV numbers are not on the same footing as the others'. (f) The class-weighted XGBoost, the imbalance variants and the L2 logistic
  are listed in the full table for completeness; they are variants of headline models, not independent competitors, and the
  headline set (one model per family) is the fair comparison.
