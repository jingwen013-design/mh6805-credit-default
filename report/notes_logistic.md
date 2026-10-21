# Notes — logistic-regression family (LI Jingwen)

Script `code/model_logistic.py` (runs in ~4.5 min). Results `results/logistic.json`, `logistic_l2.json`, `logistic_l1.json`,
`logistic_balanced.json`, `logistic_coefficients.csv`, `logistic_family_summary.csv`; figures `figures/logistic_coefficients.png`,
`logistic_l1_path.png` and the four `*_eval.png`. Same split, same `Pipeline(make_preprocessor(), LogisticRegression)`,
`cv=CV`, `scoring="roc_auc"`, seed 2607526, thresholds tuned on training out-of-fold probabilities only.

| model | setting | CV AUC (5-fold) | test AUC | test AP | Brier | acc @0.5 | recall/precision @0.5 | F1-tuned thr | recall/precision @tuned |
|---|---|---|---|---|---|---|---|---|---|
| `logistic` | unpenalised MLE (C = inf) | 0.7602 ± 0.0100 | **0.7636** | **0.5234** | 0.1383 | 0.8155 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 (F1 0.536) |
| `logistic_l2` | ridge, best C = 31.6 | 0.7602 ± 0.0100 | 0.7636 | 0.5234 | 0.1383 | 0.8155 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 (F1 0.536) |
| `logistic_l1` | lasso (saga), best C = 100, 0/32 zero | 0.7602 ± 0.0100 | 0.7636 | 0.5234 | 0.1383 | 0.8154 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 (F1 0.536) |
| `logistic_balanced` | C = inf, class_weight = balanced | 0.7609 ± 0.0107 | 0.7630 | 0.5172 | 0.1901 | 0.7479 | 0.633 / 0.450 | 0.56 | 0.598 / 0.477 (F1 0.531) |

Majority-class accuracy is 0.779; the README baseline (test AUC 0.764 / AP 0.524) is exactly the `logistic` row.

- **What was tuned.** (b) L2 and (c) L1 both ran `GridSearchCV` over `C = logspace(-3, 2, 11)` (11 values × 5 stratified folds,
  55 fits each; saga/L1 took 186 s, lbfgs/L2 6 s). (a) and (d) have no hyper-parameter; "plain" is the unpenalised MLE
  (`C = inf`, lbfgs, `tol = 1e-8`) so its coefficients coincide with statsmodels `Logit` on the same standardised design matrix
  (max |difference| 1.1e-5), which is where the p-values in `logistic_coefficients.csv` come from. sklearn 1.9 deprecates
  `penalty=`; the script uses the equivalents `C=np.inf`, `l1_ratio=0` and `l1_ratio=1` and emits no warnings.
- **Regularisation buys nothing here.** The CV curve is flat for C ≥ 0.1: L2 mean AUC goes 0.7479 (C = 0.001) → 0.7602 (C ≥ 0.1)
  and the best C = 31.6 shrinks the coefficient norm by 0.1 % (1.470 vs 1.472). Test AUC 0.76363 (L2) / 0.76363 (L1) vs 0.76363
  (plain) and the confusion matrices at 0.5 and at the tuned threshold are identical (tn/fp/fn/tp = 6693/308/1351/638 and
  5934/1067/870/1119). With n/p = 20,975/32 ≈ 655 the variance of the MLE is already tiny, so shrinkage cannot help; the
  ceiling (~0.76) is the *bias* of a model that is linear in the features, not estimation noise. Differences between the four
  models (≤ 0.0006 AUC) are an order of magnitude below the fold-to-fold SD (0.010) and should be read as ties.
- **Lasso does not select at the CV-optimal C** — best C = 100 (grid edge) keeps all 32 columns (0 exact zeros). The path
  (`logistic_l1_path.png`) is still informative: as the penalty is relaxed the first columns to enter are **PAY_1,
  N_MONTHS_DELAYED, MAX_DELAY** (all non-zero already at C = 0.001, CV AUC 0.719 with just these three), then **LIMIT_BAL**
  (C ≈ 0.0018, AUC 0.735 with four), then AGE, PAY_4–6, PAY_AMT1/4 (C ≈ 0.0056); 17 columns give 0.748 (C = 0.01), 23 give 0.757
  (C = 0.0316, the sparsest model within one standard error ≈ 0.0045 of the best), 25 give 0.7596 (C = 0.1). The six raw
  BILL_AMT columns enter last (BILL_AMT4 only at C ≈ 3.2): once utilisation is in the model the bill levels add nothing.
- **Top effects (standardised log-odds, `logistic_coefficients.png`).** Repayment behaviour dominates: N_MONTHS_DELAYED +0.634 per
  SD (1.56 months; OR 1.88, i.e. +0.41 log-odds = OR 1.51 per additional delayed month), MAX_DELAY +0.366 per SD (OR 1.44), and
  PAY_1 +0.231 per SD (OR 1.26) *on top of* the two engineered summaries — the most recent month's status carries extra weight
  (recency). Persistent utilisation matters: UTIL_MEAN +0.288 per SD (OR 1.33, p = 0.019). LIMIT_BAL −0.217 per SD of NT$130k
  (OR 0.80): a higher limit is the bank's own prior screening of a safer client. Recent payments help (PAY_AMT2 −0.152, PAY_AMT1
  −0.108 per SD, both p < 0.003). Demographics are small but significant: female OR 0.85, single OR 0.83 (vs married).
  McFadden pseudo-R² is 0.159; 15 of 32 columns are significant at 5 %.
- **Sign flips are conditional effects, not paradoxes.** PAY_2…PAY_6 have *negative* coefficients (−0.08 to −0.13 per SD, all
  p < 0.02) although each is positively associated with default on its own (see EDA). Given the same number and severity of
  delays (N_MONTHS_DELAYED, MAX_DELAY), a delay that sits in an older month rather than the latest one is less dangerous — the
  engineered features absorb the level of delinquency and the raw statuses encode its timing. Likewise UTIL_1 is −0.259 (p = 0.07)
  next to UTIL_MEAN +0.288: a latest-month balance *below* the client's own six-month average is a good sign. These two
  utilisation columns and the six BILL_AMT columns are strongly collinear (r > 0.8), which inflates their standard errors
  (0.10–0.15 vs 0.02–0.03 elsewhere); interpret the utilisation block jointly, not coefficient by coefficient.
- **The single largest coefficient is a data artefact.** EDUCATION = others (the recoded 0/5/6 "unknown" level, ~1.5 % of
  clients) has coefficient −1.07 (OR 0.34, p < 1e-5). This group has an unusually low default rate in the raw data (a known quirk of
  the Taiwan data set), not a causal education effect; it should be reported as such and would deserve a robustness check without
  the group. University vs graduate school (+0.06) and high school vs graduate school (+0.01) are both insignificant.
- **Class re-weighting = moving the threshold, at the cost of calibration.** `class_weight='balanced'` (weights 0.642 / 2.260)
  shifts the intercept from −1.305 to −0.063 and changes no slope by more than 0.075. Ranking is unchanged (test AUC 0.7630 vs
  0.7636, AP 0.5172 vs 0.5234) while at the default 0.5 cut recall jumps from 0.321 to 0.633 and precision falls from 0.674 to
  0.450 — almost exactly what the plain model gives at threshold 0.30 (recall 0.563 / precision 0.512) or 0.25. The price is
  probability quality: Brier 0.190 vs 0.138, log-loss 0.571 vs 0.440, so the balanced model's scores can no longer be read as
  default probabilities. Prefer the plain model plus an explicit business threshold.
- **Threshold choice matters more than the penalty.** For the plain model, moving from 0.5 to the F1-optimal 0.30 (chosen on
  training OOF probabilities) raises recall from 32.1 % to 56.3 % at 51.2 % precision (F1 0.435 → 0.536, accuracy 0.815 → 0.785);
  the 5:1 cost-optimal threshold 0.15 catches 73.9 % of defaulters at 36.8 % precision and cuts the expected cost from 7,063 to
  5,126 on the test set. Any of the four models gives the same trade-off curve.
- **Why logistic regression stays in the report.** It is the transparent benchmark: every score decomposes into additive,
  signed, testable contributions (the CSV lists coefficient, odds ratio, SE, z, p-value and the per-raw-unit slope), it is
  perfectly calibrated at the base rate (Brier 0.138), fits in 0.4 s, and it already recovers the same variable ranking the
  tree models split on (PAY_1, delay counts, limit, utilisation). What it cannot do is capture the non-monotone shape of
  PAY_1 (the −2/−1/0 levels) or interactions such as delay × utilisation, which is where the non-linear models should gain.
- **Caveats.** (i) The L1 optimum sits on the grid edge (C = 100); extending the grid would not change anything because the curve
  is flat from C ≈ 0.3, and a one-SE rule would legitimately prefer the 23-variable model at C = 0.0316 (CV AUC 0.757). (ii) p-values
  are Wald tests from the unpenalised fit on the standardised matrix; they are not adjusted for 32 comparisons and are unreliable
  inside the collinear bill/utilisation block. (iii) saga is slow (650 epochs at C = 100; 186 s for the grid) — the lbfgs L2 model
  is the practical choice if a penalised version is wanted. (iv) Each model was scored on the test set (8,990 rows) once, after all
  tuning, and is frozen in `results/frozen_manifest.json`; thresholds come from training OOF probabilities, so the reported precision/recall at the tuned thresholds are
  honest out-of-sample numbers.
