# Notes — generative classifiers (LDA / QDA / Gaussian naive Bayes), lecture B6 self-study part

Owner: LI Jingwen. Script: `code/model_discriminant.py` (runs in about 20 s). Results: `results/lda.json`, `results/qda.json`,
`results/naive_bayes.json`, `results/naive_bayes_log.json`. Figures: `figures/<slug>_eval.png` (one per model),
`figures/discriminant_roc_pr.png` (overlay of the four models + logistic reference), `figures/discriminant_cv_curves.png`
(CV AUC against the tuned hyper-parameter), `figures/discriminant_amount_skew.png` (why the Gaussian assumption fails).
All numbers below are copied from the JSON files; the logistic baseline is the README reference (test AUC 0.764, AP 0.524).

| model | tuned setting (5-fold CV AUC, mean ± sd) | test AUC | test AP | Brier | acc @0.5 | recall @0.5 | F1-tuned thr | recall / precision @tuned |
|---|---|---|---|---|---|---|---|---|
| Logistic baseline (README) | — (CV ≈ 0.760) | 0.764 | 0.524 | — | — | — | — | — |
| LDA (`lda`) | shrinkage = None, solver lsqr (0.7572 ± 0.0109) | **0.7592** | **0.5251** | **0.1394** | **0.8174** | 0.3746 | 0.27 | 0.5737 / 0.5064 (F1 0.5380) |
| QDA (`qda`) | reg_param = 0.5 (0.7511 ± 0.0093) | 0.7550 | 0.5163 | 0.1661 | 0.7872 | 0.5611 | 0.46 | 0.5767 / 0.4987 (F1 0.5349) |
| Gaussian NB (`naive_bayes`) | var_smoothing = 0.01 (0.7356 ± 0.0109) | 0.7398 | 0.5040 | 0.2034 | 0.7729 | 0.5752 | 0.41 | 0.5892 / 0.4741 (F1 0.5254) |
| Gaussian NB, log1p amounts (`naive_bayes_log`) | var_smoothing = 0.01 (0.7540 ± 0.0093) | 0.7563 | 0.5084 | 0.1932 | 0.7747 | 0.5390 | 0.28 | 0.5802 / 0.4588 (F1 0.5124) |

Majority-class accuracy is 0.779; thresholds are tuned on training out-of-fold probabilities only (never on the test set).

## Interpretation bullets

- **What was tuned and what won.** Every model is `Pipeline(make_preprocessor(), classifier)` (StandardScaler on the 26 numeric
  columns, one-hot on SEX/EDUCATION/MARRIAGE → 32 columns) tuned with `GridSearchCV(cv=CV, scoring="roc_auc", n_jobs=2)`.
  LDA: shrinkage over {None, 'auto', 0.05, 0.2, 0.5} → best None (CV AUC 0.7572; shrinkage only hurts, monotonically down to 0.7377
  at 0.5, because n = 20,975 ≫ p = 32 so the pooled covariance is already well estimated). QDA: reg_param over {0, 0.01, 0.1, 0.5}
  → best 0.5 (CV AUC rises monotonically 0.7394 → 0.7429 → 0.7495 → 0.7512). Gaussian NB: var_smoothing over logspace(−9, −1, 9)
  → best 0.01, but the curve is flat (0.7355 for 1e-9…1e-4, 0.7356 at 0.01, 0.7334 at 0.1) — smoothing cannot repair a wrong model.

- **Ranking of the four generative models: LDA > NB-log ≈ QDA > raw NB, and none beats logistic regression.** Test AUC 0.7592 /
  0.7563 / 0.7550 / 0.7398 versus 0.764 for the logistic baseline; test AP 0.5251 / 0.5084 / 0.5163 / 0.5040 versus 0.524.
  LDA's AP (0.5251) is actually a hair above the baseline's 0.524 and its AUC is 0.005 below — a tie within noise (fold-to-fold sd ≈ 0.01).

- **LDA versus logistic regression — the same linear model estimated two ways.** Both give log-odds that are linear in x. LDA obtains the
  weights *generatively* (class means and one pooled covariance, w = Σ⁻¹(μ₁ − μ₀)), logistic obtains them *discriminatively* by maximising
  the conditional likelihood P(y|x). On this split the two coefficient vectors have cosine similarity 0.934 and the test-set scores have
  Spearman correlation 0.990 (`results/lda.json → extra.logistic_reference`); the top features agree (N_MONTHS_DELAYED, EDUCATION=others,
  UTIL_MEAN, MAX_DELAY, PAY_1). Logistic is slightly better (0.7637 vs 0.7592 AUC on the same rows) because it does not need the
  Gaussian-with-common-covariance assumption to hold; LDA's estimates are more efficient only when that assumption is true, and with
  20,975 training rows the discriminative model has more than enough data (the Ng & Jordan 2001 argument: generative converges faster
  but to a higher asymptotic error).

- **LDA gives the best-calibrated probabilities of the four** (Brier 0.1394, log-loss 0.446; QDA 0.1661 / 0.659; NB 0.2034 / 1.212;
  NB-log 0.1932 / 1.171). At the default 0.5 threshold it has the highest accuracy (0.8174; QDA 0.7872; both NB variants are below the
  0.779 majority rate), but it then catches only 37.5 % of defaulters (precision 0.652). Moving to the F1-tuned threshold 0.27 lifts recall to 0.574 at
  precision 0.506 (F1 0.538); the 5:1 cost-tuned threshold 0.12 reaches recall 0.700 at precision 0.388. These are essentially the
  logistic-regression operating points, which is the expected outcome for a model that is its generative twin.

- **Why QDA does not gain from its extra flexibility.** QDA fits a separate 32 × 32 covariance for each class, so the decision boundary
  is quadratic. That only helps if the class-conditional distributions are (roughly) Gaussian with different spreads; here the 13
  amount columns are extremely right-skewed (training skewness: BILL_AMT ≈ 2.7–3.2, PAY_AMT 10–33), the six monthly bills are almost
  collinear (mean pairwise correlation 0.887), and 6 of the 32 columns are 0/1 dummies. The quadratic terms therefore amplify noise in
  the tails, and the CV curve says so: the more the class covariances are shrunk towards the identity (reg_param → 0.5, the edge of the
  grid) the better, i.e. the data are asking for *less* covariance structure, not more. Result: test AUC 0.7550, below LDA (0.7592),
  with worse calibration (Brier 0.166). QDA's 0.5-threshold recall (0.561) looks better than LDA's only because its probabilities are
  more spread out, not because it ranks clients better.

- **Why Gaussian naive Bayes is the weakest (test AUC 0.7398, AP 0.5040).** NB multiplies one univariate Gaussian likelihood per feature,
  assuming conditional independence. The six BILL_AMT columns (correlation 0.887) and six PAY_ status columns (0.656) are counted as
  twelve independent pieces of evidence, so the posterior becomes wildly over-confident: 35.8 % of test probabilities are below 0.01 or
  above 0.99, Brier 0.203 and log-loss 1.21 are worse than predicting the base rate (Brier 0.172 for a constant 0.221). The ranking is
  still usable (AUC 0.74) because the signal in PAY_1 / N_MONTHS_DELAYED survives, but the probabilities should not be used as risk scores.

- **The log1p variant helps ranking but not calibration.** Applying sign(x)·log1p(|x|) to the 13 amount columns (LIMIT_BAL, BILL_AMT1..6,
  PAY_AMT1..6 — signed because bills can be negative) inside the pipeline makes the within-class marginals far closer to Gaussian
  (PAY_AMT skewness 10–33 → about −0.9 to −1.3; BILL_AMT 2.7–3.2 → −1.6 to −2.1, the sign flipping because of the mass at zero).
  CV AUC rises from 0.7356 to 0.7540 and test AUC from 0.7398 to 0.7563 (+0.017), AP from 0.5040 to 0.5084 — NB-log now matches QDA.
  But the transform is monotone per column and does nothing about dependence between columns, so the over-confidence gets *worse*
  (68.9 % of test probabilities extreme; Brier 0.193, log-loss 1.17). Conclusion for the report: the two NB failure modes are separable —
  a marginal transform fixes the "not Gaussian" problem, only a model with covariance (LDA/QDA) fixes the "not independent" problem.

- **The four models tell one coherent story about this dataset.** The best generative model is the *most constrained* one (LDA, one
  pooled covariance), the quadratic and independence-based alternatives are worse, and the discriminative linear model (logistic) is
  marginally best of all. The predictable signal in credit-card default here is mostly linear in the repayment-status features; extra
  distributional flexibility buys nothing unless it is regularised back towards linearity. Tree ensembles' improvement, if any, will come
  from interactions, not from a different marginal shape.

- **Caveats.** (1) Fold-to-fold standard deviation of CV AUC is 0.009–0.011, so the LDA / QDA / NB-log / logistic differences (≤ 0.01)
  are within one sd; only "raw Gaussian NB is worse" and "log1p helps NB" are clear-cut. (2) QDA's best reg_param sits at the edge of
  the prescribed grid (0.5); a finer grid up to 1.0 might add a little, but the trend points towards LDA-like behaviour anyway. (3)
  GaussianNB treats the six one-hot dummies as Gaussian; a mixed Bernoulli/Gaussian NB would be the principled fix (not attempted).
  (4) `fit_seconds` in the JSON is the whole GridSearchCV wall time (1–3 s); all four models are essentially free to train, which is
  their one practical advantage. (5) A plain logistic regression was fitted once inside this script purely as a diagnostic reference
  for the LDA comparison (its test AUC 0.7637 / AP 0.5235 reproduces the README numbers); `evaluate()` was not called on it, and the
  official logistic results are the other member's. (6) Each final model was scored on the test set once and is frozen in `results/frozen_manifest.json`; tuning and thresholds use
  training CV / OOF only.
