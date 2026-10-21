# Notes — random forest and bagging (LU Xinyu)

Script `code/model_rf.py` (runs end-to-end in about 11 min — 672 s measured with `n_jobs=2`; 825 s in an earlier run while four other
model scripts shared the laptop. `python3 model_rf.py --plots-only` redraws the two extra figures from the saved tables). Results
`results/random_forest.json`, `results/bagging.json`, `results/rf_grid.csv`, `results/rf_importance.csv`, `results/rf_auc_vs_ntrees.csv`;
figures `figures/random_forest_eval.png`, `figures/bagging_eval.png`, `figures/rf_importance.png`, `figures/rf_auc_vs_ntrees.png`. Same
split, `Pipeline(make_preprocessor(scale=False), RandomForestClassifier(500 trees, oob_score=True))`, `cv=CV`, `scoring="roc_auc"`, seed
2607526, thresholds tuned on training out-of-fold probabilities only. 29 raw features become 32 columns after one-hot (SEX 1, EDUCATION 3,
MARRIAGE 2 dummies).

| model | setting | CV AUC (5-fold, 500 trees) | OOB AUC / OOB error | test AUC | test AP | Brier | acc @0.5 | recall/precision @0.5 | F1-tuned thr | recall/precision @tuned |
|---|---|---|---|---|---|---|---|---|---|---|
| `random_forest` | max_features = sqrt (5 of 32), min_samples_leaf = 20 | 0.7849 ± 0.0076 | 0.7849 / 0.1800 | **0.7866** | **0.5756** | 0.1321 | 0.8245 | 0.369 / 0.694 | 0.31 | 0.564 / 0.539 (F1 0.551) |
| `bagging` | max_features = None (all 32), min_samples_leaf = 20 | 0.7813 ± 0.0066 | 0.7816 / 0.1799 | 0.7833 | 0.5712 | 0.1325 | 0.8236 | 0.377 / 0.684 | 0.27 | 0.588 / 0.511 (F1 0.547) |
| `logistic` (README baseline) | unpenalised, all features | 0.7602 ± 0.0100 | — | 0.7636 | 0.5234 | 0.1383 | 0.8155 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 (F1 0.536) |

Majority-class accuracy is 0.779. Cost-tuned threshold (5:1 FN:FP): random forest t = 0.17, recall 0.757 / precision 0.369, cost 4,998
vs 5,293 at the F1 threshold and 6,598 at 0.5.

- **What was tuned and how.** `GridSearchCV` over `max_features ∈ {sqrt, 0.3, 0.5}` × `min_samples_leaf ∈ {1, 5, 20}` (9 settings × 5
  stratified folds = 45 fits), `scoring="roc_auc"`, `n_jobs=1` in the search and `n_jobs=2` in the forest. To respect the 15-minute budget
  the grid was scored with **200-tree** forests; the winner was refitted with 500 trees on the full training set and its 5-fold CV AUC
  recomputed on the shared folds. Two leakage-free checks (training rows only) show that 200 trees were enough to rank the settings:
  (a) the winner's 500-tree CV AUC is 0.7849 ± 0.0076 vs 0.7843 in the 200-tree grid, and the runner-up (`max_features=0.3`, leaf 20)
  re-scored with 500 trees on the same folds gives 0.7837 ± 0.0073 (vs 0.7832 with 200 trees) — the order is unchanged and the winner is
  ahead in **5 of 5 folds** (`extra.grid_top2_cv_auc_500trees` in the JSON); (b) the OOB AUC of the final forest as a function of the
  number of trees (`rf_auc_vs_ntrees.png`, left panel) reaches 0.7848 at 200 trees and stays within 0.7845–0.7850 up to 500, so 200 trees
  already sit on the plateau. Wall time: grid 226 s, runner-up re-score 49 s, final 500-tree forest 8 s, bagging 49 s, permutation
  importance 90 s, AUC-vs-trees curves 6 s, the two 5-fold out-of-fold passes 239 s — 672 s in total (`n_jobs=2`; wall-clock on a shared
  laptop, so the timings are indicative only).
- **Best setting: `max_features='sqrt'` (5 of 32 columns per split) and `min_samples_leaf=20`.** Leaf size is the dominant knob: CV AUC is
  0.7832–0.7843 for leaf 20, 0.7744–0.7796 for leaf 5 and only 0.7635–0.7666 for leaf 1 (fully grown trees, train AUC 1.000). Within each
  leaf size the smallest `max_features` wins by 0.001–0.005, and it is also faster: in the grid's mean fit times (200 trees, `rf_grid.csv`)
  sqrt takes 2.1–2.8 s per fit vs 3.6–4.8 s for 0.3 (1.7×) and 6.8–8.5 s for 0.5 (3.0–3.2×), because each split evaluates 5 columns
  instead of 10 or 16. With sklearn's default `min_samples_leaf=1` the forest would have scored only 0.7666 CV AUC, barely above logistic
  regression (0.7602): on this noisy target the trees must be regularised even inside an ensemble, because averaging removes variance but
  not the label noise each fully grown tree memorises.
- **Where the forest sits among the tree models (same split, `results/*.json`).** Pruned single tree 0.7755 test AUC / AP 0.5346 →
  random forest 0.7866 / 0.5756 → XGBoost 0.7887 / 0.5784 (LU Xinyu's boosting script, `results/xgboost.json`). The forest captures almost
  the whole ensemble gain over a single tree (+0.011 AUC, +0.041 AP); boosting adds only +0.002 AUC / +0.003 AP on top, still within one
  fold SD (≈ 0.007). The forest is the simplest model in the project that beats logistic regression by a clear margin, with two
  hyper-parameters and no learning-rate/early-stopping machinery.
- **Versus the logistic baseline: +0.023 test AUC (0.7866 vs 0.7636) and +0.052 AP (0.5756 vs 0.5234), Brier 0.1321 vs 0.1383.** At the
  F1-tuned threshold recall/precision are 0.564/0.539 vs 0.563/0.512 — same recall, 2.8 pp more precision (1,122 vs 1,119 defaulters
  caught with 958 vs 1,067 false alarms). The gain is about two and a half fold-SDs of the logistic CV AUC (CV +0.0247 = 2.5 SD, test
  +0.0230 = 2.3 SD, SD 0.0100) and comes from non-linearity and interactions a model linear in the features cannot express: the training
  default rate is 0.13–0.17 for PAY_1 ≤ 0, 0.34 at PAY_1 = 1 and 0.68–0.77 for PAY_1 = 2–3 (a step, not a slope), and the effect of
  utilisation depends on the limit. It is still a modest gain — the ceiling of ~0.79 AUC reflects how much next-month default is genuinely
  unpredictable from six months of history.
- **Feature subsampling (de-correlation) is worth ≈ 0.003–0.004 AUC over plain bagging, at a sixth of the fitting time.** Bagging with
  the same leaf size but `max_features=None` gives CV AUC 0.7813 vs 0.7849, test AUC 0.7833 vs 0.7866 and AP 0.5712 vs 0.5756; the random
  forest is ahead in **all five folds** (differences 0.0015–0.0071) even though the gap is smaller than the fold-to-fold SD. Fitting 500
  trees took 7.9 s vs 49.2 s. Mechanism (lecture B7): the variance of an average of B correlated trees is ρσ² + (1−ρ)σ²/B; bagged trees all
  split first on PAY_1 and are highly correlated (ρ large), so adding trees stops helping early, whereas forcing each split to choose among
  5 random columns lowers ρ. Both panels of `rf_auc_vs_ntrees.png` show it: on the training OOB rows the forest goes from 0.7787 with 50
  trees to 0.7849 with 500, bagging from 0.7759 to 0.7816, a steady gap of 0.003–0.004 at every k ≥ 50; on the test set (descriptive only)
  the forest reaches
  0.7858 with 50 trees and plateaus at 0.7865–0.7868 from 200 trees, while bagging starts lower (0.7804 at 50 trees), climbs more slowly
  (0.7838 at 400) and never closes the gap. Bagging also overfits more (train AUC 0.912 vs 0.888).
- **OOB is a free and accurate substitute for cross-validation here.** OOB AUC 0.7849 vs 5-fold CV AUC 0.7849 (out-of-fold 0.7847) for
  the forest, 0.7816 vs 0.7813 for bagging; OOB error 0.180 vs test error 0.176 at threshold 0.5. Every training row is out-of-bag for
  some tree (0 rows without an OOB prediction once k ≥ 30). In a time-constrained setting one could tune on OOB and skip the 5 refits per
  setting.
- **What matters (`rf_importance.png`, `rf_importance.csv`).** Permutation importance (AUC drop when a variable is shuffled, mean of 5
  repeats, all SDs ≤ 0.0015): **PAY_1 0.042** far ahead, then **UTIL_MEAN 0.019, UTIL_1 0.019, LIMIT_BAL 0.016, BILL_AMT1 0.016,
  PAY_AMT1 0.016, MAX_DELAY 0.014, PAY_AMT2 0.014, BILL_TREND 0.013, PAY_AMT3 0.012**. MDI (Gini share): PAY_1 0.188, MAX_DELAY 0.102,
  N_MONTHS_DELAYED 0.093, PAY_2 0.056, PAY_3 0.038, UTIL_MEAN 0.037, UTIL_1 0.031. Both measures agree on the least useful variables:
  SEX, EDUCATION, MARRIAGE (permutation 0.003–0.004, MDI ≤ 0.005) and AGE (0.008).
- **This matches credit-risk intuition and the logistic coefficients.** The most recent repayment status is the single strongest signal
  (recency: PAY_1 ≫ PAY_2 … PAY_6 under both measures); leverage / capacity comes next (utilisation of the limit, latest bill, and the
  credit limit itself — the bank's own prior screening, negative coefficient in the logit); then recent payment amounts (ability and
  willingness to pay) and a rising balance (BILL_TREND). Demographics add almost nothing once behaviour is known, which is reassuring
  from a fairness perspective. The logistic model puts the same variables at the top (N_MONTHS_DELAYED, MAX_DELAY, UTIL_MEAN, UTIL_1,
  PAY_1, LIMIT_BAL by |standardised coefficient|, `results/logistic_coefficients.csv`; the single largest coefficient is the EDUCATION_4
  dummy, a small "others" level — 343 training rows, 1.6 %, with a 6.7 % default rate — which the forest, having no use for a group that
  small, ranks near the bottom).
- **Why MDI and permutation disagree on the middle of the ranking (Spearman ρ = 0.54).** MDI credits every split, so the near-collinear
  delay variables PAY_2–PAY_6 and N_MONTHS_DELAYED collect a large Gini share between them (MDI ranks 3–14, except PAY_6 at 26), but
  permuting any one of them costs little because its correlated siblings substitute for it (permutation ranks 17–29). MAX_DELAY is the
  exception — high under both measures (MDI rank 2, permutation rank 7, AUC drop 0.0145) — because the worst delay in six months carries
  information no single PAY_k column has on its own. Conversely the six BILL_AMT and six PAY_AMT columns each carry a small unique piece of
  information: low MDI share individually, but a steady 0.008–0.016 AUC drop each when permuted. Permutation importance is the better
  "what would we lose" measure; MDI is the better "what does the forest split on" measure. MDI is also unstable *per tree* (PAY_1 share
  0.19 ± 0.14 across trees, because a tree sees PAY_1 at its root only when the 5-column draw includes it) — only the 500-tree average is
  meaningful.
- **Caveats.** (i) The grid was scored with 200-tree forests, not 500 (budget); the 500-tree re-score of the top two settings and the OOB
  curve (both on training rows) confirm the choice. (ii) Both tuned values sit at a corner of the 3 × 3 grid: `min_samples_leaf=20` is the
  largest leaf tried and `max_features='sqrt'` (5 columns) the smallest draw, and the grid trends monotonically towards that corner in both
  directions. Larger leaves (50, 100), a `max_depth` cap and a smaller draw (e.g. 3 columns) were not explored and may add a little — though
  the `max_features` edge is flat (sqrt beats 0.3 by only 0.0011 at leaf 20, well inside the fold SD). (iii) Permutation importance was
  computed on the training set (in-sample, train AUC 0.888 vs out-of-fold 0.785), so the magnitudes are optimistic; the ranks are what to
  quote. (iv) The bagging variant reuses the forest's leaf size rather than being tuned separately, so the comparison isolates
  `max_features` only. (v) The right panel of `rf_auc_vs_ntrees.png` uses the test set (first k of the fitted 500 trees) and is purely
  descriptive; the number of trees (500) was fixed a priori and nothing was chosen from that curve — the evidence for the 200-tree grid is
  the training-side OOB curve and the 500-tree CV re-score. (vi) The forest's probabilities are better calibrated than the logit's
  (Brier 0.1321 vs 0.1383) but the F1/cost thresholds (0.31 / 0.17) are far below 0.5 — at 0.5 the forest still misses 63 % of defaulters,
  so the operating threshold, not the model, decides recall.
