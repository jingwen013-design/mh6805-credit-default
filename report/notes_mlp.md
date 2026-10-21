# Notes — Feed-forward neural network / MLP (LI Jingwen, slugs `mlp`, `mlp_weighted`, `mlp_overfit`)

Script: `code/model_mlp.py` (runs end-to-end in about 5.5 min with `n_jobs=2`: 333 s on the run that produced the files below; the wall-clock
times are stored in the JSONs as `fit_seconds`, `extra.grid.grid_search_seconds`, `extra.oof_predict_seconds`, `extra.total_script_seconds`).
Results: `results/mlp.json`, `results/mlp_weighted.json`, `results/mlp_overfit.json`, `results/mlp_cv_table.csv` (CV AUC of all 24 settings),
`results/mlp_overfit_epochs.csv`, `results/mlp_comparison.csv` (the cross-model table below, written by the script from the `results/*.json`
present at run time). Figures: `figures/mlp_loss_curve.png`, `figures/mlp_overfit_epochs.png`, `figures/mlp_eval.png`, `figures/mlp_weighted_eval.png`,
`figures/mlp_overfit_eval.png`. All numbers below are copied from the JSON files (seed 2607526, shared 70/30 split, train n = 20,975, test n = 8,990).
Lecture reference: B9. PyTorch is not installed on the laptop, so the network is `sklearn.neural_network.MLPClassifier` (ReLU, Adam, mini-batches
of 200, log-loss + L2 penalty `alpha`); 32 inputs after the shared preprocessing (26 standardised numeric columns + 6 one-hot dummies —
SEX_2, EDUCATION_2/3/4, MARRIAGE_2/3 — with `drop="first"`).

"CV AUC" for `mlp` is the **best-of-grid** 5-fold mean: the maximum over the 24 settings on the same folds that are then re-used for the
out-of-fold probabilities and the threshold choice (no nested CV — the project-wide convention), so it carries a little selection optimism
(at most ≈ 0.003 given the grid spread of 0.0047). For `mlp_weighted` and `mlp_overfit` it is the mean per-fold AUC of the out-of-fold
probabilities (no grid). Test-set differences between models are compared with a paired bootstrap of the saved `results/*_p_test.npy`
(2,000 resamples of the 8,990 test rows, seed 2607526; `extra.paired_bootstrap_test` in `mlp.json`), not with the CV means.

| Model | Setting | CV AUC (5-fold, train) | Train AUC | Test AUC | Test AP | Brier | Log-loss | Acc @0.5 | Recall @0.5 | Precision @0.5 | F1-tuned thr | Recall @tuned | Precision @tuned | F1 @tuned | Fit time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP (tuned, early stopping) `mlp` | 32-64-32-1, alpha 1e-3, lr 1e-3, 4,225 weights, stopped after 23 epochs (best epoch 7) | 0.7762 ± 0.0069 (best of grid) | 0.7875 | **0.7731** | **0.5513** | 0.1353 | 0.4329 | 0.8224 | 0.3680 | 0.6828 | 0.27 | 0.5933 | 0.4882 | 0.5356 | 0.97 s |
| MLP + balanced sample weights `mlp_weighted` | same network and hyper-parameters, per-row weights 2.26 (default) / 0.64 (non-default), stopped after 19 epochs (best epoch 3) | 0.7739 ± 0.0088 (OOF folds) | 0.7817 | 0.7710 | 0.5354 | 0.1855 | 0.5578 | 0.7461 | 0.6360 | 0.4479 | 0.57 | 0.5757 | 0.5011 | 0.5358 | 0.86 s |
| MLP over-parameterised (no early stopping) `mlp_overfit` | 32-256-256-1, alpha 1e-4, 200 epochs, 74,497 weights | 0.6921 ± 0.0067 (OOF folds) | **0.9973** | 0.6776 | 0.3997 | 0.2123 | 1.3673 | 0.7556 | 0.3444 | 0.4341 | 0.05 (grid floor) | 0.5199 | 0.3511 | 0.4191 | 44.2 s |

Comparison with the other models in `results/` (test set, same split; thresholds tuned on training out-of-fold probabilities). Copied from
`results/mlp_comparison.csv`, which `model_mlp.py` builds from the `results/<slug>.json` files present when it runs (this run: JSON files
written 2026-09-16 11:11–15:33); sorted by test AP:

| Model (slug) | CV AUC | Test AUC | Test AP | Brier | F1 @tuned thr |
|---|---|---|---|---|---|
| XGBoost (`xgboost`) | 0.7863 | 0.7887 | 0.5784 | 0.1311 | 0.5498 |
| Gradient boosting, sklearn (`gradient_boosting`) | 0.7850 | 0.7872 | 0.5759 | 0.1314 | 0.5485 |
| Random forest (`random_forest`) | 0.7849 | 0.7866 | 0.5756 | 0.1321 | 0.5515 |
| Bagging (`bagging`) | 0.7813 | 0.7833 | 0.5712 | 0.1325 | 0.5470 |
| **MLP (`mlp`)** | **0.7762** | **0.7731** | **0.5513** | **0.1353** | **0.5356** |
| KNN (`knn`) | 0.7700 | 0.7706 | 0.5423 | 0.1361 | 0.5411 |
| MLP + balanced sample weights (`mlp_weighted`) | 0.7739 | 0.7710 | 0.5354 | 0.1855 | 0.5358 |
| Pruned tree (`tree_pruned`) | 0.7689 | 0.7755 | 0.5346 | 0.1332 | 0.5482 |
| Logistic regression (`logistic`, baseline) | 0.7602 | 0.7636 | 0.5234 | 0.1383 | 0.5360 |
| MLP over-parameterised (`mlp_overfit`) | 0.6921 | 0.6776 | 0.3997 | 0.2123 | 0.4191 |

## Interpretation bullets (for the report)

- **What was tuned, and the best setting.** `Pipeline(make_preprocessor(), MLPClassifier(random_state=2607526, early_stopping=True,
  validation_fraction=0.15, max_iter=300, n_iter_no_change=15))` with `GridSearchCV` over hidden_layer_sizes ∈ {(32,), (64,), (64,32), (128,64)}
  × alpha (L2) ∈ {1e-4, 1e-3, 1e-2} × learning_rate_init ∈ {1e-3, 3e-3} = 24 settings × 5 stratified folds (120 fits), scoring = ROC AUC,
  `n_jobs=2`; one early-stopped fit takes about 1 s, so the full grid ran in 77 s and neither `RandomizedSearchCV` nor subsampling was needed.
  StandardScaler + one-hot are refitted inside every fold, and inside each fit the network holds out 15 % of the rows it is given as its
  early-stopping split; the test set is touched only inside `evaluate()`. **Best: 32-64-32-1 (two hidden layers of 64 and 32 ReLU units,
  4,225 weights), alpha = 1e-3, learning rate 1e-3, best-of-grid CV AUC 0.7762 ± 0.0069**; the refit on the full training set stopped after
  23 epochs with the best internal-validation accuracy (0.8297) at epoch 7, weights restored to that epoch (`mlp_loss_curve.png`). The grid is
  flat: all 24 settings lie between CV AUC 0.7715 and 0.7762 (spread 0.0047, less than one fold-SD), with marginal means 0.7748 / 0.7736 /
  0.7747 / 0.7737 for (32,) / (64,) / (64,32) / (128,64), 0.7742 / 0.7743 / 0.7741 for alpha 1e-4 / 1e-3 / 1e-2 and 0.7741 / 0.7743 for the
  two learning rates. Grid-edge check (`extra.best_on_grid_edge`): architecture and alpha are interior values, but the selected learning rate
  1e-3 is the lower edge of the two-point grid {1e-3, 3e-3}; the marginal means (0.7741 vs 0.7743) show the choice is immaterial, so the grid
  was not extended (a 3e-4 point would be the first extension). A one-standard-error rule would pick the smallest network (32,) with
  alpha 1e-2 (CV AUC 0.7757). The ranking among the 24 settings is inside CV noise and should not be over-interpreted.
- **Early stopping is the regulariser that matters, not `alpha`.** The in-fold optimism gap (training-fold AUC minus CV AUC in
  `mlp_cv_table.csv`) grows with network size from about 0.015–0.021 for (32,) to 0.024–0.041 for (128,64), yet CV AUC does not fall, because
  every network is stopped within a handful of epochs. The epoch diagnostic (`mlp_overfit_epochs.png`: both architectures trained 200 epochs
  *without* early stopping on an 85/15 split of the training set; descriptive only, no choice is made from it) shows that even the tuned
  32-64-32-1 architecture overfits when left running — held-out AUC peaks at epoch 14 (0.7624) and decays to 0.7116 by epoch 200 while the
  training AUC climbs to 0.8913 — so alpha = 1e-3 is far too weak on its own. With early stopping the final model's train/test gap is only
  0.7875 vs 0.7731, smaller than the random forest's (0.8876 vs 0.7866).
- **Overfitting made visible (`mlp_overfit`).** A deliberately over-parameterised 32-256-256-1 network (74,497 weights, 3.55 per training row,
  sklearn default alpha = 1e-4, no early stopping, 200 epochs) drives the training log-loss from 0.45 to 0.064 and the training AUC to
  **0.9973**, but its out-of-fold AUC is 0.6916 and its **test AUC 0.6776, test AP 0.3997** — below every other model in the project except the
  unpruned tree (train AUC 1.000, test 0.6125), and 0.086 AUC below the logistic baseline. Its probabilities are extreme (test log-loss 1.37,
  Brier 0.212), so accuracy at 0.5 (0.7556) is below the majority-class rate (0.779), and both the F1- and the 5:1-cost-optimal thresholds sit
  at the 0.05 floor of the shared threshold grid (`extra.tuned_threshold_at_grid_floor`: a clipped value, not an interior optimum). In the epoch
  diagnostic its held-out AUC is highest after the *first* epoch (0.7597) and only declines (0.7000 at epoch 200) while the training AUC reaches
  0.994 — the bias–variance picture from lecture B9: memorising 21k noisy labels.
- **Against the logistic baseline: test AUC 0.7731 vs 0.7636, test AP 0.5513 vs 0.5234 (best-of-grid CV AUC 0.7762 vs 0.7602).** The
  paired bootstrap on the test set puts the difference at ΔAUC = +0.0095, 95 % CI [+0.0040, +0.0153] and ΔAP = +0.0279, 95 % CI [+0.0178,
  +0.0376] (0.15 % of resamples had ΔAUC ≤ 0, none had ΔAP ≤ 0): real, but modest, and at the F1-tuned operating point it disappears (F1 0.5356
  vs 0.5360). Default risk here is a mostly smooth, low-signal function of the repayment codes with a few threshold-type effects (ever ≥ 2
  months late; currently ≥ 2 months late; dormant zero-bill accounts) and low-order interactions that a logit approximates linearly; stopped
  at epoch 7, a 64-32 ReLU network is essentially "logistic regression plus a few learned kinks and interactions", which is what +0.01 AUC /
  +0.03 AP reflects. It is on par with the pruned tree (AUC 0.7755, AP 0.5346: −0.002 AUC, +0.017 AP) and slightly ahead of tuned KNN
  (0.7706 / 0.5423).
- **Against the tree ensembles the network loses: XGBoost 0.7887 / gradient boosting 0.7872 / random forest 0.7866 / bagging 0.7833 test AUC
  vs 0.7731, and AP 0.5712–0.5784 vs 0.5513.** Versus XGBoost the paired bootstrap gives ΔAUC = −0.0155, 95 % CI [−0.0204, −0.0108] and
  ΔAP = −0.0271, 95 % CI [−0.0384, −0.0152] (every resample negative), i.e. a larger and more certain gap than the MLP's own advantage over
  logistic regression; the gap is consistent in CV (0.7762 vs 0.7813 for bagging and 0.7849–0.7863 for the forest and the two boosters).
  Ranking by test AP: XGBoost > gradient boosting > random forest > bagging > **MLP** > KNN > pruned tree > logistic (the three top ensembles
  are within 0.003 AP of each other, so their order is not meaningful; the MLP's place below all four of them is).
- **Why a small MLP does not beat gradient boosting on tabular data of this size — and what more time would buy.** (i) *Inductive bias:* the
  strongest predictors are ordinal codes (PAY_1…PAY_6, MAX_DELAY, N_MONTHS_DELAYED) whose effect is step-like; a tree captures "PAY_1 ≥ 2" in one
  split, a ReLU network has to build each threshold from scaled inputs and gradient steps, and stage-wise boosting of shallow trees is a very
  strong prior for "a few thresholds plus low-order interactions". (ii) *Sample size:* 20,975 rows with 22 % positives is small for a neural
  network (fold-to-fold CV AUC sd 0.007–0.011) and early stopping at epoch 7 keeps it close to its linear regime. (iii) *Feature skew:* the
  monetary columns are heavy-tailed even after standardisation; tree splits are invariant to monotone transforms, an MLP is not, and the shared
  preprocessing contract allows no log/quantile transform. (iv) *Rotation invariance:* an MLP spends capacity on uninformative linear
  combinations of the 32 inputs while trees work feature-by-feature — the tabular-benchmark literature's explanation for GBDTs beating MLPs on
  small/medium tables. (v) *Tooling:* sklearn's MLP has no dropout, batch-norm or embeddings and no `class_weight` argument (though `fit()`
  accepts `sample_weight`, which we used for `mlp_weighted` — it did not help the ranking metrics: test AUC 0.7710, AP 0.5354, see the next
  bullet), and one seed was used. The flat grid (every architecture ≈ 0.773–0.776) says capacity is not the bottleneck: this family tops out
  around CV AUC 0.776 on these features versus 0.781–0.786 for bagging, forests and boosting. With more time (PyTorch, not installed here):
  entity embeddings for SEX / EDUCATION / MARRIAGE and for PAY_1…PAY_6 treated as categorical (the −2/−1/0 codes are not ordinal); dropout and
  batch-normalisation instead of a weak L2 penalty; focal loss; a log1p / quantile transform of the amounts; early stopping on validation AUC
  rather than accuracy; a learning-rate schedule and a 3e-4 start (1e-3 sits on the grid edge); averaging 5–10 seeds; Platt/isotonic calibration;
  Bayesian search. Expected gain from the flat grid and the literature: at most +0.005–0.01 AUC — enough to match, not clearly beat, boosted trees.
- **Operating points on the test set, and what balanced class weights change.** At the default 0.5 threshold the tuned network is accurate
  (0.8224 > majority-class 0.779, precision 0.6828, FPR 0.049) but catches only 36.8 % of defaulters (recall 0.3680; TN 6,661 / FP 340 /
  FN 1,257 / TP 732). The F1-optimal threshold 0.27, chosen on training out-of-fold probabilities, lifts recall to 0.5933 at precision 0.4882
  (F1 0.5356, accuracy 0.7724, balanced accuracy 0.7083); the 5:1 cost-optimal threshold 0.19 gives recall 0.7099 at precision 0.3855 and cuts
  the test cost from 6,625 (at 0.5) to 5,136. `mlp_weighted` (same tuned network refitted with balanced per-row weights, 2.26 / 0.64, passed as
  `clf__sample_weight`; not re-tuned) shows that class weighting is a threshold shift rather than a better model: at 0.5 it already catches
  63.6 % of defaulters (precision 0.4479, accuracy 0.7461), its tuned thresholds move to 0.57 (F1) and 0.42 (cost) from 0.27 / 0.19, and at those
  points it is indistinguishable from the unweighted network (F1 0.5358 vs 0.5356; cost 5,090 vs 5,136), while its ranking metrics are slightly
  lower (AUC 0.7710, AP 0.5354; per-fold CV AUC 0.7739 ± 0.0088) and its probabilities are no longer on the 22 % base-rate scale (Brier 0.1855
  vs 0.1353, log-loss 0.558 vs 0.433). Same conclusion as `notes_imbalance.md` for logistic regression and the random forest: for every model
  in the project, threshold choice moves the business outcome far more than the last 0.01 of AUC.
- **Caveats.** (a) One seed: MLP results move by a few thousandths of AUC with the initialisation and mini-batch order; the paired bootstrap
  above quantifies test-set sampling noise, not seed noise. (b) The reported CV AUC 0.7762 is a best-of-grid figure (24 settings on the same
  folds, no nested CV — the project-wide convention); the test AUC 0.7731 sits 0.003 below it, consistent with a small selection optimism.
  (c) Early stopping uses 15 % of every training fold as validation, so each network sees only 85 % of the rows the other models see, and it
  stops on accuracy (weighted = balanced accuracy for `mlp_weighted`), not AUC. (d) `mlp_overfit` is a deliberately mis-specified teaching
  example (default alpha, no early stopping); with early stopping a (256,256) network would score about the same as the small one — the grid's
  (128,64) already does (CV AUC 0.7759) — so the lesson is "regularise", not "big networks are bad". (e) The epoch diagnostic uses a single 85/15
  split of the training set and `partial_fit`, a different protocol from the 5-fold CV; it is descriptive only. (f) Timings (grid 77 s, refit
  0.97 s, weighted fit 0.86 s, over-parameterised fit 44.2 s and its out-of-fold pass 108 s, bootstrap 32 s, whole script 333 s) are laptop
  wall-clock with 2 workers × 1 BLAS thread and vary with what else is running (52 s / 275 s on an earlier quiet run); the metrics are seeded and
  reproduced to every printed digit across three runs. (g) The comparison table reflects the `results/*.json` files present on 2026-09-16 15:33;
  the SVM results (`svm_linear` 0.7506 / 0.5063, `svm_rbf` 0.7616 / 0.4944, `svm_linear_calibrated` 0.7627 / 0.5150 test AUC / AP) are in
  `results/` but not in the script's comparison list — all three rank below logistic regression on AP. If a teammate re-runs a model, re-run
  `model_mlp.py` (or read `results/mlp_comparison.csv`) before quoting the table.
- **One-liner for the results table.** Tuned MLP (32-64-32-1, alpha 1e-3, early stopping at epoch 7/23): best-of-grid CV AUC 0.776 ± 0.007,
  test AUC 0.773, test AP 0.551, F1 0.536 at threshold 0.27, accuracy 0.822 at 0.5 — beats logistic regression (0.764 / 0.523; paired ΔAUC
  +0.009 [+0.004, +0.015]) and KNN, ties the pruned tree, trails every tree ensemble (XGBoost 0.789 / 0.578, gradient boosting 0.787 / 0.576,
  random forest 0.787 / 0.576, bagging 0.783 / 0.571; paired ΔAUC vs XGBoost −0.016 [−0.020, −0.011]). Balanced sample weights only shift the
  thresholds (AUC 0.771 / AP 0.535). Over-parameterised 32-256-256-1 without early stopping: train AUC 0.997, test AUC 0.678, test AP 0.400 —
  shown only to demonstrate overfitting.
