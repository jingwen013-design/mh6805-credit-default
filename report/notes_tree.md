# Notes — Decision tree with cost-complexity pruning (LU Xinyu, slug `tree`)

Files: `code/model_tree.py` · `results/tree_pruned.json`, `results/tree_full.json`, `results/tree_rules.txt`, `results/tree_cv_grid.csv` ·
`figures/tree_cv_auc_vs_alpha.png`, `figures/tree_pruned.png`, `figures/tree_pruned_eval.png`, `figures/tree_full_eval.png`.
All numbers below are copied from the JSON files (seed 2607526, shared 70/30 split, test n = 8,990).

- **What was tuned.** `Pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(criterion="entropy"))`. A full tree grown on the
  20,975 training rows has **2,791 leaves, depth 55, 5,581 nodes and train AUC 1.000** (0.99999966). Its cost-complexity pruning path
  (lecture B7, weakest-link pruning; alpha = penalty per leaf) contains 1,283 distinct alphas; we kept 21 of them — alpha = 0 plus 20
  values log-spaced in "pruning steps remaining", which gives tree sizes 2,182 → 1,740 → … → 17 → 10 → 6 → 2 leaves — and crossed
  them with `max_depth ∈ {None, 4, 6, 8}`: 84 combinations × 5 stratified folds, `GridSearchCV(scoring="roc_auc", n_jobs=2)`, 52 s.
  No subsampling was needed. Everything (path, grid, CV, thresholds) uses the training split only.

- **Best setting.** `ccp_alpha = 6.63e-4, max_depth = 6` → a tree with **17 leaves, depth 5, 33 nodes**; CV AUC **0.7689 ± 0.0076**.
  The depth cap is effectively non-binding: the same alpha with `max_depth=None` produces the identical 17-leaf / depth-5 tree on the
  full training set and CV AUC 0.7680 (a 0.001 difference, far inside one fold-SD), so the pruning penalty does all the work. There is
  a broad plateau — every alpha between 5.0e-4 and 1.3e-3 (37 to 10 leaves) scores CV AUC 0.767–0.769 — so the result is not sensitive
  to the exact alpha. Report the model as "alpha-pruned entropy tree, 17 leaves, depth 5".

- **Overfitting, made visible (`tree_full` vs `tree_pruned`).** Unpruned tree: train AUC 1.000 but **CV AUC 0.6154, test AUC 0.6125,
  test AP 0.2897**, Brier 0.2717, log-loss 3.75. Because every leaf is pure, it emits hard 0/1 probabilities, so every threshold gives the
  same confusion matrix (recall 0.4052, precision 0.3896, F1 0.3972) and its accuracy 0.7279 is *below* the majority-class 0.779.
  Pruned tree: train AUC 0.7803, OOF AUC 0.7689, **test AUC 0.7755, test AP 0.5346**, Brier 0.1332. Removing 99.4 % of the leaves
  (2,791 → 17) *raises* test AUC by +0.163. `tree_cv_auc_vs_alpha.png` is the textbook bias–variance picture: with no depth cap, CV AUC
  is 0.615 at alpha = 0, climbs steeply through alpha ≈ 3e-4 to 6e-4 (274 → 24 leaves), peaks at ≈ 0.769 for 10–17 leaves, then decays
  slowly to 0.706 for the 2-leaf stump, while training AUC falls monotonically from 1.0. A depth cap alone (`max_depth=4`, alpha = 0)
  already reaches CV AUC 0.762; pruning adds the last +0.007.

- **Against the logistic baseline (all features): test AUC 0.7755 vs 0.764 (+0.012), test AP 0.5346 vs 0.524 (+0.011), CV AUC 0.769
  vs ≈ 0.760.** The gain is real but modest — about one fold-standard-deviation (0.0076) — so the honest wording is "on par or slightly
  better than logistic regression with a 17-rule model". Why a tiny tree can match a 29-feature GLM: default risk in this data is driven
  by *thresholds and interactions* in the repayment-status codes (ever ≥ 2 months late AND ≥ 2 months late now), which a tree captures in
  two splits while a linear logit can only approximate with linear PAY_x terms; the tree also handles the zero-bill / dormant-account
  clients as a separate regime instead of forcing one slope.

- **Operating points on the test set (thresholds tuned on training out-of-fold probabilities, never on test).**
  At 0.5: accuracy **0.8250** (majority class 0.779), precision 0.7358, recall **0.3263**, F1 0.4521, FPR 0.033 — precise but misses two
  thirds of defaulters. F1-tuned threshold **0.28**: precision **0.5314**, recall **0.5661**, F1 **0.5482**, accuracy 0.7935, balanced accuracy
  0.7121 (TN 6,008 / FP 993 / FN 863 / TP 1,126). 5:1 cost-tuned threshold 0.16: recall 0.7577, precision 0.3604, cost 5,085 vs 6,933
  at 0.5 and 7,178 for the full tree — it catches three defaulters in four at the price of flagging 38 % of good clients.

- **What the tree says — root and first splits in domain terms (`tree_pruned.png`, `tree_rules.txt`).**
  *Root:* `MAX_DELAY ≤ 1.5`, i.e. "was the client ever ≥ 2 months behind on a payment in the last six months?" 72 % of clients never
  were (default rate 12.8 %); the 28 % who were default at 46.1 %. This single split is the 2-leaf stump with CV AUC 0.706, and
  `MAX_DELAY` carries 59.8 % of the total impurity reduction (`PAY_1` 17.3 %, `BILL_AMT5` 6.3 %, `N_MONTHS_DELAYED` 3.3 %, `LIMIT_BAL` 3.1 %).
  *Right branch (ever seriously late, n = 5,880):* `PAY_1 ≤ 1.5` separates "was delinquent but ≤ 1 month late on the latest statement"
  (32.6 %) from "≥ 2 months late right now" (**68.4 %**). Within the former, `N_MONTHS_DELAYED ≤ 2.5` separates one-off (26.1 %) from
  chronic late payers (45.4 %). Within the currently-delinquent group, clients whose average repayment ratio exceeds 52 % of the bill
  (`PAY_RATIO_MEAN > 0.523`) drop to 45.3 % vs 70.3 % — paying behaviour partly offsets the delinquency flag.
  *Left branch (never seriously late, n = 15,095):* the next split is `BILL_AMT5 ≤ 789.5` (near-zero May-2005 statement). Counter-intuitively
  these low-activity accounts default *more* (21.0 %) than active ones (10.2 %); the sub-group with `UTIL_MEAN ≤ 0` (no positive balance in
  six months — dormant or newly issued cards) defaults at 35.7 % (n = 686), and above age 53.5 at 77 % (n = 31, tiny). Reading: no
  billing history = no repayment signal, and dormant/first-use cards are a known risk pocket in this dataset. Among active, never-late
  clients, `LIMIT_BAL ≤ 75,000 NT$` gives 15.4 % vs 8.4 % above — the bank's own earlier credit assessment is confirmed by the outcome.

- **Model output is coarse.** 17 leaves means only 17 distinct predicted probabilities (0.058 to 0.774), so the ROC curve is a 17-vertex
  polygon and AP/Brier comparisons with smooth models (logit, boosting) slightly disadvantage the tree. It ranks *groups* of clients, not
  individuals within a leaf; do not use it for fine risk pricing.

- **Caveats.** (a) Single trees are high-variance: a different split or seed can reshuffle the lower splits (thresholds such as
  `BILL_AMT5 ≤ 789.5` or `AGE ≤ 53.5` are data artefacts; only the MAX_DELAY → PAY_1 top structure is robust). That instability is the
  motivation for bagging / random forests / boosting in the next section, which should be compared against this 0.7755. (b) The tree leans
  on the engineered summary features (`MAX_DELAY`, `N_MONTHS_DELAYED`, `PAY_RATIO_MEAN`, `UTIL_*`) that `common.py` adds for
  interpretability; without them it would split directly on `PAY_1`/`PAY_2` with similar AUC but longer rules. (c) The alpha grid comes
  from the path of the tree grown on the whole training set; each CV fold grows a slightly different tree, so the alphas are approximate
  — standard practice, and the plateau shows it does not matter. (d) `max_depth = 6` in `best_params` is a CV tie-break within noise,
  not a meaningful constraint. (e) No class weighting was used; the imbalance is handled by threshold tuning instead.

- **One-liner for the results table.** Pruned tree (entropy, ccp_alpha 6.63e-4, 17 leaves): CV AUC 0.769 ± 0.008, test AUC 0.776,
  test AP 0.535, F1 0.548 at threshold 0.28, accuracy 0.825 at 0.5. Unpruned tree (2,791 leaves): train AUC 1.000, test AUC 0.613,
  test AP 0.290 — shown only to demonstrate overfitting.
