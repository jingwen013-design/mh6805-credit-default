# Notes — Exploratory data analysis (`eda`, LI Jingwen, shared component)

Script: `code/eda.py` (runs in ~4 s, no models fitted). Outputs: `figures/eda_*.png` (7 figures), `results/eda_summary.json`
(every number below), `results/eda_data_quality.csv`, `results/eda_variables.csv` (29 features + target).
Hygiene: data-quality checks and the target distribution describe the raw/cleaned full data; every group default rate,
distribution, correlation and univariate AUC is computed on the **training partition only** (20,975 rows) — the 8,990-row
test set was never inspected, so nothing here can leak into model evaluation.

- **Dataset, cleaning and class balance.** 30,000 clients × 23 raw features + target, all integer, **0 missing cells**. 35 exact
  duplicate rows (0.12%) dropped → 29,965, of which 6,630 default (22.1%; 3.52 non-defaulters per defaulter). Undocumented category
  codes recoded: EDUCATION 0/5/6 (14 + 280 + 51 = 345 rows, 1.15%) → 4 "others", MARRIAGE 0 (54 rows) → 3 "others". Stratified 70/30
  split with seed 2607526: train 20,975 / test 8,990, default rate 22.13% / 22.12%. Six engineered features added (UTIL_1, UTIL_MEAN,
  PAY_RATIO_MEAN, N_MONTHS_DELAYED, MAX_DELAY, BILL_TREND) → 29 features. Majority-class accuracy is 0.779, so accuracy is
  uninformative; AUC and average precision (random-classifier AP = prevalence = 0.221) are the right yardsticks — which is also why
  the logistic baseline's AP (0.524) looks low next to its AUC (0.764).
- **PAY_1 (latest repayment status) is the single most predictive raw variable.** Default rate is 12.9% for PAY_1 = 0
  (revolving, n = 10,380), 13.7% for −2 and 16.9% for −1, but jumps to 33.9% at 1 month late (n = 2,540), 67.6% at 2 months
  (n = 1,871) and 77.4% at 3 months (n = 234). Clients with PAY_1 ≥ 1 are only 22.7% of the training set yet account for
  51.2% of all defaults. Univariate AUC 0.686, Pearson r = 0.32 with the target.
- **The engineered delay counters are stronger than any raw column.** N_MONTHS_DELAYED (univariate AUC 0.724, r = 0.39) and
  MAX_DELAY (AUC 0.714, r = 0.37) top the ranking, ahead of PAY_1 (0.686). Default rate rises monotonically with the number of
  delayed months: 11.8% (0 months, n = 13,933) → 29.7% (1) → 39.1% (2) → 50.6% (3) → 56.7% (4) → 69.3% (6 months, n = 946).
  Most of the logistic baseline's lift (test AUC 0.764) therefore comes from repayment history; a single delay feature alone already
  gets to ~0.72, so the remaining 28 features add only ~0.04 AUC on top.
- **Credit limit is the strongest "capacity" variable.** Default rate falls monotonically across LIMIT_BAL quintiles: 32.1%
  (10k–50k NT\$, n = 5,429) → 25.5% → 19.5% → 17.1% → 13.7% (270k–1M, n = 4,142). Median limit is 90,000 NT\$ for defaulters vs
  150,000 for non-defaulters; univariate AUC 0.619 (higher limit → less default), r = −0.16.
- **Payment behaviour matters, bill size does not.** Clients who paid nothing in September (PAY_AMT1 = 0, n = 3,613) default at
  35.9%, vs 14.3% for 5k–10k and 12.9% for > 10k NT\$; median PAY_AMT1 is 1,604 vs 2,441. The six BILL_AMT columns are nearly
  useless on their own (univariate AUC 0.500–0.520, |r| ≤ 0.023) — what predicts default is the payment relative to the bill
  (PAY_RATIO_MEAN AUC 0.591), not the bill itself.
- **Utilisation is U-shaped, not monotone.** Default rate by UTIL_1 decile: 24.2% in the lowest decile (negative or zero bills:
  credit balances and inactive accounts), a minimum of 14.4% at 0.04–0.13, then rising steadily to 27.4% at 0.74–0.90 and 28.1%
  in the top decile (≥ 0.98, i.e. at or over the limit). Pearson r is only 0.08 and understates this. Age is similarly U-shaped
  (27.1% at 21–24, minimum 19.3% at 30–34, 26.4% at 60+; r = 0.01). Linear models cannot use these shapes without transformation;
  tree-based models can — one concrete reason to expect gradient boosting / random forest to beat logistic regression.
- **Demographics are weak.** Male 24.5% vs female 20.6%; graduate school 19.0%, university 23.9%, high school 25.2% ("others"
  6.7% but n = 343 and mostly the recoded unknown codes); married 23.7% vs single 20.8%. Each gap is a few percentage points,
  small next to the 12.9% → 67.6% swing across PAY_1. Fairness note for the report: SEX carries little signal, so dropping it would cost almost nothing.
- **Multicollinearity is severe among raw amounts.** BILL_AMT1..6 pairwise r = 0.80–0.95 (mean 0.89); 19 feature pairs have
  an absolute correlation above 0.8; UTIL_1 vs UTIL_MEAN 0.91; N_MONTHS_DELAYED vs MAX_DELAY 0.83; PAY_1..6 0.48–0.82. Consequences: individual logistic
  coefficients on monthly bills are unstable and not interpretable (use the engineered summaries, regularisation or PCA/LDA);
  tree models are unaffected; kNN/SVM distances are dominated by the 12 near-duplicate amount columns unless scaled.
- **Data-quality decisions (all "keep").** Negative bills in 1,930 rows (6.4%, 3,932 cells, min −339,603) are credit balances;
  PAY_AMT = 0 in ≥ 1 month for 14,542 rows (48.5%) and in all six months for 1,432 rows (4.8%); 866 rows (2.9%) have zero bills
  in all six months (UTIL = 0, PAY_RATIO_MEAN = 0 by construction); UTIL_1 > 1 in 2,115 rows (7.1%, max 6.46) — over-limit is a
  genuine risk signal. PAY codes −2 and 0 are absent from the UCI codebook (which lists only −1 and 1..9); we interpret them as
  "no consumption" and "revolving credit / minimum paid" and keep PAY_k ordinal. LIMIT_BAL (10k–1M, median 140k) and AGE (21–79)
  are plausible; no outlier trimming, StandardScaler inside the pipeline handles the heavy right tails.
- **Caveats.** (a) Group rates use the training partition only; full-sample values differ by < 0.5 pp. (b) PAY_1 codes 4–8 have
  6–57 clients each, so their rates (33%–78%) are noisy — treat PAY_1 as ordinal, do not one-hot it. (c) PAY_1 is the status one
  month before the label month, so it is partly a lagged version of the target; performance on this dataset should not be read
  as "predicting default from scratch". (d) Taiwan 2005, NT\$ amounts — level effects (e.g. the 50k limit cut-off) are not transferable.
  (e) The EDUCATION "others" group (6.7%) is small and heterogeneous; its low rate should not be over-interpreted.
