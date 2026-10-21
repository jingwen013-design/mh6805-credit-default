# Notes — unsupervised learning as customer segmentation (slug `unsupervised`, owner LI Jingwen)

Script `code/model_unsupervised.py` (runs in ~65 s). Results: `results/pca_variance.csv`, `pca_loadings.csv` (all 26 PCs),
`kmeans_selection.csv`, `kmeans_profiles.csv`, `hclust_agreement.csv`, `unsupervised_summary.json`, and the two `evaluate()`
outputs `logistic_plus_clusters.json` / `logistic_on_pcs.json`. Figures: `figures/pca_scree.png`, `pca_loadings.png`,
`pca_scatter.png`, `kmeans_elbow.png`, `kmeans_default_rate.png`, `hclust_dendrogram.png`, `pca_logistic_vs_ncomp.png` and the two
`*_eval.png`. The StandardScaler, PCA and k-means are fitted on the 20,975 training rows only; the 8,990 test rows are only
*assigned* with the fitted objects. In the two supervised models these steps sit inside the `Pipeline` and are refitted in every
CV fold (`cv=CV`, `scoring="roc_auc"`, seed 2607526, `n_jobs=2`). All numbers below are copied from the JSON/CSV files.

| model | inputs | CV AUC (5-fold) | test AUC | test AP | Brier | acc @0.5 | recall / precision @0.5 | F1-tuned thr | recall / precision @tuned (F1) |
|---|---|---|---|---|---|---|---|---|---|
| `logistic` (baseline, from `results/logistic.json`) | 29 features → 32 columns | 0.7602 ± 0.0100 | **0.7636** | **0.5234** | 0.1383 | 0.8155 | 0.321 / 0.674 | 0.30 | 0.563 / 0.512 (0.536) |
| `logistic_plus_clusters` | 32 columns + 3 k-means segment dummies (K = 4) | 0.7598 ± 0.0099 | 0.7632 | 0.5219 | 0.1382 | 0.8165 | 0.329 / 0.675 | 0.30 | 0.562 / 0.513 (0.536) |
| `logistic_on_pcs` | first 15 PCs of the 26 numeric features (no categoricals) | 0.7547 ± 0.0104 | 0.7617 | 0.5183 | 0.1390 | 0.8155 | 0.311 / 0.682 | 0.33 | 0.559 / 0.523 (0.540) |

- **What was done and what was tuned.** (1) PCA on the 26 standardised numeric features. (2) K-means (k-means++, 10 restarts)
  for K = 2..10 on the full training set, with the within-cluster SS (elbow) and the mean silhouette on a fixed 5,000-row
  training sample; K chosen by the kneedle elbow criterion (largest gap below the chord of the normalised inertia curve).
  (3) Ward and complete-linkage hierarchical clustering on a 1,000-row stratified training sample, compared with k-means by the
  adjusted Rand index (ARI). (4) Two supervised uses: the plain logistic (`C = inf`, lbfgs, same settings as `logistic`) with the
  K − 1 segment dummies appended, and a logistic on the first n PCs only, n ∈ {5, 10, 15} chosen by `GridSearchCV`. No grid was
  shrunk and nothing was subsampled for tuning — the whole script takes about a minute.
- **The 26 numeric features are highly redundant.** PC1 explains 30.2 % of the variance, PC2 19.6 %, PC3 8.7 %; 5 eigenvalues exceed 1
  (Kaiser), 5 PCs carry 67.3 %, 10 carry 84.3 %, 13 reach 90 % and 15 reach 95.0 % (`pca_scree.png`). The redundancy is the six
  near-identical monthly copies of each quantity: BILL_AMT1–6 load 0.27–0.28 on PC1, PAY_1–6 0.21–0.24, PAY_AMT1–6 move together on
  every component. Interpretation of the loadings (`pca_loadings.png`, signs fixed so the largest |loading| is positive):
  **PC1 = overall indebtedness** — bills, utilisation (UTIL_MEAN 0.279, UTIL_1 0.253) and the repayment statuses all positive, the
  pay-off ratio negative (−0.145); **PC2 = repayment delay vs affluence** — N_MONTHS_DELAYED 0.291, MAX_DELAY 0.285 and PAY_1–6
  (0.19–0.23) against LIMIT_BAL (−0.277), BILL_AMT1–6 (−0.24) and PAY_AMT1–6 (−0.13 to −0.15); **PC3 = recent balance build-up** —
  BILL_TREND 0.460 and UTIL_1 0.379 against PAY_RATIO_MEAN (−0.29), LIMIT_BAL (−0.27) and the delay counts (−0.24, −0.22).
- **Variance is not risk.** The second component, not the first, is the risk axis: the PC2 score alone has AUC 0.701 (correlation with
  DEFAULT 0.288), the PC1 score only 0.617, PC3 0.565. In `pca_scatter.png` the defaulters (red) concentrate in the upper part of
  the cloud (high PC2 = delayed payers) at every level of PC1, and the point cloud is a continuous arc rather than separated
  blobs — the "natural groups" this data set contains are a matter of where one cuts a continuum, which is why the silhouettes
  below are low.
- **K = 4 by the elbow; the silhouette cannot discriminate.** The inertia drops 438,238 → 373,400 → 333,756 (K = 4) and then flattens
  (drops of 65k, 40k, 18k, 14k, 15k, 10k, 9k, 10k); the kneedle gap peaks at K = 4 (0.329 vs 0.234 at K = 3 and 0.303 at K = 5).
  The silhouette on the 5,000-row sample is nearly flat for K ≤ 5 (0.217, 0.216, 0.204, 0.215) and falls to 0.159 by K = 10; its
  maximum at K = 2 merely splits the portfolio by balance size (default rates 28.5 % vs 15.7 %), which is less useful for risk than
  the 4-segment solution, the smallest K at which the delinquent group separates (it persists unchanged, 2.3–2.6 k clients at ~60 %
  default, for every K ≥ 4). A silhouette of ~0.2 says the segments are adjacent regions of one cloud, not distinct clusters.
- **The four segments and what they mean for a risk manager** (`kmeans_profiles.csv`, numbered by ascending training default rate; test
  rows assigned with the training scaler + centroids; `kmeans_default_rate.png`):
  *Segment 1 — transactors* (8,108 clients, 38.7 %; default 16.0 % train / 16.0 % test, lift 0.72): mean utilisation 4.6 % (−0.93 SD),
  median latest bill NT\$2,061 on a median limit of NT\$200,000, pay-off ratio 1.50 (they clear the statement), PAY_1 mean −0.70,
  0.28 delayed months on average; 65 % female. Low risk and low interest income — a retention / cross-sell segment, not a collections one.
  *Segment 2 — affluent heavy revolvers* (2,109, 10.1 %; 17.4 % / 17.0 %, lift 0.79): the largest exposures — median limit NT\$260,000,
  mean latest bill NT\$213,592 (+2.21 SD), utilisation 0.73, but also the largest payments (mean NT\$17,372) and few delays (0.49 months).
  Near-average probability of default on far-above-average balances: this is where *loss given default* and concentration of exposure
  matter, so limit management rather than PD is the lever.
  *Segment 3 — mass-market revolvers* (8,320, 39.7 %; 18.2 % / 17.6 %, lift 0.82): median limit NT\$80,000, utilisation 0.53, rising
  balance (BILL_TREND +0.24, +0.42 SD), pay-off ratio 0.19, 0.39 delayed months. Average risk today but the balance trend makes it the
  segment to monitor for migration into segment 4.
  *Segment 4 — delinquent* (2,438, 11.6 %; **59.7 % / 61.9 %**, lift 2.70 / 2.80): 4.5 delayed months out of 6 (+2.35 SD), maximum delay
  2.5 months, PAY_1 mean 1.55, median limit NT\$50,000, pay-off ratio 0.15, smallest payments (mean NT\$2,315). 11.6 % of clients hold
  31.4 % of all defaults — the obvious collections and limit-freeze priority. The train→test default rates agree to within 2.2 points in
  every segment, so the segmentation is stable out of sample (χ² for default rate across segments on training = 2,275, p ≈ 0).
- **Hierarchical clustering: Ward agrees, complete linkage collapses.** On the 1,000-row stratified sample Ward cut at K = 4 gives sizes
  [194, 479, 207, 120] with default rates [12.4 %, 14.8 %, 50.7 %, 17.5 %] — it rediscovers the delinquent group — and an ARI of 0.576
  against the k-means labels of the same rows (0.586 at K = 5, its best). Complete linkage gives [5, 976, 18, 1]: one giant cluster plus
  outlier singletons (at K = 2 it is [999, 1]), ARI 0.030 vs k-means and 0.020 vs Ward. Its silhouette (0.518) is *higher* than
  Ward's (0.134) or k-means' (0.200) on the same rows — a reminder that silhouette rewards isolating a few extreme balances, not
  a useful partition. Complete linkage measures the distance between the farthest members of two groups, so the heavy-tailed,
  standardised amount columns (a handful of clients with bills or payments 10–20 SD out) dominate every merge; Ward and k-means both
  minimise within-cluster variance and behave alike (`hclust_dendrogram.png`).
- **Segment dummies add nothing to the logistic regression.** `logistic_plus_clusters`: CV AUC 0.7598 ± 0.0099 vs 0.7602 ± 0.0100 for the
  same-run full logistic; test AUC 0.7632 vs 0.7636, AP 0.5219 vs 0.5234, Brier 0.1382 vs 0.1383, and the same F1-tuned threshold
  0.30 (recall 0.562 / precision 0.513 vs 0.563 / 0.512). Differences of 0.0004 are 25× smaller than the fold-to-fold SD. This is
  expected: the segment is a deterministic, coarse function of the very features already in the model, so it can only add information
  through non-linearity — and the dummy coefficients show there is little (given all features, segment 1 vs 4 −0.29 log-odds, segment 3
  vs 4 −0.20, segment 2 vs 4 +0.01). On their own the three dummies reach CV AUC 0.6378 ± 0.0060 (below PC2 alone), i.e. the
  segmentation summarises risk coarsely, not finely. Adding the top-5 PC scores on top of the 26 standardised numeric columns is an exact
  linear combination of columns already present, and the CV AUC is unchanged to four decimals (0.7598), so the evaluated model
  carries the dummies only.
- **Dimension reduction hurts, mildly at n = 15 and badly at n = 5.** `GridSearchCV` chose n = 15 (CV AUC 0.7547 ± 0.0104 vs 0.7250 for
  10 and 0.7189 for 5 PCs). Test AUC 0.7617 and AP 0.5183 — below the full logistic (0.7636 / 0.5234) and below the logistic on the
  26 numeric features without PCA (CV 0.7567, identical to n = 26: an unpenalised logistic is invariant to the rotation, so *all*
  PCs = the raw features; the last 0.0035 to 0.7602 is the six one-hot SEX/EDUCATION/MARRIAGE columns that the PC model omits).
  `pca_logistic_vs_ncomp.png` shows why the variance ordering is the wrong ordering for prediction: n = 1 gives 0.617, adding PC2 jumps
  to 0.704, n = 5…11 sit on a plateau at 0.72–0.73, and the next gains come from *low-variance* components — PC12 (+0.010, 2.5 % of
  variance), PC13 (+0.007, PAY_RATIO_MEAN 0.84), PC15 (+0.010, only 1.3 % of variance: PAY_1 +0.67 and PAY_2 +0.28 against MAX_DELAY
  −0.48 and N_MONTHS_DELAYED −0.36, i.e. the *recency* of delay that the logistic coefficients single out). With p = 26 and n = 20,975
  there is no variance to trade for bias, so PCA has nothing to offer the classifier; the smallest n within one SE of the full 26 is 15.
- **Where the unsupervised results do earn their place.** Not as inputs to the scorecard but as the map on which the scorecard is read:
  the PCA says the 26 numeric columns are three or four things (size of debt, delay vs affluence, balance build-up, plus a recency
  contrast), which explains the collinear coefficient block in `logistic_coefficients.csv` and why lasso keeps entering variables in
  families; the four segments give management a stable, plain-language partition — transactors, affluent revolvers, mass-market
  revolvers, delinquents — with default rates of 16 / 17 / 18 / 60 % that reproduce on unseen clients, on which limit policy,
  collections effort and model monitoring (segment-wise calibration of the XGBoost / logistic scores) can be organised.
- **Caveats.** (i) The amounts were standardised but not log-transformed, so a few extreme bills/payments stretch Euclidean distances
  (visible in the 88 clipped points of the scatter and in the complete-linkage collapse); a log1p transform would give rounder clusters
  but was outside the shared preprocessing contract. (ii) K-means assumes spherical, equal-variance clusters and the silhouette of ~0.2
  says the data do not really contain such clusters; the segments are a managerial partition of a continuum, chosen by an elbow
  heuristic — K = 3 or 5 would be defensible (K = 5 splits off 775 very affluent big payers at 7.1 % default). (iii) The hierarchical
  results rest on 1,000 rows, so per-cluster default rates there have wide confidence intervals (±7 points for a 200-row cluster), and
  the ARI would change with another sample. (iv) Segment numbers are assigned post hoc by training default rate; inside CV folds the
  k-means labels may permute, which is harmless because the logistic is refitted per fold. (v) The AUC differences between the three
  logistic variants (≤ 0.002) are far inside the CV standard deviation (0.010) and should be reported as ties, with the direction
  (no gain from clusters, small loss from PCs) rather than the size being the finding.
