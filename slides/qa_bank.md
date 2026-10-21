# Q&A bank — MH6805 Group B19 final presentation (27 Oct 2026, 10 min + Q&A)

Predicting credit-card default: a comparative study of machine-learning classifiers. Team: RUAN Hangjing (G2607546F) — KNN, logistic family, nested CV, PCA/K-means; LU Xinyu (G2603537D) — tree, forest/bagging, GBM/XGBoost, TreeSHAP, monotone XGBoost, PDP; LI Jingwen (G2607526K) — SVMs, MLP, LDA/QDA/NB, imbalance study, Platt recalibration, freeze machinery, aggregation.

Revised 2026-10-22 for the v2 deck (S1–S21 + A1/A2, `slides/slide_spec_v2_2026-10-22.md`) and report v3 (`report/sec_*.tex`): it now uses the 21 Oct post-freeze checks (`results/POST_REVIEW_CHECKS.md`; training CV only, never test-scored, selecting nothing) and the dated record of every test read (`results/TEST_SET_LOG.md`), and adds five items for the new slides (R15, L16, L17, J16, J17). First rewritten 2026-10-21 after the freeze (`results/frozen_manifest.json`, 46 entries), RUAN's adopted 30 Sep second pass, the Platt recalibration (19 Oct), the 1:1–20:1 cost sensitivity, the nested-CV check and the clustering-stability analysis. **Every number below is read from a file under `results/` and the file is named in the answer.** Earlier versions of this bank ranked models by test AUC in places, used "ten times" multipliers and quoted the K = 4 segments; none of that may be said on 27 Oct.

## How to use this bank

The individual mark (75 %) is decided mainly in the Q&A, and the discussion group prepares its questions from our slides. Each presenter owns the section with their name and must also be able to answer the eight general questions. The v2 deck moved three topics, so three items sit outside their speaker's block: L13 and L14 (recommendation, limitations) belong to LI's slide 21 and L15 (cluster stability) to RUAN's slide 5; the item's owner and the slide's speaker both prepare them. Every answer is 2–4 sentences, gives the exact number and the results file, and is honest about weaknesses. Three answer frameworks, tagged at the end of each answer:

- **Acknowledge · reason · leave room** — concede the premise, give the mechanism, say what would change the answer.
- **Quantify** — put the claim on the scale of a fold-paired training-CV interval (Nadeau–Bengio, `results/fold_stability.csv`) or the 5-fold CV standard deviation (0.007–0.011 AUC), with the paired test-set bootstrap as a reporting-only check (B = 500 in `results/auc_bootstrap_ci.csv`; B = 1,000 in `results/threshold_vs_model_saving.json`).
- **Admit · give direction** — when something was not done, say so and name the next step.

Four phrases to use verbatim: "chosen on training CV; each configuration frozen at its first test scoring; every test read dated; the test set only reports" · "the cut-off matters more than the model" · "weight the decision, not the data" · "segments are a map for policy, not a better score". Never say that a configuration was scored, or the test set touched, only once (the test set was read more than once, and every read is dated in `results/TEST_SET_LOG.md`); never call any model the CV "winner" (the three ensembles tie); never call the segments economically meaningful. Say "predict", "rely on" or "associated with", not causal verbs (the bank sets the limit itself), and call 5:1 a scenario borrowed from the Statlog German credit cost matrix, never a bound.

### Ten numbers to have in your head

± is the fold SD. The report and `results/fold_stability.csv` use ddof 1 (XGBoost ± 0.008, KNN ± 0.010, logit ± 0.011); the model JSONs quoted in some rows and answers below use ddof 0 (± 0.0068 / 0.0087 / 0.0100).

| Number | Value | Source |
|---|---|---|
| Sample / train / test | 29,965 clients (35 duplicates dropped from 30,000); 20,975 / 8,990; 22.1 % default (4,641 / 1,989 defaulters); majority accuracy 0.779; seed 2607526 | `results/eda_summary.json`, `results/aggregate_summary.json` |
| KNN benchmark (K = 101, uniform) | CV 0.7700 ± 0.0087, test 0.7706, AP 0.5423; nested CV 0.7684 ± 0.0073 re-selects K = 101 | `results/knn.json`, `results/nested_cv_summary_repro.json` |
| Logistic baseline (plain) | CV 0.7602 ± 0.0100, test 0.7636, AP 0.5234, Brier 0.1383; cost 7,063 at 0.5 → 5,126 at the training-OOF cut-off 0.15 | `results/logistic.json` |
| XGBoost | highest CV AUC 0.786 ± 0.008, tied with GBM (+0.0013 [−0.0005, +0.0031]) and the forest (+0.0014 [−0.0027, +0.0055]), training-fold Nadeau–Bengio intervals; test 0.789 [0.776, 0.800] | `results/fold_stability.csv`, `results/auc_bootstrap_ci.csv` |
| CV order of the 11 headline models | XGB 0.7863 · GBM 0.7850 · RF 0.7849 · MLP 0.7762 · KNN 0.7700 · pruned tree 0.7689 · RBF SVM 0.7635 (mean of shared-fold OOF AUCs; pooled 0.7606) · logistic 0.7602 · lasso 0.7602 · LDA 0.7572 · NB 0.7356 | `results/model_comparison.csv` (`select_auc`, `rank_cv_auc`), `results/svm_selection_statistics.json` |
| Paired bootstrap, test AUC differences | XGB − RF +0.0020 [−0.0010, +0.0051] (tie); XGB − GBM +0.0015 [+0.0001, +0.0030]; XGB − logistic +0.0250 [+0.0183, +0.0314] | `results/auc_bootstrap_ci.csv` |
| Threshold vs model saving at 5:1 | moving the logistic cut-off 0.5 → 0.15 saves 1,937 [1,604, 2,218]; switching logistic → XGBoost saves 145 [−18, 299]; model-saving CI includes 0 at 5:1 and 20:1; 20:1 cut-off 0.01 = flag everyone | `results/threshold_vs_model_saving.csv` / `.json` |
| Imbalance + recalibration | logit CV AUC range 0.0012 across none/balanced/SMOTE/under; SMOTE costs the forest 0.0055 CV AUC (0.780 → 0.774); weighting and under-sampling move it by under 0.001; test −0.0076 as a check; Platt returns weighted XGBoost Brier 0.1772 → 0.1311 and the 5:1 cut-off 0.37 → 0.14; balanced@0.5 = plain@0.22 on 99.4 % of test clients | `results/imbalance_comparison.csv`, `results/recalibration.json`, `results/imbalance_threshold_equivalence.csv` |
| Leakage demonstration | SMOTE before the split: forest "test" AUC 0.914 vs honest 0.774 (logit 0.770 vs 0.762); 72.6 % of the leaky test defaulters are synthetic | `results/imbalance_leakage_demo.csv` |
| Clustering (secondary) | PCA 10 PCs = 81.8 % variance; K = 5, silhouette 0.269; bootstrap ARI 0.957 [0.745, 0.990]; Jaccard 0.972 / 0.974 / 0.791 / 0.970 / 0.953; Ward ARI mean 0.523; cluster label ΔAUC +0.0001 [−0.0004, +0.0007], ΔAP −0.0003 [−0.0016, +0.0010] → not adopted | `results/unsupervised_summary_repro.json`, `results/kmeans_profiles_repro.csv` |

---

## G. General questions (all three presenters) — 8

**G1. Describe your protocol in one minute. Where exactly was the test set used?**

Every choice (settings, cut-offs, model order) is made on the 20,975 training rows by five-fold CV and out-of-fold scores. Round-1 configurations were scored on 16–18 Sep and entered in the freeze register on 19 Oct (46 entries); since then `evaluate()` refuses a changed configuration. Before that, the dated log in `results/TEST_SET_LOG.md` is the guard. It shows that until 19 Oct test AUC chose the four models in the report's figures, and that nine Platt variants were scored on 19 Oct after a 6 Oct review. So the test set is a dated report card, read more than once, never the basis of a model, feature or cut-off. *Framework: quantify.*

**G2. What does "frozen" mean, and how can we check it?**

A configuration counts as frozen at its first test scoring. `freeze.py` stores the hash of its settings and of its test-probability vector in `results/frozen_manifest.json`, and `--check` re-verifies all 46. The register itself was written on 19 Oct; earlier reads, including identical re-runs and RUAN's 30 Sep re-fits of the plain logit and KNN, are dated in `TEST_SET_LOG.md`. *Framework: quantify.*

**G3. Who did what? Give the contribution split.**

By model family, as in the proposal: RUAN — KNN, the logistic family, the nested-CV second pass, PCA/K-means and the stability analysis; LU — pruned tree, random forest and bagging, GBM and XGBoost, TreeSHAP, monotone XGBoost, partial dependence; LI — RBF and linear SVM, MLP, LDA/QDA/naive Bayes, the class-imbalance study, Platt recalibration, the shared `common.py` contract, `aggregate.py` and `freeze.py`. The per-section percentages in the report (DRAFT, group to confirm) are: Introduction 30/20/50, Data 25/15/60, Methods 30/25/45, Results 30/35/35, Interpretation 15/60/25, Operating point 40/15/45, Conclusion 34/33/33, Code 20/25/55, Presentation 33/33/34 (RUAN / LU / LI; `slides/contribution_statement_draft.md`). Each result JSON carries a `member` field, so the ownership is auditable file by file. *Framework: quantify.*

**G4. Did you use AI tools, and how?**

Yes, as a coding and typesetting assistant: drafting script skeletons, LaTeX, figure code and the first wording of the slide notes, always from our own results files. Every number in the report, slides and this bank is produced by a script in `code/` that writes to `results/` (`code/run_all.sh` regenerates all of them and `freeze.py --check` verifies the frozen test scorings), and each of us can explain the models we own and walk through the code that produced them. The modelling decisions — the KNN benchmark, the freeze, the 1:1–20:1 sweep, the Platt step, the pre-registered clustering rule — were ours and were made in response to the lectures and the professor's written comment. *(Group to confirm the exact wording of the disclosure before 26 Oct.)* *Framework: acknowledge · reason · leave room.*

**G5. Why a paired bootstrap rather than a DeLong test for the AUC differences?**

Because we wanted one procedure that works for AUC, average precision and the cost at a threshold, and DeLong covers only AUC; the paired row bootstrap (same 8,990 test rows resampled for both models, B = 500, seed 2607526) gives percentile intervals for all three (`results/auc_bootstrap_ci.csv`, `results/threshold_vs_model_saving.json` with B = 1,000). For the headline comparison the two methods would agree: XGBoost − logistic +0.0250 [+0.0183, +0.0314] is far from zero, while XGBoost − forest +0.0020 [−0.0010, +0.0051] is a tie under any test. The limitation is that with B = 500 the percentile end-points are themselves estimates: the GBM lower bound is only +0.0001 (+0.0015 [+0.0001, +0.0030], `auc_diff_2_5` in `results/auc_bootstrap_ci.csv`), so we call that one "borderline", not "significant". *Framework: acknowledge · reason · leave room.*

**G6. Why a 5:1 cost ratio?**

5:1 is a scenario borrowed from the Statlog German credit cost matrix (Hofmann 1994), not measured for this issuer. Our rough anchors (50–70 % of the median defaulter's bill lost against one to three months of a 1.5 % monthly margin on the median good client) give 9.6:1 to 40.5:1. That is an upper bound, since some defaulters later pay (`cost_anchors.json`). So we report a 1:1–20:1 sweep: from 2:1 to 20:1 the threshold saving exceeds the model saving (`threshold_vs_model_saving.csv`). *Framework: quantify.*

**G7. Everything rests on one 70/30 split. Why no repeated splits?**

The bootstrap quantifies test-sampling noise (XGBoost's own interval is [0.7760, 0.7999], about ±0.012) but not the variation another split would add; the fold-to-fold CV standard deviation (0.007–0.011) is only a proxy. Two things make us comfortable with the ranking: nested CV reproduces the plain 5-fold figures for KNN (0.7684 ± 0.0073 vs 0.7700) and the logistic family (0.7602 vs 0.7602) within one standard error (`results/nested_cv_summary_repro.json`), and the test AUC of every headline model sits within one CV standard deviation of its CV AUC (`results/model_comparison.csv`). Repeating the whole pipeline over five to ten seeds is the direct fix; it was outside the compute budget and, after the freeze, would have required a fresh hold-out to avoid turning the test set into a validation set. On the training side, the four tiers hold in all five folds, and the top three swap places within noise (`fold_stability.csv`). The ensembles were not nested, so their internal order cannot be separated from selection optimism. *Framework: admit · give direction.*

**G8. What did the professor's 30 Sep comment change in your project?**

Four things, each visible in the deliverables. (1) Test record: a freeze register (19 Oct) plus a dated log of every test read, including those before the register; nine Platt variants were scored on 19 Oct. (2) Cost: two absolute savings with intervals over a 1:1–20:1 sweep, and 5:1 labelled a borrowed scenario. (3) Clustering is secondary: K = 5 on 10 PCs, stability measured, and the label adds +0.0001 [−0.0004, +0.0007] AUC in training CV. The K = 5 model was never test-scored; the round-1 K = 4 version was scored on 16 Sep and is listed for completeness. (4) Resampling stays inside the folds, and Platt recalibration is fitted on training OOF scores. *Framework: quantify.*

---

## R. RUAN Hangjing — slides 1–7 (title, crisis data, repayment codes, protocol, segments, KNN benchmark, logit coding) — 15

**R1. Why this dataset, and is a 2005 Taiwanese sample still a reasonable basis?**

UCI *Default of Credit Card Clients* (doi:10.24432/C55S3H): one Taiwanese bank's cardholders, Apr–Sep 2005, observed as the card-debt crisis began. The central bank dates the first repayment problems to Sep 2005 (CBC FSR 2008, Box 2), and revolving balances peaked at NT\$495.3 bn in Nov 2005 (FSC). So the 22.1 % rate is consistent with a stressed book, and the PDs belong to that vintage. What should travel is the ranking logic (recent arrears predict default); levels must be re-based. *Framework: acknowledge · reason · leave room.*

**R2. What does the EDA say predicts default, and was it done on the training rows only?**

Every group rate, correlation and univariate AUC uses the 20,975 training rows; only the data-quality checks and the target count use all 29,965 (`results/eda_summary.json`, `analysis_partition`). Arrears dominate: the training default rate is 12.9 % at PAY_1 = 0, 33.9 % at code 1 ("one month late", a mixture, see below) and 67.6 % two months late; N_MONTHS_DELAYED alone has univariate AUC 0.7235 (0 months 11.8 %, 6 months 69.3 %); the limit gradient runs 32.1 % in the lowest quintile to 13.7 % in the highest; utilisation is U-shaped — 24.2 % in the lowest decile, a 14.4 % trough at 0.04–0.13, 28.1 % at the top. The six bills correlate at 0.80–0.95 (19 feature pairs above 0.8), which matters for the linear models' coefficients, not for prediction. Code 1 is a mixture: 2,540 clients in September but 21 in the five earlier months, almost half (46 %) with no positive September bill. Sex and age move the rate by at most 8 points. Bill size alone ranks at chance (univariate AUC 0.50–0.52) (`pay_code_default_rates.csv`, `eda_summary.json`). *Framework: quantify.*

**R3. Why K = 101 for KNN? That is far from the K = 5 used in class.**

The grid was K ∈ {5, 7, …, 101, 151} × {uniform, distance}, 24 settings on five stratified folds scored by AUC; CV AUC rises from 0.7061 at K = 5 to 0.7700 ± 0.0087 at K = 101 and flattens (0.7688 at K = 151), so the optimum is interior (`results/knn.json`, `results/knn_k5.json`). Default is a noisy 22 % outcome with overlapping classes, so the local default proportion needs about a hundred neighbours to be a usable probability: K = 5 has train AUC 0.883 against test 0.709 — a memorising vote — while K = 101 has train 0.787, OOF 0.770 and test 0.771. Nested CV with the inner grid re-run in each outer fold chooses K = 101 in three folds and 151 in two, and gives 0.7684 ± 0.0073 (`results/nested_cv_knn_repro.json`), so the message is "a large K", not the number 101. *Framework: quantify.*

**R4. Why is KNN the benchmark, and does it really beat logistic regression?**

KNN (CV 0.770) is ahead of the plain logit (0.760) mainly because the logit reads each status code as one evenly spaced number. Coded by meaning (months late plus flags for −2 and −1), the logit reaches 0.768: +0.0082 [+0.0021, +0.0143], 5/5 folds, level with KNN (−0.0016 [−0.0080, +0.0048]). KNN gains nothing from recoding, because a neighbourhood of discrete codes already reads each code separately (training CV; `pay_coding_cv.csv`). *Framework: quantify.*

**R5. What did your nested-CV second pass show, and why did it not change anything?**

Hyper-parameters were re-chosen inside each of the five outer folds, so the outer AUC carries no selection optimism: KNN 0.7684 ± 0.0073 against the plain 5-fold 0.7700, and the logistic family 0.7602 / 0.7601 / 0.7602 / 0.7608 / 0.7608 for plain / ridge / lasso / weighted / weighted + Platt with standard errors of about 0.005 (`results/nested_cv_summary_repro.json`, `results/logistic_nested_cv_v2.csv`). The rules were fixed in advance: the one-standard-error rule keeps the plain logit, and class weighting (with Platt recalibration) is adopted only if its cross-fitted 5:1 cost falls by more than 1 % — instead it rose by 0.43 % (12,247 against 12,195), so it was not adopted. Nested CV was run for KNN and the logistic family only; it scores almost the same, and its pre-registered rule re-selects their frozen settings (outer-fold choices vary). The ensembles were tuned by plain five-fold CV, so their 0.001 internal gaps cannot be separated from selection optimism (a stated limitation). *Framework: quantify.*

**R6. Your 30 Sep proposal said the logistic cost was 7,065 at 0.5 and 5,141 at a 0.17 cut-off; the report says 7,063 and 5,126 at 0.15. Which is right?**

Both are the same plain logit refitted twice: my 30 Sep run and the 16 Sep frozen scoring differ by solver tolerance only (test AUC 0.763679 vs 0.763634; at 0.5, 310 vs 308 false positives, cost 7,065 vs 7,063; `results/logistic_plain_v2.json`, `results/logistic.json`). The out-of-fold 5:1 cost curve is flat between 0.15 and 0.17, so each run picked a different point on the plateau; on the test rows the two cut-offs cost 5,141 and 5,126, a difference of 0.3 % (same two files; the nine-ratio view of my refit is `results/logistic_cost_sensitivity_v2.csv`). The report uses one coherent set, the 16 Sep scoring of every model, and footnotes the proposal-stage figures (threshold saving 1,924 [1,634, 2,193], model saving 160 [−8, 331] at 5:1 in `results/threshold_vs_model_saving_v2.csv`, against 1,937 [1,604, 2,218] and 145 [−18, 299] in the frozen table). *Framework: acknowledge · reason · leave room.*

**R7. Ridge and lasso give the same AUC as the plain logit. Why does regularisation buy nothing?**

CV AUC is 0.7602 for all three and test AUC 0.7636; the tuned L2 C = 31.6 shrinks almost nothing and the L1 optimum at C = 100 keeps all 32 columns (`results/logistic_family_summary.csv`). With n/p = 20,975/32 ≈ 655 the maximum-likelihood variance is already tiny, so there is no variance to trade. The 0.76 level is mostly coding, not linearity: a still-additive logit with codes by meaning reaches 0.768, and with one dummy per code 0.769 (L2 fit; `pay_coding_cv.csv`). Nested CV says the same — plain, ridge and lasso tie within one standard error (0.7602 / 0.7601 / 0.7602, `results/logistic_nested_cv_v2.csv`) — which is why the pre-registered 1-SE rule keeps the plain model, whose p-values are exact. *Framework: acknowledge · reason · leave room.*

**R8. PAY_2–PAY_6 have negative coefficients although late payment raises default, and your largest coefficient is EDUCATION = others. Paradoxes?**

Neither. Given the same number and severity of delays — absorbed by N_MONTHS_DELAYED (+0.634 per SD, odds ratio 1.88) and MAX_DELAY (+0.366) — a delay in an older month is less dangerous than the same delay now, so the raw statuses encode recency; the same logic puts UTIL_1 at −0.259 next to UTIL_MEAN at +0.288 (`results/logistic_coefficients.csv`). EDUCATION = others (−1.070, odds ratio 0.34, p = 2 × 10⁻⁶) is the recoded undocumented codes 0/5/6 — 343 training rows at a 6.7 % default rate (`results/eda_summary.json`) — a quirk of this dataset, not an education effect; the forest ranks EDUCATION 27th of 29 by MDI and 24th by permutation (18th by out-of-bag permutation; `results/rf_importance.csv`). RUAN's 30 Sep refit reproduces every sign and the ordering of the coefficients quoted here, each within 0.02 of the 16 Sep value (`results/logistic_odds_ratios_v2.csv` vs `results/logistic_coefficients.csv`: EDUCATION_4 −1.061 vs −1.070, N_MONTHS_DELAYED 0.633 vs 0.634, MAX_DELAY 0.367 vs 0.366, UTIL_MEAN 0.274 vs 0.288, UTIL_1 −0.242 vs −0.259). UTIL_1 and UTIL_MEAN correlate at 0.91, so their opposite signs must be read together. Moving both by one SD changes the log-odds by about +0.03. In the recoded logit, PAY_2–PAY_6 run −0.42 to −0.11 per month late given the counters, with all five folds agreeing in sign (`pay_coding_logit_coefficients.csv`). *Framework: acknowledge · reason · leave room.*

**R9. What did class weighting do to the logistic model, and why did you not adopt it?**

Almost exactly an intercept shift: `class_weight='balanced'` moves the intercept from −1.305 to −0.063 and changes no slope by more than 0.075 (`results/logistic_balanced.json`, `extra`); a monotone shift cannot change the ranking, so test AUC is 0.7630 against 0.7636 and the price is calibration, Brier 0.138 → 0.190. Platt scaling repairs the scale (Brier 0.1389, cut-off back to 0.15; `results/recalibration.json`) but the nested-CV rule asked for a cost gain above 1 % and the cross-fitted 5:1 cost rose by 0.43 % (12,247 against 12,195) (`results/nested_cv_summary_repro.json`), so the plain model plus an explicit business threshold is the recommendation. The balanced model at 0.5 agrees with the plain model at 0.22 on 99.4 % of test clients (`results/imbalance_threshold_equivalence.csv`). *Framework: quantify.*

**R10. Why 10 principal components, and why does a logistic regression on PCs do worse?**

Ten PCs of the 32 preprocessed columns carry 81.8 % of the variance (two PCs 48.1 %, fifteen 92.2 %; `results/unsupervised_summary_repro.json`, `pca_cum_var`), enough to make K-means distances meaningful without keeping noise components. As an input to a logit they hurt: in the 5 × 5 repeated CV on training rows a logistic on the 10 PCs loses 0.034 AUC against the full logit (−0.034 [−0.040, −0.029]) and 0.016 AP, because the predictive signal — the recency contrast of arrears — sits partly in low-variance directions that PCA discards. An unpenalised logit is rotation-invariant, so using all components equals using the raw features and dropping any can only remove signal; with n/p ≈ 655 there is no variance to trade. PCA is a map for the clustering, not a model input. *Framework: acknowledge · reason · leave room.*

**R11. Why K = 5 clusters with a silhouette of only 0.27, and how stable are they?**

K = 5 has the highest silhouette on the grid 2–8: 0.259 / 0.267 / 0.257 / **0.269** / 0.209 / 0.215 / 0.218 (`results/kmeans_selection_repro.csv`); 0.27 is weak-to-moderate and says the segments are adjacent regions of one continuous cloud, which we state. Stability over 100 bootstrap refits is high: ARI 0.957 [0.745, 0.990], per-cluster Jaccard 0.972 / 0.974 / 0.791 / 0.970 / 0.953 — the small affluent cluster (886 clients, 4.2 %) is the least reproducible — while Ward linkage agrees only moderately, ARI 0.45–0.59, mean 0.523 (`results/unsupervised_summary_repro.json`). Across random seeds the ARI falls to a mean of 0.814 and a minimum of 0.577, so the partition depends on initialisation more than the bootstrap alone suggests; that is the honest caveat. *Framework: quantify.*

**R12. Do the clusters add anything to prediction?**

No, and we tested it the way the professor asked: the one-hot cluster label added to the logit changes the out-of-sample AUC by +0.0001 [−0.0004, +0.0007] and the AP by −0.0003 [−0.0016, +0.0010] in 5 × 5 repeated CV on the training set with a Nadeau–Bengio interval (`results/unsupervised_summary_repro.json`, `results/cluster_incremental_folds_repro.csv`). The adoption rule fixed in advance required both lower bounds above zero, so the label is not used and the test set was not consulted for it. The round-1 cluster-augmented logit (`logistic_plus_clusters`) stays in the appendix as "round-1 scoring, not adopted"; we do not quote its test AUC. The label as a score on its own reaches AUC 0.648 against 0.724 for months-in-arrears alone — it is a coarse function of features the model already has. *Framework: quantify.*

**R13. Your clusters have default rates from 6.9 % to 58.5 %. Isn't that economically meaningful?**

The rates differ (16.8 / 58.5 / 6.9 / 17.9 / 18.2 %; `results/kmeans_profiles_repro.csv`) because the inputs contain the arrears and limit variables: the 58.5 % cluster averages 4.3 months delayed against 0.15–0.53 elsewhere, so the clustering has rediscovered the arrears gradient, not found a new one. A segmentation is economically distinct only if it carries information the score does not, and R12 shows it does not. What the five profiles do give is a map for policy — transactors (38 %, utilisation 4 %), mass-market revolvers (36 %, utilisation 57 %, median-sized limits), affluent heavy revolvers (9 %, limit 286k, utilisation 75 %), a small affluent low-risk group (4 %) and the delinquent 12.5 % — useful for collections priority and limit policy, descriptive only. We say "segments are a map for policy, not a better score". *Framework: acknowledge · reason · leave room.*

**R14. Your proposal used K = 4 clusters on raw features. Why did the configuration change?**

The proposal's local run clustered 26 standardised raw features with the elbow rule and found K = 4; the professor asked for stability and incremental value, so the 30 Sep configuration moved to PCA on the 32 preprocessed columns (10 PCs), chose K by silhouette and added bootstrap ARI, per-cluster Jaccard, Ward agreement and the repeated-CV incremental test (`results/unsupervised_summary_v2.json`, reproduced 21 Oct as `*_repro`). The conclusions did not change — a delinquent cluster at about 60 % default appears at every K ≥ 4 in both runs, and the cluster label adds nothing to prediction in either — so the K = 4 run survives as a one-sentence robustness note. The figures are the K = 5 ones. *Framework: admit · give direction.*

**R15. Why recode the status codes at all?**

The status codes are states: −2 no consumption, −1 paid in full, 0 revolving, k = k months late. September default is 13.7 / 16.9 / 12.9 % for −2 / −1 / 0, so no straight order exists. Months late plus two flags lifts the logit +0.0082 [+0.0021, +0.0143] CV AUC, 5/5, never test-scored (`pay_coding_cv.csv`). *Framework: quantify.*

---

## L. LU Xinyu — slides 8–13 (tree family and ensembles, depth and history ablations, SHAP, monotone constraints, personal attributes and fairness) — 17

**L1. Your unpruned tree has AUC 0.61 and accuracy below the majority class. How can a model that fits the training data perfectly be that bad?**

The tree has 2,791 leaves at depth 55 and train AUC 1.000 — it memorised 20,975 noisy labels — so test AUC is 0.6125, AP 0.290 barely above the 0.221 prevalence, and accuracy 0.728 below the 0.779 majority rate (`results/tree_full.json`, `results/model_comparison.csv`). Every leaf is pure, so it emits only 0 or 1 and the ROC curve has a single operating point. That is variance in the bias–variance sense: each tiny leaf is fitted to a handful of clients whose outcome is largely noise. Pruning to 17 leaves lifts CV AUC from 0.6154 to 0.7689 and test AUC to 0.7755, and the instability of one tree is the motivation for bagging and boosting. *Framework: acknowledge · reason · leave room.*

**L2. How did you choose the pruning parameter, and how sensitive is the result?**

Cost-complexity pruning gives 1,283 candidate alphas; 21 of them (log-spaced, 2,791 → 2 leaves) crossed with max_depth ∈ {None, 4, 6, 8} make 84 settings on five folds (`results/tree_cv_grid.csv`). The winner, alpha = 6.63 × 10⁻⁴ with depth cap 6, gives 17 leaves at depth 5 and CV AUC 0.7689 ± 0.0076; the cap is non-binding (the tree stops at depth 5). The plateau is broad — neighbouring alphas score 0.767–0.769 and a depth-4 cap alone reaches 0.763 — so only the top structure should be quoted: MAX_DELAY carries 59.8 % of the impurity reduction and PAY_1 17.3 % (`results/tree_pruned.json`, `feature_importances_top`). On CV the pruned tree is 6th of 11, tying the KNN benchmark (0.7689 vs 0.7700); on test it is ahead (0.7755 vs 0.7706) — we say "ties the benchmark on CV". *Framework: quantify.*

**L3. What does the pruned tree actually say, and why does it split on a near-zero May bill?**

The root asks whether the client was ever two or more months late (MAX_DELAY ≤ 1.5): 15,095 clients never were and default at 12.8 %, the other 5,880 at 46.1 % (`results/tree_rules.txt`, `results/tree_pruned.json` `first_splits`). On the safe side the counter-intuitive split BILL_AMT5 ≤ 789.5 isolates 3,660 low-activity accounts that default more (21.0 % vs 10.2 % elsewhere), and within them UTIL_MEAN ≤ 0 — 686 dormant or credit-balance accounts — default at 35.7 %: no billing history means no repayment signal, a known risk pocket in this dataset (`results/eda_data_quality.csv`: the non-positive-bill group defaults at 37.0 % against 21.7 % in training). The tree emits only 17 distinct probabilities, so it ranks groups, not individuals. *Framework: acknowledge · reason · leave room.*

**L4. Why min_samples_leaf = 20 in a random forest? Averaging is supposed to remove the variance of fully grown trees.**

Averaging removes variance but not the label noise each fully grown tree memorises: in the 3 × 3 grid CV AUC is 0.7832–0.7843 at leaf size 20, 0.7744–0.7796 at 5, and only 0.7635–0.7666 at 1, where every tree has train AUC 1.000 (`results/random_forest.json`, `extra.grid`). With sklearn's default leaf of 1 the forest would barely beat the logit (0.7666 vs 0.7602). Within each leaf size the smallest feature draw (sqrt = 5 of 32 columns) wins; the winning setting, refitted with 500 trees, goes from CV 0.7843 to 0.7849 and is ahead of the runner-up in all five folds, and the out-of-bag AUC 0.7849 equals the CV figure, the free check we used. Both chosen values sit at a grid corner, so leaf 50 or 100 were not tried — the honest gap. *Framework: admit · give direction.*

**L5. Random forest versus bagging: is a 0.004 AUC difference worth mentioning?**

The forest (max_features = sqrt) has CV 0.7849, test 0.7866, AP 0.5756; bagging with the same leaf size and all 32 columns has CV 0.7813, test 0.7833, AP 0.5712 (`results/bagging.json`, `results/random_forest.json`). The gap is inside one fold standard deviation, but it is consistent — the out-of-bag curve shows the forest ahead at every tree count (0.7849 vs 0.7816 at 500) — and bagging overfits more (train AUC 0.912 vs 0.888) and fits in 89.4 s against 19.7 s (`fit_seconds`). The mechanism is the lecture's variance formula ρσ² + (1 − ρ)σ²/B: bagged trees all split first on the arrears variables and are highly correlated, so adding trees stops helping; drawing five random columns per split lowers ρ. We did not bootstrap this pair on the test set, so the wording is "consistently, modestly better, and cheaper". *Framework: quantify.*

**L6. XGBoost, gradient boosting and the random forest are within 0.002 of each other. Are they distinguishable?**

No. Training-fold intervals: XGBoost − GBM +0.0013 [−0.0005, +0.0031], − forest +0.0014 [−0.0027, +0.0055] (`fold_stability.csv`). XGBoost is the reference only by the preset highest-CV rule; a bank may pick any of the three on cost, speed or explainability. *Framework: quantify.*

**L7. Why is your most flexible boosting setting the worst, and why do all ensembles stop near 0.79?**

In the 24-setting grid the worst is depth 6, learning rate 0.1, 600 rounds — CV AUC 0.7547 with train AUC 0.998 — and the best is depth 3, rate 0.03, 300 rounds with train AUC only 0.808; the learning curve at rate 0.03 peaks at 327 rounds (0.7864) and falls to 0.7794 by 1,500 rounds while training AUC keeps climbing to 0.867 (`results/xgboost.json`, `extra.grid`, `extra.learning_curve`; `results/xgb_learning_curve.csv`). Shrinkage and shallow trees are the regulariser this noisy target rewards. Four independent families reach the same ceiling — boosting 0.786, forest 0.785, MLP 0.776, KNN 0.770 on training CV (test 0.789 / 0.787 / 0.773 / 0.771 as a check) — and pushing capacity in any of them lowers CV AUC, which looks like the irreducible noise of predicting next month's default from six months of statements without income or bureau data; we cannot separate "irreducible" from "missing features" with this dataset. *Framework: acknowledge · reason · leave room.*

**L8. Class-weighted XGBoost has the same AUC but a much worse Brier score. Why, and what did Platt scaling do?**

With scale_pos_weight = 3.52 the test AUC is 0.7887 in both versions and AP 0.5788 vs 0.5784, because re-weighting rescales the positive-class gradient and shifts scores almost monotonically; what changes is the scale — Brier 0.1311 → 0.1772, log-loss 0.420 → 0.536, recall at 0.5 from 0.363 to 0.649 with precision 0.702 → 0.461 (`results/xgboost_weighted.json`, `results/xgboost.json`). LI's Platt step, fitted on the weighted model's training OOF scores, returns Brier to 0.1311 and the 5:1 cut-off from 0.37 to 0.14, with AUC unchanged at 0.7887 (`results/recalibration.json`, `xgboost_weighted_platt`). So weighting buys nothing that the unweighted model at its own threshold does not already give, and it breaks the probabilities until recalibrated. *Framework: quantify.*

**L9. What does the model rely on? Your MDI, permutation and SHAP rankings disagree.**

They agree at the top: PAY_1 is first under all three (mean |SHAP| 0.298 on the log-odds scale, MDI 0.188, permutation AUC drop 0.042), followed by MAX_DELAY (0.281) and N_MONTHS_DELAYED (0.230) under SHAP and MDI (`results/shap_importance.csv`, `results/rf_importance.csv`). They disagree in the middle — Spearman 0.43 between SHAP and MDI, 0.76–0.77 between SHAP and permutation (training-set / out-of-bag), a cross-model comparison, since SHAP comes from XGBoost and MDI and permutation from the forest — because MDI credits every split, so the correlated arrears columns collect Gini between them, while permuting one costs little when its siblings substitute; the amount columns are the mirror image. The drop-column refits make the point sharpest: of the six columns refitted, only September status costs over 0.001 OOB AUC (PAY_1 0.0058; `results/rf_refit_drop_column.csv`). Demographics (SEX, EDUCATION, MARRIAGE) sit in the bottom third under MDI (ranks 28 / 27 / 29) and training permutation (25 / 24 / 26) and never enter the top ten under SHAP (15 / 16 / 12) or out-of-bag permutation (11 / 18 / 14) (`results/rf_importance.csv`, `results/shap_importance.csv`). *Framework: acknowledge · reason · leave room.*

**L10. What is TreeSHAP, and why did you run it on the training rows?**

A feature's Shapley value is its average marginal contribution to the prediction over all orders in which features could be added, so the contributions sum exactly to the difference between this client's log-odds and the base value — TreeSHAP computes it exactly for tree ensembles, and we verified local accuracy to 3 × 10⁻⁶ on all 20,975 training rows (`results/shap_summary.json`). It is a property of the fitted model, not of a data set, so explaining the training rows costs nothing and reads no test label (until 19 Oct the script explained the test rows' features, without labels; dated in `results/TEST_SET_LOG.md`). The worked example: training client 10,000 has predicted probability 0.861; from the base value −1.263 (probability 0.220), PAY_1 = 2 contributes +0.86 log-odds, N_MONTHS_DELAYED = 6 +0.56, MAX_DELAY = 7 +0.55 and LIMIT_BAL = 10,000 +0.22. Limits: contributions are additive on the log-odds scale only, and credit between the six bills (r = 0.80–0.95) depends on the trees actually grown. *Framework: quantify.*

**L11. What did the monotone-constrained XGBoost cost, and why build it?**

The frozen constraint also forced the score up from −2 to −1 to 0 and with utilisation; the data contradict both (16.9 % vs 12.9 %; a U-shape). It costs 0.0019 CV AUC [−0.0025, −0.0013], 0/5 folds (test −0.0016 as a check). Constraining only months late (up) and the limit (down) is within 0.0005 of the free booster (training CV, never test-scored; `monotone_respec_cv.json`). That respecified version is the promise we would make, valid from revolving credit upwards. *Framework: quantify.*

**L12. What does partial dependence add, and what did it show about utilisation?**

PD averages the model's own prediction over all 20,975 training clients while one feature is swept, so it reads the shape the ensemble learned without touching a label (`results/pdp_summary.json`). Arrears are step-like: PAY_1 moves the probability from 0.21 at one month late to 0.372 at two and is flat beyond; N_MONTHS_DELAYED is monotone 0.190 → 0.321 with 97.6 % of individual ICE curves monotone. Utilisation is U-shaped — 0.228 at zero, a minimum of 0.204 at 0.084, 0.244 at the top — and the limit effect is 0.257 at 10,000 falling to 0.201 at 360,000. The logit's gap is first coding (a third of it, already inside the 86 % that follows), then one-variable shape: a depth-1 booster recovers 86 % of the XGBoost–logit gap and interactions add 0.004 (training CV, `ablation_depth.csv`). The U in utilisation is one place where that shape shows; we did not measure a share per variable. *Framework: acknowledge · reason · leave room.*

**L13. What is your recommendation to the bank, in one sentence each for model, threshold and segments?**

Model: any of the three tied ensembles (XGBoost by the preset highest-CV rule), with the respecified arrears-only monotone constraint if a lender must promise direction. Cut-off: from the bank's own cost ratio, about 1/(1 + r), re-estimated with recalibration; 5:1 is only a scenario. Segments: a map for collections and limit policy, not a model input (+0.0001 AUC). Attributes: dropping them is a documented trade-off, not a fix (S13). *Framework: quantify.*

**L14. What are the limitations, and what would you do next?**

(i) Retrospective test use, dated in `TEST_SET_LOG.md`. (ii) Borrowed cost ratio. (iii) Crisis vintage: US card delinquency ranged from 1.53 % (2021Q3) to 6.77 % (2009Q2) over 1991–2026 (Fed; a US analogy), so reuse needs PDs re-based and out-of-time validation. (iv) One split, single seeds, ensembles not nested. (v) Fairness numbers are a training-OOF diagnostic, not a compliance assessment. Next: an out-of-time hold-out, bureau and income data, measured losses for the cost ratio, a binned additive scorecard and nested multi-seed ensembles (report §7). *Framework: admit · give direction.*

**L15. The plotted K-sweep shows a bootstrap ARI of about 0.94 at K = 5, but you quote 0.957. Which is it?**

Both are correct: the sweep in the figure re-runs a smaller number of bootstrap refits at every K from 2 to 8 and reads 0.941 (least-stable cluster Jaccard 0.768) at K = 5 (`results/kmeans_selection_repro.csv`, row K = 5), while the dedicated K = 5 run with 100 refits gives 0.957 [0.745, 0.990] and a least-stable Jaccard of 0.791 (`results/unsupervised_summary_repro.json`). The two draws differ by 0.016 in mean ARI, inside the 100-refit interval, and the ordering of K is unchanged — K = 6 is the unstable one (0.79, Jaccard 0.67). Across random initialisations the ARI falls to a mean of 0.814, which is the honest caveat we state. *Framework: quantify.*

**L16. How do you know the gain is mostly shape, and how much do older months add?**

Depth-1 XGBoost, rounds re-tuned, minus logit is +0.0225 [+0.0111, +0.0338]; depth 3 minus depth 1 is +0.0036 [+0.0022, +0.0050], an upper estimate since depth 3 was chosen on these folds (`ablation_depth.csv`). History: H0 0.630 → H1 0.769; older months add +0.013 as raw months, or +0.017 via six-month summaries (`ablation_history.csv`). *Framework: quantify.*

**L17. Is XGBoost unfair to men?**

It flags men 1.18 times as often as women, against a default-rate ratio of 1.19, and its PDs match every group of ≥ 1,000 clients within 0.5 points. So men who would repay are flagged more often too (FPR 45.7 % vs 38.2 %). Removing the four columns narrows the ratio to 1.10, but under-predicts men by 1.1 points and raises the under-30 ratio from 1.11 to 1.17 (`fairness_groups.csv`, `fairness_summary.json`; training OOF diagnostic). *Framework: acknowledge · reason · leave room.*

---

## J. LI Jingwen — slides 14–21 (fold stability and ranking, four more families, imbalance and Platt, leakage, cut-off, cost ratio, one-line rule, limits) — 17

**J1. Why is the RBF-kernel SVM below the KNN benchmark and no better than the logit?**

Its fair training number is the mean of its five shared-fold OOF AUCs, 0.7635 (pooled 0.7606; the subsample CV figure 0.7681 is not comparable; `results/svm_selection_statistics.json`), against KNN 0.7700 and logistic 0.7602: 7th of 11 under every statistic, and +0.0033 [−0.0032, +0.0098] over the logit, a tie; on test 0.7616 vs 0.7636 is a tie, but AP 0.4944 vs 0.5234 and Brier 0.1448 vs 0.1383 are not, and against XGBoost the SVM's AP gap of 0.083 [0.066, 0.100] is the largest of any headline model (`results/svm_rbf.json`, `results/auc_bootstrap_ci.csv`). The mechanism is the loss: hinge loss cares only about the sign of the margin, and on this overlapping problem 13,140 training rows (62.6 %) are support vectors, so the decision values are poorly spread; Platt scaling (a = −0.974, b = 1.109) is monotone and cannot repair the ordering. Two of my choices also hurt: tuning on a 6,000-row subsample and no log transform of the heavy-tailed amounts, to which an RBF distance is most sensitive. *Framework: acknowledge · reason · leave room.*

**J2. The linear-kernel SVM scores far below every other linear model. Why?**

At the tuned C = 0.01 the exact hinge optimum is a single-feature ranker: the weight sits almost entirely on MAX_DELAY and the out-of-fold AUC is 0.7167, essentially MAX_DELAY alone; whatever the test AUC 0.7506 shows above that comes from solver residue and is not a property of the model (`results/svm_linear.json`, `notes`). Replacing hinge by squared hinge in `LinearSVC` at the same C spreads the weights over the features the logit uses and recovers logistic-level ranking: CV 0.7611, test 0.7627, AP 0.5150 (`results/svm_linear_calibrated.json`). The lesson: a linear SVM and a logistic regression share the hypothesis class w·x + b and differ only in the loss, and on a noisy 22 % target the hinge loss is the wrong one. *Framework: quantify.*

**J3. Class weighting changed nothing for the logit or XGBoost yet was decisive for your SVMs. Why?**

Logistic regression and XGBoost estimate posteriors, so re-weighting is a prior shift — for the logit an intercept move from −1.305 to −0.063 — that leaves the ranking untouched (`results/logistic_balanced.json`). The hinge loss does not estimate posteriors: with a 22 % minority the unweighted margin is placed where the majority dictates, most defaulters become bounded support vectors on the wrong side, and the decision function stops discriminating among them. Balanced weights lift the best subsample CV AUC of the RBF kernel from 0.7311 (no weighting) to 0.7681 (`results/svm_rbf.json`, `extra.best_cv_auc_by_class_weight`). Those are 6,000-row figures with fold SDs of 0.005–0.024, so the size is approximate; the direction is not. *Framework: acknowledge · reason · leave room.*

**J4. Why does your over-parameterised MLP collapse, and what does the tuned one add?**

The 32-256-256-1 network has 74,497 weights for 20,975 rows (3.6 per row), no early stopping, and reaches train AUC 0.997 with CV 0.692 and test 0.678 — worse than naive Bayes. On an internal 85/15 split of the training set the over-parameterised net's hold-out AUC is best at epoch 1 (0.760) and falls to 0.700 by epoch 200 while training AUC reaches 0.994; even the tuned 32-64-32-1 architecture run without early stopping peaks at epoch 14 (0.762) and falls to 0.712 (train 0.891) — `results/mlp_overfit_epochs.csv` (column `network`), `results/mlp_overfit.json` and `results/mlp.json` `extra.epoch_diagnostic_train_internal_split`. The tuned 32-64-32-1 network (4,225 weights, L2 alpha 0.001, early stopping at epoch 23 with best validation epoch 7) gives CV 0.7762 ± 0.0069 and test 0.7731, 4th of 11 on CV and the best non-tree model; the 24-setting grid spans only 0.0047 AUC, so architecture barely matters once capacity is controlled. Against the logit the paired bootstrap in `results/mlp.json` (`extra.paired_bootstrap_test`, B = 2,000) gives +0.0095 [+0.0040, +0.0153]; against XGBoost −0.0155 [−0.0204, −0.0108] — the aggregate B = 500 run in `results/auc_bootstrap_ci.csv` gives the same gap from XGBoost's side, mean +0.0156 [+0.0103, +0.0199], same verdict. *Framework: quantify.*

**J5. LDA, QDA and naive Bayes — why are the generative models at the bottom?**

LDA is the generative twin of the logit — same linear log-odds, parameters from class-conditional Gaussians with a pooled covariance — and lands just below it: CV 0.7572 vs 0.7602, test 0.7592, with coefficient vectors at cosine 0.934 and test scores at Spearman 0.990 to the logit (`results/lda.json`, `extra.logistic_reference`); the shared covariance is estimated on amounts with skewness 10–33, which costs the little it loses. QDA with shrinkage 0.5 reaches 0.7511 / 0.7550 because one full 32 × 32 covariance per class on skewed, dummy-laden columns is badly specified (`results/qda.json`). Gaussian naive Bayes is last (CV 0.7356, test 0.7398, Brier 0.203, worse than a constant) because the six bills correlate at a mean 0.887 and 35.8 % of its test probabilities are extreme; a signed log1p on the 13 amount columns lifts it to 0.7563 on test (`results/naive_bayes.json`, `results/model_comparison.csv`). *Framework: acknowledge · reason · leave room.*

**J6. Does re-balancing help? Which strategy won?**

None of them changes the ranking: for the C = 1 logit the CV AUC across none / balanced / SMOTE-in-folds / under-sampling spans 0.7597–0.7609, a range of 0.0012, and for the forest the four give CV 0.780 / 0.781 / 0.774 / 0.779, so SMOTE costs 0.0055 CV AUC (test −0.0076 as a check), partly because SMOTE after one-hot encoding invents fractional dummies (`imbalance_comparison.csv`). What re-balancing does change is the probability scale: mean predicted probability 0.44 instead of 0.22 and Brier 0.190 instead of 0.138 for the logit. So the strategy adopted is "none, plus a threshold chosen on training OOF for the business cost"; and every sampler ran inside the folds, as the professor asked. On CV the balanced logit is +0.0007 above the plain one — within noise — so we say "never upwards by more than noise", not "always downwards". *Framework: quantify.*

**J7. If weighting only shifts the threshold, show me.**

The balanced logit at 0.5 and the plain logit at 0.22 flag almost the same clients: agreement 99.4 % on the test set (2,794 vs 2,788 clients flagged), with the equivalent threshold 0.224 predicted from the intercept shift log(0.779/0.221) alone (`results/imbalance_threshold_equivalence.csv`). SMOTE and under-sampling give the same 0.22 equivalence at 98.9–99.0 % agreement; for the forest the equivalent plain threshold is 0.29 (balanced) or 0.34 (SMOTE) at 95–98 % agreement, lower because trees are not a pure prior shift. This figure (`s_imbalance.png`) is not on a v2 slide (slide 16 shows the Platt calibration curves), so give these numbers in words. *Framework: quantify.*

**J8. What did the Platt recalibration do, and how did you keep it leakage-free?**

Platt scaling is a one-variable logistic regression of the label on logit(p), fitted on each model's training out-of-fold probabilities and nothing else; thresholds are re-chosen on cross-fitted calibrated scores, and the map is applied to the frozen test probabilities and frozen at that first scoring on 19 Oct (`results/recalibration.json`, `method`). Platt had been pre-committed on 30 Sep, but on 6 Oct the Platt, isotonic and prior-corrected maps were compared on test Brier and cost before the freeze; that read is dated in `TEST_SET_LOG.md`. For the nine weighted / resampled models the Brier returns to the unweighted twin — weighted XGBoost 0.1772 → 0.1311, balanced logit 0.1901 → 0.1389, balanced forest 0.1606 → 0.1335, weighted MLP 0.1855 → 0.1377 — and the 5:1 cut-off returns to 0.14–0.19 (0.37 → 0.14, 0.41 → 0.15, 0.36 → 0.18, 0.42 → 0.19), around the closed form 1/6. AUC is unchanged because the map is monotone. *Framework: quantify.*

**J9. Show the leakage you are warning about.**

We ran the classic mistake on purpose and quarantined it: SMOTE on all 29,965 rows, then a split. The forest then shows "test" AUC 0.914 against the honest in-fold SMOTE 0.774, and even the logit 0.770 against 0.762 (`results/imbalance_leakage_demo.csv`). The inflation is mechanical: 72.6 % of the leaky test defaulters are synthetic points interpolated from training defaulters, so the model is asked to recognise its own neighbours — restricted to real rows the leaky forest's AUC falls to 0.845, and on synthetic-defaulters-vs-real-goods it reaches 0.940. The frozen manifest holds this run as a negative control so that nobody mistakes it for a result. *Framework: quantify.*

**J10. Your CV and test AUCs agree almost exactly for every model. Isn't that suspicious?**

It is what an honest protocol produces: for all eleven headline models the test AUC is within one CV standard deviation of the CV AUC (XGBoost 0.7863 → 0.7887, logistic 0.7602 → 0.7636, KNN 0.7700 → 0.7706, MLP 0.7762 → 0.7731), using the RBF SVM's shared-fold OOF figure (0.7635; pooled 0.7606) rather than its subsample CV (`results/model_comparison.csv`). Suspicious would be the opposite — test far above CV — which is the signature of a leak, exactly what the quarantined SMOTE demonstration shows (0.914 vs 0.774). Training error is never used to compare models; train AUC 1.000 for the unpruned tree and 0.997 for the over-parameterised MLP are quoted only to show the optimism gap. The saved `results/*_p_oof.npy` and `*_p_test.npy` vectors let anyone recompute every number. *Framework: acknowledge · reason · leave room.*

**J11. Why does the ranking slide order models by CV and not by test AUC?**

Because a test set that ranks becomes a validation set. Honestly: until 19 Oct `aggregate.py` ordered by test AUC, and the top four then had the tree in place of the MLP. Since then the order is training CV AUC (`select_auc`), and every model ranked above the logit beats it in 5/5 folds (`fold_stability.csv`). *Framework: quantify.*

**J12. How do you know a difference between two models is real?**

Two scales. The fold-to-fold CV standard deviation (0.007–0.011 AUC) says whether a grid choice matters; the paired bootstrap of the same 8,990 test rows (B = 500, seed 2607526) says whether a test difference exceeds sampling noise (`results/auc_bootstrap_ci.csv`). Against XGBoost (bootstrap means, `auc_diff_mean`): forest +0.0020 [−0.0010, +0.0051] (a tie — XGBoost fails to beat it in 11 % of replicates), GBM +0.0015 [+0.0001, +0.0030] (borderline), then clearly below — pruned tree +0.0132 [+0.0082, +0.0184], MLP +0.0156 [+0.0103, +0.0199], KNN +0.0181 [+0.0125, +0.0237], logistic +0.0249 [+0.0183, +0.0314], SVM +0.0271, LDA +0.0293, naive Bayes +0.0486. Differences inside the linear family (≤ 0.005) are noise. Training-fold paired intervals (Nadeau–Bengio): XGBoost − KNN +0.0163 [+0.0055, +0.0270], − logit +0.0261 [+0.0145, +0.0376], − MLP +0.0101 [+0.0058, +0.0144]; MLP − KNN +0.0062 [−0.0003, +0.0127], suggestive, not decisive. *Framework: quantify.*

**J13. Why is the threshold more important than the model, and how big is each effect?**

In the 5:1 scenario (borrowed from the Statlog German credit cost matrix, not this issuer's cost) the logistic regression at the default 0.5 costs 7,063 cost units on the 8,990 test clients (flag nobody 9,945, flag everybody 7,001); moving its cut-off to the training-OOF optimum 0.15 saves 1,937 [1,604, 2,218], 27 % of the cost, while switching to XGBoost at its own cut-off saves a further 145 [−18, 299] — an interval that includes zero (`results/threshold_vs_model_saving.csv`, paired bootstrap B = 1,000). The mechanism: if a missed defaulter costs five false alarms, the optimal rule flags anyone above probability 1/(1 + r) = 0.167, far below 0.5; at 0.5 every calibrated headline model catches only 32–37 % of defaulters (`recall_1_0_5` in `results/model_comparison.csv`; the exceptions are the over-confident naive Bayes, 58 %, and the RBF SVM, 7 %), at 0.15 the logit catches 74 % and XGBoost 78 % (`results/logistic.json`, `results/xgboost.json`, `at_tuned_cost`). No multiplier is used on the slide — two absolute savings with intervals. *Framework: quantify.*

**J14. Is the 5:1 conclusion robust to the cost ratio? What happens at 20:1?**

The threshold saving grows with the ratio — 238 [141, 338] at 2:1, 684 at 3:1, 1,937 at 5:1, 7,121 at 10:1, 20,335 at 20:1 — while the model saving stays between 87 and 450 and its interval includes zero at 5:1 and 20:1 (`results/threshold_vs_model_saving.csv`). At 20:1 the logistic cut-off drops to 0.01 and 99.9 % of clients are flagged, so the rule has degenerated into "flag everyone" (cost 6,993 against the 7,001 of flagging literally everyone); the threshold "saving" there is the trivial rule, not a model result, and we say so. The searched cut-offs track the closed form 1/(1 + r) at every ratio (0.33 vs 0.333 at 2:1, 0.10 vs 0.091 at 10:1), which is the sanity check that the OOF search is doing what theory says. *Framework: quantify.*

**J15. Why did you build the freeze machinery yourself instead of simply not touching the test set?**

Because by 18 Sep the test set had already scored 34 configurations across eight model families, and the professor's comment was precisely that further decisions would turn it into a validation set. A rule in people's heads is not auditable; a manifest is: `code/freeze.py` records each configuration's hash and its test-probability hash at first scoring, `common.evaluate()` refuses a changed configuration under a frozen slug before any label is read, and `--check` re-verifies all 46 entries (`results/frozen_manifest.json`). The two decisions made after the comment — adopting RUAN's nested-CV choices and the Platt variants — were taken on training rows and then frozen at their first test scoring as new registered configurations; the dated log covers the reads before the register existed; the drop-column refits and TreeSHAP were moved to out-of-bag and training rows on 19 Oct. *Framework: acknowledge · reason · leave room.*

**J16. Isn't a one-line rule enough?**

On training rows, "any delay in six months" flags 33.6 %, recall 0.645, cost 585.6 per 1,000 at 5:1. XGBoost (cross-fitted) costs 540.3: 7.7 % less, +45 [+25, +66] per 1,000, 5/5 folds. The logit saves only +5 [−4, +14]. The rule was the cheapest of four candidates on these rows, a mild selection effect in its favour (`simple_rule_benchmark.csv`). *Framework: acknowledge · reason · leave room.*

**J17. Which model is cheapest?**

It rotates: GBM at 1:1 (cut-off 0.49), forest at 2:1, XGBoost at 3:1–7:1, forest at 10:1–15:1, KNN at 20:1 (0.03). The top three are never more than 1.03 % apart, and no fold interval excludes zero (`preferred_model_by_ratio_oof.csv`; training OOF, selects nothing). So we name no cost winner. *Framework: quantify.*

---

*Question counts: General 8 · RUAN 15 · LU 17 · LI 17 = 57. Every number above traces to the named file under `results/`; re-check this bank if any `results/` file is regenerated after 2026-10-22.*
