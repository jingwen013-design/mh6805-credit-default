# Slides outline — MH6805 Group B19 final presentation (27 Oct 2026, morning; written 2026-10-21)

Companion to `slide_spec_2026-10-21.md` (full on-slide text, figures, notes EN/中文) and `qa_bank.md`. Canva deck DAHYnjt1H8I, MH6810 template style. PPT due 26 Oct noon on NTU Learn. 10 minutes + Q&A; all three present.
Protocol wording on every slide that needs it: "chosen on training CV; each frozen configuration scored once on the test set; the test set only reports". Ordering is always training CV AUC (OOF AUC for the subsample-tuned RBF SVM). AUC to 3 decimals, costs as integers. Every number below is in a file under `results/` (named per slide).

## Running order (words = Notes EN in the spec; 140 words/min)

| # | Title | Section | Presenter | Words | Time | Cum. |
|---|---|---|---|---|---|---|
| 1 | Credit-card default: eleven classifiers, one honest protocol | Data & Problem | RUAN | 51 | 0:22 | 0:22 |
| 2 | Data: 29,965 clients, 22.1% default next month | Data & Problem | RUAN | 77 | 0:33 | 0:55 |
| 3 | EDA: repayment behaviour is the signal, demographics are noise | Data & Problem | RUAN | 84 | 0:36 | 1:31 |
| 4 | Protocol: chosen on training CV, frozen, scored once | Protocol | RUAN | 108 | 0:46 | 2:17 |
| 5 | Benchmark and linear family: KNN 0.770, logistic 0.760 | Protocol | RUAN | 100 | 0:43 | 3:00 |
| 6 | Tree family: pruning, averaging, boosting reach CV 0.786 | Models | LU | 96 | 0:41 | 3:41 |
| 7 | Interpretation: repayment delay drives the score, constraints cost 0.002 | Models | LU | 100 | 0:43 | 4:24 |
| 8 | SVM, MLP, discriminants: only the network clears the benchmark | Models | LI | 114 | 0:49 | 5:13 |
| 9 | Imbalance: re-balancing moves the threshold, not the ranking | Models | LI | 111 | 0:48 | 6:00 |
| 10 | Leakage control: SMOTE before the split fakes 0.914 AUC | Models | LI | 71 | 0:30 | 6:31 |
| 11 | Ranking by training CV: boosting leads logistic by 0.025 | Results | LI | 79 | 0:34 | 7:05 |
| 12 | Operating point: the cut-off saves 1,937, the model 145 | Results | LI | 108 | 0:46 | 7:51 |
| 13 | Business: a map for policy, a cut-off for decisions | Business & Future | LU | 112 | 0:48 | 8:39 |
| 14 | Limitations and next: one split, one assumed cost ratio | Business & Future | LU | 92 | 0:39 | 9:18 |
| A1 | Appendix: nested CV, Platt reliability, nine-ratio sweep | appendix, Q&A only | RUAN (nested CV) / LI (Platt, sweep) | 77 | — | — |

Totals (re-computed 21 Oct evening after the review fixes): RUAN 1–5 = 420 words / 3:00 · LU 6–7, 13–14 = 400 / 2:51 · LI 8–12 = 483 / 3:27 · spoken 1,303 words = 9:18; buffer to 9:20 = 0:02 — the cut rules below are not optional if any presenter runs long; three hand-overs and slide changes use the 40 s to 10:00.

## Per-slide must-say and cut rule
Format: **must-say** = the sentence(s) that cannot be dropped (numbers from the file named). **Cut** = what to drop first if the running clock is ≥ 10 s behind the cumulative column.

1. **Must-say:** question (who misses next month's payment), eleven classifiers, one protocol, who presents what. The slide carries the labelled contribution line ("Contribution (draft, to confirm): RUAN 29 % · LU 29 % · LI 42 %") and a separate presenting line — not spoken, but keep the word "draft" until RUAN and LU confirm. **Cut:** the per-person listing — say only "three of us, in turn". (aggregate_summary.json, contribution_statement_draft.md)
2. **Must-say:** 29,965 clients, 22.1 % default, majority accuracy 0.779 so accuracy is not the yardstick; one stratified 70/30 split, 20,975 / 8,990, seed 2607526; 5:1 is one point of a range. **Cut:** the list of the 23 raw fields and 6 engineered features. (eda_summary.json, eda_data_quality.csv)
3. **Must-say:** training rows only; two months late 67.6 % vs revolving 12.9 %; the utilisation U (24 / 14 / 28 %); so we expect trees to beat lines. **Cut:** the credit-limit quintile gradient (32.1 % → 13.7 %). (eda_summary.json)
4. **Must-say:** every choice on training rows; resampling and scaling inside folds; rank by 5-fold CV AUC; the arithmetic that closes to 46 — 34 models + 1 leakage control scored once by 18 Sep, 2 re-scored in RUAN's 30 Sep second pass, 9 Platt versions 19 Oct; guard refuses changed configurations; nested CV (hyper-parameters chosen inside each outer fold) after the professor's 30 Sep comment re-selected KNN and logistic, nothing re-scored. **Cut:** the description of how the guard works — keep "46 frozen entries; `freeze.py --check` passes". (frozen_manifest.json, nested_cv_summary_repro.json)
5. **Must-say:** KNN K = 101 uniform is the benchmark, best of 24 settings (K = 5 only 0.706 — spoken, no longer on the slide), CV 0.770 / test 0.771; plain logistic CV 0.760 / test 0.764; lasso and ridge change nothing; nested CV 0.768 and 0.760 confirm both. The table takes the full width (no figure unless the owner renders figures/slides/s_knn_k.png at 1920×960). **Cut:** the K = 5 aside and the 1-SE / −0.4 % cost detail (it lives on A1). (knn.json, knn_k5.json, logistic.json, nested_cv_summary_repro.json)
6. **Must-say:** full tree memorises (2,791 leaves, CV 0.615); pruned 17 leaves ties KNN (CV 0.769, test 0.776); forest CV 0.785 / 0.787; XGBoost CV 0.786 / 0.789 is the CV winner; XGB − forest +0.002 [−0.001, +0.005] = tie. **Cut:** fit times (0.5 s vs 19.7 s) and the XGBoost hyper-parameters. (tree_pruned.json, random_forest.json, xgboost.json, auc_bootstrap_ci.csv)
7. **Must-say:** TreeSHAP on training rows (test not read); PAY_1 0.298, MAX_DELAY 0.281, N_MONTHS_DELAYED 0.230 lead; monotone XGBoost removes 24 wrong-direction steps for 0.002 AUC; these tools explain, never select. **Cut:** the drop-column OOB figure (0.785 → 0.779) and the PDP minimum. (shap_importance.csv, monotone_summary.json)
8. **Must-say:** MLP 32-64-32-1 with early stopping is the only one above KNN (CV 0.776 / test 0.773); RBF SVM OOF 0.761 / 0.762, tuned on a 6,000-row subsample; LDA 0.757, QDA 0.751, NB 0.736 below — Gaussian densities fail on skewed amounts and bills correlated at 0.80–0.95. The twin's numbers are its own: 74,497 weights, hold-out AUC peaks at epoch 1 (0.760) and falls to 0.700 by epoch 200; the "epoch 14 → 0.712" curve is the tuned architecture without stopping (left panel). **Cut:** the over-parameterised twin sentence — point at the figure instead. (mlp.json, mlp_overfit.json, mlp_overfit_epochs.csv, svm_rbf.json, lda.json, naive_bayes.json)
9. **Must-say:** samplers inside folds only; logistic AUC moves by about 0.001 across four strategies (CV 0.760–0.761, test 0.762–0.764; range 0.0012), forest with SMOTE CV 0.780 → 0.774, test 0.782 → 0.774; balanced@0.5 = plain@0.22 on 99.4 % of clients; weighting breaks calibration (Brier 0.138 → 0.190); Platt on training OOF repairs it (weighted XGBoost 0.177 → 0.131; cut-offs back to 0.14–0.19), nine versions frozen 19 Oct. **Cut:** the mean-probability 0.44 vs 0.22 sentence. (imbalance_comparison.csv, recalibration.csv)
10. **Must-say:** SMOTE before the split gives the forest 0.914 "test" AUC against an honest 0.774; 73 % of the leaky test defaulters are synthetic; logistic barely moves (0.770 vs 0.762); quarantined negative control. **Cut:** nothing — this is already the shortest slide; if desperate, drop the logistic comparison. (imbalance_leakage_demo.csv)
11. **Must-say:** ordered by training CV AUC, bars are test AUC, whiskers CV ± 1 sd; KNN K = 101 is the benchmark line; above it: XGBoost, GBM, forest, MLP; pruned tree ties; SVM, logistic, LDA, NB below; XGBoost is the CV winner (test numbers reported beside it, never "top on both"); XGB − logistic +0.025 [+0.018, +0.031]; XGB − forest +0.002 includes zero, so the top three are one group. **Cut:** the XGB − KNN +0.018 and XGB − GBM figures. (aggregate_summary.json, auc_bootstrap_ci.csv)
12. **Must-say:** at 5:1 logistic at 0.5 costs 7,063; cut-off 0.15 (training OOF, near 1/(1+5)) saves 1,937 [1,604, 2,218]; switching to XGBoost saves 145 [−18, 299], interval includes zero; 5:1 is one point of 1:1–20:1; at 20:1 the cut-off 0.01 flags almost everyone; fix the cut-off first, the model second. **Cut:** the 3:1 and 10:1 rows — speak only 1:1, 5:1, 20:1. (threshold_vs_model_saving.csv)
13. **Must-say:** recommendation first — rank with XGBoost, cut-off t ≈ 1/(1+r) from the bank's own ratio; at 5:1 t = 0.15 flags 49 % and takes the test cost 6,642 → 4,981; re-estimate r and recalibrate quarterly; explain with logistic or the 17-rule tree. Then "clustering is secondary" (the slide block is labelled so): PCA 10 PCs = 81.8 %; K = 5, silhouette 0.269, bootstrap ARI 0.957 (0.941 in the plotted K-sweep — both on the slide); as an extra predictor the label changes AUC by one ten-thousandth (ΔAUC +0.1 [−0.4, +0.7] ×10⁻³, ΔAP −0.3 [−1.6, +1.0] ×10⁻³) with intervals across zero, so by the pre-fixed rule it is not used — "a map for policy, not a better model" (never "economically meaningful"). **Cut:** the 0.941 sweep aside and "re-estimate quarterly". (threshold_vs_model_saving.csv, aggregate_summary.json, unsupervised_summary_repro.json, kmeans_selection_repro.csv)
14. **Must-say:** one split, one seed — the top-three gaps (≤ 0.002) sit inside or at the edge of bootstrap noise (forest gap includes zero; GBM gap only just excludes it); 5:1 assumed, anchors imply 9.6–40, hence the sweep, which stops at 20:1 because the rule is already "flag everyone" there (cut-off 0.01, 99.9 % flagged); Taiwan 2005, one-month horizon; next: repeated splits, nested CV for every family, LGD pricing, drift monitoring; thank you, questions. **Cut:** the 20:1 aside and the per-client-interval item. (cost_anchors.json, threshold_vs_model_saving.csv, auc_bootstrap_ci.csv)
A1. Not spoken. Shown only when asked about the professor's comment (nested CV table: KNN 0.768 ± 0.007, logistic family 0.760–0.761, 1-SE rule, −0.4 % < 1 %) or calibration (Platt on training OOF, applied once on 19 Oct). (nested_cv_summary_repro.json, logistic_nested_cv_v2.csv, recalibration.csv)

## Hand-overs (spoken by the outgoing presenter; next presenter already standing at the laptop)
- After slide 5 (RUAN → LU): "LU will now show what trees add on top of this linear ceiling."
- After slide 7 (LU → LI): "LI takes over with the models that did not beat the benchmark — and why."
- After slide 12 (LI → LU): "LU closes with what this means for the bank and what we would do next."
- After slide 14 (LU → floor): "Thank you — we welcome your questions." LI opens `qa_bank.md`; A1 is shown by whoever is asked.

## Rehearsal checklist (two full runs on 25 Oct, one on 26 Oct after upload; 12 lines)
1. Time each presenter against the totals above (RUAN 3:00, LU 2:51, LI 3:27); stop at 9:20 of speech — if over, apply the cut rules in slide order from the back of each block.
2. Every slide: say the mechanism (why), then the number; never read the body text aloud.
3. Say "chosen on training CV; scored once on the frozen test set" at least on slides 4, 11 and 12; never say "touched once" or rank by test AUC.
4. Place every model relative to KNN 0.770 / 0.771 (slides 5, 6, 8, 11).
5. Costs: say the two absolute savings with intervals (1,937 [1,604, 2,218]; 145 [−18, 299]); never say "N times".
6. Clustering: "a map for policy, not a better model"; never "economically meaningful" or "actionable segment".
7. Numbers on slides match `results/` files: re-check 0.786 / 0.789, 0.770 / 0.771, 0.760 / 0.764, 0.914 / 0.774, 1,937, 145, 81.8 %, 0.957 against the source lines.
8. Beyond-syllabus items (XGBoost, TreeSHAP, monotone XGBoost, PDP, Platt, paired bootstrap) are named once each, not dwelt on.
9. Hand-over sentences spoken verbatim; the laptop stays with one person; presenter clicks, does not describe clicking.
10. Each presenter rehearses their 12+ questions from `qa_bank.md` and the ten numbers to have in your head; A1 opens in under 5 s.
11. Figures: only the 1920×960 renders in figures/slides/ go on slides. Re-renders requested from the figure owner by 25 Oct — s_xgb_learning_curve.png (slide 6), s_mlp_overfit.png (slide 8), s_auc_headline.png with an un-clipped x-label (slide 11), s_imbalance.png with an un-clipped legend (slide 9), optional s_knn_k.png (slide 5); if any is missing, apply the Canva fallback written in that slide's Figure field of the spec (full-width placement + text box, left-panel crop, edge crop).
12. Export the Canva deck to PPTX on 26 Oct before noon; check the five section tabs, slide numbers and NTU logo survive the export; upload to NTU Learn and confirm receipt; bring one printed copy of the running order and this checklist; the first speaker holds the timer.
