# Slides outline — MH6805 Group B19 final presentation (27 Oct 2026, morning; v2, written 2026-10-22)

Companion to `slide_spec_v2_2026-10-22.md` (full on-slide text, figures and sizes, notes EN/中文, the v1 → v2 slide map) and `qa_bank.md` (revised 22 Oct for v2). This replaces the v1 outline of 21 Oct (14 slides + A1, `slide_spec_2026-10-21.md`). v2 has 21 spoken slides + 2 appendix slides; each slide makes one point, and its notes give the conclusion first, then the reason, with at most two result numbers. Canva: v1 is the 15-page design DAHYodggd94 (exported to `MH6805_B19_Presentation.pdf` / `.pptx` in this folder, 21 Oct 17:23); v2 is built in the working copy DAHYnjt1H8I, MH6810 template style, by duplicating v1 layouts as the spec's slide map says. PPT due 26 Oct noon on NTU Learn; the discussion group gets the PDF in advance. 10 minutes + Q&A; all three present.
Protocol wording on every slide that needs it: "chosen on training CV; each configuration frozen at its first test scoring; every test read dated; the test set only reports". Post-freeze (21 Oct) checks are training CV, never test-scored, and select nothing; their source lines say so. Ordering is always training CV AUC, and the three ensembles tie. 5:1 is a scenario borrowed from the Statlog German credit cost matrix. AUC to 3 decimals, costs as integers, no 4-decimal number on any slide. Every number below is in a file under `results/` (named per slide).

## Running order (words = Notes EN in the spec; 140 words/min)

| # | Title | Section | Presenter | Words | Time | Cum. |
|---|---|---|---|---|---|---|
| S1 | Card default: the cut-off matters more than the model | Data & Problem | RUAN | 65 | 0:28 | 0:28 |
| S2 | Our data come from Taiwan's 2005 card-debt crisis | Data & Problem | RUAN | 64 | 0:27 | 0:55 |
| S3 | Repayment codes are states, not a number line | Data & Problem | RUAN | 57 | 0:24 | 1:20 |
| S4 | Every choice made on training folds; test only reports | Protocol | RUAN | 66 | 0:28 | 1:48 |
| S5 | Segments describe the book; they do not price risk | Models | RUAN | 56 | 0:24 | 2:12 |
| S6 | The KNN benchmark needs about a hundred neighbours | Models | RUAN | 43 | 0:18 | 2:30 |
| S7 | Why the logit trails: coding first, shape second | Models | RUAN | 66 | 0:28 | 2:59 |
| S8 | Pruned tree ties KNN; ensembles tie at the top | Models | LU | 67 | 0:29 | 3:27 |
| S9 | XGBoost's edge is mostly shape, not interactions | Models | LU | 71 | 0:30 | 3:58 |
| S10 | Last month's record carries most of the signal | Results | LU | 51 | 0:22 | 4:20 |
| S11 | Arrears push scores up; a larger limit pulls down | Results | LU | 71 | 0:30 | 4:50 |
| S12 | Monotone constraints: promise only what credit logic guarantees | Results | LU | 70 | 0:30 | 5:20 |
| S13 | Personal attributes add little, but force a fairness choice | Results | LU | 72 | 0:31 | 5:51 |
| S14 | Four tiers hold in every fold; top three tie | Results | LI | 51 | 0:22 | 6:13 |
| S15 | Four more families: close in ranking, not in probabilities | Results | LI | 46 | 0:20 | 6:33 |
| S16 | Class imbalance: weight the decision, not the data | Results | LI | 63 | 0:27 | 7:00 |
| S17 | Leakage control: SMOTE before the split fakes 0.914 AUC | Results | LI | 47 | 0:20 | 7:20 |
| S18 | The cut-off matters more than the model | Business & Future | LI | 69 | 0:30 | 7:49 |
| S19 | The bank's own cost ratio should set the cut-off | Business & Future | LI | 59 | 0:25 | 8:15 |
| S20 | A one-line rule gets most of the way | Business & Future | LI | 53 | 0:23 | 8:37 |
| S21 | Limits: one crisis vintage, a re-read test set, borrowed costs | Business & Future | LI | 55 | 0:24 | 9:01 |
| A1 | Appendix · Nested CV, Platt, Cost Sweep, Settings and References | appendix, Q&A only | RUAN (nested CV) / LI (Platt, sweep) / LU (settings) | — | — | — |
| A2 | Appendix · Cost sweep and real-world context | appendix, Q&A only | LI (sweep) / RUAN (crisis, cycle) / LU (regulation) | — | — | — |

Totals (spec v2.1, 22 Oct): RUAN S1–S7 = 417 words / 2:59 · LU S8–S13 = 402 / 2:52 · LI S14–S21 = 443 / 3:10 · spoken 1,262 words = 9:01 at 140 wpm (9:42 at 130 wpm). That is not the run time: the 22 spoken result numbers add about 25 s and the three hand-overs about 15 s, so a run at 140 wpm is ≈ 9:41 before slide changes. At 130 wpm it would be ≈ 10:24, so hold about 140 wpm; the rehearsal target is 9:45, and the cut-first list below is the only planned cut.

## Per-slide message and cut rule
Format: **Message** = the one point the slide makes; the notes say it first (numbers from the files named). **Cut** = one of the four cut-first sentences, dropped in the order 1 → 4 only if the timed rehearsal runs past 9:45 (≈ 29 words, ≈ 12 s in all); make the same cut in Notes 中文. LU has no cuts, so LU stays above 2:50.

S1. **Message:** where the bank sets its cut-off matters more than which model it uses; RUAN introduces herself and the split (RUAN data, protocol and simpler models; LU trees; LI comparison and the bank's decision). The slide keeps v1's contribution mini-table ("draft, to confirm") and adds the small presenting line "RUAN 1–7 · LU 8–13 · LI 14–21". (aggregate_summary.json, contribution_statement_draft.md)
S2. **Message:** the data are a crisis vintage (one Taiwanese bank, inputs up to Sep 2005, as the card-debt bubble burst), and with 22.1 % defaulting, "nobody defaults" is 77.9 % right yet catches no one, so we judge ranking and cost, not accuracy. **Cut 2:** "We judge ranking, and the cost of mistakes." (eda_summary.json; FSC index-7.xlsx; CBC FSR Jun 2008)
S3. **Message:** repayment codes are states, not a scale: two months late, two-thirds default, and "paid in full" defaults more than "revolving" (16.9 % vs 12.9 %), so evenly spaced numbers mislead. (pay_code_default_rates.csv, eda_summary.json; training rows)
S4. **Message:** every choice (settings, cut-offs) is made by five-fold CV on training rows; each model is frozen at its first test scoring, and a dated log lists every test read. (frozen_manifest.json, TEST_SET_LOG.md, nested_cv_summary_repro.json)
S5. **Message:** K-means finds five overlapping segments (silhouette 0.27) and the label adds nothing measurable to the logit: segments are a map for policy, not a better score. **Cut 3:** ", where one would mean cleanly separated". (unsupervised_summary_repro.json, kmeans_selection_repro.csv; training rows)
S6. **Message:** the KNN benchmark needs about a hundred neighbours because default is noisy and a small vote is luck; its AUC is 0.77 (K = 101; nested CV 0.768). (knn.json, knn_k5.json, nested_cv_knn_frozen_grid.json)
S7. **Message:** the logit trails mainly because it reads each status code as one evenly spaced number; coded by meaning it rises from 0.760 to 0.768 and catches up with KNN (training CV), but most of the gap to the trees remains. Hand-over to LU follows. (pay_coding_cv.csv, pay_coding_summary.json; post-freeze)
S8. **Message:** a pruned 17-rule tree already ties KNN, and the forest, gradient boosting and XGBoost tie at the top (CV 0.785–0.786), about 0.025 above the logit. (tree_pruned.json, random_forest.json, gradient_boosting.json, xgboost.json, fold_stability.csv)
S9. **Message:** most of XGBoost's edge is curved one-variable effects, not interactions: one-split trees recover 86 % of its lead over the logit and interactions add only 0.004, so a banded scorecard could keep most of the gain. (ablation_depth.csv, pdp_summary.json; post-freeze training CV)
S10. **Message:** next month's default is largely the next step of a delinquency process: September's record lifts XGBoost from 0.63 to 0.77, and older months add a little through six-month summaries. (ablation_history.csv, pay_code_default_rates.csv; post-freeze training CV)
S11. **Message:** arrears push XGBoost's scores towards default and a larger limit pushes them away; the three delay columns are one signal, and this describes the model, not causes, because the bank sets the limit. (shap_importance.csv, shap_summary.json, rf_refit_drop_column.csv; training rows)
S12. **Message:** promise only what credit logic guarantees: the frozen constraint also ordered −2/−1/0 and utilisation against the data and cost 0.002 CV AUC; constraining only lateness and the limit is almost free. (monotone_check_consistent.csv, monotone_respec_cv.json; post-freeze training CV)
S13. **Message:** sex, age, marriage and education barely help ranking (0.002) but force a fairness choice: no score can match each group's rate and equalise errors, and dropping the columns trades one gap for another, so the bank must choose and document. Hand-over to LI follows. (fairness_groups.csv, fairness_summary.json, ablation_demographics.csv; training-OOF diagnostic)
S14. **Message:** the four tiers (ensembles, network, KNN, logit) hold in every fold and the top three tie; XGBoost's frozen test AUC, 0.789, is close to its CV score. (fold_stability.csv, fold_stability_summary.json; test reporting only: aggregate_summary.json, auc_bootstrap_ci.csv)
S15. **Message:** the network, SVM, LDA and naive Bayes rank near the benchmark, but LDA and naive Bayes give probabilities too extreme to price (calibration slopes 0.749, 0.141). (fold_stability.csv, mlp.json, svm_selection_statistics.json, lda.json, naive_bayes.json)
S16. **Message:** weight the decision, not the data: weighting shifts the logit's intercept and inflates every probability; Platt restores them, yet weighted + Platt is still 0.4 % costlier than the plain logit, so keep the real proportions and move the cut-off. **Cut 4:** ", the dotted lines". (recalibration.csv, imbalance_comparison.csv, nested_cv_summary_repro.json)
S17. **Message:** resampling before the split fakes success (0.914 against an honest 0.774), so every resampler runs inside the training folds. (imbalance_leakage_demo.csv; negative control in frozen_manifest.json)
S18. **Message:** where the bank draws the line matters more than which model draws it: in the 5:1 scenario borrowed from the Statlog German credit cost matrix, moving the logit's cut-off saves 1,937; switching to XGBoost saves 145 more [−18, 299], which could be zero. (threshold_vs_model_saving.csv; cut-offs from training OOF, test reporting only)
S19. **Message:** the bank's true ratio is unknown, so we swept it: the cut-off should follow the bank's own ratio (about 1 / (1 + r)), and the cheapest model keeps changing within about 1 %, so we name no cost winner. (preferred_model_by_ratio_oof.csv, cost_anchors.json; training OOF, selects nothing)
S20. **Message:** a one-line rule, "flag any payment delay in six months", gets most of the way: XGBoost costs 7.7 % less on training rows, the logit almost nothing less; the rule misses the defaulters who were never late. (simple_rule_benchmark.csv, topk_capture_oof.csv; post-freeze, never test-scored)
S21. **Message:** the answer holds within three limits (a crisis vintage, a test set read more than once with every read dated, a borrowed cost ratio): rank with any ensemble, set the cut-off from costs, re-base before reuse. **Cut 1:** "So rank with any ensemble, and set the cut-off from costs." (report §7; TEST_SET_LOG.md; Federal Reserve delinquency series)
A1. Not spoken. Shown when asked about nested CV (KNN and the logistic family only; the ensembles were not nested), Platt (maps fitted on training OOF; each version frozen at its first test scoring, 19 Oct; weighted + Platt +0.4 % cost, not adopted), the cost sweep, settings, the post-freeze checks line or the references. (nested_cv_summary_repro.json, recalibration.csv, threshold_vs_model_saving.csv, POST_REVIEW_CHECKS.md, r_crosscheck.csv)
A2. Not spoken. Shown when asked about the cost sweep (threshold saving against model saving, 1:1–20:1, test reporting only) or the real-world context (US card delinquency, EU AI Act Annex III 5(b), BoE–FCA 2024 survey; present-day benchmarks, no compliance claimed). (threshold_vs_model_saving.csv; source lines on the slide)

## Hand-overs (spoken by the outgoing presenter, not counted in the words; about 5 s each)
- After S7 (RUAN → LU): "Tree models can learn what a straight line misses. LU, over to you."
- After S13 (LU → LI): "LI will now line up the models and turn scores into decisions."
- After S21 (LI closes): "Thank you — we welcome your questions." Then open the floor: each presenter takes questions on their own slides, LI opens `qa_bank.md`, and A1 / A2 are shown by whoever is asked.

## Rehearsal checklist (one timed full run with the clicker and hand-overs before 26 Oct, one after upload; 12 lines)
1. Time each block against the totals above (RUAN 2:59, LU 2:52, LI 3:10 of notes, plus numbers and hand-overs); aim for 9:45 in all. If over, drop the four cut-first sentences in order (S21, S2, S5, S16) before cutting anything else, and make the same cut in Notes 中文.
2. Hold about 140 wpm (130 wpm runs ≈ 10:24); saying decimals without the leading zero ("point seven eight nine") saves about 5 s.
3. Every slide: conclusion first, then the reason, at most two result numbers; never read the body text aloud.
4. Protocol (S4, S14, S21): "each configuration frozen at its first test scoring; every test read dated". The test set was read more than once; never say it was scored or touched only once, and never rank by test AUC.
5. Ensembles (S8, S14): "the three ensembles tie"; XGBoost is the reference only by the preset highest-CV rule, never "the winner".
6. Costs (S18–S20): 5:1 is "a scenario borrowed from the Statlog German credit cost matrix" (the S18 notes say "borrowed from German credit data"), never a bound; say the two savings (1,937; 145 [−18, 299]) and never "N times".
7. Wording: "predict / rely on / associated with", never causal verbs (S10, S11: the bank sets the limit); segments are "a map for policy, not a better score", never "economically meaningful".
8. Post-freeze slides (S7, S9, S10, S12, S13, S19, S20) say "training CV" or "training rows" aloud at least once: none of these checks was ever test-scored.
9. Numbers on slides match `results/`: re-check 0.786 / 0.789, 0.770, 0.760 → 0.768, 86 % and 0.004, 0.63 → 0.77, 0.002, 0.914 / 0.774, 1,937 and 145, 7.7 % against each source line.
10. Figures: only the 22 Oct renders in `figures/slides/` named in the spec, plus `figures/shap_beeswarm.png` (S11) and the context images (S1, S2, A2), sized as the spec's Figure sizing says; never `monotone_pay1.png`; a slide's fallback only if a later re-render breaks.
11. Q&A: each presenter rehearses their block of `qa_bank.md` (RUAN R1–R15, LU L1–L17, LI J1–J17) and the eight general questions; L13 / L14 (on S21) and L15 (on S5) are prepared by both the owner and the slide's speaker; A1 and A2 open in under 5 s.
12. Export the Canva deck to PPTX and PDF on 26 Oct before noon; check that the five section tabs, page numbers and NTU logo survive the export; upload to NTU Learn and confirm receipt; bring one printed copy of the running order and this checklist; the first speaker holds the timer.
