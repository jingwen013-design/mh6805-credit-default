# Contribution statement — DRAFT (group to confirm)

**Status: DRAFT — group to confirm.** Percentages were drafted by LI Jingwen on 2026-10-21 from the ownership in `Project/README.md` (§Ownership) and the work log in the brief; every number is a proposal until RUAN and LU confirm or amend. In the report the numbers are typeset in red until confirmed (see `report/sec_front.tex` and the caption of Table `tab:a-workload`).

Order of columns: RUAN Hangjing (G2607546F) / LU Xinyu (G2603537D) / LI Jingwen (G2607526K). Each row sums to 100.

| Section | RUAN | LU | LI | Justification (ownership facts) |
|---|---:|---:|---:|---|
| §1 Introduction & objectives | 30 | 20 | 50 | LI drafted the framing and the four objectives and did the final aggregation of the text; RUAN wrote the proposal (v2, submitted 30 Sep) whose problem statement the introduction carries forward; LU contributed the tree/boosting objective. |
| §2 Data | 25 | 15 | 60 | LI wrote `common.py` (cleaning, 35-duplicate drop, 6 engineered features, stratified 70/30 split with seed 2607526, preprocessor) and `eda.py`; RUAN's proposal EDA and PCA inputs fed the section; LU checked the tree-friendly encodings. |
| §3 Methods | 30 | 25 | 45 | LI wrote the shared protocol (`evaluate()`, freeze guard, imbalance-inside-folds design, Platt recalibration) and the SVM/MLP/discriminant methods; RUAN wrote KNN, the logistic family and the nested-CV / pre-registered selection rules; LU wrote tree, forest/bagging, GBM/XGBoost methods. |
| §4 Results | 30 | 35 | 35 | RUAN: KNN benchmark, logistic family, 30 Sep second pass (nested CV); LU: pruned tree, random forest, GBM, XGBoost (highest CV AUC, tied with GBM and the forest); LI: SVM, MLP, LDA/QDA/NB, imbalance study, leakage demo, aggregation and paired bootstrap. |
| §5 Interpretation | 15 | 60 | 25 | LU owns TreeSHAP on training rows, monotone-constrained XGBoost, PDP/ICE and the importance/drop-column analysis; LI wrote the "why it loses" mechanisms for SVM/MLP/discriminant models and the calibration reading; RUAN the logistic odds ratios and KNN mechanism. |
| §6 Operating point & cost | 40 | 15 | 45 | RUAN built the cost-ratio sensitivity table (threshold saving vs model saving per ratio, proposal v2 Table 2) and the nested-CV check; LI wrote the recalibration, threshold-vs-model-saving reproduction (`threshold_vs_model_saving.py`) and the closed-form 1/(1+r) argument; LU supplied the XGBoost operating point. |
| §7 Conclusion | 34 | 33 | 33 | Jointly written and revised; one-point rounding assigned to RUAN as proposal author. |
| Code & reproducibility | 20 | 25 | 55 | LI: `common.py`, `aggregate.py`, `freeze.py` + manifest (46 entries), `recalibrate.py`, `run_all.sh`, GitHub repo; LU: five tree/forest/boosting/SHAP/monotone scripts; RUAN: KNN, logistic, unsupervised scripts and the v2 nested-CV / clustering-stability code reassembled as `nested_cv_check.py` and `cluster_stability.py`. |
| Presentation (slides & delivery) | 33 | 33 | 34 | Three presenters, about equal speaking time (v2 deck: RUAN slides 1–7, LU slides 8–13, LI slides 14–21); LI assembled the deck, notes and Q&A bank, hence the extra point. |

Model-family ownership (unchanged from `README.md`): RUAN — KNN (benchmark), logistic family, PCA + K-means / hierarchical segmentation; LU — pruned tree, random forest / bagging, GBM + XGBoost; LI — SVM (linear, RBF), feed-forward network, LDA / QDA / naive Bayes, class-imbalance study. Shared: EDA, aggregation, threshold/cost analysis, report, slides. Not shared with any other module.

## Slide-1 one-line version

> Contribution (draft, to be confirmed): RUAN 29 % — KNN benchmark, logistic family, clustering, nested-CV second pass · LU 29 % — tree, forest, boosting, SHAP/PDP/monotone · LI 42 % — SVM/MLP/discriminant, imbalance & recalibration, freeze/aggregation code, report assembly.

(The three headline figures are the unweighted means of the nine rows above: RUAN 29.2, LU 29.1, LI 41.8 → 29 / 29 / 42; adjust if the table changes.)

## Slide-1 form (as set on the slide: small heading + 3 × 3 mini-table, each row ≤ 12 words)

**Contribution (draft, to confirm)**

| RUAN | 29 % | KNN benchmark, logistic family, clustering, nested-CV pass |
| LU | 29 % | tree, forest, boosting, SHAP/PDP/monotone |
| LI | 42 % | SVM/MLP/discriminants, imbalance & Platt, freeze/aggregation, report |

On slide 1 the statement is set as the mini-table above under the small heading "Contribution (draft, to confirm)" (the one-line version is for the report and messages; as a single bullet its 38 words would set at about 14 pt on a 1920 × 1080 slide), so that the deck itself carries the contribution statement required by guideline §4; the presenting order (RUAN 1–7 · LU 8–13 · LI 14–21) is a separate smaller line. The word "draft" stays on the slide until RUAN and LU confirm (deadline 25 Oct evening; PPT upload 26 Oct noon). Slide 1 and the report (`sec_front.tex`, Table `tab:a-workload`) must change together if any row changes.

## 给 RUAN 和 LU 的确认留言（中文，3 行）

1. 两位好，这是报告和 PPT 都要放的分工百分比草稿（按 README 的模型族归属起草，红字表示待确认），请看上面每一行。
2. 如果哪一行你觉得比例不对，直接回复「§X 改成 a/b/c」即可，我会同步改报告和第 1 页幻灯片；无异议请回「确认」。
3. 麻烦在 10 月 25 日（周六）晚前回复，26 日中午 PPT 要交 NTU Learn，29 日交报告。
