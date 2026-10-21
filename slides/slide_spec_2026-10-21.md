# Slide spec — MH6805 Group B19 final presentation (written 2026-10-21)

Deck: Canva design DAHYnjt1H8I, MH6810 template style (NTU logo top-right, "Topic: claim" title, one bold claim line, body, grey source line bottom-left, bottom progress bar with five section tabs + slide number). 27 Oct 2026 morning session, 10 minutes + Q&A; PPT due 26 Oct noon on NTU Learn.
Section tabs: `Data & Problem` (1–3) · `Protocol` (4–5) · `Models` (6–10) · `Results` (11–12) · `Business & Future` (13–14) · A1 appendix (not spoken).
Rules applied: every on-slide number is in a file under `results/` (named in the source line); AUC to 3 decimals, costs as integers, no 4-decimal number on any slide; title ≤ 9 words, claim line ≤ 16, bullets ≤ 12 (percentages printed closed up, e.g. 67.6%); figures placed on a slide are the 1920 × 960 renders in `figures/slides/` with 16–20 pt text — report-sized PNGs are not placed (the 21 Oct 16:46 renders for slides 5, 6 and 8 are checked and assigned; re-render requests and mandatory Canva fallbacks are written in the Figure field of slides 9 and 11; slide 12 carries a one-line Canva unit note); ordering is always training CV AUC (OOF AUC for the subsample-tuned RBF SVM), test numbers are reported beside it; honest wording everywhere: "chosen on training CV; each frozen configuration scored once on the test set; the test set only reports". Speech budget ≤ 9:20 at 140 words/min; the Time field of each slide is its Notes EN word count ÷ 140.

## Running order (words = Notes EN; time = words ÷ 140 wpm)

| # | On-slide title | Section | Presenter | Words | Time | Cumulative |
|---|---|---|---|---|---|---|
| 1 | Credit-card default: eleven classifiers, one honest protocol | Data & Problem | RUAN Hangjing | 51 | 0:22 | 0:22 |
| 2 | Data: 29,965 clients, 22.1% default next month | Data & Problem | RUAN Hangjing | 77 | 0:33 | 0:55 |
| 3 | EDA: repayment behaviour is the signal, demographics are noise | Data & Problem | RUAN Hangjing | 84 | 0:36 | 1:31 |
| 4 | Protocol: chosen on training CV, frozen, scored once | Protocol | RUAN Hangjing | 108 | 0:46 | 2:17 |
| 5 | Benchmark and linear family: KNN 0.770, logistic 0.760 | Protocol | RUAN Hangjing | 100 | 0:43 | 3:00 |
| 6 | Tree family: pruning, averaging, boosting reach CV 0.786 | Models | LU Xinyu | 96 | 0:41 | 3:41 |
| 7 | Interpretation: repayment delay drives the score, constraints cost 0.002 | Models | LU Xinyu | 100 | 0:43 | 4:24 |
| 8 | SVM, MLP, discriminants: only the network clears the benchmark | Models | LI Jingwen | 114 | 0:49 | 5:13 |
| 9 | Imbalance: re-balancing moves the threshold, not the ranking | Models | LI Jingwen | 111 | 0:48 | 6:00 |
| 10 | Leakage control: SMOTE before the split fakes 0.914 AUC | Models | LI Jingwen | 71 | 0:30 | 6:31 |
| 11 | Ranking by training CV: boosting leads logistic by 0.025 | Results | LI Jingwen | 79 | 0:34 | 7:05 |
| 12 | Operating point: the cut-off saves 1,937, the model 145 | Results | LI Jingwen | 108 | 0:46 | 7:51 |
| 13 | Business: a map for policy, a cut-off for decisions | Business & Future | LU Xinyu | 112 | 0:48 | 8:39 |
| 14 | Limitations and next: one split, one assumed cost ratio | Business & Future | LU Xinyu | 92 | 0:39 | 9:18 |
| A1 | Appendix: nested CV, Platt reliability, nine-ratio sweep | A1 appendix (not spoken; shown only if asked) | whoever is asked (RUAN for nested CV, LI for Platt and the sweep) | 77 | — | — |

**Spoken total (slides 1–14): 1303 words = 9:18 at 140 wpm; buffer to 9:20 = 0:02.**

| Presenter | Slides | Words | Time |
|---|---|---|---|
| RUAN Hangjing | 1–5 | 420 | 3:00 |
| LU Xinyu | 6–7, 13–14 | 400 | 2:51 |
| LI Jingwen | 8–12 | 483 | 3:27 |

Three hand-overs (after 5, 7, 12) ≈ 5 s each and slide changes sit in the 40 s between 9:20 and the 10-minute limit. Word counts, times and the running order were re-computed on 21 Oct (evening) after the review fixes; the same table is in slides_outline.md. A1 is shown only in Q&A.

---

## Slide 1 — Credit-card default: eleven classifiers, one honest protocol

Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:22 (51 words)
Claim line: Threshold choice matters more than the model; chosen on training CV, the test set only reports.
Body:
- MH6805 Group B19 — RUAN Hangjing · LU Xinyu · LI Jingwen
- Who will miss next month's card payment? UCI Taiwan 2005, 29,965 clients
- Presenting: RUAN 1–5 · LU 6–7, 13–14 · LI 8–12
**Contribution (draft, to confirm)** — small heading over a 3 × 3 mini-table (name · share · work), each row ≤ 12 words:
| RUAN | 29 % | KNN benchmark, logistic family, clustering, nested-CV pass |
| LU | 29 % | tree, forest, boosting, SHAP/PDP/monotone |
| LI | 42 % | SVM/MLP/discriminants, imbalance & Platt, freeze/aggregation, report |
Figure: none — title slide; NTU logo top-right per template. The mini-table is the on-slide contribution statement required by guideline §4 (heading carries the word "Contribution"; percentages are the drafted 29 / 29 / 42 from slides/contribution_statement_draft.md; each row ≤ 12 words, so no body line exceeds the 12-word rule and nothing sets below 16 pt). Place it small (16–18 pt) under the three bullets. Keep the word "draft" in the heading unless RUAN and LU have confirmed the split by 26 Oct noon; bullet 3 is a smaller line, presenting order only.
Source line: MH6805 Group B19 · UCI Default of Credit Card Clients (Yeh & Lien 2009), CC BY 4.0 · results/aggregate_summary.json
Notes EN: Good morning, we are Group B19. Our question: who will miss next month's card payment? Eleven classifiers, one protocol, and what a risk manager should do with the result. I cover data and protocol; LU the tree family and the business reading; LI the other models, imbalance and the operating point.
Notes 中文: 早上好，我们是 B19 组。问题是：谁会在下个月逾期还卡？十一个分类器、一套规程，以及风控经理该怎么用结果。我讲数据和规程，LU 讲树模型家族和业务解读，LI 讲其余模型、类别不平衡和决策阈值。
Hand-over: (none — same presenter continues)

---

## Slide 2 — Data: 29,965 clients, 22.1% default next month

Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:33 (77 words)
Claim line: Accuracy is not the yardstick — "never default" already scores 0.779.
Body:
- 29,965 clients — 35 exact duplicates dropped from 30,000
- 22.1% default next month — majority-class accuracy 0.779
- 29 predictors — 23 raw + 6 engineered (utilisation, repayment, delay)
- 20,975 train / 8,990 test — one stratified 70/30 split, seed 2607526
Figure: none — four stat tiles (big number + label) in a 2 × 2 grid, percentages printed closed up (22.1%) as on the EDA figure; optional small inset figures/eda_target_distribution.png (class balance bar) if the tile grid leaves space.
Source line: results/eda_summary.json, eda_data_quality.csv; UCI Default of Credit Card Clients, CC BY 4.0
Notes EN: Thirty thousand Taiwanese card holders from 2005; 35 duplicates dropped, 29,965 left, no missing cells. Twenty-three raw fields plus six engineered from utilisation, repayment and delay. One stratified 70/30 split, saved to disk: 20,975 training, 8,990 test. Twenty-two per cent default, so "never default" is already 77.9 per cent accurate. We judge by ROC-AUC and average precision and price the errors: a missed defaulter costs five false alarms — one point of a range we sweep later.
Notes 中文: 2005 年台湾 3 万名信用卡客户；去掉 35 条重复后剩 29,965 人，没有缺失值。23 个原始字段，加 6 个由使用率、还款和逾期构造的特征。分层 70/30 只切一次并存盘：训练 20,975，测试 8,990。违约率 22%，"全判不违约"已有 77.9% 的准确率。我们用 ROC-AUC 和平均精度评价，并给错误定价：漏掉一个违约者等于五次误报——这只是稍后要扫描的区间中的一个点。
Hand-over: (none — same presenter continues)

---

## Slide 3 — EDA: repayment behaviour is the signal, demographics are noise

Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:36 (84 words)
Claim line: Two months late → 67.6% default, revolving 12.9%: a step no line draws.
Body:
- PAY_1: revolving 12.9%, one month late 33.9%, two months late 67.6%
- Credit limit protects: 32.1% default in lowest quintile, 13.7% in highest
- Months delayed alone: univariate AUC 0.724 — the strongest single feature
- Utilisation is U-shaped: 24% at zero, 14% moderate, 28% at the limit
- All rates from the 20,975 training rows; test set not inspected
Figure: figures/slides/s_eda_gradients.png — both panels (left: default rate by September repayment status; right: by credit-limit quintile). Caption: "Default rate along the two strongest gradients, training partition only".
Source line: Training partition n = 20,975; results/eda_summary.json (default_rate_by, univariate_auc); UCI CC BY 4.0
Notes EN: Everything here uses training rows only. Behaviour beats demographics: one month late defaults 34 per cent, two months late 68, revolving clients 13. Credit limit protects — 32 per cent default in the lowest quintile, 14 in the highest. Two shapes matter: the jump from one to two months late, and the U in utilisation — 24 per cent at zero, 14 at moderate use, 28 near the limit. A straight line in log-odds draws neither, so we expect trees to beat linear models.
Notes 中文: 这页只用训练行。行为比人口属性有用：逾期一个月违约 34%，两个月 68%，循环信用客户 13%。额度是保护因素——最低五分位违约 32%，最高 14%。两个形状要紧：从逾期一个月到两个月的跳跃，以及使用率的 U 形——零使用 24%，中等 14%，接近满额 28%。对数几率里的直线两个都画不出，所以我们预期树模型赢过线性模型。
Hand-over: (none — same presenter continues)

---

## Slide 4 — Protocol: chosen on training CV, frozen, scored once

Section: Protocol
Presenter: RUAN Hangjing
Time: 0:46 (108 words)
Claim line: Every choice on 20,975 training rows; 46 frozen configurations; the test set only reports.
Body:
- One stratified split, seed 2607526; scaling and resampling inside every fold
- Rank by 5-fold training CV AUC; thresholds from training out-of-fold scores
- Frozen after test scoring: 34 models + leakage control by 18 Sep
- +2 re-scored 30 Sep, +9 Platt 19 Oct = 46 frozen entries
- After 30 Sep feedback: nested CV re-selects frozen KNN and logistic settings
Figure: diagram drawn in Canva, no PNG — five boxes left to right: raw data → one split saved to disk → Pipeline (preprocess + resample + fit) inside 5-fold CV → out-of-fold probabilities → thresholds and model choice → freeze → one scoring on the 8,990 test rows (box drawn with a padlock; small text under it: "a changed configuration is refused before any test label is read"). Side box below the freeze box: "30 Sep second pass (after the proposal feedback): nested CV, hyper-parameters chosen inside each outer fold → same KNN and logistic choices, nothing re-scored". Caption: "Training rows decide everything; the frozen test scoring only reports". Arithmetic shown on the slide must close: 35 (34 models + 1 leakage control, first scored 16–18 Sep) + 2 (30 Sep) + 9 (19 Oct) = 46.
Source line: results/frozen_manifest.json (46 entries: 35 by 18 Sep, 2 on 30 Sep, 9 on 19 Oct), nested_cv_summary_repro.json (agrees_with_frozen = true for KNN and logistic), knn_v2.json; code/freeze.py --check passes
Notes EN: One split, saved to disk; the test set enters no decision. Preprocessing and resampling sit inside every fold. Models are ranked by five-fold training CV AUC; cut-offs come from training out-of-fold probabilities. Each configuration was scored once on the test set and frozen: 34 models plus the leakage control by 18 September, two re-scored in RUAN's 30 September second pass, nine Platt versions on 19 October — 46 entries; a guard refuses changed configurations. After the professor's comment we added nested CV, hyper-parameters chosen inside each outer fold: it re-selected the same KNN and logistic settings, so first-round tuning did not flatter CV. The test set only reports.
Notes 中文: 只切一次并存盘，测试集不参与任何决策。预处理和重采样都放在折内。模型按训练集五折 CV 的 AUC 排序，阈值取自训练集折外概率。每个配置在测试集上只打分一次并冻结：9 月 18 日前 34 个模型加 1 个泄漏对照，RUAN 9 月 30 日第二轮复核了 2 个，10 月 19 日 9 个 Platt 版本——共 46 条，配置一改动守卫程序就拒绝重新打分。收到教授评语后我们加了嵌套 CV，超参数在每个外折内部选：它重新选出同样的 KNN 和逻辑回归设置，说明第一轮调参没有美化 CV 数字。测试集只负责报告。
Hand-over: (none — same presenter continues)

---

## Slide 5 — Benchmark and linear family: KNN 0.770, logistic 0.760

Section: Protocol
Presenter: RUAN Hangjing
Time: 0:43 (100 words)
Claim line: KNN, K = 101, is the bar; plain, lasso and ridge logistic sit one point below.
Body:
| Model | CV AUC (sd) | Test AUC | Test AP | Why it sits there |
|---|---|---|---|---|
| KNN K = 101, uniform — benchmark | 0.770 (0.009) | 0.771 | 0.542 | heavy smoothing in 32 dimensions |
| Logistic, plain | 0.760 (0.010) | 0.764 | 0.523 | cannot bend the U or step |
| Logistic L1 / L2 | 0.760 / 0.760 | 0.764 / 0.764 | 0.523 | no shrinkage needed at 20,975 rows |
| Nested CV (30 Sep): KNN | 0.768 ± 0.007 | — | — | K = 101 uniform re-selected |
| Nested CV (30 Sep): logistic family | 0.760 (five variants within 1 SE) | — | — | 1-SE keeps plain; weighting −0.4 % |
Figure: figures/slides/s_knn_k.png — single panel, right of the table at 40 % width (1920 × 960 render of 21 Oct 16:46, checked: 5-fold CV AUC vs K for uniform and distance weights, K = 101 circled and labelled "CV AUC 0.770", K = 5 annotated 0.706, nothing clipped). Caption: "CV AUC climbs with K and plateaus at K = 101 (training folds)". The K = 5 row (CV 0.706, test 0.709; results/knn_k5.json) stays out of the table and is spoken in the notes ("best of 24 settings").
Source line: results/knn.json, logistic.json, logistic_family_summary.csv, nested_cv_summary_repro.json; test n = 8,990
Notes EN: KNN is the course benchmark; every model is placed against it. Best of 24 settings: K equals 101, uniform — CV 0.770, test 0.771, average precision 0.542; K equals 5 would score only 0.706. Plain logistic sits one point lower, CV 0.760, test 0.764; lasso and ridge change nothing, since 20,975 rows and 32 columns need no shrinkage — additive log-odds cannot bend the utilisation U or the two-months step. Nested CV gives KNN 0.768 and logistic 0.760 and re-selects both frozen choices; class weighting would cut cross-fitted cost by only 0.4 per cent, below our 1 per cent rule.
Notes 中文: KNN 是课程基准，所有模型都和它比。24 个设置里最好的是 K = 101、等权重——CV 0.770，测试 0.771，平均精度 0.542；K = 5 只有 0.706。朴素逻辑回归低一个点，CV 0.760、测试 0.764；lasso 和 ridge 毫无变化，因为 20,975 行、32 列不需要收缩——可加的对数几率弯不出使用率的 U 形和逾期两个月的台阶。嵌套 CV 给出 KNN 0.768、逻辑回归 0.760，重新选中两个冻结配置；加类别权重只能把交叉拟合成本降 0.4%，低于预设的 1% 门槛。
Hand-over: "LU will now show what trees add on top of this linear ceiling."

---

## Slide 6 — Tree family: pruning, averaging, boosting reach CV 0.786

Section: Models
Presenter: LU Xinyu
Time: 0:41 (96 words)
Claim line: Pruned tree ties KNN; 500 averaged or 300 boosted depth-3 trees add 0.016–0.017 CV AUC.
Body:
- Full tree: 2,791 leaves, depth 55, CV AUC 0.615 — pure memorisation
- Pruned to 17 leaves, root MAX_DELAY > 1.5: CV 0.769, test 0.776
- Forest, 500 trees, ≥ 20 clients per leaf: CV 0.785, test 0.787
- XGBoost, depth 3, rate 0.03, 300 rounds: CV 0.786, test 0.789
- XGBoost − forest +0.002 [−0.001, +0.005]: tied; 0.5 s vs 19.7 s
Figure: figures/slides/s_xgb_learning_curve.png — single panel under the bullets (1920 × 960 render of 21 Oct 16:46, checked: the rate-0.1 annotation "peak 0.786 at 122 rounds" now sits at x ≈ 180 rounds, y ≈ 0.81, clear of the y-axis label; "chosen: 300 rounds, depth 3, rate 0.03" at the top of the dotted line; legend bottom-right; no 4-decimal number). Caption: "CV AUC peaks near 300 rounds while training AUC keeps climbing". If the file is re-rendered again before 26 Oct, re-check that no text crosses the y-axis label or the legend; if it does, cover the colliding text with a Canva text box reading "peak 0.786 at 122 rounds (rate 0.1)".
Source line: results/tree_full.json, tree_pruned.json, tree_rules.txt, random_forest.json, xgboost.json, auc_bootstrap_ci.csv; test n = 8,990
Notes EN: A tree grown to purity has 2,791 leaves and CV AUC 0.615 — it memorises. Cost-complexity pruning on training CV cuts it to 17 leaves, root split maximum delay above one month: CV 0.769, tying KNN, test 0.776. Averaging 500 bootstrap trees with at least 20 clients per leaf removes the variance: forest CV 0.785, test 0.787. Boosting shallow trees, depth three, rate 0.03, 300 rounds, gives XGBoost CV 0.786, test 0.789, our CV winner. Its bootstrap gap to the forest, 0.002, includes zero: a tie, but XGBoost fits in half a second instead of twenty.
Notes 中文: 长到纯的树有 2,791 个叶子，CV AUC 0.615——它在死记硬背。按训练 CV 做代价复杂度剪枝，砍到 17 个叶子，根节点是"最大逾期超过一个月"：CV 0.769 与 KNN 打平，测试 0.776。500 棵自助树取平均、每叶至少 20 个客户，把方差消掉：森林 CV 0.785、测试 0.787。提升浅树，深度 3、学习率 0.03、300 轮，得到 XGBoost 的 CV 0.786、测试 0.789，CV 冠军。它和森林的自助差距 0.002 含零：打平，但 XGBoost 半秒拟合完，森林要二十秒。
Hand-over: (none — same presenter continues)

---

## Slide 7 — Interpretation: repayment delay drives the score, constraints cost 0.002

Section: Models
Presenter: LU Xinyu
Time: 0:43 (100 words)
Claim line: Three delay features carry the model; monotone constraints remove 24 wrong-direction steps for −0.002 AUC.
Body:
- TreeSHAP on 20,975 training rows: PAY_1 0.298, MAX_DELAY 0.281, N_MONTHS_DELAYED 0.230
- Dropping PAY_1 cuts forest out-of-bag AUC 0.785 → 0.779, the largest loss
- Partial dependence keeps the utilisation U: minimum 0.204 at 0.08
- Monotone XGBoost: wrong-direction steps 24 → 0; test AUC 0.789 → 0.787
- Beyond syllabus: XGBoost, TreeSHAP, PDP/ICE, monotone constraints, Platt, paired bootstrap
Figure: figures/slides/s_importance.png — single panel. Caption: "Mean |SHAP| of the ten strongest predictors, with direction and OOB rank".
Source line: results/shap_importance.csv (training rows), rf_refit_drop_column.csv, pdp_summary.json, monotone_summary.json
Notes EN: What drives the score? TreeSHAP on the training rows — the test set is not read — puts the three delay features first: last month's status, maximum delay, months delayed, mean absolute SHAP 0.298, 0.281, 0.230; bill, utilisation and limit follow at a third of that. A drop-column refit of the forest agrees: removing PAY_1 costs the most out-of-bag AUC, 0.785 to 0.779. For a regulator we fitted a monotone-constrained XGBoost — eleven features forced up, two down: all 24 wrong-direction steps vanish for 0.002 AUC. These tools go beyond the syllabus; we use them to explain, never to select.
Notes 中文: 分数由什么决定？在训练行上做 TreeSHAP——不读测试集——前三都是逾期特征：上月状态、最大逾期、逾期月数，平均绝对 SHAP 0.298、0.281、0.230；账单、使用率和额度只有三分之一。森林的逐列删除重拟合给出同样结论：去掉 PAY_1 损失的袋外 AUC 最大，0.785 到 0.779。为监管需要我们拟合了单调约束的 XGBoost——11 个特征强制向上、2 个向下：24 步反向移动全部消失，代价 0.002 的 AUC。这些工具超出大纲；我们用它们解释，从不用来选模型。
Hand-over: "LI takes over with the models that did not beat the benchmark — and why."

---

## Slide 8 — SVM, MLP, discriminants: only the network clears the benchmark

Section: Models
Presenter: LI Jingwen
Time: 0:49 (114 words)
Claim line: MLP CV 0.776 beats KNN; RBF SVM 0.761, LDA 0.757, NB 0.736 fall below.
Body:
| Model | Selection AUC (sd) | Test AUC | Test AP | Mechanism |
|---|---|---|---|---|
| MLP 32-64-32-1, early stopping | CV 0.776 (0.007) | 0.773 | 0.551 | 4,225 weights; grid spread 0.005 |
| MLP 32-256-256-1, no stopping | CV 0.692 (0.007) | 0.678 | 0.400 | 74,497 weights memorise: 0.760 → 0.700 |
| RBF SVM C = 1, γ = 0.01, balanced | OOF 0.761 (subsample CV 0.768) | 0.762 | 0.494 | tuned on 6,000 rows; Platt |
| LDA / QDA | CV 0.757 (0.011) / 0.751 (0.009) | 0.759 / 0.755 | 0.525 / 0.516 | Gaussian densities on skewed amounts |
| Gaussian naive Bayes | CV 0.736 (0.011) | 0.740 | 0.504 | bills correlate 0.80–0.95 (mean 0.89) |
Figure: figures/slides/s_mlp_overfit.png — both panels at ≥ 45 % width beside the table (1920 × 960 render of 21 Oct 16:46, checked: left = tuned 32-64-32-1 run without stopping, annotated "peak 0.762 at epoch 14 → 0.712 at epoch 200"; right = over-parameterised 32-256-256-1, "peak 0.760 at epoch 1 → 0.700 at epoch 200"; both match results/mlp_overfit_epochs.csv; panel titles no longer collide). The 32-256-256-1 row's mechanism cell (0.760 → 0.700) is the right panel's hold-out curve (train AUC 0.994). Caption: "Held-out AUC peaks at epoch 14 (left), then falls: early stopping regularises".
Source line: results/mlp.json, mlp_overfit.json, mlp_overfit_epochs.csv, svm_rbf.json, lda.json, qda.json, naive_bayes.json, eda_summary.json; test n = 8,990
Notes EN: Three more families, all against KNN. The tuned network — 32-64-32-1, early stopping, 4,225 weights — is the only one above it: CV 0.776, test 0.773. Its over-parameterised twin, 74,497 weights and no stopping, shows why: held-out AUC peaks at epoch one and falls to 0.700 by epoch 200; CV 0.692. The RBF SVM was tuned on a 6,000-row subsample because kernel SVMs scale quadratically: out-of-fold AUC 0.761, test 0.762, and the lowest average precision, 0.494 — a margin classifier ranks the tail poorly. LDA tracks the logit at 0.757; QDA 0.751; naive Bayes 0.736 — Gaussian densities fail on skewed amounts and bills correlated at 0.80 to 0.95. Flexibility pays only when regularised.
Notes 中文: 再看三个家族，仍和 KNN 比。调好的网络——32-64-32-1、早停、4,225 个权重——是唯一超过基准的：CV 0.776，测试 0.773。它的过参数化孪生版本，74,497 个权重、不早停，说明了原因：留出集 AUC 第 1 轮就封顶，到第 200 轮跌到 0.700；CV 0.692。RBF 核 SVM 在 6,000 行子样本上调参，因为核 SVM 开销按平方增长：折外 AUC 0.761，测试 0.762，平均精度 0.494 全组最低——间隔分类器尾部排序差。LDA 紧跟逻辑回归，0.757；QDA 0.751；朴素贝叶斯 0.736——高斯密度遇上偏态金额和相关 0.80 到 0.95 的账单列就失准。灵活性只有加了正则才有回报。
Hand-over: (none — same presenter continues)

---

## Slide 9 — Imbalance: re-balancing moves the threshold, not the ranking

Section: Models
Presenter: LI Jingwen
Time: 0:48 (111 words)
Claim line: Four strategies move logistic AUC by about 0.001; Platt scaling on training OOF restores calibration.
Body:
- Logistic, four strategies inside folds: CV AUC 0.760–0.761, test 0.762–0.764
- Forest with SMOTE: CV 0.780 → 0.774, test 0.782 → 0.774
- Balanced at 0.5 = plain at 0.22: 99.4% agreement
- Weighting breaks calibration: mean p 0.44 vs 0.22; Brier 0.138 → 0.190
- Platt on training OOF: weighted XGBoost Brier 0.177 → 0.131; cut-offs 0.14–0.19
Figure: figures/slides/s_imbalance.png — both panels (left: test AUC by re-balancing strategy; right: test Brier raw vs Platt-recalibrated for the nine variants). The right-panel legend is STILL clipped at the image edge in the 21 Oct 16:46 render ("after Platt scaling on training OC"): REQUEST a re-render with the legend centred below both panels (ncol = 2) or the label shortened to "after Platt (training OOF)". If no re-render by 25 Oct, the fallback is mandatory, not optional: crop the right 3 % of the image in Canva and add the Canva legend line under the figure, "red = raw after weighting/resampling; blue = after Platt on training OOF". Caption: "AUC barely moves (left); Platt returns Brier to the unweighted level (right)".
Source line: results/imbalance_comparison.csv, imbalance_threshold_equivalence.csv, recalibration.csv (nine Platt versions frozen 19 Oct); test n = 8,990
Notes EN: Weights, SMOTE and under-sampling ran inside the folds only. For logistic the four strategies move CV and test AUC by about 0.001; for the forest SMOTE loses 0.006 to 0.008 — synthetic points blur the leaves. Re-balancing does not re-rank clients: the balanced logit at 0.5 decides like the plain logit at 0.22 for 99.4 per cent of test clients. What it breaks is calibration — mean predicted probability 0.44 against a 0.22 prevalence, Brier 0.138 to 0.190. Platt scaling fitted on training out-of-fold scores repairs it: weighted XGBoost's Brier returns from 0.177 to 0.131 and the 5:1 cut-offs return to 0.14–0.19; nine versions scored once on 19 October and frozen.
Notes 中文: 类别权重、SMOTE 和欠采样都只在折内做。逻辑回归的四种策略 CV 和测试 AUC 只差约 0.001；森林用 SMOTE 反而 CV 掉 0.006、测试掉 0.008——合成点把叶子糊掉了。重新平衡不改变客户排序：加权逻辑回归在 0.5 的判决与朴素逻辑回归在 0.22 的判决，在 99.4% 的测试客户上一致。它破坏的是校准——平均预测概率 0.44，真实违约率 0.22，Brier 从 0.138 到 0.190。在训练集折外分数上拟合的 Platt 缩放把它修回来：加权 XGBoost 的 Brier 从 0.177 回到 0.131，5:1 阈值回到 0.14 到 0.19；九个版本 10 月 19 日各打分一次并冻结。
Hand-over: (none — same presenter continues)

---

## Slide 10 — Leakage control: SMOTE before the split fakes 0.914 AUC

Section: Models
Presenter: LI Jingwen
Time: 0:30 (71 words)
Claim line: Honest forest 0.774; the same forest "tested" on 73 % synthetic defaulters scores 0.914.
Body:
- 0.914 — leaky "test" AUC, forest, SMOTE before the split
- 0.774 — honest test AUC, same forest, SMOTE inside the folds
- 73 % — of the leaky test defaulters are synthetic copies
- 0.770 vs 0.762 — logistic barely moves: a line cannot memorise
Figure: figures/slides/s_leakage.png — single panel. Caption: "Same learners, same data: only the order of SMOTE and split differs".
Source line: results/imbalance_leakage_demo.csv — quarantined negative control (imb_leaky_smote_demo in frozen_manifest.json); 16,705 synthetic rows
Notes EN: One deliberate mistake, quarantined as a negative control. Over-sample first, split second, and the forest reports a test AUC of 0.914 against an honest 0.774 — 73 per cent of its "test" defaulters are synthetic copies of training defaulters, so it recognises its own neighbours. Logistic barely moves, 0.770 versus 0.762: a line cannot memorise. Leakage flatters the flexible models most, which is why every sampler here sits inside the fold.
Notes 中文: 故意犯一次错，隔离成负对照。先过采样再切分，森林报出 0.914 的"测试" AUC，诚实的数字是 0.774——它"测试集"里 73% 的违约者是训练违约者的合成副本，它认出的是自己的邻居。逻辑回归几乎不动，0.770 对 0.762：直线背不住样本。泄漏最会吹捧灵活模型，所以这里所有采样器都放在折内。
Hand-over: (none — same presenter continues)

---

## Slide 11 — Ranking by training CV: boosting leads logistic by 0.025

Section: Results
Presenter: LI Jingwen
Time: 0:34 (79 words)
Claim line: Above KNN: XGBoost, GBM, forest, MLP; pruned tree ties it; SVM, logistic, LDA, NB below.
Body:
- Order: training CV AUC; bars: test AUC; whiskers: CV ± 1 sd
- XGBoost CV 0.786, test 0.789, AP 0.578, Brier 0.131: CV winner
- XGBoost − logistic +0.025 [+0.018, +0.031]; − KNN (benchmark) +0.018 [+0.012, +0.024]
- XGBoost − forest +0.002 [−0.001, +0.005]; − GBM +0.002 [>0, +0.003]
Figure: figures/slides/s_auc_headline.png — single panel (four bullets only; the figure draws no interval on the bars — its whiskers are 5-fold CV AUC ± 1 sd around the CV marker, and the SVM marker is its subsample CV 0.768 ± 0.006 while its bar is test 0.762, so no "within one sd" claim is made on the slide). The 21 Oct 16:46 render still has two defects: on the bottom row the NB value label runs into the dotted-reference label and prints as "0.740logistic 0.764", and the x-axis label is clipped ("… marker = full-training O"). REQUEST a re-render with the two reference labels ("logistic 0.764", "KNN 0.771") moved to the top of the axes above the XGBoost bar (or into the legend) and the x-label shortened to "ROC-AUC on the test set († SVM: tuned on 6,000 rows; marker = full-training OOF)". If no re-render by 26 Oct, BOTH Canva fallbacks are mandatory: (i) cover "logistic 0.764" and "KNN 0.771" at the foot of the axes with a white rectangle and add a text box above the bars reading "dotted = logistic 0.764 · dashed = KNN 0.771 (benchmark)"; (ii) crop the clipped tail of the x-label so no half-word shows. KNN K = 101 is the benchmark line in the figure. Bullet 4 prints the GBM lower bound as ">0": results/auc_bootstrap_ci.csv gives auc_diff_2_5 = 0.000085 with auc_diff_excludes_zero = True, and three-decimal rounding would print "+0.000" and contradict the notes (only the forest gap includes zero). Caption: "CV order; bars = test AUC; whiskers = CV ± 1 sd".
Source line: results/aggregate_summary.json, model_comparison_main.md, auc_bootstrap_ci.csv (B = 500 paired bootstrap of the 8,990 test rows)
Notes EN: The whole field, ordered by the selection score — training CV AUC — with test AUC as bars. Four models beat the benchmark: XGBoost, gradient boosting, random forest, MLP; the pruned tree ties it; RBF SVM, logistic, LDA and naive Bayes fall below. On the paired bootstrap of 8,990 test rows XGBoost beats logistic by 0.025 and KNN by 0.018, intervals clear of zero; its gap to the forest, 0.002, includes zero — the top three are one group.
Notes 中文: 全部模型按选择分数——训练 CV AUC——排序，柱子是测试 AUC。四个模型超过基准：XGBoost、梯度提升、随机森林、MLP；剪枝树与之打平；RBF SVM、逻辑回归、LDA 和朴素贝叶斯在基准之下。在 8,990 行测试集的配对自助法上，XGBoost 领先逻辑回归 0.025、领先 KNN 0.018，区间不含零；它与森林的差距 0.002 含零——前三名是一个梯队。
Hand-over: (none — same presenter continues)

---

## Slide 12 — Operating point: the cut-off saves 1,937, the model 145

Section: Results
Presenter: LI Jingwen
Time: 0:46 (108 words)
Claim line: At 5:1, re-thresholding logistic saves 1,937 [1,604, 2,218]; switching to XGBoost saves 145 [−18, 299].
Body:
| Cost ratio FN:FP | Logistic cut-off (closed form 1/(1+r)) | Threshold saving [95 % CI] | Model saving [95 % CI] |
|---|---|---|---|
| 1:1 | 0.50 (0.50) | 0 | 87 [44, 131] |
| 3:1 | 0.30 (0.25) | 684 [541, 830] | 127 [47, 214] |
| 5:1 | 0.15 (0.17) | 1,937 [1,604, 2,218] | 145 [−18, 299] |
| 10:1 | 0.10 (0.09) | 7,121 [6,395, 7,791] | 420 [187, 646] |
| 20:1 | 0.01 (0.05) — flags 99.9 % | 20,335 [18,909, 21,621] | 150 [−52, 333] |
Figure: figures/slides/s_cost_vs_threshold.png — single panel, left of the table. Its y-axis is the 5:1 cost per 1,000 test clients (dots at ≈ 785 and ≈ 740) while its own annotations (7,063 → 5,126 logistic, 6,642 → 4,981 XGBoost) and every cost in the table are totals over the 8,990 test clients (results/threshold_vs_model_saving.csv, cost_logistic_at_0_5 = 7,063). Add one small grey Canva line directly under the figure: "axis = cost per 1,000 clients; annotated values = totals over 8,990 test clients" (12 words), so the nine-fold difference between dot height and printed number is explained on the slide, not only if asked. The diamonds are the cut-offs chosen on training OOF (t = 0.15); do not point at the test curve's own minimum (the test set only reports). Caption: "5:1 cost against the cut-off; diamonds = cut-offs chosen on training OOF".
Source line: results/threshold_vs_model_saving.csv (cut-offs on training OOF; paired bootstrap B = 1,000 on test rows, reporting only; costs = totals over 8,990 test clients), aggregate_summary.json, cost_sensitivity.csv
Notes EN: At 5:1, logistic at the default 0.5 costs 7,063 units. Moving the cut-off to 0.15 — chosen on training out-of-fold scores, near the closed form one over one plus five — saves 1,937 units, interval 1,604 to 2,218. Switching to XGBoost at its own cut-off saves a further 145, interval minus 18 to 299: it includes zero. Five to one is one point of a 1:1 to 20:1 sweep: the threshold saving grows with the ratio, the model saving stays in the low hundreds, and at 20:1 the optimal cut-off is 0.01 — flag almost everyone. Fix the cut-off from the cost ratio first; the model comes second.
Notes 中文: 5:1 下，逻辑回归用默认 0.5，成本 7,063 个单位。把阈值移到 0.15——在训练集折外分数上选的，接近闭式解 1/(1+5)——节省 1,937 个单位，区间 1,604 到 2,218。再换成 XGBoost 用它自己的阈值，只多省 145，区间负 18 到 299，含零。5:1 只是 1:1 到 20:1 扫描中的一个点：阈值节省随比率增长，换模型的节省始终停在一两百，到 20:1 最优阈值是 0.01——几乎标记所有人。先按成本比率定阈值，选模型是第二位。
Hand-over: "LU closes with what this means for the bank and what we would do next."

---

## Slide 13 — Business: a map for policy, a cut-off for decisions

Section: Business & Future
Presenter: LU Xinyu
Time: 0:48 (112 words)
Claim line: Five stable K-means segments on 10 PCs add no AUC: a map, not a score.
Body: (two labelled groups; the group labels are small headings, not bullets)
**Recommendation**
- Deploy XGBoost; cut-off t ≈ 1/(1+r) from the bank's own FN:FP ratio
- At 5:1, t = 0.15 flags 49%: cost 6,642 → 4,981
**Secondary analysis — PCA + K-means, training rows only**
- 10 PCs = 81.8% variance; K = 5, silhouette 0.269 (weak-to-moderate)
- Stable: bootstrap ARI 0.957 [0.745, 0.990] (K-sweep 0.941); Jaccard 0.79–0.97
- Extra predictor (×10⁻³): ΔAUC +0.1 [−0.4, +0.7], ΔAP −0.3 [−1.6, +1.0]; not adopted
Figure: figures/slides/s_cluster_stability.png — RIGHT panel only (bootstrap ARI and least-stable-cluster Jaccard by K), cropped in Canva to about 40 % width beside the secondary block so the recommendation has equal weight. Caption: "K-sweep stability by K; at K = 5: ARI 0.941, Jaccard 0.768" — it names the plotted orange point (results/kmeans_selection_repro.csv row K = 5: boot_ari_mean 0.9413, min_cluster_jaccard 0.7679). Bullet 4 keeps the dedicated K = 5 run with B = 100 (ARI 0.957 [0.745, 0.990]; Jaccard 0.791–0.974; results/unsupervised_summary_repro.json). If asked: the sweep uses a smaller bootstrap draw per K (kmeans_selection_repro.csv), which is why its point sits just below the dedicated-run range.
Source line: results/threshold_vs_model_saving.csv (t_xgb 0.15, share_flagged_xgb 0.488, cost_xgb_tuned 4,981), aggregate_summary.json (XGBoost cost 6,642 at 0.5; per 8,990 test clients), unsupervised_summary_repro.json (extra-predictor test: 5×5 repeated CV on training rows, Nadeau–Bengio interval; ×10⁻³ on the slide), kmeans_selection_repro.csv (RUAN's 30 Sep analysis reproduced 21 Oct)
Notes EN: Recommendation: rank with XGBoost; set the cut-off from the bank's own cost ratio, about one over one plus r. At 5:1 that is 0.15: it flags 49 per cent of clients and takes XGBoost's test cost from 6,642 to 4,981. Re-estimate the ratio and recalibrate quarterly; explain with logistic or the 17-rule tree. Clustering is secondary. Ten principal components keep 81.8 per cent of variance; K equals 5 has the best silhouette, 0.269, and is stable: bootstrap ARI 0.957, 0.941 in the sweep plotted. As an extra predictor the label moves AUC by one ten-thousandth, interval across zero: by the pre-fixed rule, not used. A map for policy, not a better model.
Notes 中文: 先说建议：用 XGBoost 排序，阈值按银行自己的成本比率定，约等于 1/(1+r)。5:1 下就是 0.15：标记 49% 的客户，XGBoost 的测试成本从 6,642 降到 4,981。每季度重估比率并重校准；用逻辑回归或 17 条规则的树解释。聚类是次要分析。10 个主成分保留 81.8% 的方差；K = 5 的轮廓系数最高，0.269——弱到中等——而且稳定：自助 ARI 0.957，图上的 K 扫描是 0.941。把簇标签当额外预测变量，AUC 只变万分之一，区间跨零，按预先定好的规则不采用。它是做政策的地图，不是更好的模型。
Hand-over: (none — same presenter continues)

---

## Slide 14 — Limitations and next: one split, one assumed cost ratio

Section: Business & Future
Presenter: LU Xinyu
Time: 0:39 (92 words)
Claim line: Honest but narrow: one split, one seed, a 2005 Taiwan sample, an assumed 5:1.
Body:
- One split, one seed: top-three gaps ≤ 0.002, forest gap includes zero
- 5:1 assumed; anchors imply 9.6–40; sweep stops at 20:1 (0.01 flags 99.9%)
- One-month horizon, Taiwan 2005; PAY_1 is observed one month before the label
- Next: repeated splits and nested CV for every family; loss-given-default pricing
- Next: drift monitoring with periodic recalibration; per-client uncertainty intervals
Figure: figures/slides/s_threshold_vs_model.png — single panel. Caption: "Threshold saving vs model saving across 1:1–20:1, with paired-bootstrap bands".
Source line: results/cost_anchors.json (implied ratio 9.6–40), threshold_vs_model_saving.csv (ratio 20: cut-off 0.01, share flagged 0.999), auc_bootstrap_ci.csv
Notes EN: Limitations: one split, one seed — the top-three gaps, 0.002 or less, sit inside or at the edge of bootstrap noise. The 5:1 ratio is assumed; the data's bill and margin anchors imply 9.6 to 40, hence the sweep; it stops at 20:1 because the rule is already "flag everyone" there. Taiwan 2005, one-month horizon, PAY_1 observed one month before the label: early warning, not scoring from scratch. Next: repeat the split, extend nested CV to every family, price errors from loss-given-default data, monitor drift. Thank you — we welcome your questions.
Notes 中文: 局限：一次切分、一个种子——前三名之间 0.002 以内的差距都在自助噪声之内或边缘。5:1 是假设；数据里账单和利差的锚点推出 9.6 到 40，所以展示整段扫描；扫描到 20:1 为止，因为那里的规则已经是「全部标记」。2005 年台湾、一个月窗口，PAY_1 在标签前一个月观察到：这是预警，不是从零评分。下一步：重复切分、把嵌套 CV 推广到每个家族、用违约损失率数据给错误定价、做漂移监控。谢谢大家，欢迎提问。
Hand-over: (end of talk — open the floor; LI has the Q&A bank ready)

---

## Slide A1 — Appendix: nested CV, Platt reliability, nine-ratio sweep

Section: A1 appendix (not spoken; shown only if asked)
Presenter: whoever is asked (RUAN for nested CV, LI for Platt and the sweep)
Time: 0:00 (77 words, Q&A only — not in the 9:20 budget; would be 0:33 if spoken)
Claim line: Second-pass evidence: nested CV confirms the frozen choices; Platt restores calibration.
Body:
| Nested CV (training only, 30 Sep; reproduced 21 Oct) | Nested AUC (SE) | 5:1 cut-off OOF | Cross-fitted 5:1 cost | Decision |
|---|---|---|---|---|
| KNN, K = 101 uniform (5 outer folds) | 0.768 (sd 0.007) | — | — | frozen choice confirmed |
| Logistic, plain | 0.760 (0.005) | 0.17 | 12,195 | kept by the 1-SE rule |
| Logistic, ridge / lasso | 0.760 / 0.760 (0.005) | 0.17 / 0.15 | 12,205 / 12,187 | within 1 SE |
| Logistic, class-weighted | 0.761 (0.005) | 0.39 | 12,263 | mis-calibrated (mean p 0.44) |
| Logistic, weighted + Platt | 0.761 (0.005) | 0.15 | 12,247 | −0.4 % cost < 1 % rule → not adopted |
Figure: figures/slides/s_calibration_platt.png — single panel (reliability of class-weighted XGBoost and logistic before/after Platt, training OOF). Caption: "Weighted models before and after Platt scaling, training out-of-fold deciles". If a second appendix panel is allowed, add figures/threshold_vs_model_saving.png with the full nine-ratio table from results/threshold_vs_model_saving.csv.
Source line: results/nested_cv_summary_repro.json (21 Oct reproduction; all five cost cells from it — RUAN's 30 Sep logistic_nested_cv_v2.csv agrees to ±1 unit), knn_v2.json, recalibration.csv, threshold_vs_model_saving.csv
Notes EN: Not spoken. If asked about the professor's comment: the nested layer chooses hyper-parameters inside each outer fold and reproduces the plain five-fold figures within one standard error — KNN 0.768 against 0.770, logistic 0.760 against 0.760 — and the pre-registered rules re-select the configurations frozen on 16 September, so nothing was re-scored. If asked about calibration: the Platt map is fitted on training out-of-fold scores only and applied once to the frozen test scores on 19 October.
Notes 中文: 不讲，备问。若被问到教授评语：嵌套层在每个外折内部选超参数，复现出与普通五折相差不到一个标准误的数字——KNN 0.768 对 0.770，逻辑回归 0.760 对 0.760——预先登记的规则重新选中了 9 月 16 日冻结的配置，因此没有任何模型被重新打分。若被问到校准：Platt 映射只在训练集折外分数上拟合，10 月 19 日对冻结的测试分数只应用一次。
Hand-over: (none)
