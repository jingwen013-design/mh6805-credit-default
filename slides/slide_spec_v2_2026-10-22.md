# Slide spec v2 — MH6805 Group B19 final presentation (written 2026-10-22)

Deck: Canva design DAHYnjt1H8I, MH6810 template style (NTU logo top-right; title 46 px bold navy #18395b; claim line 37 px bold; body 24–27 px; grey source line 18 px #5e7182; bottom progress bar with five tabs `Data & Problem` · `Protocol` · `Models` · `Results` · `Business & Future` + page number). Presentation 27 Oct 2026 morning, 10 minutes + Q&A, all three speak; slides due 26 Oct noon; the discussion group gets the PDF in advance.

**What changed from v1 (21 Oct, 14 + A1).** The student asked for notes that explain instead of listing numbers, more of the report's figures, and more slides if needed. v2 has **21 spoken slides + 2 appendix slides**. Each slide makes one point. The notes give the conclusion first, then the reason, with at most two result numbers. The body is either a figure with a one-line caption or at most three short bullets. All content follows the rewritten report (`report/sec_*.tex`, `report/MH6805_Group_Project_Report.pdf`), `results/POST_REVIEW_CHECKS.md`, `results/*` and the verified context brief (`CONTEXT_SOURCES.md`).

**v2.1 fixes (22 Oct, after the audience/timing, fact and figure reviews).**
- Timing: notes cut from 1,321 to 1,262 words (9:01; RUAN 2:59, LU 2:52, LI 3:10); hand-over estimate corrected to about 5 s; spoken-number cost stated; 9:45 rehearsal target and a cut-first list added.
- Clarity: S12 ("predicted risk", not "score"; utilisation glossed), S11 (why the limit is not a cause; delay columns named in the caption), S16 ("slightly costlier than the plain logit"), S18 (cost unit = one needless call), S19 (one caught defaulter is worth r false alarms), S20 (no idiom, grammar fixed), S14 (why a stable ranking matters), S1 (RUAN names herself and the real split), S4 ("frozen" explained), S15 ("generative" dropped from the title; Platt defined once, on S16), S21 (columns match the three spoken limits), S3/S6 (off-thread sentences moved to Q&A), S7 (no "a third" in the notes, so it cannot be added to S9's 86 %).
- Facts: S2 "last month of our inputs"; S4 nested-CV wording (its rule re-selects, folds vary); S5 ARI 0.94 chart vs 0.957 dedicated run; S13 calibration-within-groups wording; S11 MAS FEAT source; S17 "test AUCs" source; S18 "monthly" margin; A1 "> 1 %"; A2 §2.6–2.7 and quoted wording; R2 46 %; R5; R7 0.769; L14 dates; S15 "7th of the 11 headline configurations"; S20 fallback integers.
- Figures: S9/S10 now use two separately rendered panels (s_ablation_depth.png, s_ablation_history.png; no crop); the history panel plots the window-summary series the report quotes (+0.017); S12 uses the new s_monotone.png (consistent sweep + CV price) instead of the monotone_pay1.png crop; s_cost_vs_threshold.png and s_threshold_vs_model.png re-rendered without collisions and with the scenario wording; s_fairness.png and s_pay_coding_gain.png wording fixed; palette fixes in s_leakage, s_knn_k, s_cluster_stability (0.75 line labelled) and s_pay_codes; BoE image cropped (ctx_boe_fca_2024_explainability_crop.png); figure sizes changed to fit between the claim line and the source line (see Figure sizing).

**Wording rules applied throughout** (from the report and the post-review checks):
- Never write "scored once": each configuration is **frozen at its first test scoring**, and the test set was read more than once. Every read is dated in `results/TEST_SET_LOG.md`.
- Never write "CV winner": the three ensembles **tie**.
- Post-freeze (21 Oct) analyses are **training CV**, never test-scored, and select nothing. Their source lines say so.
- 5:1 is a **scenario borrowed from the Statlog German credit data**, not this issuer's cost.
- Class weighting + Platt makes the logit **0.4 % costlier** (so it was not adopted).
- Use "predict / rely on / associated with", not "drive". The bank sets the limit itself.
- British spelling. AUC to 3 decimals, costs as integers, no 4-decimal number on any slide.
- Titles are given in sentence case for readability. If the deck keeps v1's Title Case, apply it to every slide.

**Counting conventions.** "Words" = tokens in Notes EN that contain a letter or digit. Time = words ÷ 140 wpm (130 wpm shown for the presenter check). Hand-over lines are spoken but not counted; allow about 5 s each. Spoken numbers take longer than one token each (see the note under the running order). The "at most two numbers" limit counts result numbers in the notes. It does not count the scenario label (five to one), counts of models, folds or months, or dates. Numbers in Notes 中文 match Notes EN exactly. Claim lines are ≤ 95 characters so they stay on one line.

**Story for classmates (8 sentences, from the report).**
1. Our 29,965 Taiwanese cardholders were observed just as the 2005–06 card-debt crisis began. That makes this a crisis vintage: 22.1 % default next month, and "nobody defaults" is already 77.9 % accurate, so we judge ranking and cost, not accuracy.
2. Repayment behaviour beats demographics because next month's default is mostly the next step of a delinquency process: adding September's record lifts XGBoost from 0.63 to 0.77 CV AUC, while sex, age, marriage and education together add only 0.002.
3. Each family lands where its assumptions put it. KNN with about a hundred neighbours is the benchmark (0.770). The logit trails mainly because it reads repayment codes as evenly spaced numbers; recoded by meaning it reaches 0.768, level with KNN. The forest, gradient boosting and XGBoost tie at the top (0.785–0.786), and most of their edge is one-variable shape: interactions add only 0.004. LDA and naive Bayes rank close but give distorted probabilities.
4. Every choice was made by cross-validation on training rows. Each configuration was frozen at its first test scoring, and every read of the test set is dated, so the test set is an honest if retrospective report card: XGBoost scores 0.789 there.
5. Re-weighting moves the intercept, not the ranking; resampling before the split fakes a 0.914 AUC. So weight the decision, not the data.
6. The cut-off matters more than the model. In the borrowed 5:1 scenario, moving the logit's cut-off saves 1,937 cost units, while switching to XGBoost saves only 145 more [−18, 299]. The cut-off should follow the bank's own cost ratio, and the cheapest model keeps changing as that ratio moves.
7. A one-line "any delay" rule gets most of the way (XGBoost costs 7.7 % less). Dropping personal attributes trades one fairness gap for another, so the bank must choose and document.
8. Limits: one bank at the onset of a crisis, one split, a test set read more than once and a borrowed cost ratio. Re-base before any reuse.

---

## Running order (words = Notes EN; time = words ÷ 140 wpm)

| # | On-slide title | Section | Presenter | Status | Words | Time | Cumulative |
|---|---|---|---|---|---|---|---|
| S1 | Card default: the cut-off matters more than the model | Data & Problem | RUAN Hangjing | kept v1 #1 | 65 | 0:28 | 0:28 |
| S2 | Our data come from Taiwan's 2005 card-debt crisis | Data & Problem | RUAN Hangjing | new (layout v1 #3) | 64 | 0:27 | 0:55 |
| S3 | Repayment codes are states, not a number line | Data & Problem | RUAN Hangjing | kept v1 #3 | 57 | 0:24 | 1:20 |
| S4 | Every choice made on training folds; test only reports | Protocol | RUAN Hangjing | kept v1 #4 | 66 | 0:28 | 1:48 |
| S5 | Segments describe the book; they do not price risk | Models | RUAN Hangjing | kept v1 #13 | 56 | 0:24 | 2:12 |
| S6 | The KNN benchmark needs about a hundred neighbours | Models | RUAN Hangjing | kept v1 #5 | 43 | 0:18 | 2:30 |
| S7 | Why the logit trails: coding first, shape second | Models | RUAN Hangjing | new (layout v1 #5) | 66 | 0:28 | 2:59 |
| S8 | Pruned tree ties KNN; ensembles tie at the top | Models | LU Xinyu | kept v1 #6 | 67 | 0:29 | 3:27 |
| S9 | XGBoost's edge is mostly shape, not interactions | Models | LU Xinyu | new (layout v1 #7) | 71 | 0:30 | 3:58 |
| S10 | Last month's record carries most of the signal | Results | LU Xinyu | new (layout v1 #7) | 51 | 0:22 | 4:20 |
| S11 | Arrears push scores up; a larger limit pulls down | Results | LU Xinyu | kept v1 #7 | 71 | 0:30 | 4:50 |
| S12 | Monotone constraints: promise only what credit logic guarantees | Results | LU Xinyu | new (layout v1 #3) | 70 | 0:30 | 5:20 |
| S13 | Personal attributes add little, but force a fairness choice | Results | LU Xinyu | new (layout v1 #3) | 72 | 0:31 | 5:51 |
| S14 | Four tiers hold in every fold; top three tie | Results | LI Jingwen | kept v1 #11 | 51 | 0:22 | 6:13 |
| S15 | Four more families: close in ranking, not in probabilities | Results | LI Jingwen | kept v1 #8 | 46 | 0:20 | 6:33 |
| S16 | Class imbalance: weight the decision, not the data | Results | LI Jingwen | kept v1 #9 | 63 | 0:27 | 7:00 |
| S17 | Leakage control: SMOTE before the split fakes 0.914 AUC | Results | LI Jingwen | kept v1 #10 | 47 | 0:20 | 7:20 |
| S18 | The cut-off matters more than the model | Business & Future | LI Jingwen | kept v1 #12 | 69 | 0:30 | 7:49 |
| S19 | The bank's own cost ratio should set the cut-off | Business & Future | LI Jingwen | new (layout v1 #3) | 59 | 0:25 | 8:15 |
| S20 | A one-line rule gets most of the way | Business & Future | LI Jingwen | new (layout v1 #3) | 53 | 0:23 | 8:37 |
| S21 | Limits: one crisis vintage, a re-read test set, borrowed costs | Business & Future | LI Jingwen | kept v1 #14 | 55 | 0:24 | 9:01 |
| A1 | Appendix · Nested CV, Platt, Cost Sweep, Settings and References | appendix (not spoken) | whoever is asked | kept v1 A1 | — | — | — |
| A2 | Appendix · Cost sweep and real-world context | appendix (not spoken) | whoever is asked | new (layout v1 A1) | — | — | — |

**Spoken total (S1–S21): 1,262 words = 9:01 at 140 wpm** (cap 1,330 words ≈ 9:30; at 130 wpm 9:42). That is not the run time. Read aloud, the 22 numbers in the notes add about 55–60 words ("0.789" is "zero point seven eight nine"; "1,937" is six words), about 25 s; the three hand-over lines take about 5 s each (≈ 15 s); 20 slide changes add a few seconds more. Realistic run time ≈ 9:41 at 140 wpm before slide changes. At 130 wpm it would be about 10:24, over the limit, so the presenters need to hold about 140 wpm (or use the cut-first list); saying decimals without the leading zero ("point seven eight nine") saves about 5 s. So: (1) hold one timed full rehearsal with the clicker and hand-overs before 26 Oct and aim for **9:45**; (2) if it runs over, drop the sentences in the cut-first list below, in order (≈ 12 s), before cutting anything else.

| Presenter | Slides | Words | Time | Owns (report) |
|---|---|---|---|---|
| RUAN Hangjing | S1–S7 | 417 | 2:59 (3:12 at 130 wpm) | data, KNN, logistic family, nested CV, clustering |
| LU Xinyu | S8–S13 | 402 | 2:52 (3:06 at 130 wpm) | trees, forest, boosting, SHAP, monotone, depth/history ablations; fairness (balance) |
| LI Jingwen | S14–S21 | 443 | 3:10 (3:24 at 130 wpm) | SVM, MLP, LDA/QDA/NB, imbalance, leakage, Platt, freeze, ranking, cost/operating point; simple rule (balance) |

**Cut-first list (only if the timed rehearsal runs past 9:45; ≈ 29 words, ≈ 12 s).** In this order; no LU cuts, so LU stays above 2:50:
1. S21: "So rank with any ensemble, and set the cut-off from costs." (11; the claim line says it).
2. S2: "We judge ranking, and the cost of mistakes." (8; S4 and S18 make the point).
3. S5: ", where one would mean cleanly separated" (6).
4. S16: ", the dotted lines" (3).
Make the same cut in Notes 中文.

Hand-overs (spoken, not counted; about 5 s each):
- after S7, RUAN → LU: "Tree models can learn what a straight line misses. LU, over to you."
- after S13, LU → LI: "LI will now line up the models and turn scores into decisions."
- after S21, LI closes: "Thank you — we welcome your questions."

**Why this split of slides.**
- Each presenter speaks one contiguous block. The order is RUAN → LU → LI, with two hand-overs, after S7 and after S13.
- **Fairness (S13) goes to LU.** It is the last paragraph of the report's Interpretation section, which LU's block follows. It reuses LU's demographics ablation (the same out-of-fold vectors) and LU's XGBoost. Without it LU would speak about 2:30 and LI about 3:50.
- **The simple-rule benchmark (S20) goes to LI.** It belongs to the report's operating-point section, which LI owns: it uses the cross-fitted 5:1 cost machinery. Keeping it with LI also keeps the Business & Future block in one pair of hands.
- **Protocol (S4) stays with RUAN, as in v1.** It comes right after RUAN's data slides and includes RUAN's nested CV. LI's freeze work is credited in the source line, and LI answers freeze questions in Q&A.

## v1 → v2 slide map (for duplicating layouts in Canva)

v1 numbers are the slides of `slides/MH6805_B19_Presentation.pdf`. If template pages 1–3 are still at the front of the Canva design, v1 slide k is Canva page k + 3.

Layout types in v1:
- title: v1 #1
- tiles: v1 #2, #12
- wide figure + text lines: v1 #3
- diagram: v1 #4
- bullets left, figure right: v1 #5
- three boxes + figure: v1 #6
- figure left, bullets right: v1 #7
- numbered table: v1 #8
- two text blocks + figure: v1 #9
- text column + figure: v1 #10
- large figure + four tiles: v1 #11
- figure panel + bullet box: v1 #13
- three columns: v1 #14
- appendix text: A1

**Duplicate before you edit.** Three kept slides lend their layout to new ones, so copy them first:
- v1 #3 (wide figure): make 5 copies, for S2, S12, S13, S19 and S20; then edit the original into S3.
- v1 #7 (figure left, text right): make 2 copies, for S9 and S10; then edit the original into S11.
- v1 #5 (bullets left, figure right): make 1 copy, for S7; then edit the original into S6.
- A1: make 1 copy, for A2.
Then reorder the pages to S1–S21, A1, A2 and delete v1 #2.

| v2 | Status | From v1 | What happens |
|---|---|---|---|
| S1 | KEPT | #1 title | new title and claim; title-slide figure swapped to the Taipei Times headline; presenting-order line updated |
| S2 | NEW | layout from #3 (wide figure) | Taiwan revolving-balance chart |
| S3 | KEPT | #3 EDA | rewritten: figure swapped to s_pay_codes.png; 67.6 % callout and two text lines deleted |
| S4 | KEPT | #4 protocol diagram | wording fixes ("frozen at first scoring", dated log, post-freeze line) |
| S5 | KEPT | #13 clusters | recommendation bullets deleted; becomes the clustering slide |
| S6 | KEPT | #5 KNN + linear | bullets deleted (logit content moves to S7); figure enlarged |
| S7 | NEW | layout from #5 (bullets left, figure right; figure only) | pay-coding gain |
| S8 | KEPT | #6 tree family | "CV WINNER" removed; three boxes rewritten; figure and banner deleted |
| S9 | NEW | layout from #7 (figure left, text right) | depth ablation (s_ablation_depth.png) |
| S10 | NEW | layout from #7 | history ablation (s_ablation_history.png) |
| S11 | KEPT | #7 interpretation | figure swapped to shap_beeswarm.png; five bullets deleted |
| S12 | NEW | layout from #3 (wide figure) | monotone check (s_monotone.png) |
| S13 | NEW | layout from #3 (wide figure) | fairness |
| S14 | KEPT | #11 ranking | figure swapped to s_fold_stability.png; four tiles deleted |
| S15 | KEPT | #8 table | four rows → three; title "only the network clears the benchmark" removed; new title without "generative" |
| S16 | KEPT | #9 imbalance | figure swapped to s_calibration_platt.png; text blocks deleted |
| S17 | KEPT | #10 leakage | left text column and italic line deleted; figure enlarged |
| S18 | KEPT | #12 operating point | four tiles deleted; figure enlarged |
| S19 | NEW | layout from #3 (wide figure) | cut-off by cost ratio |
| S20 | NEW | layout from #3 (wide figure) | one-line rule |
| S21 | KEPT | #14 three columns | columns become VINTAGE · TEST RECORD · COSTS, one line each; "XGBoost CV 0.786 … top three tied" removed |
| A1 | KEPT | A1 appendix | wording fixes; post-freeze line and references added |
| A2 | NEW | layout from A1 (appendix) | 2×2 grid: cost sweep + three context images |
| — | RETIRED | #2 data tiles | its numbers move to S2's claim line; delete the slide |

**Figure sizing (all slides; measured on the v1 PDF, 22 Oct).** Title y 56–112; a one-line claim y 143–196 (a two-line claim reaches 242); grey source line from y 955; progress bar from y 1032. Every claim line in this spec is ≤ 95 characters, so it stays on one line at 37 px. That leaves about 650 px for figure plus caption:
- wide figures (1920 × 960 renders): about 1300 × 650 px, centred, top edge at y ≈ 215; caption, one line of 22–24 px, at y ≈ 880–910, clear of the source line;
- figure-left layouts (S9, S10, S11): figure up to 700 px tall, top edge at y ≈ 215 (bottom ≤ 915); the caption goes in the right column, not under the figure;
- S1 headline image and S2 chart as stated on their slides (S2: about 1400 × 645, top edge at y ≈ 215, caption at y ≈ 880);
- A2: four cells of at most 900 × 400 (see A2).
At 1300 px wide, the 15–17 pt text of a 1920-px render (dpi 150) comes out at about 21–24 px, and 14 pt legends at about 20 px, so no text falls below 16 px.

**Fallbacks.** All seven new `s_*.png` renders were delivered and checked against this spec on 22 Oct; the fixer pass the same day re-rendered or added the files listed under "v2.1 fixes" above. The fallbacks only apply if a later re-render breaks one. Each slide's Figure field names a report figure, or a small table built from the same results file.

---
## S1 — Card default: the cut-off matters more than the model

Status: KEPT from v1 #1 (title). Text changes: replace the title "Predicting Credit-Card Default: Eleven Classifiers, One Protocol" and the claim. Swap the right-hand figure (s_auc_headline) for the Taipei Times headline. Change the presenting-order line. Keep the names, student IDs, module/date lines and the contribution line exactly as in v1 (it stays marked "draft, to confirm" until RUAN and LU confirm).
Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:28 (65 words)
Title: Card default: the cut-off matters more than the model
Claim line: Ten classifiers compared on training CV; three ensembles tie; the frozen test set only reports.
Body (title block, as v1):
- MH6805 Machine Learning in Finance · Group B19 · 27 October 2026
- RUAN Hangjing G2607546F · LU Xinyu G2603537D · LI Jingwen G2607526K
- Presenting: RUAN 1–7 · LU 8–13 · LI 14–21
(The presenting line is set small, under the v1 contribution line.)
Figure: figures/context/ctx_news_taipeitimes_2006-09-29.png. Place it right of the name block at about 820 × 300 px (keep the aspect ratio of 1500 × 546; no crop). Caption: "Taipei Times, 29 Sep 2006: the crisis our data come from".
Source line: Taipei Times, 29 Sep 2006, p. 12, https://www.taipeitimes.com/News/biz/archives/2006/09/29/2003329739 · UCI Default of Credit Card Clients (Yeh & Lien 2009), doi:10.24432/C55S3H, CC BY 4.0 · results/aggregate_summary.json
Notes EN: Good morning, we are Group B19. Our question: which cardholders will miss next month's payment, and what should the bank do? Our short answer: where the bank sets its cut-off, the line for flagging clients, matters more than which model it uses. I'm RUAN: data, protocol and the simpler models; LU: the trees and what they rely on; LI: the comparison and the bank's decision.
Notes 中文: 早上好，我们是 B19 组。我们的问题是：哪些持卡人下个月会逾期还款，银行又该怎么办？一句话答案：银行把判定线——也就是标记客户的那条线——划在哪里，比用哪个模型更重要。我是 RUAN：讲数据、实验规程和较简单的模型；LU：树模型以及模型依赖哪些输入；LI：模型比较和银行的决策。
Hand-over: (none — same presenter continues)
If asked: "Why ten classifiers?" — Ten families in the report's headline table; the lasso logit ties the plain logit and is not shown. The project has 46 frozen configurations in all (results/frozen_manifest.json).

---

## S2 — Our data come from Taiwan's 2005 card-debt crisis

Status: NEW. Layout from v1 #3 (wide figure + text lines). Duplicate v1 #3, delete the 67.6 % callout and its two text lines, and put the figure in.
Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:27 (64 words)
Title: Our data come from Taiwan's 2005 card-debt crisis
Claim line: 29,965 cardholders; 22.1% default next month; "nobody defaults" is 77.9% right, catches none.
Body: figure + caption.
Figure: figures/context/ctx_taiwan_card_revolving.png (1920 × 883), full image, about 1400 × 645 px, centred. Caption: "Taiwan's card revolving balances peaked in Nov 2005, then fell 49% by 2008".
Source line: FSC Banking Bureau, credit-card statistics (index-7.xlsx, OGDL v1.0), https://www.banking.gov.tw/webdowndoc?file=/stat/cc/index-7.xlsx · CBC Financial Stability Report, Jun 2008, Box 2, https://www.cbc.gov.tw/en/public/Attachment/931716211071.pdf · results/eda_summary.json
Notes EN: Our data come from a crisis: one Taiwanese bank's cardholders in 2005, as Taiwan's card-debt bubble burst; the central bank dates the first repayment problems to September 2005, the last month of our inputs. About 22 per cent default next month, so accuracy is useless: "nobody defaults" is 78 per cent right and catches no one. We judge ranking, and the cost of mistakes.
Notes 中文: 我们的数据来自一场危机：台湾一家银行 2005 年的持卡人，正赶上台湾卡债泡沫破裂；台湾央行把还款问题最早浮现的时间定在 2005 年 9 月，正是我们输入变量的最后一个月。下个月约 22% 的客户违约，所以准确率没有意义：「没人违约」就有 78% 的准确率，却一个违约者也抓不到。我们看的是排序能力，以及犯错的代价。
Hand-over: (none — same presenter continues)
If asked: "Is 22 % a normal default rate?" — No. US card delinquency ranged 1.53–6.77 % over 1991–2026 (Federal Reserve, a US analogy). Our rate fits a stressed book or an undocumented, defaulter-heavy sample (report §1, §7 iii).

---

## S3 — Repayment codes are states, not a number line

Status: KEPT from v1 #3 (EDA), rewritten:
- new title and claim;
- figure s_eda_gradients.png → s_pay_codes.png;
- delete the "67.6%" callout, its explanation line and the two lines under it;
- source line updated.
Section: Data & Problem
Presenter: RUAN Hangjing
Time: 0:24 (57 words)
Title: Repayment codes are states, not a number line
Claim line: Two months late: 67.6% default; "paid in full" defaults more than "revolving" (16.9% vs 12.9%).
Body: figure + caption.
Figure: figures/slides/s_pay_codes.png (new; render of 22 Oct checked against this spec), full width, about 1300 × 650 px. Caption: "September default rate by status code, training rows: −2, −1 and 0 have no order". Fallback: figures/slides/s_eda_gradients.png, LEFT panel only (crop at x = 960), with the same caption.
Source line: Training rows only (n = 20,975); results/pay_code_default_rates.csv, eda_summary.json; codes −2 and 0 are not defined in the UCI codebook
Notes EN: The strongest signal is the monthly repayment status: two months late, two-thirds default next month. But the codes are labels, not a scale: minus two is no spending, minus one paid in full, zero revolving, that is carrying a balance, and paid-in-full clients default more than revolving ones. So reading the codes as evenly spaced numbers misleads.
Notes 中文: 最强的信号是每个月的还款状态：逾期两个月的客户，下个月有三分之二违约。但这些代码是标签，不是刻度：−2 是没有消费，−1 是全额还款，0 是循环信用、也就是带着余额，而全额还款的客户违约率反而高于循环信用的客户。所以把这些代码当成等距数字来读会误导。
Hand-over: (none — same presenter continues)
If asked: "Do sex and age matter here?" — Much less: they move the default rate by at most 8 points (results/eda_summary.json); slide 13 has the ranking test. "What is odd about code 1?" — It appears 2,540 times in September but only 21 times in the five earlier months. It mixes dormant accounts with clients recovering from two months late, so compare codes within a month (results/pay_code_default_rates.csv).

---

## S4 — Every choice made on training folds; test only reports

Status: KEPT from v1 #4 (protocol diagram). Text changes:
- (a) Title: was "Protocol: Chosen on Training CV, Frozen, Scored Once on Test".
- (b) Claim line: as below.
- (c) Test box: "TEST SET 8,990 CLIENTS — SCORED ONCE PER FROZEN CONFIGURATION, THEN READ-ONLY" becomes "TEST SET 8,990 CLIENTS — EACH CONFIGURATION FROZEN AT FIRST SCORING · EVERY READ DATED".
- (d) Small text under the padlock: "register since 19 Oct; earlier reads dated in TEST_SET_LOG.md".
- (e) "TRAINING ROWS DECIDE EVERYTHING" box, second line: "round 1 scored 16–18 Sep, registered 19 Oct · 2 re-fits 30 Sep · 9 Platt 19 Oct = 46 entries".
- (f) Nested-CV side box: change to "Nested CV (30 Sep, RUAN): almost the same AUC; its rule re-selects the frozen KNN and logit settings" and add the line "Post-freeze checks (21 Oct): training CV only; select nothing".
- (g) Keep the bottom rule line as it is.
Section: Protocol
Presenter: RUAN Hangjing
Time: 0:28 (66 words)
Title: Every choice made on training folds; test only reports
Claim line: Five-fold training CV picks settings and cut-offs; each model frozen at its first test scoring.
Body: the v1 diagram (five boxes + padlock + nested-CV side box) with the text changes above.
Figure: none (Canva diagram; in Canva it is one picture, re-rendered from HTML on 22 Oct with every label at 22 px or more).
Source line: results/frozen_manifest.json (46 entries), TEST_SET_LOG.md (every test read, dated), nested_cv_summary_repro.json; code/freeze.py --check passes (freeze machinery: LI)
Notes EN: We made every choice on training data; the test set only reports. We locked away 30 per cent of clients as a test set, and chose settings and cut-offs by cross-validation on the rest: train on four folds, check on the fifth, rotate. Each model was frozen at its first test scoring, meaning its settings never changed afterwards, and a dated log lists every test read.
Notes 中文: 所有选择都在训练数据上做，测试集只负责报告。我们把 30% 的客户锁起来做测试集，超参数和判定线都在其余数据上用交叉验证来选：四折训练、一折检验、轮换进行。每个模型在第一次测试打分时冻结，也就是此后设置再也不改；一份带日期的日志记下每一次读取测试集。
Hand-over: (none — same presenter continues)
If asked: "Was the test set really untouched?" — No, it was read more than once. Until 19 Oct, test AUC chose the four models shown in the report's figures, and nine Platt variants were scored after a 6 Oct review. Every read is dated in results/TEST_SET_LOG.md, so treat the test set as a dated report card. "Did nested CV pick the same settings?" — Nested cross-validation, which re-tunes inside each fold, scores almost the same (KNN 0.768 against 0.770), and its pre-registered rule, applied to all training rows, still selects K = 101 and the plain logit. The outer folds themselves chose K = 101 uniform in only 2 of 5, so do not say every fold picked the same (results/nested_cv_summary_repro.json, nested_cv_knn_frozen_grid.json).

---

## S5 — Segments describe the book; they do not price risk

Status: KEPT from v1 #13. Text changes:
- new title and claim;
- delete the "WHAT WE FOUND" bullet box and the bottom line "A map for policy, not a better model: rank with XGBoost …" (that recommendation is now on S21);
- enlarge the figure to full width.
Section: Models
Presenter: RUAN Hangjing
Time: 0:24 (56 words)
Title: Segments describe the book; they do not price risk
Claim line: K-means finds five overlapping segments (silhouette 0.27); the label adds no measurable AUC.
Body: figure + caption.
Figure: figures/slides/s_cluster_stability.png, BOTH panels, about 1300 × 650 px. Caption: "Silhouette (left) and bootstrap stability (right) by K, training rows; dashed line = usual stability bar 0.75". (Re-rendered 22 Oct: metric lines in ink, not model-family colours; the 0.75 line is labelled.)
Source line: Training rows only (RUAN's 30 Sep analysis, reproduced 21 Oct); results/unsupervised_summary_repro.json, kmeans_selection_repro.csv, cluster_incremental_folds_repro.csv
Notes EN: Segments describe clients but do not improve prediction. K-means, grouping clients by similarity, finds five segments that overlap: silhouette 0.27, where one would mean cleanly separated. Their default rates differ mainly because the inputs include arrears, and the segment label adds nothing measurable to logistic regression. Segments are a map for policy, not a better score.
Notes 中文: 分群能描述客户，但不能改进预测。K-means 按相似度给客户分组，找到五个彼此重叠的群：轮廓系数 0.27，1 才代表分得很开。各群违约率不同，主要是因为输入里本来就有逾期变量；把群标签加进逻辑回归，看不出任何可测的改进。分群是做政策的地图，不是更好的评分。
Hand-over: (none — same presenter continues)
If asked: "How stable are the segments?" — Bootstrap ARI at K = 5 is about 0.94–0.96: the chart's grid run shows 0.94, and a dedicated 100-resample run gives 0.957 [0.745, 0.990]. Across random starts it falls to a minimum of 0.58. The label adds +0.0001 [−0.0004, +0.0007] AUC in 5 × 5 repeated training CV, so it was not adopted and never test-scored (results/unsupervised_summary_repro.json).

---

## S6 — The KNN benchmark needs about a hundred neighbours

Status: KEPT from v1 #5. Text changes:
- new title and claim;
- delete all five bullets and the bottom "Rule: …" line (the logit content moves to S7);
- enlarge s_knn_k.png to full width.
Section: Models
Presenter: RUAN Hangjing
Time: 0:18 (43 words)
Title: The KNN benchmark needs about a hundred neighbours
Claim line: K = 101 reaches CV AUC 0.770; nested CV, re-tuning inside each fold, gives 0.768.
Body: figure + caption.
Figure: figures/slides/s_knn_k.png, about 1300 × 650 px. Caption: "CV AUC rises with K and levels off near K = 101 (training folds)". (Re-rendered 22 Oct: distance weights now dashed green with open squares, not the forest colour.)
Source line: Training 5-fold CV; results/knn.json, knn_k5.json, nested_cv_knn_frozen_grid.json, knn_dimensionality_cv.csv
Notes EN: Our benchmark is K-nearest neighbours: a client's risk is how often her most similar clients defaulted. It needs about a hundred neighbours, because default is noisy and a small vote is luck. Its AUC, the chance a defaulter outranks a payer, is 0.77.
Notes 中文: 我们的基准是 K 近邻：一个客户的风险，就是和他最相似的那些客户里有多少人违约。它需要大约一百个邻居，因为违约很「吵」，邻居太少的投票靠运气。它的 AUC——随机一个违约者排在随机一个正常还款者前面的概率——是 0.77。
Hand-over: (none — same presenter continues)
If asked: "Why not K = 5, as in class?" — CV AUC is 0.706 at K = 5 against 0.770 at K = 101. Compressing the inputs to 15 principal components gives 0.768, so there is no gain (results/knn.json, knn_dimensionality_cv.csv; training CV).

---

## S7 — Why the logit trails: coding first, shape second

Status: NEW. Layout from v1 #5 (bullets left, figure right). Delete the bullets and enlarge the figure to full width.
Section: Models
Presenter: RUAN Hangjing
Time: 0:28 (66 words)
Title: Why the logit trails: coding first, shape second
Claim line: Coding status by meaning lifts the logit from 0.760 to 0.768, level with KNN (training CV).
Body: figure + caption.
Figure: figures/slides/s_pay_coding_gain.png (new; render of 22 Oct checked against this spec), about 1300 × 650 px. Caption: "CV AUC, codes as numbers vs read by meaning: only the linear models gain". (Re-rendered 22 Oct: the KNN line now reads "recoded logit level with it (within noise)", matching the claim line.) Fallback: a 4-row Canva table built from results/pay_coding_cv.csv, columns "codes as numbers → by meaning": logit 0.760 → 0.768; LDA 0.757 → 0.765; KNN 0.770 → 0.768; MLP 0.776 → 0.773.
Source line: Post-freeze training CV, never test-scored; results/pay_coding_cv.csv, pay_coding_summary.json
Notes EN: The logistic regression, our transparent scorecard, reads each status code as one evenly spaced number. Coded by meaning instead, months late plus flags for "no spending" and "paid in full", it catches up with KNN in training cross-validation. KNN and the network gain nothing; they already treat each code as its own group. A cheap upgrade, but most of the gap to the tree models remains.
Notes 中文: 逻辑回归是我们的透明评分卡，它把每个还款代码当成一个等距的数字来读。按含义重新编码——逾期月数，加上「无消费」和「全额还款」两个标记——它在训练集交叉验证上追平了 KNN。KNN 和神经网络本来就把每个代码当作单独的一组，所以没有提升。这是个便宜的升级，但与树模型的差距大部分还在。
Hand-over: RUAN: "Tree models can learn what a straight line misses. LU, over to you." / 中文：「树模型能学到直线学不到的东西。LU，交给你。」
If asked: "Does a one-hot logit close the gap?" — No. Free to give each code any risk, it still trails XGBoost by 0.017 CV AUC in 5/5 folds, so the remaining shape sits in other columns (results/pay_coding_cv.csv). "How much of the gap does recoding close?" — About a third (31 %) of XGBoost's lead over the plain logit. The 86 % on slide 9 includes that part, so the two do not add up (results/pay_coding_summary.json, ablation_depth.csv).

---

## S8 — Pruned tree ties KNN; ensembles tie at the top

Status: KEPT from v1 #6. Text changes:
- new title and claim;
- rewrite the three box headings and texts as below; "XGBOOST — CV WINNER" becomes "BOOSTING";
- delete the learning-curve figure and the bottom banner "XGBoost − forest … XGBoost kept as CV winner, forest as fallback";
- widen the three boxes into three equal columns.
Section: Models
Presenter: LU Xinyu
Time: 0:29 (67 words)
Title: Pruned tree ties KNN; ensembles tie at the top
Claim line: Forest, gradient boosting and XGBoost reach CV AUC 0.785–0.786, about 0.025 above the logit.
Body (three boxes, one line each):
- ONE TREE — 17 rules, first split on worst delay; ties KNN (0.769)
- RANDOM FOREST — averages 500 trees grown on resampled clients: errors cancel
- BOOSTING — adds small trees, each fixing the errors left
Figure: figures/slides/s_tree_top.png (added in the 22 Oct polish; the learning curve stays in Q&A: figures/slides/s_xgb_learning_curve.png). Top two levels of the pruned tree, drawn from the training leaf counts in results/tree_rules.txt by make_slide_figures_v2.py: all clients 22 % default; at most 1 month late in six months 72 % of clients, 13 %; 2+ months late at least once 28 %, 46 %; then May bill ≤ NT$790 21 % vs card in use 10 %, and not 2+ late in September 33 % vs still 2+ late 68 %. Place it left at about 1090 × 652 px from y ≈ 225; the three boxes become a stacked right column (about 590 px wide). No caption needed (the figure carries its title). It illustrates "a bank can audit it line by line"; the notes are unchanged.
Source line: Training 5-fold CV; results/tree_pruned.json, tree_rules.txt, random_forest.json, gradient_boosting.json, xgboost.json, fold_stability.csv
Notes EN: A single tree, pruned to 17 if-then rules, already ties the KNN benchmark, and a bank can audit it line by line. Ensembles, many trees combined, go further: a random forest averages hundreds of trees grown on resampled clients, so errors cancel; boosting adds small trees one by one, each fixing the errors left. All three land about 0.025 above the logit and tie with each other.
Notes 中文: 一棵决策树剪枝到 17 条「如果…那么…」规则，就已经和 KNN 基准打平，而且银行可以逐条审核。集成模型——把很多棵树组合起来——走得更远：随机森林把几百棵在重抽样客户上长出来的树取平均，误差互相抵消；提升法一棵接一棵地加小树，每棵修正前面剩下的错误。三个集成模型都比逻辑回归高约 0.025，彼此打平。
Hand-over: (none — same presenter continues)
If asked: "Which ensemble would you deploy?" — Any of the three. On training folds XGBoost leads GBM by +0.0013 [−0.0005, +0.0031] and the forest by +0.0014 [−0.0027, +0.0055]. Choose on cost, speed (XGBoost fits in 0.5 s) or explainability (results/fold_stability.csv).

---

## S9 — XGBoost's edge is mostly shape, not interactions

Status: NEW. Layout from v1 #7 (figure left, text right). The figure takes about 62 % of the width (left); the right column holds only the caption, set at 26 px as a reading guide.
Section: Models
Presenter: LU Xinyu
Time: 0:30 (71 words)
Title: XGBoost's edge is mostly shape, not interactions
Claim line: One-split trees recover 86% of XGBoost's lead; interactions add only 0.004 (training CV).
Body: figure + caption.
Figure: figures/slides/s_ablation_depth.png (1200 × 700, re-rendered 22 Oct as its own file with its own y-axis title, so no crop is needed). Place it at about 1190 × 694 px, left, top edge at y ≈ 215. Caption (right column): "CV AUC by tree depth, rounds re-tuned; depth 1 means no interactions". Fallback: a 5-row Canva table from results/ablation_depth.csv — logit 0.760 · depth 1 0.783 · depth 2 0.785 · depth 3 0.786 · depth 4 0.786. Do not crop s_ablation.png: its two panels share no clean vertical cut.
Source line: Post-freeze training CV, never test-scored; results/ablation_depth.csv, pdp_summary.json; Friedman (2001)
Notes EN: Most of XGBoost's edge comes from curved one-variable effects, not interactions, where one variable changes another's effect. We tested this by limiting tree depth. A booster of one-split trees can learn any curve but no interaction, and it already recovers 86 per cent of XGBoost's lead over the plain logit. Allowing interactions adds only 0.004. So a points-based scorecard with curved bands should recover most of the gain and stay explainable.
Notes 中文: XGBoost 的优势大部分来自单个变量的弯曲效应，而不是交互效应——也就是一个变量改变另一个变量的作用。我们用限制树的深度来检验。只做一次分裂的树组成的提升模型能学任何曲线，但学不了交互，它已经找回了 XGBoost 领先普通逻辑回归幅度的 86%。允许交互只再多 0.004。所以一个按区间计分、带弯曲分段的评分卡，应该能拿回大部分收益，同时保持可解释。
Hand-over: (none — same presenter continues)
If asked: "Is 0.004 reliable?" — It is if anything an upper estimate, because depth 3 was chosen on the same folds. On the 23 raw columns the gain is 0.005, and Friedman's H for status × limit is only 0.061 (results/ablation_depth.csv, pdp_summary.json).

---

## S10 — Last month's record carries most of the signal

Status: NEW. Layout from v1 #7 (figure left, text right), as S9.
Section: Results
Presenter: LU Xinyu
Time: 0:22 (51 words)
Title: Last month's record carries most of the signal
Claim line: September's record lifts XGBoost from 0.63 to 0.77; older months add 0.017 (training CV).
Body: figure + caption.
Figure: figures/slides/s_ablation_history.png (1200 × 700, re-rendered 22 Oct as its own file; no crop). The solid line is now raw months plus summaries of the same months (H1+W1 0.769, H3+W3 0.784, H6+W6 0.786), the series the report quotes: older months add +0.013 as raw months and +0.017 with summaries. Place as on S9. Caption (right column): "CV AUC as repayment history is added: raw months only (dashed, open circles) vs plus summaries of the same months (solid)". Fallback: a 4-row Canva table from results/ablation_history.csv, columns raw months · with summaries: limit + demographics 0.630 · —; + Sep 0.769 · 0.769; + Jul–Sep 0.780 · 0.784; + Apr–Sep 0.782 · 0.786. Do not use figures/ablation_summary.png panel (b): it plots the all-six-month-summary series and 4-decimal labels.
Source line: Post-freeze training CV, never test-scored; results/ablation_history.csv, pay_code_default_rates.csv (roll rates)
Notes EN: Behaviour beats demographics because next month's default is largely the next step of a delinquency process: arrears persist or deepen. Limit and demographics alone give XGBoost 0.63; adding September's record lifts it to 0.77. Older months add a little, through six-month summaries that tell a new slip from a chronic one.
Notes 中文: 还款行为胜过人口属性，因为下个月违约很大程度上就是逾期过程的下一步：逾期会持续或加深。只用额度和人口属性，XGBoost 是 0.63；加上 9 月的还款记录，升到 0.77。更早的月份只多一点，通过六个月汇总变量起作用，用来区分偶尔失手和长期拖欠。
Hand-over: (none — same presenter continues)
If asked: "Does this travel to calmer years?" — Not as is: a crisis-era model learns crisis-speed deterioration (how fast arrears deepen) and may overstate risk in calmer years (report §5, §7 iii). "How persistent are arrears?" — 59.1 % of clients two months late in September were already two months late in August, and so were 85.9 % of those three months late (results/pay_code_default_rates.csv; training rows).

---

## S11 — Arrears push scores up; a larger limit pulls down

Status: KEPT from v1 #7. Text changes:
- new title and claim;
- figure s_importance.png → figures/shap_beeswarm.png (a report figure);
- delete all five bullets;
- the figure moves to the left (about 55 % width); the right column holds the caption only, as on S9.
Section: Results
Presenter: LU Xinyu
Time: 0:30 (71 words)
Title: Arrears push scores up; a larger limit pulls down
Claim line: SHAP splits each score into input contributions; September status ranks first on every measure.
Body: figure + caption.
Figure: figures/slides/s_shap_beeswarm.png (1350 × 1050; 22 Oct polish: the same model, training rows and 2,000-row subsample as figures/shap_beeswarm.png, redrawn by code/make_slide_shap.py with labels at about 24 px on the slide), full image, about 900 × 700 px, left, top edge at y ≈ 215. Caption (right column, 24–26 px, a reading guide for the raw column names): "TreeSHAP for XGBoost, 2,000 training clients; red = high value. PAY_1 = September status · MAX_DELAY = worst delay in six months · N_MONTHS_DELAYED = months in arrears (these three carry one signal) · BILL_AMT1 / PAY_AMT1 = September bill / payment · UTIL = bill ÷ limit · LIMIT_BAL = credit limit · MARRIAGE_2 = single · SEX_2 = female". Fallback if the beeswarm reads too small: figures/slides/s_importance.png (full width).
Source line: Training rows only; results/shap_importance.csv, shap_summary.json, rf_refit_drop_column.csv; Lundberg & Lee (2017); MAS FEAT (2018) Principle 13, https://www.mas.gov.sg/publications/monographs-or-information-paper/2018/feat; BoE & FCA (2024) §2.7, https://www.bankofengland.co.uk/report/2024/artificial-intelligence-in-uk-financial-services-2024
Notes EN: Arrears push XGBoost's scores towards default; a larger limit pushes them away. SHAP splits each client's score into contributions from each input; each dot is a client, red a high value. Two cautions: the three delay columns repeat one signal, so read them together; and this describes the model, not causes: the bank gives large limits to clients it already trusts. Still, these are reasons a bank can give a customer.
Notes 中文: 逾期把 XGBoost 的分数推向违约，额度越高则把分数往回拉。SHAP 把每个客户的分数拆成各个输入的贡献；每个点是一个客户，红色表示取值高。两点提醒：三个逾期变量重复的是同一个信号，要放在一起读；而且这描述的是模型，不是因果：银行本来就把高额度给它已经信任的客户。不过，这些正是银行可以向客户说明的理由。
Hand-over: (none — same presenter continues)
If asked: "Do the importance measures agree?" — SHAP (XGBoost) and permutation importance (forest) agree, rank correlation 0.77. Impurity importance disagrees because correlated delay columns share credit. Of six columns refitted, only September status costs over 0.001 OOB AUC (results/shap_importance.csv, rf_importance.csv, rf_refit_drop_column.csv).

---

## S12 — Monotone constraints: promise only what credit logic guarantees

Status: NEW. Layout from v1 #3 (wide figure). Figure replaced on 22 Oct: the monotone_pay1.png crop showed the fixed-counter sweep (rows that cannot exist; POST_REVIEW 1.3, 7.10), counted the −1→0 fall as wrong and printed 4-decimal test AUCs. Do not use monotone_pay1.png on any slide.
Section: Results
Presenter: LU Xinyu
Time: 0:30 (70 words)
Title: Monotone constraints: promise only what credit logic guarantees
Claim line: Frozen version also orders −2/−1/0: costs 0.002 CV AUC; lateness and limit only: almost free.
Body: figure + caption.
Figure: figures/slides/s_monotone.png (new 22 Oct, 1920 × 960, training-side files only), full image at about 1300 × 650 px, centred, top edge at y ≈ 215. Left panel: 200 training clients moved one September status step at a time with the delay counters updated (consistent sweep); bars = clients whose free-XGBoost risk falls: 19 and 194 among the not-late codes (where the data fall too, 17% → 13%), 0 at 0→1, 1→2 and 2→3, 160 and 200 in the thin 3→4 and 4→5 cells (< 100 clients). Right panel: CV AUC price against free XGBoost, frozen constraint −0.002 [all 5 folds worse], lateness-and-limit-only under 0.001 (a tie). Caption: "Left: 200 training clients, one status step worse. Right: CV AUC price of each constraint". Fallback, if the figure reads poorly, three bullets (≤ 12 words each):
- Free booster never lowers risk from revolving into arrears
- Frozen constraint also ordered −2/−1/0 and utilisation: costs 0.002
- Lateness-and-limit-only constraint costs under 0.001
Source line: Post-freeze training CV and training-row sweeps, never test-scored; results/monotone_check_consistent.csv (consistent sweep, 200 training clients), monotone_respec_cv.json; Chen & Guestrin (2016)
Notes EN: A monotone constraint fixes a direction: all else equal, falling further behind never lowers your predicted risk. From revolving credit into arrears, the free booster already obeys. Our frozen version also forced an order on "no spending", "paid in full" and "revolving", and made risk rise with utilisation, the bill over the limit; the data contradict both, and it cost 0.002. Constraining only lateness and the limit is almost free.
Notes 中文: 单调约束是固定一个方向：其他条件不变时，拖欠得越久，预测风险绝不会下降。从循环信用进入逾期这一段，不加约束的提升模型本来就遵守这一点。我们冻结的版本还强行给「无消费」「全额还款」「循环信用」排了顺序，并让风险随使用率（账单除以额度）只升不降；数据并不支持这两点，代价是 0.002。只约束逾期程度和额度几乎没有代价。
Hand-over: (none — same presenter continues)
If asked: "Does the free model ever lower a score as a client slips into arrears?" — Moving 200 training clients from revolving credit to 1, 2 and 3 months late (counters updated) never lowers a score. Its wrong-way steps lie among non-delinquent codes or in the thin 4- and 5-month cells (57 and 16 clients) (results/monotone_check_consistent.csv). "Why does the free model lower risk from −1 to 0?" — Because the data do too: paid-in-full clients default at 16.9 %, revolving ones at 12.9 % (results/pay_code_default_rates.csv).

---

## S13 — Personal attributes add little, but force a fairness choice

Status: NEW. Layout from v1 #3 (wide figure).
Section: Results
Presenter: LU Xinyu
Time: 0:31 (72 words)
Title: Personal attributes add little, but force a fairness choice
Claim line: Dropping sex, age, marriage, education costs 0.002 CV AUC; no option is fair on every measure.
Body: figure + caption.
Figure: figures/slides/s_fairness.png (new; render of 22 Oct checked against this spec), about 1300 × 650 px. Caption: "Flag-rate and default-rate ratios to the reference group at the cost-based cut-off (5:1 scenario, slide 18), training out-of-fold" (checked 22 Oct: rows men/women, married/single, under-30/30–39; series observed default, XGBoost, logit, XGBoost without demographics; title re-rendered 22 Oct to "…the logit distorts it", since the logit under-flags the young: 1.05 against 1.11). Fallback: figures/fairness_flag_rates.png, Sex and Age panels only (crop x ≈ 0–1010 px of 2025).
Source line: Post-freeze training-OOF diagnostic, not a compliance assessment; results/fairness_groups.csv, fairness_summary.json, ablation_demographics.csv; EU AI Act Annex III 5(b), https://eur-lex.europa.eu/eli/reg/2024/1689/oj; Kleinberg et al. (2017)
Notes EN: Sex, age, marriage and education barely help ranking: removing them costs XGBoost 0.002. Men default more often than women, and XGBoost's probabilities match each group, so it flags more men, including men who would repay. When groups differ in default rates, no score can match each group's rate and also make equal errors across groups. Dropping the columns narrows the sex gap but flags more under-thirties. A bank must choose, and document.
Notes 中文: 性别、年龄、婚姻和教育对排序几乎没帮助：去掉它们，XGBoost 只损失 0.002。男性违约率高于女性，XGBoost 的概率与各群体吻合，所以它标记的男性更多，其中也包括本会按时还款的男性。当各群体违约率不同时，没有一个评分能既与各群体的违约率吻合、又让各群体的误判率相等。去掉这些列能缩小性别差距，却会标记更多 30 岁以下的人。银行必须自己选择，并记录在案。
Hand-over: LU: "LI will now line up the models and turn scores into decisions." / 中文：「接下来 LI 把各个模型放在一起比较，并把分数变成决策。」
If asked: "Isn't the logit fairer?" — No. It flags men 1.36 times as often as women, against 1.18 for XGBoost and a default-rate ratio of 1.19. One SEX coefficient shifts every man's log-odds, and more of the logit's scores sit near the cut-off (results/fairness_summary.json, fairness_groups.csv). "Does dropping the columns remove the information?" — No: the limit and other inputs still carry part of it, so it is a documented trade-off, not a fix (report §5).

---

## S14 — Four tiers hold in every fold; top three tie

Status: KEPT from v1 #11. Text changes:
- new title and claim;
- figure s_auc_headline.png → s_fold_stability.png;
- delete the four numbered tiles under the figure; the figure takes their space.
Section: Results
Presenter: LI Jingwen
Time: 0:22 (51 words)
Title: Four tiers hold in every fold; top three tie
Claim line: Ensembles, network, KNN, then the logit, in every fold; XGBoost's frozen test AUC is 0.789.
Body: figure + caption.
Figure: figures/slides/s_fold_stability.png (checked 22 Oct: left = per-fold CV AUC of the six core models; right = fold-paired gap to XGBoost with 95 % interval), both panels, about 1300 × 650 px, top edge at y ≈ 215. Caption: "Per-fold training CV AUC (left) and fold-paired gap to XGBoost (right)". Fallback: figures/fold_stability.png, LEFT panel with the rank column (crop x ≈ 0–1000 px of 1764).
Source line: Training OOF of the frozen configurations; results/fold_stability.csv, fold_stability_summary.json; test (n = 8,990, reporting only): aggregate_summary.json, auc_bootstrap_ci.csv
Notes EN: The ranking holds in all five folds: ensembles first, then the neural network, then KNN, then the logit, so the gaps between tiers are not the luck of one split. Within the top three the order shuffles: a tie. On the test set XGBoost scores 0.789, close to its cross-validation score.
Notes 中文: 五折中每一折的排名都一样：集成模型第一，然后是神经网络，然后是 KNN，然后是逻辑回归，所以梯队之间的差距不是某一次切分的运气。前三名的先后在各折之间来回变：打平。在测试集上 XGBoost 得到 0.789，与交叉验证分数很接近。
Hand-over: (none — LI continues)
If asked: "Does XGBoost beat the logit on test too?" — Yes, by +0.0250 [+0.0183, +0.0314] in a paired bootstrap of 8,990 test clients (B = 500). That is reporting only; each configuration was frozen at its first scoring (results/auc_bootstrap_ci.csv).

---

## S15 — Four more families: close in ranking, not in probabilities

Status: KEPT from v1 #8 (numbered table). Text changes:
- new title (v1's "Only the Network Clears the Benchmark" over-claims; "generative" dropped because the notes never explain it) and claim;
- the table drops from four rows to three (QDA moves to Q&A);
- column headings become "MODEL · HOW IT LEARNS · CV AUC · CALIBRATION SLOPE";
- delete the small grey sub-lines under each row.
Section: Results
Presenter: LI Jingwen
Time: 0:20 (46 words)
Title: Four more families: close in ranking, not in probabilities
Claim line: Network 4th in every fold; LDA and naive Bayes too extreme (calibration slopes 0.749, 0.141).
Body (table, three rows; mechanism cells ≤ 12 words):
| Model | How it learns | CV AUC | Calibration slope (1 = honest) |
|---|---|---|---|
| Neural network (64–32) | layers of curved boundaries, stopped early | 0.776 | 0.936 |
| SVM (RBF kernel) | widest margin; Platt turns scores into probabilities | 0.764† | 0.997 |
| LDA / naive Bayes | bell-shaped amounts / independent bills: both false here | 0.757 / 0.736 | 0.749 / 0.141 |
Figure: none (table). † = mean of shared-fold OOF AUCs (tuned on 6,000 rows). The SVM row is shown, not spoken; Platt is defined once, on S16.
Source line: Training CV / OOF; results/fold_stability.csv (calibration slopes, post-freeze), mlp.json, svm_rbf.json, svm_selection_statistics.json, lda.json, naive_bayes.json
Notes EN: These models rank near the benchmark, but some give distorted probabilities. The neural network, layers of curved boundaries, is fourth in every fold. LDA assumes bell-shaped amounts, yet bills are skewed; naive Bayes treats correlated bills as independent, so their probabilities are too extreme to price.
Notes 中文: 这些模型的排序接近基准，但有的给出的概率是扭曲的。神经网络由层层弯曲的边界组成，每一折都排第四。LDA 假设金额呈钟形分布，可账单严重偏斜；朴素贝叶斯把相关的几期账单当成相互独立，所以它们的概率过于极端，不能用来定价。
Hand-over: (none — LI continues)
If asked: "Why was the SVM tuned on only 6,000 rows?" — Kernel SVM cost grows roughly with the square of the rows. On the shared folds its mean OOF AUC is 0.764 (pooled 0.761), and it ranks 7th of the 11 headline configurations (lasso logit included) under every statistic (results/svm_selection_statistics.json).

---

## S16 — Class imbalance: weight the decision, not the data

Status: KEPT from v1 #9. Text changes:
- new title and claim;
- figure s_imbalance.png (test AUCs) → s_calibration_platt.png (training OOF);
- delete "WHAT WE COMPARED", "WHAT HAPPENED" and both bottom lines;
- the figure goes full width.
Section: Results
Presenter: LI Jingwen
Time: 0:27 (63 words)
Title: Class imbalance: weight the decision, not the data
Claim line: Weights move the intercept, not the ranking; weighted logit + Platt costs 0.4% more than plain.
Body: figure + caption.
Figure: figures/slides/s_calibration_platt.png, about 1300 × 650 px. Caption: "Weighted models before (dotted) and after Platt scaling (solid), training out-of-fold".
Source line: Training OOF / CV; results/recalibration.csv, imbalance_comparison.csv, nested_cv_summary_repro.json (cross-fitted 5:1 cost +0.4%); Platt (1999); Chawla et al. (2002)
Notes EN: Weighting changes the probabilities, not the ranking: up-weighting defaulters mostly shifts the logit's intercept, its baseline log-odds, inflating every probability, the dotted lines. Platt scaling, a one-input logistic regression on training scores, restores them, yet the result is still slightly costlier than the plain logit. SMOTE, synthetic defaulters between real neighbours, even lowers the forest's AUC. Keep the real proportions; move the cut-off.
Notes 中文: 加权改变的是概率，不是排序：给违约者加权，主要是平移了逻辑回归的截距，也就是基准对数几率，把所有概率都抬高（图中虚线）。Platt 缩放是在训练集分数上拟合的单输入逻辑回归，能把概率拉回来，但结果仍比普通逻辑回归略贵。SMOTE 在真实邻居之间造出合成违约者，反而让随机森林的 AUC 下降。保留真实比例，改动判定线就好。
Hand-over: (none — LI continues)
If asked: "How much costlier is the weighted logit plus Platt?" — 0.4 % in cross-fitted 5:1 cost on training rows (12,247 against 12,195 for the plain logit), so the pre-registered > 1 % rule did not adopt it (results/nested_cv_summary_repro.json). "How big is the intercept shift?" — +1.24 log-odds. The balanced logit at 0.5 flags the same clients as the plain logit at 0.22 for 99.4 % of test clients. SMOTE lowers the forest's CV AUC from 0.780 to 0.774 (results/recalibration.csv, imbalance_threshold_equivalence.csv, imbalance_comparison.csv).

---

## S17 — Leakage control: SMOTE before the split fakes 0.914 AUC

Status: KEPT from v1 #10. The title (same words; case only) and the claim are unchanged. Delete the "WHY IT LEAKS" column (its three lines are now spoken) and the italic bottom line; enlarge s_leakage.png to full width.
Section: Results
Presenter: LI Jingwen
Time: 0:20 (47 words)
Title: Leakage control: SMOTE before the split fakes 0.914 AUC
Claim line: Honest forest 0.774; the same forest "tested" on 73% synthetic defaulters scores 0.914.
Body: figure + caption.
Figure: figures/slides/s_leakage.png, about 1300 × 650 px. Caption: "Same learners, same data: only the order of SMOTE and split differs". (Re-rendered 22 Oct: honest bars navy, leaky bars hatched grey, no model-family colours.)
Source line: Test AUCs, reporting only (honest = frozen SMOTE-in-folds forest, CV also 0.774; leaky = "test" split of SMOTE'd data, n = 14,001); results/imbalance_leakage_demo.csv; negative control imb_leaky_smote_demo in frozen_manifest.json
Notes EN: Resampling before the split fakes success. Create synthetic defaulters first, and many "test" defaulters are near-copies of training ones: the forest recognises its neighbours and scores 0.914, not an honest 0.774. Flexible models gain most from such leaks, so every resampler here runs inside the training folds.
Notes 中文: 在切分之前重采样，会造出虚假的成功。如果先造合成违约者，很多「测试」违约者其实是训练违约者的近似副本：随机森林认出了自己的邻居，得到 0.914，而诚实的数字是 0.774。越灵活的模型越会从这种泄漏里「得益」，所以我们所有的重采样都只在训练折内部进行。
Hand-over: (none — LI continues)
If asked: "Does the logit leak too?" — Barely: 0.770 against an honest 0.762, because a straight line cannot memorise neighbours. Restricted to real rows, the leaky forest still shows 0.845 (results/imbalance_leakage_demo.csv).

---

## S18 — The cut-off matters more than the model

Status: KEPT from v1 #12. Text changes:
- new title and claim;
- delete the four tiles and the bottom paragraph;
- enlarge s_cost_vs_threshold.png to full width (re-rendered 22 Oct: every label sits in clear space with a white box, the legend is below the axes, a summary box states the two savings, and the title says "5:1 scenario borrowed from German credit data"; y-axis = totals over 8,990 test clients, so v1's per-1,000 Canva note is no longer needed — delete it).
Section: Business & Future
Presenter: LI Jingwen
Time: 0:30 (69 words)
Title: The cut-off matters more than the model
Claim line: At 5:1, moving the logit's cut-off saves 1,937; switching to XGBoost saves 145 more [−18, 299].
Body: figure + caption.
Figure: figures/slides/s_cost_vs_threshold.png, about 1300 × 650 px. Caption: "Test cost at 5:1 against the cut-off; diamonds = cut-offs chosen on training OOF".
Source line: Cut-offs chosen on training OOF; test n = 8,990, reporting only; results/threshold_vs_model_saving.csv (paired bootstrap B = 1,000); 5:1 = Statlog German Credit cost matrix (Hofmann 1994), doi:10.24432/C5NC77
Notes EN: Where the bank draws the line matters more than which model draws it. The usual coin-flip cut-off assumes both mistakes cost the same. Suppose, as a scenario borrowed from German credit data, a missed defaulter costs as much as five needless reminder calls. Moving the logit's cut-off saves the cost of 1,937 needless calls on the test set; switching to XGBoost saves 145 more, and that could be zero.
Notes 中文: 银行把线划在哪里，比用哪个模型来划更重要。通常那条抛硬币式的判定线，假设两种错误代价相同。假设——这是借自德国信用数据的情景——漏掉一个违约者的代价，相当于五次不必要的提醒电话。那么相应下调逻辑回归的判定线，在测试集上能省下相当于 1,937 次不必要电话的成本；换成 XGBoost 再多省 145，而这个数也可能是零。
Hand-over: (none — LI continues)
If asked: "Why 5:1?" — It is the Statlog German-credit cost matrix: a scenario, not this issuer's cost. Our rough anchors (50–70 % of the median defaulter's bill lost, against one to three months of a 1.5 % monthly margin on the median good client's bill) give 9.6:1 to 40.5:1, and that is an upper bound (results/cost_anchors.json).

---

## S19 — The bank's own cost ratio should set the cut-off

Status: NEW. Layout from v1 #3 (wide figure).
Section: Business & Future
Presenter: LI Jingwen
Time: 0:25 (59 words)
Title: The bank's own cost ratio should set the cut-off
Claim line: Cut-off falls from 0.49 at 1:1 to 0.03 at 20:1; the cheapest model keeps changing.
Body: figure + caption.
Figure: figures/slides/s_ratio_cutoff.png (new; render of 22 Oct checked against this spec), about 1300 × 650 px. Caption: "Training OOF: cost-minimising cut-off and cheapest model at each cost ratio". Fallback: a Canva table from results/preferred_model_by_ratio_oof.csv with columns ratio · cut-off · cheapest · top-3 spread, rows 1:1 GBM 0.49 0.69% · 3:1 XGB 0.27 0.80% · 5:1 XGB 0.15 1.03% · 10:1 RF 0.09 0.90% · 20:1 KNN 0.03 0.64%.
Source line: Training OOF, post-freeze, selects nothing; results/preferred_model_by_ratio_oof.csv, preferred_model_by_ratio_oof_long.csv, cost_anchors.json
Notes EN: The bank's true ratio is unknown, so we swept it. At r to one, one caught defaulter is worth r false alarms, so flag a client once the default chance tops one in r plus one. As the ratio rises the cut-off falls, and the cheapest model keeps changing, within about one per cent: we name no cost winner.
Notes 中文: 银行真实的成本比是未知的，所以我们把它扫了一遍。比例为 r 比 1 时，抓到一个违约者值 r 次误报，所以只要违约概率超过 r+1 分之一就标记。比例越高，判定线越低；而最便宜的模型一直在换，差距只在百分之一左右：我们不宣布成本冠军。
Hand-over: (none — LI continues)
If asked: "What should the bank do first?" — Estimate its own cost ratio, then set the cut-off from it (about 1 / (1 + r) for calibrated scores). "Why sweep at all?" — The true ratio is unknown, and our rough loss anchors point above 5:1 (9.6:1 to 40.5:1, an upper bound; results/cost_anchors.json). "What happens at very high ratios?" — Above about 15:1 almost everyone is flagged (96 % at 20:1). The model's value then lies in ordering a graded action: reminders for most, calls for the riskiest (results/preferred_model_by_ratio_oof.csv).

---

## S20 — A one-line rule gets most of the way

Status: NEW. Layout from v1 #3 (wide figure).
Section: Business & Future
Presenter: LI Jingwen
Time: 0:23 (53 words)
Title: A one-line rule gets most of the way
Claim line: XGBoost costs 7.7% less than "flag any delay" at 5:1; the logit, almost nothing less.
Body: figure + caption.
Figure: figures/slides/s_simple_rule.png (new; render of 22 Oct checked against this spec), about 1300 × 650 px. Caption: "5:1 cost per 1,000 training clients: no-parameter rules vs cross-fitted models". Fallback: a Canva table from results/simple_rule_benchmark.csv, cost per 1,000: any delay 586 · logit 581 · XGBoost 540 · flag everybody 779.
Source line: Training rows, post-freeze, never test-scored; results/simple_rule_benchmark.csv, topk_capture_oof.csv
Notes EN: A one-line rule gets most of the way: flag anyone with any payment delay in six months. On training data XGBoost costs 7.7 per cent less, the logit almost nothing less. The rule misses the third of defaulters who were never late; that is where bills, the limit and utilisation earn their place.
Notes 中文: 一条一句话的规则就能走完大半路程：六个月里只要有过一次逾期就标记。在训练数据上，XGBoost 只比它便宜 7.7%，逻辑回归几乎不比它便宜。这条规则漏掉的是从未逾期过的那三分之一违约者，账单、额度和使用率正是在那里发挥作用。
Hand-over: (none — LI continues)
If asked: "So is the model worth it?" — Yes, but modestly: 7.7 % cheaper than the rule at 5:1 on training rows, and the logit barely beats it (results/simple_rule_benchmark.csv). "What if the team can only call the riskiest 10 %?" — XGBoost's top 10 % catches 31.5 % of defaulters, and September status alone catches 30.9 %, because the top of any list is clients already two months late (results/topk_capture_oof.csv).

---

## S21 — Limits: one crisis vintage, a re-read test set, borrowed costs

Status: KEPT from v1 #14 (three columns). Text changes:
- new title and claim;
- column headings "WHAT WE FOUND · LIMITATIONS · NEXT STEPS" become "VINTAGE · TEST RECORD · COSTS" (the same three limits the notes list, in the same order; each line ends with its fix);
- each column is cut to the one line below;
- delete v1's lines "XGBoost CV 0.786, test 0.789: top three tied", "One 70/30 split, one seed: gaps below 0.005 …" and the rest.
Section: Business & Future
Presenter: LI Jingwen
Time: 0:24 (55 words)
Title: Limits: one crisis vintage, a re-read test set, borrowed costs
Claim line: Rank with any ensemble, set the cut-off from costs, re-base before reuse.
Body (three columns, one line each):
- VINTAGE — One bank, crisis onset: re-base probabilities, test a later period
- TEST RECORD — Test read more than once, every read dated; one split; ensembles not nested
- COSTS — 5:1 borrowed: measure the bank's own losses, then reset the cut-off
Figure: none (the US delinquency chart is on A2 for Q&A).
Source line: Report §7; results/TEST_SET_LOG.md; Federal Reserve, charge-off and delinquency rates, https://www.federalreserve.gov/releases/chargeoff/delallsa.htm
Notes EN: Our answer holds within three limits. One bank at a crisis onset, and default rates swing over a cycle: rescale probabilities and test a later period. One test set read more than once, every read dated: a report card. And a borrowed cost ratio. So rank with any ensemble, and set the cut-off from costs.
Notes 中文: 我们的结论在三个局限之内成立。一，这是危机刚开始时一家银行的数据，而违约率会随周期大幅波动：要重新校准概率，并在更晚的时期上检验。二，测试集被读过不止一次，每次都记了日期：它是一份成绩单。三，成本比是借来的。所以，用任意一个集成模型排序，用成本来定判定线。
Hand-over: LI closes: "Thank you — we welcome your questions." / 中文：「谢谢，欢迎提问。」 Then open the floor; each presenter takes questions on their own slides.
If asked: "How much do default rates swing?" — Several-fold: US card delinquency ran from 1.53 % to 6.77 % over 1991–2026 (Federal Reserve, a US analogy; A2). "What would you do with more time?" (v1's NEXT column) — An out-of-time hold-out, bureau and income data, measured losses for the cost ratio, a binned additive scorecard, and nested multi-seed ensembles (report §7).

---

## A1 — Appendix · Nested CV, Platt, Cost Sweep, Settings and References (not spoken)

Status: KEPT from v1 A1. Text changes:
- PLATT block: "each calibrated version scored once on the test set and frozen" → "each calibrated version frozen at its first test scoring (19 Oct); maps fitted on training OOF only".
- NESTED CV block: "weighted + Platt 0.761, cost 12,247: −0.4% < 1% rule → not adopted" → "weighted + Platt 0.761, cross-fitted 5:1 cost +0.4% (12,247 vs 12,195; rule needs > 1% saving) → not adopted". Add "nested CV run for KNN and the logistic family only; ensembles not nested".
- Add a line above REFERENCES: "POST-FREEZE CHECKS (21 Oct; training CV only, never test-scored, select nothing): pay-code recoding, interaction depth, history window, demographics and fairness, monotone respecification, simple rules, cost-ratio preference, fold stability; R refits match 13 of 15 Python CV AUCs within 0.005 — results/POST_REVIEW_CHECKS.md".
- REFERENCES: append "Friedman (2001) · Nadeau & Bengio (2003) · Kleinberg, Mullainathan & Raghavan (2017) · EU AI Act 2024/1689 Annex III 5(b) · BoE & FCA (2024) · FSC Banking Bureau index-7.xlsx · CBC FSR Jun 2008".
Section: A1 appendix (not spoken; shown only if asked)
Presenter: whoever is asked (RUAN for nested CV, LI for Platt and the sweep, LU for settings)
Time: 0:00 (not spoken)
Title: Appendix · Nested CV, Platt, Cost Sweep, Settings and References
Claim line: (none; appendix text layout)
Body: v1 A1 text blocks with the changes above.
Figure: none.
Source line: results/nested_cv_summary_repro.json, recalibration.csv, threshold_vs_model_saving.csv, POST_REVIEW_CHECKS.md, r_crosscheck.csv
Notes EN: (not spoken)
Notes 中文: （不讲，备问）
Hand-over: (none)
If asked: Use the block that matches the question; every cell names its results file.

---

## A2 — Appendix · Cost sweep and real-world context (not spoken)

Status: NEW. Layout from v1 A1 (appendix): delete its text and place four images in a 2 × 2 grid. Each image fits inside a 900 × 400 px cell, aspect kept (rows at y ≈ 135 and ≈ 575), with a 16 px source line under each (y ≈ 540 and ≈ 980), clear of the progress bar at 1032.
Section: A1 appendix (not spoken; shown only if asked)
Presenter: whoever is asked (LI for the sweep, RUAN for the crisis/cycle, LU for regulation)
Time: 0:00 (not spoken)
Title: Appendix · Cost sweep and real-world context
Claim line: (none)
Body: four images.
Figure:
- top-left: figures/slides/s_threshold_vs_model.png (threshold saving vs model saving, 1:1–20:1, test set, reporting only; re-rendered 22 Oct at 1620 × 720 for this cell, so it fills 900 × 400 and its text stays ≥ 17 px; labels boxed, true minus signs, thousands separators, "5:1 = scenario borrowed from German credit data");
- top-right: figures/context/ctx_us_card_delinquency.png (fits at 870 × 400);
- bottom-left: figures/context/ctx_eu_ai_act_annex3_5b.png (fits at 900 × 341);
- bottom-right: figures/context/ctx_boe_fca_2024_explainability_crop.png (new 22 Oct: the original with its clipped heading row and clipped URL line cropped off; fits at 900 × 146; the 16 px source line below it carries the citation).
Source line:
- results/threshold_vs_model_saving.csv;
- Federal Reserve, https://www.federalreserve.gov/releases/chargeoff/delallsa.htm (US analogy, 30+ days past due);
- Regulation (EU) 2024/1689 Annex III 5(b), https://eur-lex.europa.eu/eli/reg/2024/1689/oj (application of Annex III obligations moved to 2 Dec 2027 by Regulation (EU) 2026/1744);
- BoE & FCA, AI in UK financial services 2024, §2.6–2.7 (excerpt of §2.7; blue box added), https://www.bankofengland.co.uk/report/2024/artificial-intelligence-in-uk-financial-services-2024.
Notes EN: (not spoken)
Notes 中文: （不讲，备问）
Hand-over: (none)
If asked: "Do banks really use boosting and SHAP?" — In the 2024 BoE–FCA survey, gradient boosting was the most common model type (32 % of use cases, §2.6), and feature importance (72 %) and Shapley additive explanations (64 %) were the most popular explainability approaches (§2.7, quoted as worded). These are present-day benchmarks; we claim no compliance.
---

## Q&A items in `slides/qa_bank.md` that must change

The bank is not rewritten here. Only the items whose wording or numbers now contradict the report or v2 are listed, with replacement answers, plus five new items for the new slides. Presenter headers change to **RUAN slides 1–7 · LU slides 8–13 · LI slides 14–21**; the Presentation row of `slides/contribution_statement_draft.md` must say the same.

| Item | Why it must change | New answer (2–4 sentences, numbers from the named files) |
|---|---|---|
| Header, line 15: the four verbatim phrases | "scored once on the frozen test set" contradicts `TEST_SET_LOG.md` | Use: "chosen on training CV; each configuration frozen at its first test scoring; every test read dated; the test set only reports" · "the cut-off matters more than the model" · "weight the decision, not the data" · "segments are a map for policy, not a better score". Never: "scored once", "CV winner", "touched once", "economically meaningful segments". |
| Ten-numbers table, XGBoost row ("CV winner, 1st of 11 on CV") | winner language | "XGBoost: highest CV AUC 0.786 ± 0.008, tied with GBM (+0.0013 [−0.0005, +0.0031]) and the forest (+0.0014 [−0.0027, +0.0055]), training-fold Nadeau–Bengio intervals; test 0.789 [0.776, 0.800]" (`fold_stability.csv`, `auc_bootstrap_ci.csv`). |
| Ten-numbers table, imbalance row ("RF −0.008 test AUC with SMOTE") | lead with CV (POST_REVIEW 4.10) | "SMOTE costs the forest 0.0055 CV AUC (0.780 → 0.774); weighting and under-sampling move it by under 0.001; test −0.0076 as a check" (`imbalance_comparison.csv`). |
| G1 (protocol) | "each frozen configuration scored once" understates the test history | "Every choice (settings, cut-offs, model order) is made on the 20,975 training rows by five-fold CV and out-of-fold scores. Round-1 configurations were scored on 16–18 Sep and entered in the freeze register on 19 Oct (46 entries); since then `evaluate()` refuses a changed configuration. Before that, the dated log in `results/TEST_SET_LOG.md` is the guard. It shows that until 19 Oct test AUC chose the four models in the report's figures, and that nine Platt variants were scored on 19 Oct after a 6 Oct review. So the test set is a dated report card, read more than once, never the basis of a model, feature or cut-off." |
| G2 (what "frozen" means) | same | "A configuration counts as frozen at its first test scoring. `freeze.py` stores the hash of its settings and of its test-probability vector in `results/frozen_manifest.json`, and `--check` re-verifies all 46. The register itself was written on 19 Oct; earlier reads, including identical re-runs and RUAN's 30 Sep re-fits of the plain logit and KNN, are dated in `TEST_SET_LOG.md`." |
| G6 (why 5:1) | calls 5:1 "a lower bound"; the report calls the 9.6–40.5:1 anchors an upper bound | "5:1 is a scenario borrowed from the Statlog German credit cost matrix (Hofmann 1994), not measured for this issuer. Our rough anchors (50–70 % of the median defaulter's bill lost against one to three months of a 1.5 % monthly margin on the median good client) give 9.6:1 to 40.5:1. That is an upper bound, since some defaulters later pay (`cost_anchors.json`). So we report a 1:1–20:1 sweep: from 2:1 to 20:1 the threshold saving exceeds the model saving (`threshold_vs_model_saving.csv`)." |
| G7 (one split) | add fold stability | Append: "On the training side, the four tiers hold in all five folds, and the top three swap places within noise (`fold_stability.csv`). The ensembles were not nested, so their internal order cannot be separated from selection optimism." |
| G8 (what the professor's comment changed) | "nothing was re-scored after 18 Sep" and "withdrew the test-set evaluation of the cluster-augmented logit" are wrong (POST_REVIEW 4.1, 4.9) | "(1) Test record: a freeze register (19 Oct) plus a dated log of every test read, including those before the register; nine Platt variants were scored on 19 Oct. (2) Cost: two absolute savings with intervals over a 1:1–20:1 sweep, and 5:1 labelled a borrowed scenario. (3) Clustering is secondary: K = 5 on 10 PCs, stability measured, and the label adds +0.0001 [−0.0004, +0.0007] AUC in training CV. The K = 5 model was never test-scored; the round-1 K = 4 version was scored on 16 Sep and is listed for completeness. (4) Resampling stays inside the folds, and Platt recalibration is fitted on training OOF scores." |
| R1 (why this dataset) | "drive default in any card book" is causal; crisis vintage missing | "UCI *Default of Credit Card Clients* (doi:10.24432/C55S3H): one Taiwanese bank's cardholders, Apr–Sep 2005, observed as the card-debt crisis began. The central bank dates the first repayment problems to Sep 2005 (CBC FSR 2008, Box 2), and revolving balances peaked at NT$495.3 bn in Nov 2005 (FSC). So the 22.1 % rate is consistent with a stressed book, and the PDs belong to that vintage. What should travel is the ranking logic (recent arrears predict default); levels must be re-based." |
| R2 (EDA) | "33.9 % one month late" unqualified; demographics missing | Add: "Code 1 is a mixture: 2,540 clients in September but 21 in the five earlier months, almost half (46 %) with no positive September bill. Sex and age move the rate by at most 8 points. Bill size alone ranks at chance (univariate AUC 0.50–0.52)" (`pay_code_default_rates.csv`, `eda_summary.json`). |
| R4 (KNN vs logit) | blames linearity | "KNN (CV 0.770) is ahead of the plain logit (0.760) mainly because the logit reads each status code as one evenly spaced number. Coded by meaning (months late plus flags for −2 and −1), the logit reaches 0.768: +0.0082 [+0.0021, +0.0143], 5/5 folds, level with KNN (−0.0016 [−0.0080, +0.0048]). KNN gains nothing from recoding, because a neighbourhood of discrete codes already reads each code separately (training CV; `pay_coding_cv.csv`)." |
| R5 (nested CV) | "nested CV confirms" generalised (POST_REVIEW 3.2) | Replace the last sentence: "Nested CV was run for KNN and the logistic family only; it scores almost the same, and its pre-registered rule re-selects their frozen settings (outer-fold choices vary). The ensembles were tuned by plain five-fold CV, so their 0.001 internal gaps cannot be separated from selection optimism (a stated limitation)." |
| R7 (no gain from regularisation) | "the 0.76 ceiling is the bias of linearity" | "…so there is no variance to trade. The 0.76 level is mostly coding, not linearity: a still-additive logit with codes by meaning reaches 0.768, and with one dummy per code 0.769 (L2 fit; `pay_coding_cv.csv`)." |
| R8 (odd coefficient signs) | read coefficients as a block | Add: "UTIL_1 and UTIL_MEAN correlate at 0.91, so their opposite signs must be read together. Moving both by one SD changes the log-odds by about +0.03. In the recoded logit, PAY_2–PAY_6 run −0.42 to −0.11 per month late given the counters, with all five folds agreeing in sign (`pay_coding_logit_coefficients.csv`)." |
| L6 (are the ensembles distinguishable?) | "XGBoost is the reference because it is first on training CV" reads as a winner | "No. Training-fold intervals: XGBoost − GBM +0.0013 [−0.0005, +0.0031], − forest +0.0014 [−0.0027, +0.0055] (`fold_stability.csv`). XGBoost is the reference only by the preset highest-CV rule; a bank may pick any of the three on cost, speed or explainability." |
| L9 (what drives the score) | "drives" causal; only six columns were refitted | Retitle "What does the model rely on?" and change "removing any other single feature" to "of the six columns refitted, only September status costs over 0.001 OOB AUC". Say SHAP and MDI come from different models (cross-model comparison). |
| L11 (monotone) | "the version we would deploy"; ignores that the frozen constraint orders −2/−1/0 and utilisation against the data | "The frozen constraint also forced the score up from −2 to −1 to 0 and with utilisation; the data contradict both (16.9 % vs 12.9 %; a U-shape). It costs 0.0019 CV AUC [−0.0025, −0.0013], 0/5 folds (test −0.0016 as a check). Constraining only months late (up) and the limit (down) is within 0.0005 of the free booster (training CV, never test-scored; `monotone_respec_cv.json`). That respecified version is the promise we would make, valid from revolving credit upwards." |
| L12 (PDP) | "logit's additive log-odds plateau … cannot bend the U" | "…The logit's gap is first coding (a third of it, already inside the 86 % that follows), then one-variable shape: a depth-1 booster recovers 86 % of the XGBoost–logit gap and interactions add 0.004 (training CV, `ablation_depth.csv`). The U in utilisation is one place where that shape shows; we did not measure a share per variable." |
| L13 (recommendation) | "deploy the XGBoost … or its monotone twin" | "Model: any of the three tied ensembles (XGBoost by the preset highest-CV rule), with the respecified arrears-only monotone constraint if a lender must promise direction. Cut-off: from the bank's own cost ratio, about 1/(1 + r), re-estimated with recalibration; 5:1 is only a scenario. Segments: a map for collections and limit policy, not a model input (+0.0001 AUC). Attributes: dropping them is a documented trade-off, not a fix (S13)." |
| L14 (limitations) | missing vintage, test history and fairness | "(i) Retrospective test use, dated in `TEST_SET_LOG.md`. (ii) Borrowed cost ratio. (iii) Crisis vintage: US card delinquency ranged from 1.53 % (2021Q3) to 6.77 % (2009Q2) over 1991–2026 (Fed; a US analogy), so reuse needs PDs re-based and out-of-time validation. (iv) One split, single seeds, ensembles not nested. (v) Fairness numbers are a training-OOF diagnostic, not a compliance assessment." |
| J6 (does re-balancing help?) | leads with test "SMOTE loses 0.008" | "…for the forest the four give CV 0.780 / 0.781 / 0.774 / 0.779, so SMOTE costs 0.0055 CV AUC (test −0.0076 as a check), partly because SMOTE after one-hot encoding invents fractional dummies (`imbalance_comparison.csv`)." |
| J8 (Platt) | "scored once, as a new configuration" | "…applied to the frozen test probabilities and frozen at that first scoring on 19 Oct. Platt had been pre-committed on 30 Sep, but on 6 Oct the Platt, isotonic and prior-corrected maps were compared on test Brier and cost before the freeze; that read is dated in `TEST_SET_LOG.md`." |
| J11 (why order by CV) | "the CV winner happens to be the test leader"; hides that aggregate.py ranked by test AUC until 19 Oct | "Because a test set that ranks becomes a validation set. Honestly: until 19 Oct `aggregate.py` ordered by test AUC, and the top four then had the tree in place of the MLP. Since then the order is training CV AUC (`select_auc`), and every model ranked above the logit beats it in 5/5 folds (`fold_stability.csv`)." |
| J12 (is a difference real?) | add the training-side scale | Add: "Training-fold paired intervals (Nadeau–Bengio): XGBoost − KNN +0.0163 [+0.0055, +0.0270], − logit +0.0261 [+0.0145, +0.0376], − MLP +0.0101 [+0.0058, +0.0144]; MLP − KNN +0.0062 [−0.0003, +0.0127], suggestive, not decisive." |
| J15 (freeze machinery) | "scored once as new, registered configurations" | "…then frozen at their first test scoring as new registered configurations; the dated log covers the reads before the register existed." |
| NEW R15 (S7) | new slide | "Why recode at all? The status codes are states: −2 no consumption, −1 paid in full, 0 revolving, k = k months late. September default is 13.7 / 16.9 / 12.9 % for −2 / −1 / 0, so no straight order exists. Months late plus two flags lifts the logit +0.0082 [+0.0021, +0.0143] CV AUC, 5/5, never test-scored (`pay_coding_cv.csv`)." |
| NEW L16 (S9–S10) | new slides | "How do you know the gain is mostly shape? Depth-1 XGBoost, rounds re-tuned, minus logit is +0.0225 [+0.0111, +0.0338]; depth 3 minus depth 1 is +0.0036 [+0.0022, +0.0050], an upper estimate since depth 3 was chosen on these folds (`ablation_depth.csv`). History: H0 0.630 → H1 0.769; older months add +0.013 as raw months, or +0.017 via six-month summaries (`ablation_history.csv`)." |
| NEW L17 (S13) | new slide | "Is XGBoost unfair to men? It flags men 1.18 times as often as women, against a default-rate ratio of 1.19, and its PDs match every group of ≥ 1,000 clients within 0.5 points. So men who would repay are flagged more often too (FPR 45.7 % vs 38.2 %). Removing the four columns narrows the ratio to 1.10, but under-predicts men by 1.1 points and raises the under-30 ratio from 1.11 to 1.17 (`fairness_groups.csv`, `fairness_summary.json`; training OOF diagnostic)." |
| NEW J16 (S20) | new slide | "Isn't a rule enough? 'Any delay in six months' flags 33.6 %, recall 0.645, cost 585.6 per 1,000 at 5:1. XGBoost (cross-fitted) costs 540.3: 7.7 % less, +45 [+25, +66] per 1,000, 5/5 folds. The logit saves only +5 [−4, +14]. The rule was the cheapest of four candidates on these rows, a mild selection effect in its favour (`simple_rule_benchmark.csv`)." |
| NEW J17 (S19) | new slide | "Which model is cheapest? It rotates: GBM at 1:1 (cut-off 0.49), forest at 2:1, XGBoost at 3:1–7:1, forest at 10:1–15:1, KNN at 20:1 (0.03). The top three are never more than 1.03 % apart, and no fold interval excludes zero (`preferred_model_by_ratio_oof.csv`). So we name no cost winner." |
