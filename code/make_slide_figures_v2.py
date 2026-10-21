"""
make_slide_figures_v2.py — seven extra slide figures for deck v2 (22 Oct 2026), same style as make_slide_figures.py.
Owner: shared. TRAINING-SIDE FILES ONLY: every number drawn comes from a post-freeze (21 Oct) training check, i.e.
5-fold CV on the 20,975 training rows or their pooled / cross-fitted out-of-fold (OOF) scores. Nothing is fitted,
tuned or scored here. No test file, no *_p_test.npy, no load_split() — every read goes through rd(), which refuses any
file not in ALLOWED.

Why a separate script: make_slide_figures.py calls load_split() and reads *_p_test.npy at import time, so it cannot be
imported for its constants; its style block (palette, ink, rcParams, 1920 x 960 px at dpi 150) is copied verbatim below,
with the text sizes raised to 16-20 pt for the projector.

Writes figures/slides/:
  s_pay_codes.png        default rate by September repayment code (-2 / -1 / 0 are not an ordered scale)
  s_pay_coding_gain.png  CV AUC with the PAY codes as one number vs read by meaning (logit, LDA, KNN, MLP)
  s_fold_stability.png   per-fold CV AUC of six headline models | XGBoost minus each, with Nadeau-Bengio 95% interval
  s_ablation_depth.png   interaction depth 1-4, rounds re-tuned (slide S9; own y-axis title, 1200 x 700 px)
  s_ablation_history.png months of repayment history: raw months vs raw months + summaries of the same window (S10)
  s_ablation.png         both panels side by side (kept for reference; the deck uses the two files above)
  s_fairness.png         flag-rate ratio vs default-rate ratio by sex / marital status / age (XGBoost, logit, no-demo XGB)
  s_simple_rule.png      5:1 cost per 1,000 clients: one-line arrears rules vs logit vs XGBoost
  s_ratio_cutoff.png     cost-minimising cut-off and cheapest model at cost ratios 1:1-20:1
  s_monotone.png         consistent PAY_1 sweep of 200 training clients (free XGBoost) | CV AUC price of two constraints (S12)

22 Oct revision (fixer pass): history panel now plots the window-summary series (H1+W1, H3+W3, H6+W6) that the report
quotes (+0.017), not the all-six-month-summary series; the two ablation panels are also written as separate files;
s_monotone.png added; fairness title and pay-coding annotation reworded; pay-code highlight bars in the deck navy.

Run from Project/code:  /opt/miniconda3/bin/python3 make_slide_figures_v2.py      (a few seconds)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "slides")
os.makedirs(OUT, exist_ok=True)

ALLOWED = {"pay_code_default_rates.csv", "pay_coding_cv.csv", "fold_stability.csv", "ablation_depth.csv",
           "ablation_history.csv", "fairness_groups.csv", "simple_rule_benchmark.csv",
           "preferred_model_by_ratio_oof.csv", "preferred_model_by_ratio_oof_long.csv", "topk_capture_oof.csv",
           "monotone_check_consistent.csv", "monotone_respec_cv.json"}


def rd(name):
    assert name in ALLOWED, f"{name} is not a training-side file on the allow-list"
    if name.endswith(".json"):
        import json
        with open(os.path.join(RESULTS, name)) as fh:
            return json.load(fh)
    return pd.read_csv(os.path.join(RESULTS, name))


# ---- style: copied from make_slide_figures.py (palette = report's validated categorical palette, colour = model family)
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#4a3aa7", red="#e34948")
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"
NEUTRAL, WASH = "#b4b3ad", "#f3f2ef"          # muted bars for context; light band behind a highlighted region
NAVY = "#18395b"                              # deck title colour: highlights that are not a model family
FAMILY_COLOR = {"boosting": PAL["blue"], "linear": PAL["orange"], "forest / bagging": PAL["aqua"], "single tree": PAL["yellow"],
                "naive Bayes": PAL["magenta"], "KNN": PAL["green"], "SVM": PAL["violet"], "neural net": PAL["red"]}
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "legend.fontsize": 16, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.dpi": 150, "lines.linewidth": 2.5, "font.family": "DejaVu Sans", "axes.axisbelow": True})
SIZE = (12.8, 6.4)          # 1920 x 960 px at dpi 150
TITLE = 20                  # figure title, left-aligned
NOTE = 16                   # annotation text
MINUS = "−"


def sgn(v, d=3):
    """Signed number with a true minus sign, e.g. +0.008 / −0.002."""
    s = f"{v:+.{d}f}"
    return s.replace("-", MINUS)


def num(v, d=3):
    return f"{v:.{d}f}".replace("-", MINUS)


def save(fig, name, rect=None):
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def left_title(fig, text):
    fig.suptitle(text, x=0.012, y=0.985, ha="left", va="top", fontsize=TITLE)


# ====================================================================================================================
# 1. s_pay_codes.png — default rate by September repayment code (training rows)
# ====================================================================================================================
pc = rd("pay_code_default_rates.csv")
p1 = pc[pc.variable == "PAY_1"].copy()
rows = []
for c in (-2, -1, 0, 1, 2):
    r = p1[p1.code == c].iloc[0]
    rows.append((str(c).replace("-", MINUS), int(r.n), int(r.n_default)))
hi = p1[p1.code >= 3]
rows.append(("3+", int(hi.n.sum()), int(hi.n_default.sum())))
meaning = ["no\nconsumption", "paid\nin full", "revolving\ncredit", "1 month late\n(mixed group)", "2 months\nlate", "3+ months\nlate"]
rate = np.array([nd / n for _, n, nd in rows])

fig, ax = plt.subplots(figsize=SIZE)
x = np.arange(len(rows))
ax.axvspan(-0.5, 2.5, color=WASH, zorder=0, lw=0)
cols = [NAVY] * 3 + [NEUTRAL] * 3
ax.bar(x, rate, width=0.56, color=cols, zorder=2)
for xi, v in zip(x, rate):
    ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=17, color=INK, fontweight="bold" if xi < 3 else "normal")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n{m}\nn = {n:,}" for (c, n, _), m in zip(rows, meaning)], fontsize=15)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_xlim(-0.5, len(rows) - 0.5)
ax.set_ylim(0, 0.86)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Default rate next month (training rows)")
ax.grid(axis="x", visible=False)
ax.text(1.0, 0.80, "All three codes mean 'not late',\nyet the rate goes up, then down",
        ha="center", va="top", fontsize=NOTE + 1, color=INK)
ax.text(1.0, 0.56, "Read as one evenly spaced number,\nthe code forces " + f"{MINUS}2 < {MINUS}1 < 0 < 1 < 2\nonto a straight line",
        ha="center", va="top", fontsize=NOTE, color=INK2)
ax.set_title("Default rate by September repayment code: " + f"{MINUS}2, {MINUS}1 and 0 are not in order", loc="left", fontsize=TITLE, pad=14)
save(fig, "s_pay_codes.png")

# ====================================================================================================================
# 2. s_pay_coding_gain.png — CV AUC, PAY codes as one number vs read by meaning (post-freeze training CV)
# ====================================================================================================================
pcv = rd("pay_coding_cv.csv")
fs = rd("fold_stability.csv").set_index("slug")
spec = [("logit_plain", "Logistic regression", FAMILY_COLOR["linear"]),
        ("lda", "LDA", FAMILY_COLOR["linear"]),
        ("knn", "KNN (K = 101)", FAMILY_COLOR["KNN"]),
        ("mlp", "MLP (neural net)", FAMILY_COLOR["neural net"])]
fig, ax = plt.subplots(figsize=SIZE)
ys = np.arange(len(spec))[::-1] * 1.0
for (m, lab, c), y in zip(spec, ys):
    a = pcv[(pcv.model == m) & (pcv.coding == "numeric")].iloc[0]
    b = pcv[(pcv.model == m) & (pcv.coding == "delay_flags")].iloc[0]
    v0, v1 = a.mean_auc, b.mean_auc
    if abs(v1 - v0) > 0.004:
        ax.annotate("", xy=(v1, y), xytext=(v0, y), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5, mutation_scale=22,
                                                                     shrinkA=9, shrinkB=9), zorder=3)
    else:
        ax.plot([v0, v1], [y, y], color=c, lw=2.5, zorder=3)
    ax.plot(v0, y, "o", ms=13, mfc="white", mec=c, mew=2.5, zorder=4)
    ax.plot(v1, y, "o", ms=13, mfc=c, mec="white", mew=2, zorder=5)
    lo, hi_ = (v0, v1) if v0 < v1 else (v1, v0)
    WB = dict(boxstyle="square,pad=0.1", fc="white", ec="none")
    ax.text(lo - 0.0009, y, num(lo), ha="right", va="center", fontsize=NOTE, color=INK2, bbox=WB, zorder=6)
    ax.text(hi_ + 0.0009, y, num(hi_), ha="left", va="center", fontsize=NOTE, color=INK2, bbox=WB, zorder=6)
    k = int(b.diff_auc_folds_positive)
    verdict = f"{sgn(b.diff_auc_vs_numeric_mean)}, better in {k}/5 folds" + ("" if k == 5 else ": no gain")
    ax.text((v0 + v1) / 2, y + 0.36, verdict, ha="center", va="center", fontsize=NOTE, color=INK,
            fontweight="bold" if k == 5 else "normal", bbox=WB, zorder=6)
xgb_cv = fs.loc["xgboost", "auc_mean"]; knn_cv = fs.loc["knn", "auc_mean"]
ax.axvline(xgb_cv, color=INK3, lw=1.5, ls="--", zorder=1)
ax.text(xgb_cv - 0.0006, ys[-1] - 0.55, f"XGBoost {xgb_cv:.3f}\n(codes as numbers)", ha="right", va="center", fontsize=NOTE - 1, color=INK2)
ax.axvline(knn_cv, color=PAL["green"], lw=1.2, ls=":", zorder=1)
ax.text(knn_cv - 0.0004, ys[0] + 0.9, f"KNN benchmark {knn_cv:.3f}\nrecoded logit level with it (within noise)", ha="right", va="center",
        fontsize=NOTE - 1, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([s[1] for s in spec], fontsize=17)
ax.tick_params(axis="y", length=0)
ax.set_ylim(ys[-1] - 0.9, ys[0] + 1.2)
ax.set_xlim(0.745, 0.791)
ax.grid(axis="y", visible=False)
ax.set_xlabel("5-fold CV ROC-AUC (training folds, post-freeze check)")
fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=12, mfc="white", mec=INK2, mew=2.2, label="codes as one number (as frozen)"),
                    Line2D([], [], ls="none", marker="o", ms=12, mfc=INK2, mec="white", label=f"codes read by meaning (months late + flags for {MINUS}2, {MINUS}1)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, handletextpad=0.3, columnspacing=1.6, fontsize=15)
left_title(fig, "Reading the codes by meaning helps the linear models, not the flexible ones")
save(fig, "s_pay_coding_gain.png", rect=[0, 0.06, 1, 1])

# ====================================================================================================================
# 3. s_fold_stability.png — per-fold CV AUC | XGBoost minus each model (Nadeau-Bengio 95% interval)
# ====================================================================================================================
six = [("xgboost", "XGBoost", PAL["blue"], "-", "o"),
       ("gradient_boosting", "Gradient boosting", PAL["blue"], "--", "s"),
       ("random_forest", "Random forest", PAL["aqua"], "-", "D"),
       ("mlp", "MLP", PAL["red"], "-", "o"),
       ("knn", "KNN (K = 101)", PAL["green"], "-", "o"),
       ("logistic", "Logistic regression", PAL["orange"], "-", "o")]
fig, axes = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
folds = np.arange(1, 6)
for s, lab, c, ls, mk in six:
    v = fs.loc[s, [f"auc_fold{i}" for i in folds]].values.astype(float)
    ax.plot(folds, v, color=c, ls=ls, marker=mk, ms=8, mec="white", mew=1.5, lw=2.2, label=lab, zorder=3)
top3 = fs.loc[["xgboost", "gradient_boosting", "random_forest"], [f"auc_fold{i}" for i in folds]].values.astype(float)
spread = (top3.max(0) - top3.min(0)).max()
ax.set_xticks(folds); ax.set_xticklabels([f"Fold {i}" for i in folds])
ax.set_xlim(0.75, 5.25); ax.set_ylim(0.740, 0.810)
ax.set_ylabel("CV ROC-AUC (training folds)")
ax.set_title("AUC in each of the five folds", loc="left", fontsize=18)
ax.text(3.0, 0.8045, f"top three within {spread:.3f}\nof each other in every fold", fontsize=NOTE - 1, color=INK, ha="center", va="center")
ax.grid(axis="x", visible=False)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=14,
           handlelength=2.2, columnspacing=1.2, handletextpad=0.4)

ax = axes[1]
rows5 = [r for r in six if r[0] != "xgboost"]
yy = np.arange(len(rows5))[::-1]
for (s, lab, c, ls, mk), y in zip(rows5, yy):
    m, lo, hi_ = fs.loc[s, ["xgb_minus_auc_mean", "xgb_minus_auc_nb_lo", "xgb_minus_auc_nb_hi"]].astype(float)
    k = int(fs.loc[s, "xgb_auc_wins_of_5"])
    ax.plot([lo, hi_], [y, y], color=c, lw=3.5, solid_capstyle="round", zorder=3)
    ax.plot(m, y, marker=mk, ms=12, color=c, mec="white", mew=2, zorder=4)
    tie = lo < 0
    txt = f"{sgn(m)}: a tie" if tie else f"{sgn(m)}, ahead in {k}/5 folds"
    ax.text(0.0012, y + 0.36, txt, ha="left", va="center", fontsize=NOTE - 1, color=INK,
            fontweight="bold" if tie else "normal")
ax.axvline(0, color=INK, lw=1.2, zorder=2)
ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in rows5])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.006, 0.045); ax.set_ylim(-0.5, len(rows5) - 0.25)
ax.set_xticks([0, 0.02, 0.04])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else sgn(v, 2)))
ax.set_xlabel("XGBoost minus model (95% interval)")
ax.set_title("Gap to XGBoost, paired by fold", loc="left", fontsize=18)
ax.grid(axis="y", visible=False)
left_title(fig, "The three tree ensembles tie in every fold; their lead over KNN and the logit holds")
save(fig, "s_fold_stability.png", rect=[0, 0.05, 1, 1])

# ====================================================================================================================
# 4. ablation (XGBoost, training CV): s_ablation_depth.png (S9) | s_ablation_history.png (S10) | s_ablation.png (both)
# ====================================================================================================================
ad = rd("ablation_depth.csv")
auc = ad[ad.metric == "auc"]
dep = auc[(auc.question == "a_depth_rounds_tuned") & (auc.kind == "cv") & (auc.feature_set == "full29")].sort_values("max_depth")
logit29 = float(auc[(auc.question == "a_depth") & (auc.kind == "cv") & (auc.model == "logit") & (auc.feature_set == "full29")]["mean"].iloc[0])
pdiff = auc[auc.kind == "paired_diff"]
add_gain = pdiff[(pdiff.spec == "depth 1, rounds tuned | 29 predictors (engineered)") & (pdiff.spec_ref == "logit | 29 predictors (engineered)")].iloc[0]
int_gain = pdiff[(pdiff.spec == "depth 3, rounds tuned | 29 predictors (engineered)") & (pdiff.spec_ref == "depth 1, rounds tuned | 29 predictors (engineered)")].iloc[0]
total = add_gain["mean"] + int_gain["mean"]

ah = rd("ablation_history.csv")
ahu = ah[ah.metric == "auc"]
hx = ahu[(ahu.kind == "cv") & (ahu.model == "XGBoost")].drop_duplicates("feature_set").set_index("feature_set")["mean"]
raw = [hx[k] for k in ("H0", "H1", "H3", "H6")]
win = [hx[k] for k in ("H1+W1", "H3+W3", "H6+W6")]          # raw months + summaries recomputed over the same months
hd = ahu[ahu.kind == "paired_diff"]
older_raw = float(hd[(hd.spec == "H6 raw months only") & (hd.spec_ref == "H1 raw months only")]["mean"].iloc[0])
older_win = float(hd[(hd.spec == "H6 + window summaries (months 1-k)") & (hd.spec_ref == "H1 + window summaries (months 1-k)")]["mean"].iloc[0])
sept = float(hd[(hd.spec == "H1 raw months only") & (hd.spec_ref == "H0 raw months only")]["mean"].iloc[0])
PANEL = (8.0, 4.667)        # 1200 x 700 px: the left 62 % of a slide, placed at about 1:1


def draw_depth(ax, ylabel=True):
    d = dep.max_depth.values.astype(int); v = dep["mean"].values.astype(float)
    ax.axhline(logit29, color=PAL["orange"], lw=2.2, ls="--", zorder=2)
    ax.text(4.25, logit29 + 0.0012, f"logit {logit29:.3f}", ha="right", va="bottom", fontsize=NOTE, color=INK2)
    ax.plot(d, v, color=PAL["blue"], marker="o", ms=10, mec="white", mew=1.8, zorder=4)
    for di, vi in zip(d, v):
        ax.text(di, vi + 0.0016, f"{vi:.3f}", ha="center", va="bottom", fontsize=NOTE - 1, color=INK2)
    bx = 0.72
    ax.annotate("", xy=(bx, v[0]), xytext=(bx, logit29), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5, shrinkA=0, shrinkB=0))
    ax.text(bx + 0.07, (v[0] + logit29) / 2, f"{sgn(add_gain['mean'])}  ({add_gain['mean'] / total:.0%})\nshapes of single variables,\nno interactions",
            ha="left", va="center", fontsize=NOTE, color=INK)
    ax.hlines(v[0], 1, 3.3, color=INK3, lw=1.2, ls=":", zorder=1)
    ax.annotate("", xy=(3.18, v[2]), xytext=(3.18, v[0]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5, shrinkA=0, shrinkB=0))
    ax.annotate(f"{sgn(int_gain['mean'])}  ({int_gain['mean'] / total:.0%})\ninteractions", xy=(3.18, (v[0] + v[2]) / 2), xytext=(3.45, 0.7765),
                ha="center", va="center", fontsize=NOTE, color=INK, arrowprops=dict(arrowstyle="-", color=INK3, lw=1.2))
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["1\n(additive)", "2", "3\n(frozen)", "4"])
    ax.set_xlim(0.55, 4.35); ax.set_ylim(0.755, 0.795)
    ax.set_xlabel("XGBoost tree depth (rounds re-tuned per depth)")
    if ylabel:
        ax.set_ylabel("5-fold CV ROC-AUC (training)")
    ax.grid(axis="x", visible=False)


def draw_history(ax, ylabel=True, compact=False):
    hxv = np.arange(4)
    ax.plot(hxv, raw, color=PAL["blue"], ls="--", marker="o", ms=10, mfc="white", mec=PAL["blue"], mew=2.2, lw=2.2,
            label="raw months only" if compact else "raw monthly columns only", zorder=3)
    ax.plot(hxv[1:], win, color=PAL["blue"], ls="-", marker="o", ms=10, mec="white", mew=1.8, lw=2.5,
            label="plus summaries" if compact else "plus summaries of the same months", zorder=4)
    WB = dict(boxstyle="square,pad=0.1", fc="white", ec="none")
    ax.text(0.10, raw[0], f"{raw[0]:.3f}", ha="left", va="center", fontsize=NOTE, color=INK2, bbox=WB)
    ax.text(1.0, raw[1] + 0.007, f"{raw[1]:.3f}", ha="center", va="bottom", fontsize=NOTE, color=INK2, bbox=WB)
    ax.text(3.0, raw[3] - 0.007, f"{raw[3]:.3f}", ha="center", va="top", fontsize=NOTE, color=INK2, bbox=WB)
    ax.text(3.0, win[2] + 0.007, f"{win[2]:.3f}", ha="center", va="bottom", fontsize=NOTE, color=INK2, bbox=WB)
    ax.annotate(f"+ September: {raw[0]:.2f} \u2192 {raw[1]:.2f}", xy=(0.5, (raw[0] + raw[1]) / 2), xytext=(1.15, 0.700),
                fontsize=NOTE, color=INK, ha="left", va="center", fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK3, lw=1.2))
    ax.text(1.15, 0.733, f"older months: {sgn(older_raw)} as raw months,\n{sgn(older_win)} with summaries",
            ha="left", va="center", fontsize=NOTE - 1, color=INK2)
    ax.set_xticks(hxv)
    ax.set_xticklabels(["limit +\ndemographics", "+ Sep\n(1 month)", "+ Jul\u2013Sep\n(3 months)", "+ Apr\u2013Sep\n(6 months)"],
                       fontsize=13 if compact else 16)
    ax.set_xlim(-0.25, 3.3); ax.set_ylim(0.60, 0.81)
    ax.set_xlabel("Repayment history given to XGBoost")
    if ylabel:
        ax.set_ylabel("5-fold CV ROC-AUC (training)")
    ax.legend(loc="lower right", fontsize=14 if compact else 15, handlelength=3.2)
    ax.grid(axis="x", visible=False)


fig, ax = plt.subplots(figsize=PANEL)
draw_depth(ax)
ax.set_title("Interactions add little", loc="left", fontsize=18)
save(fig, "s_ablation_depth.png")

fig, ax = plt.subplots(figsize=PANEL)
draw_history(ax)
ax.set_title("Recent history carries the signal", loc="left", fontsize=18)
save(fig, "s_ablation_history.png")

fig, axes = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[1, 1.12]))
draw_depth(axes[0]); axes[0].set_title("(a) Interactions add little", loc="left", fontsize=18)
draw_history(axes[1], compact=True); axes[1].set_title("(b) Recent history carries the signal", loc="left", fontsize=18)
left_title(fig, "Where the gain over the logit comes from: one-variable shapes, latest month")
save(fig, "s_ablation.png", rect=[0, 0, 1, 1])

# ====================================================================================================================
# 5. s_fairness.png — flag-rate ratio vs default-rate ratio by group (training OOF, 5:1 cut-offs)
# ====================================================================================================================
fg = rd("fairness_groups.csv")
comps = [("SEX", "male", "Men vs women"), ("MARRIAGE", "married", "Married vs single"), ("AGE", "<30", "Under 30 vs 30–39")]
series = [("xgb_frozen", "default", "Observed default rate", INK2, "-", True),
          ("xgb_frozen", "flag", "XGBoost: flag rate", PAL["blue"], "-", True),
          ("logit_frozen", "flag", "Logit: flag rate", PAL["orange"], "-", True),
          ("xgb_nodemo", "flag", "XGBoost, no demographics: flag rate", PAL["blue"], "--", False)]
fig, ax = plt.subplots(figsize=SIZE)
ytick, ylab = [], []
gap = 1.0
y0 = 0.0
for attr, grp, name in comps[::-1]:
    sub = []
    for j, (var, what, lab, c, ls, filled) in enumerate(series[::-1]):
        r = fg[(fg.variant == var) & (fg.attribute == attr) & (fg.group == grp)].iloc[0]
        val = r.default_rate_ratio_to_ref if what == "default" else r.flag_rate_ratio_to_ref
        y = y0 + j * 0.62
        sub.append((y, val, c, ls, filled, what))
    dref = [s for s in sub if s[5] == "default"][0][1]
    ax.plot([dref, dref], [sub[0][0] - 0.3, sub[-1][0] + 0.3], color=INK3, lw=1.3, ls=":", zorder=1)
    for y, val, c, ls, filled, what in sub:
        ax.plot([1.0, val], [y, y], color=c, lw=2.6, ls=ls, solid_capstyle="butt", zorder=2)
        ax.plot(val, y, "o", ms=12, mfc=c if filled else "white", mec=c if not filled else "white", mew=2.2, zorder=4)
        side = 1 if val >= 1 else -1
        ax.text(val + side * 0.012, y, f"{val:.2f}", ha="left" if side > 0 else "right", va="center", fontsize=NOTE,
                color=INK, fontweight="bold" if what == "default" else "normal",
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=5)
    ytick.append(np.mean([s[0] for s in sub])); ylab.append(name)
    y0 = sub[-1][0] + 0.62 + gap
ax.axvline(1.0, color=INK, lw=1.2, zorder=1)
ax.set_yticks(ytick); ax.set_yticklabels(ylab, fontsize=17)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.92, 1.47); ax.set_ylim(-0.7, y0 - gap + 0.15)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Ratio to the reference group: women / single / 30–39   (1.0 = same rate)\nflags at each model's 5:1 cut-off, training out-of-fold")
handles = [Line2D([], [], color=c, ls=ls, lw=2.6, marker="o", ms=11, mfc=c if f else "white", mec=c if not f else "white", mew=2, label=lab)
           for _, _, lab, c, ls, f in series]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=14, handlelength=3.2, columnspacing=1.0,
           handletextpad=0.4)
ax.annotate("without the attributes,\nthe age gap widens", xy=(1.205, 0.0), xytext=(1.235, 0.12), ha="left", va="center",
            fontsize=NOTE - 1, color=INK2)
left_title(fig, "Who gets flagged: XGBoost tracks each group's default rate; the logit distorts it")
save(fig, "s_fairness.png", rect=[0, 0.05, 1, 1])

# ====================================================================================================================
# 6. s_simple_rule.png — 5:1 cost per 1,000 clients: one-line rules vs logit vs XGBoost (training OOF, cross-fitted)
# ====================================================================================================================
sr = rd("simple_rule_benchmark.csv").set_index("rule_or_model")
XG, LG, AD = "XGBoost at OOF 5:1 cut-off", "Logistic (plain) at OOF 5:1 cut-off", "N_MONTHS_DELAYED >= 1"
lab = {XG: "XGBoost", LG: "Logistic regression", AD: "Rule: any delay, 6 months",
       "MAX_DELAY >= 2": "Rule: 2+ months late, 6 months", "PAY_1 >= 1": "Rule: late in September",
       "flag everybody": "Flag everybody", "PAY_1 >= 2": "Rule: 2+ months late, September"}
sr = sr.loc[list(lab)]
cost = sr.cost_5to1_crossfitted_per_1000.astype(float)
order = cost.sort_values().index[::-1]          # cheapest on top
colr = {XG: PAL["blue"], LG: PAL["orange"], AD: INK2}
fig, ax = plt.subplots(figsize=SIZE)
yy = np.arange(len(order))
ax.barh(yy, cost[order], height=0.58, color=[colr.get(k, NEUTRAL) for k in order], zorder=2)
for y, k in zip(yy, order):
    ax.text(cost[k] + 10, y, f"{cost[k]:,.0f}", ha="left", va="center", fontsize=17, color=INK,
            fontweight="bold" if k in colr else "normal")
rule = sr.loc[AD]
yx = list(order).index(XG); yl = list(order).index(LG)
xs = rule.xgb_saving_vs_rule_per_1000_mean
ax.text(675, yx + 0.1,
        f"{xs:.0f} below the any-delay rule\n"
        f"({xs / rule.cost_5to1_crossfitted_per_1000:.1%}), cheaper in {int(rule.xgb_cheaper_folds_of_5)}/5 folds",
        ha="left", va="center", fontsize=NOTE, color=INK)
ax.text(675, yl - 0.1,
        (f"{rule.logit_saving_vs_rule_per_1000_mean:.0f} below the rule\n"
         f"(interval {rule.logit_saving_vs_rule_nb_lo:.0f} to {rule.logit_saving_vs_rule_nb_hi:+.0f}): a tie").replace("-", MINUS),
        ha="left", va="center", fontsize=NOTE, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([lab[k] for k in order], fontsize=17)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1150); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xticks([0, 200, 400, 600, 800])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(axis="y", visible=False)
ax.set_xlabel("Cost per 1,000 clients (missed defaulter = 5, false alarm = 1)\n5:1 scenario, training out-of-fold")
left_title(fig, "A one-line arrears rule is a hard benchmark: the logit ties it, XGBoost beats it")
save(fig, "s_simple_rule.png")

# ====================================================================================================================
# 7. s_ratio_cutoff.png — cost-minimising cut-off and cheapest model by cost ratio (pooled training OOF)
# ====================================================================================================================
pr = rd("preferred_model_by_ratio_oof.csv")
short = {"Gradient boosting": "Gradient\nboosting", "Random forest": "Random\nforest", "XGBoost": "XGBoost", "KNN (K=101)": "KNN"}
fam = {"gradient_boosting": "boosting", "xgboost": "boosting", "random_forest": "forest / bagging", "knn": "KNN"}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=SIZE, sharex=True, gridspec_kw=dict(height_ratios=[3.0, 1.15]))
rr = np.geomspace(0.9, 22, 200)
ax.plot(rr, 1 / (1 + rr), color=INK3, lw=1.8, ls="--", zorder=2, label="theory for calibrated scores: 1 / (1 + ratio)")
ax.plot(pr.ratio, pr.preferred_cutoff, color=INK2, lw=2.2, zorder=3, label="cut-off that minimises cost (training out-of-fold)")
for _, r in pr.iterrows():
    ax.plot(r.ratio, r.preferred_cutoff, "o", ms=14, color=FAMILY_COLOR[fam[r.preferred_slug]], mec="white", mew=2, zorder=5)
first, last, five = pr.iloc[0], pr.iloc[-1], pr[pr.ratio == 5].iloc[0]
lbl = lambda r_: f"cut-off {r_.preferred_cutoff:.2f}, flags {r_.preferred_flag_rate:.0%}"
ax.annotate(lbl(first), (first.ratio, first.preferred_cutoff), xytext=(16, 2), textcoords="offset points", ha="left", va="bottom",
            fontsize=NOTE, color=INK, fontweight="bold")
ax.annotate(lbl(five), (five.ratio, five.preferred_cutoff), xytext=(14, 8), textcoords="offset points", ha="left", va="bottom",
            fontsize=NOTE, color=INK, fontweight="bold")
ax.annotate(lbl(last), (last.ratio, last.preferred_cutoff), xytext=(24, 0.27), textcoords="data", ha="right", va="bottom",
            fontsize=NOTE, color=INK, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK3, lw=1.2, shrinkB=8))
ax.axvline(5, color=INK3, lw=1.2, ls=":", zorder=1)
ax.text(4.8, 0.465, "5:1 scenario, borrowed\nfrom German credit data", fontsize=NOTE - 1, color=INK2, ha="right", va="top")
ax.text(0.86, 0.0, f"lead over the runner-up: {pr.margin_pct.min():.1f}\u2013{pr.margin_pct.max():.1f}%;\n"
        "no fold-level interval excludes zero", fontsize=NOTE, color=INK, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc=WASH, ec="none"))
ax.set_xscale("log"); ax.minorticks_off()
ax.set_xlim(0.8, 25); ax.set_ylim(-0.02, 0.62)
ax.set_ylabel("Cut-off on P(default)")
ax.legend(loc="upper right", fontsize=15)
ax.grid(axis="x", visible=False)
# strip: cheapest model and its lead at each ratio
for _, r in pr.iterrows():
    ax2.plot(r.ratio, 0.86, "o", ms=13, color=FAMILY_COLOR[fam[r.preferred_slug]], mec="white", mew=2, clip_on=False)
    ax2.text(r.ratio, 0.68, short[r.preferred_oof], ha="center", va="top", fontsize=15, color=INK, linespacing=0.9)
    ax2.text(r.ratio, 0.02, f"+{r.margin_pct:.1f}%", ha="center", va="bottom", fontsize=15, color=INK2)
ax2.set_ylim(0, 1); ax2.set_yticks([0.6, 0.1]); ax2.set_yticklabels(["cheapest\nmodel", "its lead"], fontsize=15)
ax2.tick_params(axis="y", length=0)
ax2.grid(False); ax2.spines["left"].set_visible(False)
ax2.set_xticks(pr.ratio); ax2.set_xticklabels([f"{int(r)}:1" for r in pr.ratio])
ax2.set_xlabel("Assumed cost ratio, missed defaulter : false alarm")
left_title(fig, "Cut-off falls from 0.49 to 0.03 as misses get costlier; no model stays cheapest")
save(fig, "s_ratio_cutoff.png", rect=[0, 0, 1, 1])
# ====================================================================================================================
# 8. s_monotone.png — consistent PAY_1 sweep, free XGBoost (200 training clients) | CV AUC price of two constraints
# ====================================================================================================================
mc = rd("monotone_check_consistent.csv")
sw = mc[(mc.model == "free_numeric") & (mc.profiles == "random_200") & (mc["mode"] == "consistent") & (mc.feature == "PAY_1")].set_index("step")
steps = ["-2->-1", "-1->0", "0->1", "1->2", "2->3", "3->4", "4->5"]
sw = sw.loc[steps]
npro = int(sw.n_profiles.iloc[0])
mj = rd("monotone_respec_cv.json")["paired_vs_frozen_oof"]
price = [("mono_frozen_numeric", "Frozen constraint:\nalso orders \u22122/\u22121/0\nand utilisation"),
         ("mono_respec", "Lateness and limit\nonly")]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[2.15, 1]))
xs = np.arange(len(steps))
for lo_, hi_, lab_, shade in [(-0.5, 1.5, "not late:\ndata fall too", True), (1.5, 4.5, "into arrears\nand deeper", False),
                              (4.5, 6.5, "thin cells (< 100):\ndata fall too", True)]:
    if shade:
        ax.axvspan(lo_, hi_, color=WASH, zorder=0, lw=0)
    ax.text((lo_ + hi_) / 2, 284, lab_, ha="center", va="top", fontsize=NOTE, color=INK2)
nv = sw.n_violating.values.astype(int)
ax.bar(xs, nv, width=0.56, color=PAL["blue"], zorder=2)
for xi, n_ in zip(xs, nv):
    ax.text(xi, n_ + 4, f"{n_}", ha="center", va="bottom", fontsize=17, color=INK, fontweight="bold" if n_ == 0 else "normal")
ax.text(3.0, 110, f"0 of {npro} at each step:\nnever lowers\npredicted risk", ha="center", va="center", fontsize=NOTE, color=INK,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK3, lw=1))
ticks = []
for a, (_, r) in zip(steps, sw.iterrows()):
    code_from, code_to = a.split("->")
    ticks.append(code_from.replace("-", MINUS) + " \u2192 " + code_to.replace("-", MINUS) + "\n"
                 + f"{r.default_rate_from:.0%}".rstrip("%") + "\u2192" + f"{r.default_rate_to:.0%}")
ax.set_xticks(xs)
ax.set_xticklabels(ticks, fontsize=15)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_xlim(-0.5, len(steps) - 0.5); ax.set_ylim(0, 290)
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_ylabel(f"Clients whose predicted risk falls\n(of {npro} training clients)")
ax.set_xlabel("Status step (below it: observed default rate)", fontsize=16)
ax.set_title("Free XGBoost, September status one step worse", loc="left", fontsize=18)
ax.grid(axis="x", visible=False)

yy = np.array([1.45, 0.0])
for (key, lab_), y in zip(price, yy):
    r = mj[key]["auc_minus_frozen_free"]
    ax2.plot([r["nb_lo"], r["nb_hi"]], [y, y], color=PAL["blue"], lw=3.5, solid_capstyle="round", zorder=3)
    ax2.plot(r["mean"], y, "o", ms=13, color=PAL["blue"], mec="white", mew=2, zorder=4)
    val = sgn(r["mean"]) if abs(r["mean"]) >= 0.001 else "under 0.001"
    ax2.text(-0.00345, y + 0.25, val + ("" if r["nb_hi"] < 0 else ": a tie"), ha="left", va="bottom", fontsize=NOTE + 1, color=INK,
             fontweight="bold")
    ax2.text(-0.00345, y - 0.17, lab_, ha="left", va="top", fontsize=NOTE - 1, color=INK2)
ax2.axvline(0, color=INK, lw=1.2, zorder=2)
ax2.text(0.0001, 2.35, "free\nXGBoost", ha="left", va="top", fontsize=NOTE - 2, color=INK2)
ax2.set_xlim(-0.0035, 0.0011); ax2.set_ylim(-0.95, 2.4)
ax2.set_xticks([-0.002, 0])
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else sgn(v, 3)))
ax2.set_yticks([])
ax2.spines["left"].set_visible(False)
ax2.set_xlabel("CV AUC vs free XGBoost\n(95% interval, training folds)", fontsize=15)
ax2.set_title("Price of a constraint", loc="left", fontsize=18)
left_title(fig, "Free XGBoost already respects lateness; forcing an order on \u22122/\u22121/0 costs AUC")
save(fig, "s_monotone.png", rect=[0, 0, 1, 1])
print("done ->", OUT)
