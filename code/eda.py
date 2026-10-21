"""
eda.py — Exploratory data analysis for the credit-card default project (NO models are fitted here).
Owner: shared.  Run from Project/code:  python3 eda.py

Data-hygiene convention used throughout this script
---------------------------------------------------
* Data-quality checks and the target distribution describe the RAW file (30,000 rows) and the cleaned
  file (29,965 rows) — these are facts about the dataset, not about any model.
* Every bivariate / relationship analysis (default rate by group, distributions by class, correlations,
  univariate discriminative power) is computed on the TRAINING partition only (20,975 rows from
  common.load_split()). The held-out test set is never looked at, so nothing learnt here (e.g. which
  bands to use, which features look strong) can leak into model evaluation.

Outputs (all names start with eda_):
  figures/eda_target_distribution.png        (1) class balance: full / train / test, counts + %
  figures/eda_default_rate_by_pay1.png       (2) default rate by PAY_1 status, counts as labels
  figures/eda_default_rate_by_groups.png     (3) 2x3 panel: LIMIT_BAL quintile, AGE band, EDUCATION, MARRIAGE, SEX, PAY_AMT1 band
  figures/eda_distributions_by_default.png   (4) LIMIT_BAL, BILL_AMT1, PAY_AMT1 by class (log scale where needed)
  figures/eda_correlation_heatmap.png        (5) Pearson correlation of the 26 numeric features + target
  figures/eda_default_rate_util_delay.png    (6) default rate vs UTIL_1 decile and vs N_MONTHS_DELAYED
  figures/eda_univariate_strength.png        (extra) univariate AUC of every numeric feature (train only)
  results/eda_data_quality.csv               (7) compact data-quality table
  results/eda_summary.json                   every number a report writer needs
  results/eda_variables.csv                  variable description table (29 features + target)
"""
import json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from sklearn.metrics import roc_auc_score
from common import (load_raw, clean, load_split, save_extra_figure, RESULTS, FIGURES, SEED, TARGET,
                    CAT_COLS, PAY_COLS, BILL_COLS, PAYAMT_COLS, ENG_COLS, NUM_COLS)

MEMBER = "shared"
np.random.seed(SEED)                          # nothing random is used, but keep the project convention
T0 = time.time()

# ----------------------------------------------------------------------------- palette & chrome
# Validated reference palette (dataviz skill): categorical slot 1 blue / slot 2 orange for the two classes,
# one blue hue for single-series rate bars, blue <-> gray <-> red diverging ramp for the correlation heatmap.
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_NO, C_YES = "#2a78d6", "#eb6834"            # non-default (0) / default (1)
C_RATE = "#2a78d6"                            # single-series bars
C_ENG = "#eb6834"                             # engineered features in the univariate figure
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, color=AXIS)

def pct_axis(ax, top=None):
    ax.set_ylabel("default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if top is not None:
        ax.set_ylim(0, top)

def rate_bar(ax, tab, title, xlabel, overall, top=None, rotate=0, n_fmt="n={n:,}"):
    """Single-series bar of default rate per group; rate above each bar, group size as a small second line."""
    x = np.arange(len(tab))
    ax.bar(x, tab["rate"].values, width=0.78, color=C_RATE, linewidth=0)
    ax.axhline(overall, color=MUTED, linewidth=0.9, label=f"overall default rate {overall:.1%}")
    ax.legend(loc="upper center", handlelength=1.6)
    ymax = max(tab["rate"].max(), overall)
    for i, (n, r) in enumerate(zip(tab["n"].values, tab["rate"].values)):
        ax.text(i, r + ymax * 0.02, f"{r:.0%}\n" + n_fmt.format(n=int(n)), ha="center", va="bottom", fontsize=7.2, color=INK2, linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in tab.index], rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title); ax.set_xlabel(xlabel)
    pct_axis(ax, top or ymax * 1.28)
    style(ax)

def group_rate(g, y):
    t = pd.DataFrame({"g": g, "y": y}).groupby("g", observed=True)["y"].agg(n="size", rate="mean")
    return t

def to_json_table(t):
    return {str(k): {"n": int(v["n"]), "rate": round(float(v["rate"]), 4)} for k, v in t.iterrows()}

def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

# ----------------------------------------------------------------------------- data
raw = load_raw()                               # 30,000 x 24 (ID dropped): 23 raw features + target
df = clean(raw)                                # 29,965 x 30: duplicates dropped, codes recoded, 6 engineered features added
n_dup = int(df.attrs["n_duplicates_dropped"])
X_tr, X_te, y_tr, y_te, feature_names = load_split()
tr = X_tr.assign(**{TARGET: y_tr.values})     # TRAINING partition: all relationship analysis below uses `tr` only
y = tr[TARGET].astype(int)
overall_tr = float(y.mean())
print(f"raw {raw.shape} | cleaned {df.shape} (dropped {n_dup} duplicates) | train {X_tr.shape} | test {X_te.shape}")
assert list(X_tr.columns) == feature_names and len(feature_names) == 29

# ----------------------------------------------------------------------------- (7) data-quality table (raw + cleaned)
edu_codes = raw["EDUCATION"].value_counts().to_dict(); mar_codes = raw["MARRIAGE"].value_counts().to_dict()
neg_bill_rows = int((raw[BILL_COLS] < 0).any(axis=1).sum())
neg_bill_cells = int((raw[BILL_COLS] < 0).sum().sum())
zero_pay_any = int((raw[PAYAMT_COLS] == 0).any(axis=1).sum())
zero_pay_all = int((raw[PAYAMT_COLS] == 0).all(axis=1).sum())
zero_bill_all = int((raw[BILL_COLS] == 0).all(axis=1).sum())
nonpos_bill_all_clean = int((df[BILL_COLS] <= 0).all(axis=1).sum())
pay_undoc = {c: {"-2": int((raw[c] == -2).sum()), "0": int((raw[c] == 0).sum())} for c in PAY_COLS}
over_limit = int((df["UTIL_1"] > 1).sum())
dq_rows = [
    ("Missing values", "0 cells (all 24 columns complete)", "None needed — no imputation step in the pipeline"),
    ("Exact duplicate rows", f"{n_dup} of 30,000 (0.12%)", "Dropped in common.clean() → 29,965 rows"),
    ("Data types", "All 24 columns integer; no strings, no dates", "None"),
    ("EDUCATION undocumented codes 0/5/6", f"0: {edu_codes.get(0,0)}, 5: {edu_codes.get(5,0)}, 6: {edu_codes.get(6,0)} (total {sum(edu_codes.get(k,0) for k in (0,5,6))}, 1.15%)", "Recoded to 4 = others/unknown"),
    ("MARRIAGE undocumented code 0", f"{mar_codes.get(0,0)} rows (0.18%)", "Recoded to 3 = others"),
    ("PAY_1..6 undocumented codes -2 and 0", f"PAY_1: -2 in {pay_undoc['PAY_1']['-2']:,} rows, 0 in {pay_undoc['PAY_1']['0']:,} rows (codebook lists only -1 and 1..9)", "Kept as ordinal: -2 = no consumption, -1 = paid in full, 0 = revolving credit / minimum paid, k>0 = k months late"),
    ("Negative bill amounts", f"{neg_bill_cells:,} cells in {neg_bill_rows:,} rows (6.4%); min BILL_AMT = {int(raw[BILL_COLS].min().min()):,}", "Kept — a negative bill is a credit balance (overpayment / refund)"),
    ("Zero payments", f"PAY_AMT = 0 in at least one month for {zero_pay_any:,} rows (48.5%); all six months zero for {zero_pay_all:,} rows (4.8%)", "Kept — 'no payment' is informative; PAY_RATIO_MEAN handles zero bills"),
    ("Zero bills in all six months", f"{zero_bill_all:,} of the raw 30,000 rows "
     f"({100 * zero_bill_all / len(raw):.1f}%); {nonpos_bill_all_clean:,} of the {len(df):,} cleaned rows "
     f"({100 * nonpos_bill_all_clean / len(df):.2f}%) have all six bills NON-POSITIVE",
     "Kept — inactive or credit-balance accounts. For the non-positive group PAY_RATIO_MEAN is "
     "undefined and imputed 0; in the training partition that group defaults at "
     f"{100 * tr.loc[(tr[BILL_COLS] <= 0).all(axis=1), TARGET].mean():.1f}% against "
     f"{100 * tr.loc[~(tr[BILL_COLS] <= 0).all(axis=1), TARGET].mean():.1f}% for the rest, so the "
     "imputed value is informative rather than neutral"),
    ("Bill above credit limit (UTIL_1 > 1)", f"{over_limit:,} rows (7.1%); max UTIL_1 = {df['UTIL_1'].max():.2f}", "Kept — over-limit usage is a genuine risk signal"),
    ("LIMIT_BAL range", f"{int(raw['LIMIT_BAL'].min()):,} – {int(raw['LIMIT_BAL'].max()):,} NT$ (median {int(raw['LIMIT_BAL'].median()):,})", "Plausible; heavy right tail → StandardScaler inside the pipeline, no trimming"),
    ("AGE range", f"{int(raw['AGE'].min())} – {int(raw['AGE'].max())} years (median {int(raw['AGE'].median())})", "Plausible; no outlier removal"),
    ("Class imbalance", f"{int(df[TARGET].sum()):,} defaults of {len(df):,} = {df[TARGET].mean():.1%}", "Kept as is; addressed downstream by class_weight / threshold tuning; AUC & AP as metrics"),
    ("Train / test split", f"{len(X_tr):,} / {len(X_te):,} rows, stratified 70/30, seed {SEED}; default rate {y_tr.mean():.2%} / {y_te.mean():.2%}", "Fixed once in data/processed; test set untouched in this EDA"),
]
dq = pd.DataFrame(dq_rows, columns=["check", "finding", "action"])
dq.to_csv(os.path.join(RESULTS, "eda_data_quality.csv"), index=False)

# ----------------------------------------------------------------------------- (1) target distribution
fig, ax = plt.subplots(figsize=(7.2, 4))
parts = [("Full (cleaned)", df[TARGET]), ("Train", y_tr), ("Test", y_te)]
w = 0.36; x = np.arange(len(parts))
for j, (lab, col) in enumerate([("No default (0)", C_NO), ("Default (1)", C_YES)]):
    vals = [int((s == j).sum()) for _, s in parts]
    tot = [len(s) for _, s in parts]
    bars = ax.bar(x + (j - 0.5) * (w + 0.03), vals, width=w, color=col, linewidth=0, label=lab)
    for b, v, t in zip(bars, vals, tot):
        ax.text(b.get_x() + b.get_width() / 2, v + 250, f"{v:,}\n({v / t:.1%})", ha="center", va="bottom", fontsize=8, color=INK2, linespacing=1.1)
ax.set_xticks(x); ax.set_xticklabels([f"{lab}\nn = {len(s):,}" for lab, s in parts])
ax.set_ylabel("clients"); ax.set_ylim(0, 27500)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
ax.set_title("Target distribution — 22.1% of clients default on next month's payment")
ax.legend(loc="upper right"); style(ax)
save_extra_figure(fig, "eda_target_distribution.png")

# ----------------------------------------------------------------------------- (2) default rate by PAY_1 (train)
pay1_tab = group_rate(tr["PAY_1"], y)
fig, ax = plt.subplots(figsize=(9.6, 4.4))
rate_bar(ax, pay1_tab, "Default rate by PAY_1 — repayment status in September 2005, the most recent month (training set)",
         "PAY_1:  −2 no consumption · −1 paid in full · 0 revolving / minimum paid · k ≥ 1 = k months late", overall_tr, top=1.0)
save_extra_figure(fig, "eda_default_rate_by_pay1.png")

# ----------------------------------------------------------------------------- (3) 2x3 panel of group default rates (train)
lim_q = pd.qcut(tr["LIMIT_BAL"], 5, duplicates="drop")
lim_lab = {iv: f"{iv.left / 1000:.0f}k–{iv.right / 1000:.0f}k" for iv in lim_q.cat.categories}
lim_tab = group_rate(lim_q.map(lim_lab).astype(pd.CategoricalDtype([lim_lab[iv] for iv in lim_q.cat.categories], ordered=True)), y)
age_bins = [20, 25, 30, 35, 40, 50, 60, 100]; age_labels = ["21–24", "25–29", "30–34", "35–39", "40–49", "50–59", "60+"]
age_tab = group_rate(pd.cut(tr["AGE"], bins=age_bins, labels=age_labels, right=False), y)
edu_tab = group_rate(tr["EDUCATION"].map({1: "grad school", 2: "university", 3: "high school", 4: "others"}).astype(pd.CategoricalDtype(["grad school", "university", "high school", "others"], ordered=True)), y)
mar_tab = group_rate(tr["MARRIAGE"].map({1: "married", 2: "single", 3: "others"}).astype(pd.CategoricalDtype(["married", "single", "others"], ordered=True)), y)
sex_tab = group_rate(tr["SEX"].map({1: "male", 2: "female"}).astype(pd.CategoricalDtype(["male", "female"], ordered=True)), y)
pa_bins = [-1, 0, 1000, 2000, 5000, 10000, np.inf]; pa_labels = ["0", "1–1k", "1k–2k", "2k–5k", "5k–10k", ">10k"]
pa_tab = group_rate(pd.cut(tr["PAY_AMT1"], bins=pa_bins, labels=pa_labels), y)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
rate_bar(axes[0, 0], lim_tab, "by LIMIT_BAL quintile", "credit limit (NT$)", overall_tr, top=0.46)
rate_bar(axes[0, 1], age_tab, "by AGE band", "age (years)", overall_tr, top=0.46)
rate_bar(axes[0, 2], edu_tab, "by EDUCATION", "", overall_tr, top=0.46)
rate_bar(axes[1, 0], mar_tab, "by MARRIAGE", "", overall_tr, top=0.46)
rate_bar(axes[1, 1], sex_tab, "by SEX", "", overall_tr, top=0.46)
rate_bar(axes[1, 2], pa_tab, "by PAY_AMT1 band (latest payment)", "payment in September 2005 (NT$)", overall_tr, top=0.46)
fig.suptitle("Default rate by client characteristics (training set, n = 20,975)", fontsize=12, fontweight="bold")
save_extra_figure(fig, "eda_default_rate_by_groups.png")

# ----------------------------------------------------------------------------- (4) distributions by class (train)
def hist_by_class(ax, series, title, xlabel, bins, transform=None, xfmt=None):
    vals = series if transform is None else transform(series)
    for j, (lab, col) in enumerate([("no default", C_NO), ("default", C_YES)]):
        v = vals[y.values == j]
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=col, linewidth=0, label=lab)
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, linewidth=1.4)
        med = np.median(series[y.values == j])
        ax.axvline(np.median(v), color=col, linewidth=1.2)
        ax.text(np.median(v), ax.get_ylim()[1] * (0.97 - 0.09 * j), f" median {lab}: {med:,.0f}", color=col, fontsize=8, va="top")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("density")
    if xfmt: ax.xaxis.set_major_formatter(xfmt)
    style(ax)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
hist_by_class(axes[0], tr["LIMIT_BAL"].values, "LIMIT_BAL (credit limit)", "NT$ (thousands)", bins=np.arange(0, 801, 20),
              transform=lambda s: s / 1000, xfmt=matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
axes[0].set_xlim(0, 800)
lg = lambda s: np.log10(1 + np.clip(s, 0, None))
hist_by_class(axes[1], tr["BILL_AMT1"].values, "BILL_AMT1 (latest bill), log scale", "log10(1 + bill), negative/zero bills shown at 0", bins=np.linspace(0, 6.2, 50), transform=lg)
hist_by_class(axes[2], tr["PAY_AMT1"].values, "PAY_AMT1 (latest payment), log scale", "log10(1 + payment), zero payments shown at 0", bins=np.linspace(0, 6.2, 50), transform=lg)
axes[0].legend(loc="center right")
fig.suptitle("Distributions by outcome (training set): defaulters have lower limits, similar bills and much smaller payments", fontsize=11.5, fontweight="bold")
save_extra_figure(fig, "eda_distributions_by_default.png")

# ----------------------------------------------------------------------------- (5) correlation heatmap (train)
corr_cols = NUM_COLS + [TARGET]
corr = tr[corr_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90, fontsize=8); ax.set_yticklabels(corr_cols, fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.values[i, j]
        rgb = DIVERGING((v + 1) / 2); lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6, color="white" if lum < 0.55 else INK)
# thin separators between variable blocks
for pos in np.cumsum([2, 6, 6, 6, 6]) - 0.5:
    ax.axhline(pos, color=SURF, linewidth=2); ax.axvline(pos, color=SURF, linewidth=2)
ax.axhline(len(corr_cols) - 1.5, color=SURF, linewidth=3); ax.axvline(len(corr_cols) - 1.5, color=SURF, linewidth=3)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Pearson r"); cb.outline.set_visible(False)
ax.set_title("Pearson correlation — 26 numeric features (incl. 6 engineered) and the target, training set", pad=12)
ax.tick_params(length=0)
save_extra_figure(fig, "eda_correlation_heatmap.png")

# ----------------------------------------------------------------------------- (6) UTIL_1 decile & N_MONTHS_DELAYED (train)
util_q = pd.qcut(tr["UTIL_1"], 10, duplicates="drop")
util_lab = {iv: f"{max(iv.left, tr['UTIL_1'].min()):.2f}–{iv.right:.2f}" for iv in util_q.cat.categories}
util_tab = group_rate(util_q.map(util_lab).astype(pd.CategoricalDtype([util_lab[iv] for iv in util_q.cat.categories], ordered=True)), y)
nmd_tab = group_rate(tr["N_MONTHS_DELAYED"], y)
maxd_tab = group_rate(tr["MAX_DELAY"], y)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
rate_bar(axes[0], util_tab, f"Default rate by UTIL_1 decile (latest bill / credit limit) — {len(util_tab)} bins after tie-dropping", "UTIL_1 range", overall_tr, top=0.9, rotate=30)
rate_bar(axes[1], nmd_tab, "Default rate by N_MONTHS_DELAYED (delayed months out of 6)", "number of months with PAY_k > 0", overall_tr, top=0.9)
save_extra_figure(fig, "eda_default_rate_util_delay.png")

# ----------------------------------------------------------------------------- (extra) univariate discriminative power (train)
uni = []
for c in NUM_COLS:
    a = float(roc_auc_score(y, tr[c]))
    uni.append(dict(feature=c, auc_raw=round(a, 4), auc_oriented=round(max(a, 1 - a), 4), direction="higher → more default" if a >= 0.5 else "higher → less default",
                    engineered=c in ENG_COLS))
uni_df = pd.DataFrame(uni).sort_values("auc_oriented", ascending=True)
fig, ax = plt.subplots(figsize=(8, 7.6))
cols = [C_ENG if e else C_RATE for e in uni_df["engineered"]]
ax.barh(uni_df["feature"], uni_df["auc_oriented"] - 0.5, left=0.5, color=cols, height=0.72, linewidth=0)
for i, (v, d) in enumerate(zip(uni_df["auc_oriented"], uni_df["direction"])):
    ax.text(v + 0.003, i, f"{v:.3f}  ({'+' if d.startswith('higher → more') else '−'})", va="center", fontsize=7.8, color=INK2)
ax.axvline(0.5, color=AXIS, linewidth=0.9)
ax.set_xlim(0.5, 0.78); ax.set_xlabel("univariate AUC, orientation-corrected:  (+) higher → more default,  (−) higher → less default")
ax.set_title("Univariate AUC of each numeric feature (training set, no model fitted)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_RATE, label="raw feature"), Patch(color=C_ENG, label="engineered feature")], loc="lower right")
ax.tick_params(axis="y", labelsize=8.5); style(ax, ygrid=False); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
save_extra_figure(fig, "eda_univariate_strength.png")

# ----------------------------------------------------------------------------- variable description table
month = {1: "Sep 2005", 2: "Aug 2005", 3: "Jul 2005", 4: "Jun 2005", 5: "May 2005", 6: "Apr 2005"}
var_rows = [
    ("LIMIT_BAL", "numeric (NT$)", "Amount of credit given (individual + family/supplementary credit)", "feature — numeric, standardised"),
    ("SEX", "categorical", "1 = male, 2 = female", "feature — categorical, one-hot"),
    ("EDUCATION", "categorical", "1 = graduate school, 2 = university, 3 = high school, 4 = others (raw codes 0/5/6 recoded to 4)", "feature — categorical, one-hot"),
    ("MARRIAGE", "categorical", "1 = married, 2 = single, 3 = others (raw code 0 recoded to 3)", "feature — categorical, one-hot"),
    ("AGE", "numeric (years)", "Age in years (21–79)", "feature — numeric, standardised"),
]
for i in range(1, 7):
    var_rows.append((f"PAY_{i}", "ordinal integer (−2..8)", f"Repayment status in {month[i]}: −2 no consumption, −1 paid in full, 0 revolving credit / minimum paid, k ≥ 1 = payment k months late (UCI name PAY_{0 if i == 1 else i})", "feature — numeric (ordinal), standardised"))
for i in range(1, 7):
    var_rows.append((f"BILL_AMT{i}", "numeric (NT$)", f"Bill statement amount in {month[i]} (negative = credit balance)", "feature — numeric, standardised"))
for i in range(1, 7):
    var_rows.append((f"PAY_AMT{i}", "numeric (NT$)", f"Amount paid in {month[i]} (previous payment)", "feature — numeric, standardised"))
var_rows += [
    ("UTIL_1", "engineered numeric", "Credit utilisation, latest month: BILL_AMT1 / LIMIT_BAL", "feature — engineered, standardised"),
    ("UTIL_MEAN", "engineered numeric", "Mean utilisation over 6 months: mean(BILL_AMT1..6) / LIMIT_BAL", "feature — engineered, standardised"),
    ("PAY_RATIO_MEAN", "engineered numeric", "Mean of PAY_AMTk / BILL_AMTk over months with a positive bill, clipped to [0, 5]; 0 if no positive bill", "feature — engineered, standardised"),
    ("N_MONTHS_DELAYED", "engineered integer (0–6)", "Number of the 6 months with a payment delay (PAY_k > 0)", "feature — engineered, standardised"),
    ("MAX_DELAY", "engineered integer (0–8)", "Worst repayment status over the 6 months: max(PAY_1..6) clipped at 0", "feature — engineered, standardised"),
    ("BILL_TREND", "engineered numeric", "Balance growth relative to limit: (BILL_AMT1 − BILL_AMT6) / LIMIT_BAL", "feature — engineered, standardised"),
    ("DEFAULT", "binary target", "1 = defaulted on the payment due in October 2005, 0 = paid (UCI name 'default payment next month')", "target"),
]
vars_df = pd.DataFrame(var_rows, columns=["name", "type", "description", "role"])
assert set(vars_df["name"]) == set(feature_names) | {TARGET} and len(vars_df) == 30
vars_df.to_csv(os.path.join(RESULTS, "eda_variables.csv"), index=False)

# ----------------------------------------------------------------------------- summary JSON
corr_t = corr[TARGET].drop(TARGET).sort_values(ascending=False)
spear_t = tr[corr_cols].corr(method="spearman")[TARGET].drop(TARGET).sort_values(ascending=False)
feat_corr = corr.drop(index=TARGET, columns=TARGET)
pairs = [(a, b, float(feat_corr.loc[a, b])) for i, a in enumerate(feat_corr.index) for b in feat_corr.columns[i + 1:]]
pairs.sort(key=lambda t: -abs(t[2]))
n_pairs_over_08 = sum(abs(p[2]) > 0.8 for p in pairs)
bill_block = feat_corr.loc[BILL_COLS, BILL_COLS].values[np.triu_indices(6, 1)]
pay_block = feat_corr.loc[PAY_COLS, PAY_COLS].values[np.triu_indices(6, 1)]
payamt_block = feat_corr.loc[PAYAMT_COLS, PAYAMT_COLS].values[np.triu_indices(6, 1)]

def by_class(cols):
    out = {}
    for c in cols:
        g = tr.groupby(TARGET)[c]
        out[c] = {"mean_0": r4(g.mean()[0]), "mean_1": r4(g.mean()[1]), "median_0": r4(g.median()[0]), "median_1": r4(g.median()[1]),
                  "min": r4(tr[c].min()), "max": r4(tr[c].max()), "std": r4(tr[c].std())}
    return out

eng_summary = {c: {**{k: r4(v) for k, v in tr[c].describe().to_dict().items()}, "pearson_with_target": r4(corr_t[c]), "spearman_with_target": r4(spear_t[c]),
                   "univariate_auc": next(u["auc_oriented"] for u in uni if u["feature"] == c)} for c in ENG_COLS}
summary = {
    "component": "eda", "member": MEMBER, "seed": SEED,
    "analysis_partition": "Data quality and target distribution: raw/cleaned full data. All group default rates, distributions, "
                          "correlations and univariate AUCs: TRAINING partition only (test set never inspected).",
    "dataset": {"source": "UCI Default of Credit Card Clients (Yeh & Lien 2009), Taiwan, Apr–Sep 2005", "n_raw_rows": int(len(raw)),
                "n_raw_cols_excl_id": int(raw.shape[1]), "n_raw_features": int(raw.shape[1] - 1), "n_duplicates_dropped": n_dup,
                "n_after_dedup": int(len(df)), "n_features_final": len(feature_names), "n_engineered": len(ENG_COLS),
                "n_numeric_features": len(NUM_COLS), "n_categorical_features": len(CAT_COLS), "missing_cells": int(raw.isna().sum().sum())},
    "class_balance": {
        "raw": {"n": int(len(raw)), "n_default": int(raw[TARGET].sum()), "rate": r4(raw[TARGET].mean())},
        "cleaned": {"n": int(len(df)), "n_default": int(df[TARGET].sum()), "n_no_default": int((df[TARGET] == 0).sum()), "rate": r4(df[TARGET].mean())},
        "train": {"n": int(len(y_tr)), "n_default": int(y_tr.sum()), "rate": r4(y_tr.mean())},
        "test": {"n": int(len(y_te)), "n_default": int(y_te.sum()), "rate": r4(y_te.mean())},
        "majority_class_accuracy": r4(1 - df[TARGET].mean()), "imbalance_ratio_neg_per_pos": r4((df[TARGET] == 0).sum() / df[TARGET].sum()),
    },
    "split": {"test_size": 0.30, "stratified": True, "seed": SEED, "n_train": int(len(X_tr)), "n_test": int(len(X_te))},
    "data_quality": {
        "missing_cells": 0, "duplicates_dropped": n_dup,
        "education_raw_codes": {str(k): int(v) for k, v in sorted(edu_codes.items())}, "education_recoded_rows": int(sum(edu_codes.get(k, 0) for k in (0, 5, 6))),
        "marriage_raw_codes": {str(k): int(v) for k, v in sorted(mar_codes.items())}, "marriage_recoded_rows": int(mar_codes.get(0, 0)),
        "pay_status_undocumented_codes": pay_undoc, "pay1_value_counts_raw": {str(k): int(v) for k, v in raw["PAY_1"].value_counts().sort_index().items()},
        "negative_bill_rows": neg_bill_rows, "negative_bill_cells": neg_bill_cells, "min_bill_amt": int(raw[BILL_COLS].min().min()),
        "zero_payment_rows_any_month": zero_pay_any, "zero_payment_rows_all_months": zero_pay_all, "zero_bill_rows_all_months": zero_bill_all,
        "util_1_over_1_rows": over_limit, "util_1_negative_rows": int((df["UTIL_1"] < 0).sum()),
        "limit_bal": {"min": int(raw["LIMIT_BAL"].min()), "median": int(raw["LIMIT_BAL"].median()), "mean": r4(raw["LIMIT_BAL"].mean()), "max": int(raw["LIMIT_BAL"].max())},
        "age": {"min": int(raw["AGE"].min()), "median": int(raw["AGE"].median()), "mean": r4(raw["AGE"].mean()), "max": int(raw["AGE"].max())},
        "sex_counts_raw": {str(k): int(v) for k, v in raw["SEX"].value_counts().sort_index().items()},
    },
    "default_rate_by": {"overall_train": r4(overall_tr), "PAY_1": to_json_table(pay1_tab), "LIMIT_BAL_quintile": to_json_table(lim_tab),
                        "AGE_band": to_json_table(age_tab), "EDUCATION": to_json_table(edu_tab), "MARRIAGE": to_json_table(mar_tab),
                        "SEX": to_json_table(sex_tab), "PAY_AMT1_band": to_json_table(pa_tab), "UTIL_1_decile": to_json_table(util_tab),
                        "N_MONTHS_DELAYED": to_json_table(nmd_tab), "MAX_DELAY": to_json_table(maxd_tab)},
    "pay1_headline": {"rate_pay1_le_0": r4(y[tr["PAY_1"] <= 0].mean()), "n_pay1_le_0": int((tr["PAY_1"] <= 0).sum()),
                      "rate_pay1_ge_2": r4(y[tr["PAY_1"] >= 2].mean()), "n_pay1_ge_2": int((tr["PAY_1"] >= 2).sum()),
                      "rate_pay1_eq_1": r4(y[tr["PAY_1"] == 1].mean()), "n_pay1_eq_1": int((tr["PAY_1"] == 1).sum()),
                      "share_of_defaults_with_pay1_ge_1": r4(y[tr["PAY_1"] >= 1].sum() / y.sum()), "share_of_clients_with_pay1_ge_1": r4((tr["PAY_1"] >= 1).mean())},
    "correlations": {
        "pearson_with_target": {k: r4(v) for k, v in corr_t.items()}, "spearman_with_target": {k: r4(v) for k, v in spear_t.items()},
        "top_feature_pairs_abs": [{"a": a, "b": b, "r": round(r, 4)} for a, b, r in pairs[:15]],
        "n_feature_pairs_abs_r_over_0_8": int(n_pairs_over_08),
        "bill_amt_block": {"min_r": r4(bill_block.min()), "mean_r": r4(bill_block.mean()), "max_r": r4(bill_block.max())},
        "pay_status_block": {"min_r": r4(pay_block.min()), "mean_r": r4(pay_block.mean()), "max_r": r4(pay_block.max())},
        "pay_amt_block": {"min_r": r4(payamt_block.min()), "mean_r": r4(payamt_block.mean()), "max_r": r4(payamt_block.max())},
        "strongest_positive_with_target": {"feature": corr_t.index[0], "r": r4(corr_t.iloc[0])},
        "strongest_negative_with_target": {"feature": corr_t.index[-1], "r": r4(corr_t.iloc[-1])},
    },
    "univariate_auc": sorted(uni, key=lambda u: -u["auc_oriented"]),
    "engineered_features": eng_summary,
    "by_class_summary": by_class(["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_1"] + ENG_COLS),
    "figures": ["eda_target_distribution.png", "eda_default_rate_by_pay1.png", "eda_default_rate_by_groups.png", "eda_distributions_by_default.png",
                "eda_correlation_heatmap.png", "eda_default_rate_util_delay.png", "eda_univariate_strength.png"],
    "tables": ["eda_data_quality.csv", "eda_variables.csv"],
    "runtime_seconds": round(time.time() - T0, 1),
}
json.dump(summary, open(os.path.join(RESULTS, "eda_summary.json"), "w"), indent=1, ensure_ascii=False)
print(f"done in {summary['runtime_seconds']}s — figures: {len(summary['figures'])}, tables: {len(summary['tables'])}, summary: results/eda_summary.json")
