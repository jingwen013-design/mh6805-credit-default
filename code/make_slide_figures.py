"""
make_slide_figures.py — slide-sized versions (1920 x 960 px, 16-20 pt text) of the figures the presentation uses.
Owner: shared. Reads results/*.csv / *.json / *.npy and data/processed/train.csv only; nothing is fitted or selected.
Every number drawn here is the same number as in the report (same frozen files). Writes figures/slides/s_*.png.

Palette = the report's validated categorical palette (colour = model family, fixed slot order); text never wears a
series colour; bars thin with square baseline; legends present whenever two or more series are drawn.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from common import COST_FN, COST_FP, FIGURES, RESULTS, ROOT, load_split

OUT = os.path.join(FIGURES, "slides")
os.makedirs(OUT, exist_ok=True)
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#4a3aa7", red="#e34948")
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"
FAMILY_COLOR = {"boosting": PAL["blue"], "linear": PAL["orange"], "forest / bagging": PAL["aqua"], "single tree": PAL["yellow"],
                "naive Bayes": PAL["magenta"], "KNN": PAL["green"], "SVM": PAL["violet"], "neural net": PAL["red"]}
SHORT = {"knn": "KNN (K = 101) — benchmark", "logistic": "Logistic — baseline", "logistic_l1": "Logistic L1", "lda": "LDA",
         "naive_bayes": "Gaussian NB", "tree_pruned": "Pruned tree", "random_forest": "Random forest",
         "gradient_boosting": "Gradient boosting", "xgboost": "XGBoost", "svm_rbf": "SVM (RBF)", "mlp": "MLP"}
plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 17, "legend.fontsize": 15, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.dpi": 150, "lines.linewidth": 2.5, "font.family": "DejaVu Sans"})
SIZE = (12.8, 6.4)


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white"); plt.close(fig); print("wrote", name)


X_train, X_test, y_train, y_test, _ = load_split()
yte = y_test.values.astype(int); ytr = y_train.values.astype(int)
P_TEST = lambda s: np.load(os.path.join(RESULTS, f"{s}_p_test.npy")).astype(float)
P_OOF = lambda s: np.load(os.path.join(RESULTS, f"{s}_p_oof.npy")).astype(float)
comp = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
head = comp[comp.headline].sort_values("select_auc", ascending=True)
R = {s: json.load(open(os.path.join(RESULTS, f"{s}.json"))) for s in head.slug}

# ------------------------------------------------------------------ A. headline AUC, ranked by CV AUC
fig, ax = plt.subplots(figsize=SIZE)
yy = np.arange(len(head))
cols = [FAMILY_COLOR[f] for f in head.family]
ax.barh(yy, head.test_auc, color=cols, height=0.55, zorder=2)
ok = head.cv_auc.notna()
ax.errorbar(head.cv_auc[ok], yy[ok.values], xerr=head.cv_auc_std[ok].fillna(0), fmt="o", color=INK, ms=7, capsize=4, lw=1.6,
            zorder=4, label="5-fold CV AUC ± 1 sd (training) — the selection score")
for y_, v, c_, sd in zip(yy, head.test_auc, head.cv_auc.fillna(head.test_auc), head.cv_auc_std.fillna(0)):
    ax.text(max(v, c_ + sd) + 0.0015, y_, f"{v:.3f}", va="center", ha="left", fontsize=14, color=INK2)
knn, lr = R["knn"]["test_auc"], R["logistic"]["test_auc"]
ax.axvline(knn, color=INK3, lw=1.2, ls="--", zorder=1); ax.text(knn + 0.0008, -0.45, "KNN 0.771", fontsize=12, color=INK2, va="bottom")
ax.axvline(lr, color=INK3, lw=1.2, ls=":", zorder=1); ax.text(lr - 0.0008, -0.45, "logistic 0.764", fontsize=12, color=INK2, va="bottom", ha="right")
ax.set_yticks(yy); ax.set_yticklabels([SHORT[s] + (" †" if s == "svm_rbf" else "") for s in head.slug])
ax.set_xlim(0.72, 0.805)
ax.set_title("Headline models ordered by CV AUC (the selection score); bars = test AUC", loc="left")
for f_, c_ in FAMILY_COLOR.items():
    if f_ in set(head.family): ax.barh([-5], [0], color=c_, label=f_)
ax.set_ylim(-0.6, len(head) - 0.4); ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=4, columnspacing=1.0, handlelength=1.3, fontsize=12)
ax.set_xlabel("ROC-AUC on the test set   († SVM tuned on a 6,000-row subsample; marker = full-training OOF AUC)")
save(fig, "s_auc_headline.png")

# ------------------------------------------------------------------ B. ROC + PR for the three story models
fig, axes = plt.subplots(1, 2, figsize=SIZE)
trio = [("knn", PAL["green"], "-"), ("logistic", PAL["orange"], "-"), ("xgboost", PAL["blue"], "-")]
for s, c, ls in trio:
    p = P_TEST(s); fx, ty, _ = roc_curve(yte, p); pr, rc, _ = precision_recall_curve(yte, p)
    axes[0].plot(fx, ty, color=c, ls=ls, label=f"{SHORT[s].split(' —')[0]} — AUC {R[s]['test_auc']:.3f}")
    axes[1].plot(rc, pr, color=c, ls=ls, label=f"{SHORT[s].split(' —')[0]} — AP {R[s]['test_ap']:.3f}")
axes[0].plot([0, 1], [0, 1], color=INK3, lw=1.2, ls="--", label="chance (AUC 0.5)")
axes[1].axhline(yte.mean(), color=INK3, lw=1.2, ls="--", label=f"prevalence {yte.mean():.3f} (no skill)")
axes[0].set_xlabel("False-positive rate (good clients flagged)"); axes[0].set_ylabel("True-positive rate (defaulters caught)"); axes[0].set_title("ROC curves (test set)")
axes[1].set_xlabel("Recall (defaulters caught)"); axes[1].set_ylabel("Precision (flagged clients who default)"); axes[1].set_title("Precision–recall curves (test set)")
for ax in axes: ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="lower right" if ax is axes[0] else "upper right")
save(fig, "s_roc_pr.png")

# ------------------------------------------------------------------ C. cost vs threshold, logistic and XGBoost
grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
def cost_curve(y, p):
    P = p[None, :] >= grid[:, None]
    fn = (~P & (y == 1)).sum(1); fp = (P & (y == 0)).sum(1)
    return (COST_FN * fn + COST_FP * fp) / len(y) * 1000.0
prev = yte.mean(); flag_none = COST_FN * prev * 1000; flag_all = COST_FP * (1 - prev) * 1000
fig, ax = plt.subplots(figsize=SIZE)
for s, c in [("logistic", PAL["orange"]), ("xgboost", PAL["blue"])]:
    cc = cost_curve(yte, P_TEST(s)); ax.plot(grid, cc, color=c, label=SHORT[s].split(" —")[0])
    t = R[s]["at_tuned_cost"]["threshold"]; i = int(np.argmin(np.abs(grid - t)))
    ax.plot([t], [cc[i]], marker="D", ms=11, color=c, mec="white", mew=1.5, zorder=5)
    i5 = int(np.argmin(np.abs(grid - 0.5))); ax.plot([0.5], [cc[i5]], marker="o", ms=11, color=c, mec="white", mew=1.5, zorder=5)
    dy = 130 if s == "logistic" else -70; dx = -0.02 if s == "logistic" else 0.06
    ax.annotate(f"{SHORT[s].split(' —')[0]}: {R[s]['at_tuned_cost']['cost']:,.0f} at t = {t:.2f}", (t, cc[i]), xytext=(t + dx, cc[i] + dy), fontsize=14, color=INK2,
                ha="left", arrowprops=dict(arrowstyle="-", color=INK3))
    ax.annotate(f"{R[s]['at_0_5']['cost']:,.0f} at 0.5", (0.5, cc[i5]), xytext=(0.53 if s == "logistic" else 0.40, cc[i5] + (40 if s == "logistic" else -60)), fontsize=14, color=INK2)
ax.axhline(flag_none, color=INK3, lw=1.2, ls="--"); ax.text(0.98, flag_none + 8, f"flag nobody: {flag_none:,.0f}", ha="right", fontsize=13, color=INK2)
ax.axhline(flag_all, color=INK3, lw=1.2, ls="--"); ax.text(0.98, flag_all + 8, f"flag everybody: {flag_all:,.0f}", ha="right", fontsize=13, color=INK2)
ax.plot([], [], marker="o", color=INK, ls="none", ms=9, label="default cut-off 0.5"); ax.plot([], [], marker="D", color=INK, ls="none", ms=9, label="cut-off chosen on training OOF (5:1)")
ax.set_xlim(0, 1); ax.set_ylim(450, 1200); ax.set_xlabel("Decision threshold on P(default)"); ax.set_ylabel("Expected cost per 1,000 clients (5·FN + 1·FP)")
ax.set_title("Expected cost against the decision threshold (test set, 5:1 cost assumption)", loc="left")
ax.legend(loc="upper left")
save(fig, "s_cost_vs_threshold.png")

# ------------------------------------------------------------------ D. threshold saving vs model saving over cost ratios
tab = pd.read_csv(os.path.join(RESULTS, "threshold_vs_model_saving.csv"))
fig, ax = plt.subplots(figsize=SIZE)
for col, lab, c in [("threshold_saving", "move the logistic cut-off from 0.5 to the tuned value", PAL["orange"]),
                    ("model_saving", "then switch logistic → XGBoost (both tuned)", PAL["blue"])]:
    ax.plot(tab.ratio, tab[col], marker="o", ms=9, color=c, label=lab)
    ax.fill_between(tab.ratio, tab[col + "_lo"], tab[col + "_hi"], color=c, alpha=0.15)
r5 = tab.set_index("ratio").loc[5]
ax.annotate(f"5:1 — threshold saves {r5.threshold_saving:,.0f} [{r5.threshold_saving_lo:,.0f}, {r5.threshold_saving_hi:,.0f}]",
            (5, r5.threshold_saving), xytext=(1.6, 9000), fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
ax.annotate(f"model saves {r5.model_saving:,.0f} [{r5.model_saving_lo:,.0f}, {r5.model_saving_hi:,.0f}] — interval includes 0",
            (5, r5.model_saving), xytext=(1.6, 4500), fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
ax.axhline(0, color=INK, lw=1); ax.axvline(5, color=INK3, lw=1.2, ls="--")
ax.set_xscale("log"); ax.set_xticks(tab.ratio); ax.set_xticklabels([f"{r}:1" for r in tab.ratio])
ax.set_xlabel("Assumed cost ratio  C_FN : C_FP  (5:1 is the report's reference, not the truth)")
ax.set_ylabel("Test cost saved (C_FP = 1)")
ax.set_title("Cost saved by re-thresholding vs by changing model, with 95% paired-bootstrap bands", loc="left")
ax.legend(loc="upper left")
save(fig, "s_threshold_vs_model.png")

# ------------------------------------------------------------------ E. Platt recalibration, training OOF (cross-fitted)
fig, ax = plt.subplots(figsize=SIZE)
for s, lab, c in [("xgboost_weighted", "class-weighted XGBoost", PAL["blue"]), ("logistic_balanced", "class-weighted logistic", PAL["orange"])]:
    raw = P_OOF(s); cal = np.load(os.path.join(RESULTS, "recalibrated", f"{s}_platt_p_oof_crossfit.npy"))
    fr, mr = calibration_curve(ytr, raw, n_bins=10, strategy="quantile"); fc, mc = calibration_curve(ytr, cal, n_bins=10, strategy="quantile")
    ax.plot(mr, fr, ls=":", marker="o", ms=9, color=c, label=f"{lab} — raw (mean p {raw.mean():.2f})")
    ax.plot(mc, fc, ls="-", marker="o", ms=9, color=c, label=f"{lab} + Platt (mean p {cal.mean():.2f})")
ax.plot([0, 1], [0, 1], color=INK3, lw=1.2, ls="--", label="perfect calibration")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted P(default), decile"); ax.set_ylabel("Observed default rate (training, out-of-fold)")
ax.set_title("Reliability of the class-weighted models before and after Platt scaling (training, out-of-fold)", loc="left")
ax.legend(loc="upper left")
save(fig, "s_calibration_platt.png")

# ------------------------------------------------------------------ F. leakage demo
lk = pd.read_csv(os.path.join(RESULTS, "imbalance_leakage_demo.csv")).set_index("learner")
fig, ax = plt.subplots(figsize=SIZE)
labs = ["Logistic", "Random forest"]; x = np.arange(2); w = 0.3
hon = [lk.loc["logistic", "honest_test_auc_smote"], lk.loc["rf", "honest_test_auc_smote"]]
leak = [lk.loc["logistic", "leaky_test_auc"], lk.loc["rf", "leaky_test_auc"]]
ax.bar(x - w / 2 - 0.01, hon, width=w, color=PAL["aqua"], label="SMOTE inside the folds (honest test AUC)")
ax.bar(x + w / 2 + 0.01, leak, width=w, color=PAL["red"], label="SMOTE before the split (leaky \"test\" AUC)")
for i in range(2):
    ax.text(x[i] - w / 2 - 0.01, hon[i] + 0.004, f"{hon[i]:.3f}", ha="center", fontsize=15, color=INK2)
    ax.text(x[i] + w / 2 + 0.01, leak[i] + 0.004, f"{leak[i]:.3f}", ha="center", fontsize=15, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.70, 0.95); ax.set_ylabel("Test ROC-AUC")
ax.set_title(f"SMOTE before the split inflates the forest by +{leak[1] - hon[1]:.3f} AUC "
             f"({lk.loc['rf', 'share_synthetic_in_leaky_test_defaulters']:.0%} of its 'test' defaulters are synthetic)", loc="left")
ax.legend(loc="upper left")
save(fig, "s_leakage.png")

# ------------------------------------------------------------------ G. importance: mean |SHAP| top 10 with direction and OOB rank
imp = pd.read_csv(os.path.join(RESULTS, "shap_importance.csv")).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=SIZE)
ax.barh(np.arange(len(imp)), imp.mean_abs_shap, color=PAL["blue"], height=0.55)
for i, r in enumerate(imp.itertuples()):
    sign = "↑ default" if r.corr_value_vs_shap > 0 else "↓ default"
    ax.text(r.mean_abs_shap + 0.004, i, f"{r.mean_abs_shap:.3f}   {sign}   (OOB-permutation rank {int(r.oob_perm_rank)})", va="center", fontsize=14, color=INK2)
ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels(imp.feature); ax.set_xlim(0, imp.mean_abs_shap.max() * 1.9)
ax.set_xlabel("Mean |SHAP| on the 20,975 training clients (log-odds), tuned XGBoost")
ax.set_title("Mean |SHAP| of the top-10 predictors, tuned XGBoost, with the direction of the effect", loc="left")
save(fig, "s_importance.png")

# ------------------------------------------------------------------ H. PDP of utilisation (U-shape)
pdp = pd.read_csv(os.path.join(RESULTS, "pdp_values.csv"))
u = pdp[(pdp.kind == "one_way") & (pdp.feature_1 == "UTIL_MEAN")].sort_values("value_1")
if len(u):
    fig, ax = plt.subplots(figsize=SIZE)
    ax.fill_between(u.value_1, u.ice_p25, u.ice_p75, color=PAL["blue"], alpha=0.12, label="ICE inter-quartile band")
    ax.plot(u.value_1, u.pd, color=PAL["blue"], label="partial dependence, tuned XGBoost (training rows)")
    if u.empirical_default_rate.notna().any():
        ax.plot(u.value_1, u.empirical_default_rate, marker="o", ms=8, ls="none", color=PAL["orange"], label="empirical default rate at the grid value (training)")
    i = int(np.argmin(u.pd.values)); ax.plot([u.value_1.values[i]], [u.pd.values[i]], marker="v", ms=14, color=INK, zorder=5)
    ax.annotate(f"minimum {u.pd.values[i]:.3f} at utilisation {u.value_1.values[i]:.2f}", (u.value_1.values[i], u.pd.values[i]),
                xytext=(u.value_1.values[i] + 0.15, u.pd.values[i] - 0.03), fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
    ax.set_xlabel("UTIL_MEAN — average bill / credit limit"); ax.set_ylabel("P(default)")
    ax.set_title("Partial dependence of P(default) on average utilisation (tuned XGBoost, training rows)", loc="left")
    ax.legend(loc="upper left"); save(fig, "s_pdp_util.png")

# ------------------------------------------------------------------ I. clustering: silhouette and stability vs K (RUAN's 30 Sep run)
ks = pd.read_csv(os.path.join(RESULTS, "kmeans_selection_v2.csv"))
fig, axes = plt.subplots(1, 2, figsize=SIZE)
axes[0].plot(ks.K, ks.silhouette, marker="o", ms=9, color=PAL["blue"]); axes[0].set_title("Silhouette (10 PCs, training set)"); axes[0].set_ylabel("silhouette")
axes[1].plot(ks.K, ks.boot_ari_mean, marker="o", ms=9, color=PAL["aqua"], label="mean bootstrap ARI")
axes[1].plot(ks.K, ks.min_cluster_jaccard, marker="s", ms=9, color=PAL["orange"], label="least stable cluster (Jaccard)")
axes[1].axhline(0.75, color=INK3, lw=1.2, ls="--"); axes[1].set_title("Stability under bootstrap refits"); axes[1].legend(loc="lower left")
for ax in axes: ax.axvline(5, color=INK3, lw=1.2, ls=":"); ax.set_xlabel("K"); ax.set_xticks(ks.K)
k5 = ks.set_index("K").loc[5]
axes[0].annotate(f"K = 5: {k5.silhouette:.2f}\nweak-to-moderate separation", (5, k5.silhouette), xytext=(2.2, 0.222), fontsize=13, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
fig.suptitle("K-means on 10 principal components: silhouette and bootstrap stability by K (training set)", fontsize=17)
save(fig, "s_cluster_stability.png")

# ------------------------------------------------------------------ J. imbalance study: AUC by strategy and Brier raw vs Platt
imb = pd.read_csv(os.path.join(RESULTS, "imbalance_comparison.csv")); rc = pd.read_csv(os.path.join(RESULTS, "recalibration.csv"))
fig, axes = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[1, 1.35]))
strat = ["none", "balanced", "smote", "under"]; x = np.arange(len(strat)); w = 0.32
for j, (lrn, lab, c) in enumerate([("logistic", "logistic", PAL["orange"]), ("rf", "random forest", PAL["aqua"])]):
    t = imb[imb.learner == lrn].set_index("strategy").loc[strat]
    axes[0].bar(x + (j - 0.5) * (w + 0.02), t.test_auc, width=w, color=c, label=lab)
    for xi, v in zip(x + (j - 0.5) * (w + 0.02), t.test_auc): axes[0].text(xi, v + 0.001, f"{v:.3f}", ha="center", fontsize=12, color=INK2)
axes[0].set_xticks(x); axes[0].set_xticklabels(["none", "class\nweights", "SMOTE", "under-\nsampling"]); axes[0].set_ylim(0.74, 0.80)
axes[0].set_ylabel("Test ROC-AUC"); axes[0].set_title("Test AUC by re-balancing strategy", loc="left"); axes[0].legend(loc="upper right")
lab = {"logistic_balanced": "logistic, weighted", "imb_logistic_smote": "logistic, SMOTE", "imb_logistic_under": "logistic, under-sampled",
       "imb_rf_balanced": "forest, weighted", "imb_rf_smote": "forest, SMOTE", "imb_rf_under": "forest, under-sampled",
       "xgboost_weighted": "XGBoost, weighted", "mlp_weighted": "MLP, weighted"}
rc = rc[rc.slug.isin(lab)].reset_index(drop=True); yy = np.arange(len(rc))
axes[1].barh(yy + 0.2, rc.test_brier_raw, height=0.36, color=PAL["red"], label="raw, after weighting / resampling")
axes[1].barh(yy - 0.2, rc.test_brier_platt, height=0.36, color=PAL["blue"], label="after Platt scaling on training OOF")
for i, r in rc.iterrows():
    axes[1].text(r.test_brier_raw + 0.002, i + 0.2, f"{r.test_brier_raw:.3f}", va="center", fontsize=12, color=INK2)
    axes[1].text(r.test_brier_platt + 0.002, i - 0.2, f"{r.test_brier_platt:.3f}", va="center", fontsize=12, color=INK2)
axes[1].set_yticks(yy); axes[1].set_yticklabels([lab[s] for s in rc.slug]); axes[1].set_xlim(0.10, 0.22); axes[1].invert_yaxis()
axes[1].set_xlabel("Test Brier score (lower = better calibrated)"); axes[1].set_title("Brier score, raw vs Platt-recalibrated", loc="left")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=13)
save(fig, "s_imbalance.png")

# ------------------------------------------------------------------ K. EDA: default rate by September status and by limit quintile (training rows)
d = X_train.assign(y=ytr)
pay = d.groupby(d.PAY_1.clip(lower=-2, upper=3)).y.agg(["mean", "size"])
pay_lab = {-2: "no consumption", -1: "paid in full", 0: "revolving credit", 1: "1 month late", 2: "2 months late", 3: "3+ months late"}
quint = d.groupby(pd.qcut(d.LIMIT_BAL, 5, duplicates="drop")).y.agg(["mean", "size"])
fig, axes = plt.subplots(1, 2, figsize=SIZE)
axes[0].bar(np.arange(len(pay)), pay["mean"], color=PAL["orange"], width=0.6)
for i, (m, n) in enumerate(zip(pay["mean"], pay["size"])): axes[0].text(i, m + 0.012, f"{m:.1%}\n(n = {n:,})", ha="center", fontsize=12, color=INK2)
axes[0].set_xticks(np.arange(len(pay))); axes[0].set_xticklabels([pay_lab.get(int(k), str(k)) for k in pay.index], rotation=20, ha="right")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Default rate next month"); axes[0].set_title("Repayment status in September (PAY_1)")
axes[1].bar(np.arange(len(quint)), quint["mean"], color=PAL["blue"], width=0.6)
for i, (m, n) in enumerate(zip(quint["mean"], quint["size"])): axes[1].text(i, m + 0.006, f"{m:.1%}", ha="center", fontsize=13, color=INK2)
axes[1].set_xticks(np.arange(len(quint))); axes[1].set_xticklabels([f"NT${int(round(iv.left/1000)):,}k–{int(round(iv.right/1000)):,}k" for iv in quint.index], rotation=20, ha="right")
axes[1].set_ylim(0, 0.45); axes[1].set_title("Credit-limit quintile")
fig.suptitle("Default rate along the two strongest gradients (training partition only, n = 20,975)", fontsize=17)
save(fig, "s_eda_gradients.png")
# ------------------------------------------------------------------ L. XGBoost learning curve (slide version: no 4-decimal annotation block)
lc = pd.read_csv(os.path.join(RESULTS, "xgb_learning_curve.csv"))
fig, ax = plt.subplots(figsize=SIZE)
for lr, c, ls in [(0.03, PAL["blue"], "-"), (0.1, PAL["violet"], "-")]:
    t = lc[np.isclose(lc.learning_rate, lr)].sort_values("n_rounds")
    ax.plot(t.n_rounds, t.cv_auc, color=c, ls=ls, label=f"5-fold CV AUC, learning rate {lr}")
    ax.fill_between(t.n_rounds, t.cv_auc - t.cv_auc_std, t.cv_auc + t.cv_auc_std, color=c, alpha=0.12, linewidth=0)
    ax.plot(t.n_rounds, t.train_auc, color=c, ls="--", linewidth=1.8, label=f"training-fold AUC, learning rate {lr}")
    i = int(t.cv_auc.values.argmax()); ax.plot(t.n_rounds.values[i], t.cv_auc.values[i], "o", color=c, ms=8)
    ax.annotate(f"peak {t.cv_auc.values[i]:.3f} at {int(t.n_rounds.values[i])} rounds", (t.n_rounds.values[i], t.cv_auc.values[i]),
                xytext=(-60, 70) if lr == 0.1 else (12, 16), textcoords="offset points", fontsize=14, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK3))
ax.axvline(300, color=INK3, ls=":", linewidth=1.5); ax.text(308, 0.852, "chosen: 300 rounds,\ndepth 3, rate 0.03", fontsize=14, color=INK2, va="top")
ax.set_ylim(0.74, 0.86); ax.set_xlim(0, lc.n_rounds.max()); ax.set_xlabel("Boosting rounds (trees)"); ax.set_ylabel("ROC-AUC (training folds)")
ax.set_title("XGBoost: CV AUC peaks and then declines while training AUC keeps climbing", loc="left")
ax.legend(loc="lower right", ncol=2, fontsize=13)
save(fig, "s_xgb_learning_curve.png")

# ------------------------------------------------------------------ M. MLP over-fitting without early stopping (two networks)
ep = pd.read_csv(os.path.join(RESULTS, "mlp_overfit_epochs.csv"))
fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
for ax, net, ttl in zip(axes, ["32-64-32-1", "32-256-256-1"], ["Tuned 32-64-32-1, run without early stopping", "Over-parameterised 32-256-256-1, no stopping"]):
    t = ep[ep.network == net].sort_values("epoch")
    ax.plot(t.epoch, t.train_auc, color=PAL["red"], ls="--", linewidth=1.8, label="training AUC")
    ax.plot(t.epoch, t.holdout_auc, color=PAL["red"], label="held-out AUC (15 % of training rows)")
    i = int(t.holdout_auc.values.argmax()); e0, h0, hN = int(t.epoch.values[i]), t.holdout_auc.values[i], t.holdout_auc.values[-1]
    ax.plot(e0, h0, "o", color=PAL["red"], ms=8)
    ax.annotate(f"peak {h0:.3f} at epoch {e0} \u2192 {hN:.3f} at epoch {int(t.epoch.values[-1])}", (e0, h0), xytext=(10, 40), textcoords="offset points", fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
    ax.set_xlabel("Epoch"); ax.set_title(ttl, loc="left", fontsize=15); ax.set_ylim(0.6, 1.0)
axes[0].set_ylabel("ROC-AUC"); axes[0].legend(loc="lower right", fontsize=13)
fig.suptitle("Early stopping is the regulariser: held-out AUC peaks early, then decays as the network memorises", fontsize=17)
save(fig, "s_mlp_overfit.png")
# ------------------------------------------------------------------ N. KNN: 5-fold CV AUC against K (the benchmark's tuning curve, training folds only)
kn = json.load(open(os.path.join(RESULTS, "knn.json")))
ct = pd.DataFrame(kn["extra"]["cv_table"])
fig, ax = plt.subplots(figsize=SIZE)
for wgt, c, mk in [("uniform", PAL["green"], "o"), ("distance", PAL["aqua"], "s")]:
    t = ct[ct.weights == wgt].sort_values("K")
    ax.errorbar(t.K, t.cv_auc_mean, yerr=t.cv_auc_std, color=c, marker=mk, ms=7, capsize=4, label=f"{wgt} weights (mean ± 1 sd over 5 folds)")
best = ct.sort_values("cv_auc_mean", ascending=False).iloc[0]
ax.plot(best.K, best.cv_auc_mean, "o", ms=16, mfc="none", mec=INK, mew=2)
ax.annotate(f"chosen: K = {int(best.K)}, {best.weights} — CV AUC {best.cv_auc_mean:.3f}", (best.K, best.cv_auc_mean), xytext=(-330, -110),
            textcoords="offset points", fontsize=14, color=INK2, arrowprops=dict(arrowstyle="-", color=INK3))
k5 = ct[(ct.K == 5) & (ct.weights == "uniform")]
if len(k5): ax.annotate(f"course default K = 5: {float(k5.cv_auc_mean.iloc[0]):.3f}", (5, float(k5.cv_auc_mean.iloc[0])), xytext=(16, -6), textcoords="offset points", fontsize=14, color=INK2)
ax.set_xlabel("K (neighbours)"); ax.set_ylabel("5-fold CV ROC-AUC (training folds)"); ax.set_title("KNN benchmark: CV AUC rises with K and plateaus near K = 101", loc="left")
ax.legend(loc="lower right", fontsize=13)
save(fig, "s_knn_k.png")
print("done ->", OUT)
