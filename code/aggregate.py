"""
aggregate.py — aggregation of every model result of the project (slug "aggregate").

Run from Project/code:  python3 aggregate.py            (about 1 minute; nothing is fitted, nothing is tuned)

What it does
  1. loads every results/*.json written by common.evaluate() through common.load_all_results(), dropping the two
     non-model summaries (eda_summary.json, unsupervised_summary.json) and any slug starting with "imb_leaky" or "eda";
  2. writes results/model_comparison.csv / .md (one row per model, sorted by the selection score below) and the per-class
     "main table" of the eleven headline models, results/model_comparison_main.csv / .md (test set, class 0 and class 1
     precision / recall / F1 at the F1-tuned threshold and at 0.5, laid out like the sample report);
  3. draws figures/all_roc.png, all_pr.png, all_auc_bar.png (+ all_auc_bar_full.png for every model), threshold_curves.png,
     cost_curves.png and calibration.png from results/<slug>_p_test.npy against the shared y_test (load_split());
  4. paired bootstrap (B = 500, seed 2607526) of the test-set AUC (and AP) difference between the selected headline model
     and every other headline model -> results/auc_bootstrap_ci.csv;
  5. writes results/aggregate_summary.json with the headline numbers, the selection, rankings, bootstrap verdicts and short
     summaries of the imbalance and unsupervised studies.

Contract: nothing here fits or tunes anything. The test set is only used to *report* metrics of models whose
hyper-parameters and thresholds were fixed on training data by the individual model scripts. Model selection — the best
headline model and the top-4 drawn in the threshold / cost / calibration figures — uses the training-set selection score
only: the 5-fold CV AUC, or the pooled training out-of-fold AUC for a script whose CV ran on a tuning subsample (its CV AUC
is not comparable). The test AUC never ranks or selects anything ("we don't use the test data to observe the training —
that is actually kind of cheating", lecture of 2026-10-07). The bootstrap resamples test rows to attach uncertainty to the differences between the
already-selected model and the others; it does not select anything.
"""
import sys, json, os, time
import numpy as np
import pandas as pd
from common import *                        # RESULTS, FIGURES, SEED, COST_FN, COST_FP, load_split, load_all_results, ...
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

T0 = time.time()
HEADLINE = ["knn", "logistic", "logistic_l1", "lda", "naive_bayes", "tree_pruned", "random_forest",
            "gradient_boosting", "xgboost", "svm_rbf", "mlp"]
EXCLUDE_PREFIX = ("imb_leaky", "eda")
B_BOOT = 500
LOGISTIC_REF = dict(test_auc=0.764, test_ap=0.524)      # README reference numbers for the plain logistic baseline

# ----------------------------------------------------------------------------- palette / style
# Validated categorical palette (fixed slot order). Colour = model family, linestyle = member within a family,
# so the same model has the same colour in every figure. Text always uses ink tokens, never a series colour.
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#4a3aa7", red="#e34948")
INK, INK2, INK3, GRID_C = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"
FAMILY_COLOR = {"boosting": PAL["blue"], "linear": PAL["orange"], "forest / bagging": PAL["aqua"],
                "single tree": PAL["yellow"], "naive Bayes": PAL["magenta"], "KNN": PAL["green"],
                "SVM": PAL["violet"], "neural net": PAL["red"]}
LINESTYLE = {"logistic": "-", "logistic_l1": "--", "lda": ":", "gradient_boosting": "--", "xgboost": "-"}
SHORT = {"knn": "KNN (K=101)", "logistic": "Logistic (baseline)", "logistic_l1": "Logistic L1", "lda": "LDA",
         "naive_bayes": "Gaussian NB", "tree_pruned": "Pruned tree", "random_forest": "Random forest",
         "gradient_boosting": "Gradient boosting", "xgboost": "XGBoost", "svm_rbf": "SVM (RBF)", "mlp": "MLP"}

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.dpi": 140, "savefig.dpi": 140, "axes.titlecolor": INK})


def family_of(slug):
    s = slug
    if s.startswith("knn"): return "KNN"
    if s.startswith(("xgboost", "gradient_boosting")): return "boosting"
    if s.startswith(("random_forest", "bagging", "imb_rf")): return "forest / bagging"
    if s.startswith("tree"): return "single tree"
    if s.startswith("naive_bayes"): return "naive Bayes"
    if s.startswith("svm"): return "SVM"
    if s.startswith("mlp"): return "neural net"
    if s.startswith(("logistic", "imb_logistic", "lda", "qda")): return "linear"
    return "other"


def color_of(slug):
    return FAMILY_COLOR.get(family_of(slug), INK3)


def ls_of(slug):
    return LINESTYLE.get(slug, "-")


# ----------------------------------------------------------------------------- 1. load
X_train, X_test, y_train, y_test, feature_names = load_split()
y_te = y_test.values.astype(int); y_tr = y_train.values.astype(int)
n_test = len(y_te); prevalence = float(y_te.mean())

raw = load_all_results()
results = [r for r in raw if isinstance(r, dict) and "test_auc" in r and "slug" in r
           and not str(r["slug"]).startswith(EXCLUDE_PREFIX) and not str(r["slug"]).endswith("_v2")]   # *_v2 = RUAN's 30 Sep
                                                                                                      # re-scoring of frozen configs
                                                                                                      # (adopted record, not new models)
dropped = [(r.get("slug") or r.get("component") or "?") for r in raw if r not in results]
R = {r["slug"]: r for r in results}
missing = [s for s in HEADLINE if s not in R]
if missing:
    sys.exit(f"headline result(s) missing from results/: {missing} — run the corresponding model script first")
P_TEST = {s: np.load(os.path.join(RESULTS, f"{s}_p_test.npy")).astype(float) for s in R}
P_OOF = {s: np.load(os.path.join(RESULTS, f"{s}_p_oof.npy")).astype(float)
         for s in R if os.path.exists(os.path.join(RESULTS, f"{s}_p_oof.npy"))}
for s, p in P_TEST.items():                      # sanity: the saved probabilities must reproduce the JSON numbers
    assert len(p) == n_test, f"{s}: p_test has {len(p)} rows, y_test has {n_test}"
    assert abs(roc_auc_score(y_te, p) - R[s]["test_auc"]) < 1e-9, f"{s}: p_test does not reproduce the JSON test AUC"
print(f"loaded {len(results)} model results ({len(HEADLINE)} headline); dropped non-model / excluded files: {dropped}")


# ----------------------------------------------------------------------------- 2. comparison tables
def tune_seconds_of(r):
    e = r.get("extra") or {}
    for k in ("grid_seconds", "cv_seconds", "tuning_seconds", "total_script_seconds"):
        if isinstance(e.get(k), (int, float)):
            return float(e[k])
    t = e.get("tuning")
    if isinstance(t, dict):
        for k in ("seconds", "grid_seconds"):
            if isinstance(t.get(k), (int, float)):
                return float(t[k])
    return np.nan


def cv_note_of(r):
    e = r.get("extra") or {}
    t = e.get("tuning")
    if isinstance(t, dict) and t.get("subsample_rows"):
        return f"cv_auc measured on a {int(t['subsample_rows']):,}-row tuning subsample (not comparable; use oof_auc)"
    return ""


def select_auc_of(r):
    """Selection score, training data only: the 5-fold CV AUC; for a script whose CV ran on a row subsample (see
    cv_note_of) the pooled out-of-fold AUC on the full training set instead. Never the test AUC."""
    v = r.get("oof_auc") if cv_note_of(r) else r.get("cv_auc")
    return np.nan if v is None else float(v)


def row_of(r):
    a5, af, ac = r["at_0_5"], r["at_tuned_f1"], r["at_tuned_cost"]
    return dict(
        name=r["name"], slug=r["slug"], member=r.get("member", ""), family=family_of(r["slug"]),
        headline=r["slug"] in HEADLINE,
        select_auc=select_auc_of(r), select_source="oof_auc" if cv_note_of(r) else "cv_auc",
        cv_auc=r.get("cv_auc"), cv_auc_std=r.get("cv_auc_std"), oof_auc=r.get("oof_auc"), train_auc=r.get("train_auc"),
        test_auc=r["test_auc"], test_ap=r["test_ap"], brier=r["test_brier"], logloss=r.get("test_logloss"),
        accuracy_0_5=a5["accuracy"], recall_1_0_5=a5["recall_1"], precision_1_0_5=a5["precision_1"], f1_1_0_5=a5["f1_1"],
        cost_0_5=a5["cost"],
        thr_f1=af["threshold"], recall_1_tuned=af["recall_1"], precision_1_tuned=af["precision_1"], f1_1_tuned=af["f1_1"],
        accuracy_tuned=af["accuracy"], cost_f1=af["cost"],
        thr_cost=ac["threshold"], recall_1_cost=ac["recall_1"], precision_1_cost=ac["precision_1"],
        fpr_cost=ac["fpr"], fnr_cost=ac["fnr"], cost_cost=ac["cost"],
        fit_seconds=r.get("fit_seconds"), tune_seconds=tune_seconds_of(r), cv_note=cv_note_of(r),
        best_params=json.dumps(r.get("best_params") or {}, ensure_ascii=False))


full = (pd.DataFrame([row_of(r) for r in results])
        .sort_values("select_auc", ascending=False, na_position="last", kind="mergesort").reset_index(drop=True))
full.insert(0, "rank_cv_auc", np.arange(1, len(full) + 1))
full.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)

head = full[full["headline"]].copy()           # inherits the selection-score order
if head["select_auc"].isna().any():
    sys.exit(f"headline model(s) without a CV / OOF AUC, cannot select: {list(head.loc[head['select_auc'].isna(), 'slug'])}")
head["order"] = head["slug"].map({s: i for i, s in enumerate(HEADLINE)})
head_by_order = head.sort_values("order")
BEST = head.iloc[0]["slug"]                    # selected headline model: highest CV AUC on the training set
TOP4 = list(head["slug"].iloc[:4])             # top-4 by CV AUC, drawn in the threshold / cost / calibration figures
TEST_LEADER = head.sort_values("test_auc", ascending=False).iloc[0]["slug"]   # reported only, never used to select
LOG = R["logistic"]


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if isinstance(v, str): return v
    if isinstance(v, (bool, np.bool_)): return "yes" if v else ""
    if isinstance(v, (int, np.integer)): return f"{v:,}"
    return f"{v:.{nd}f}"


def md_table(df, cols, labels=None, nd=None):
    labels = labels or cols; nd = nd or {}
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], nd.get(c, 4)) for c in cols) + " |")
    return "\n".join(lines)


# per-class main table (test set) — class 0 and class 1 rows per headline model, like the sample report's summary table
main_rows = []
for _, h in head_by_order.iterrows():
    r = R[h["slug"]]; af, a5 = r["at_tuned_f1"], r["at_0_5"]
    for cls in (0, 1):
        main_rows.append(dict(
            model=SHORT[h["slug"]], slug=h["slug"], member=h["member"], rank_cv_auc=int(h["rank_cv_auc"]),
            test_auc=h["test_auc"], test_ap=h["test_ap"], cv_auc=h["cv_auc"], **{"class": cls},
            tuned_threshold=af["threshold"],
            precision_tuned=af[f"precision_{cls}"], recall_tuned=af[f"recall_{cls}"], f1_tuned=af[f"f1_{cls}"],
            accuracy_tuned=af["accuracy"],
            precision_0_5=a5[f"precision_{cls}"], recall_0_5=a5[f"recall_{cls}"], f1_0_5=a5[f"f1_{cls}"],
            accuracy_0_5=a5["accuracy"], support=int(a5["tn"] + a5["fp"]) if cls == 0 else int(a5["fn"] + a5["tp"])))
main = pd.DataFrame(main_rows)
main.to_csv(os.path.join(RESULTS, "model_comparison_main.csv"), index=False)


def main_md():
    L = ["| Model | test AUC | test AP | thr (F1, OOF) | class | precision @thr | recall @thr | F1 @thr | accuracy @thr | precision @0.5 | recall @0.5 | F1 @0.5 | accuracy @0.5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, row in main.iterrows():
        first = row["class"] == 0
        L.append("| " + " | ".join([
            f"**{row['model']}** (#{row['rank_cv_auc']})" if first else "",
            f"{row['test_auc']:.4f}" if first else "", f"{row['test_ap']:.4f}" if first else "",
            f"{row['tuned_threshold']:.2f}" if first else "", str(row["class"]),
            f"{row['precision_tuned']:.2f}", f"{row['recall_tuned']:.2f}", f"{row['f1_tuned']:.2f}",
            f"{row['accuracy_tuned']:.2%}" if first else "",
            f"{row['precision_0_5']:.2f}", f"{row['recall_0_5']:.2f}", f"{row['f1_0_5']:.2f}",
            f"{row['accuracy_0_5']:.2%}" if first else ""]) + " |")
    return "\n".join(L)


FULL_COLS = ["rank_cv_auc", "name", "member", "select_auc", "cv_auc", "cv_auc_std", "oof_auc", "test_auc", "test_ap", "brier",
             "accuracy_0_5", "recall_1_0_5", "precision_1_0_5", "f1_1_0_5", "thr_f1", "recall_1_tuned", "precision_1_tuned",
             "f1_1_tuned", "thr_cost", "fpr_cost", "fnr_cost", "cost_cost", "fit_seconds"]
FULL_LABELS = ["#", "model", "member", "select AUC", "CV AUC", "CV sd", "OOF AUC", "test AUC", "test AP", "Brier", "acc@0.5", "recall1@0.5",
               "prec1@0.5", "F1@0.5", "thr F1", "recall1@thr", "prec1@thr", "F1@thr", "thr cost", "FPR@cost", "FNR@cost",
               "cost@cost thr", "fit s"]
ND = {"thr_f1": 2, "thr_cost": 2, "cost_cost": 0, "fit_seconds": 1, "accuracy_0_5": 3, "recall_1_0_5": 3, "precision_1_0_5": 3,
      "f1_1_0_5": 3, "recall_1_tuned": 3, "precision_1_tuned": 3, "f1_1_tuned": 3, "fpr_cost": 3, "fnr_cost": 3}
md = [
    "# Model comparison — all evaluate() results (test set n = 8,990, default rate 22.1 %; train n = 20,975)",
    "",
    f"Generated by `code/aggregate.py` from `results/*.json` (seed {SEED}). {len(full)} model results; the eleven headline models are "
    "marked in the per-class table below. Thresholds: `thr F1` and `thr cost` were tuned on TRAINING out-of-fold probabilities "
    f"(F1 criterion / {COST_FN:.0f}:1 FN:FP cost criterion) by each model script; all metrics are then measured once on the test set. "
    "`cost` = 5·FN + FP on the 8,990 test rows (flag-nobody cost = 9,940; flag-everybody = 7,002). CV AUC = 5-fold stratified CV on the "
    "training set (mean ± sd); OOF AUC = AUC of the pooled training out-of-fold probabilities. Excluded: the leakage demonstration "
    "(`imb_leaky_*`) and the EDA / unsupervised summaries, which are not model results. "
    "**Selection uses training data only**: `select AUC` = CV AUC, or OOF AUC where the CV ran on a tuning subsample; models are "
    f"ranked and the best headline model (`{BEST}`) is chosen on it. Test metrics are reported once and never used to rank or select.",
    "",
    "## Headline models — per-class test metrics (sample-report layout)",
    "",
    "Test set only. `@thr` = at the F1-tuned threshold (chosen on training OOF probabilities), `@0.5` = at the default cut-off. "
    "Class 1 = default. Majority-class accuracy is 0.779. Sorted in report-section order; `#` = rank by the training-set "
    f"selection score (CV AUC) among all {len(full)} results.",
    "",
    main_md(),
    "",
    "## Every model result, ranked by the selection score (training-set CV AUC; test metrics reported, not ranked on)",
    "",
    md_table(full, FULL_COLS, FULL_LABELS, ND),
    "",
    "Notes: " + "; ".join(f"`{s}`: {n}" for s, n in zip(full["slug"], full["cv_note"]) if n) + ".",
    "`fit_seconds` is what each script recorded — for some models the final refit only (xgboost 0.5 s, mlp 1.0 s), for others the "
    "whole GridSearchCV wall time (lda, naive_bayes, logistic_l1, tree_pruned); see `tune_seconds` in the CSV and the model notes. "
    "KNN and SVM pay at prediction time instead (KNN 0.38 s, RBF-SVM 7.8 s for the 8,990 test rows).",
]
open(os.path.join(RESULTS, "model_comparison.md"), "w").write("\n".join(md) + "\n")
MAIN_NOTE = md.index("## Headline models — per-class test metrics (sample-report layout)") + 2   # the explanatory paragraph
open(os.path.join(RESULTS, "model_comparison_main.md"), "w").write(
    "# Headline models — per-class test metrics\n\n" + md[MAIN_NOTE] + "\n\n" + main_md() + "\n")
print(f"tables written; selected headline model by CV AUC: {BEST} (CV {R[BEST]['cv_auc']:.4f}, test {R[BEST]['test_auc']:.4f}); "
      f"top-4 by CV AUC: {TOP4}; test-AUC leader (reported only): {TEST_LEADER}")


# ----------------------------------------------------------------------------- 3. figures
PANELS = [("Linear, generative and kernel models", ["logistic", "logistic_l1", "lda", "naive_bayes", "svm_rbf"]),
          ("Trees, ensembles, KNN and MLP (logistic baseline for reference)",
           ["tree_pruned", "random_forest", "gradient_boosting", "xgboost", "knn", "mlp", "logistic"])]


def plot_curves(kind):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, (title, slugs) in zip(axes, PANELS):
        for s in slugs:
            p = P_TEST[s]; ref = (s == "logistic" and "logistic" != slugs[0])
            if kind == "roc":
                x, y, _ = roc_curve(y_te, p); lab = f"{SHORT[s]} — AUC {R[s]['test_auc']:.3f}"
            else:
                y, x, _ = precision_recall_curve(y_te, p); lab = f"{SHORT[s]} — AP {R[s]['test_ap']:.3f}"
            ax.plot(x, y, color=(INK3 if ref else color_of(s)), ls=(":" if ref else ls_of(s)), lw=(1.4 if ref else 2.0),
                    label=lab, alpha=0.95, zorder=3 if s == BEST else 2)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=INK3, lw=0.9, ls="--", label="chance (AUC 0.5)")
            ax.set_xlabel("False positive rate (share of non-defaulters flagged)"); ax.set_ylabel("True positive rate (recall of defaulters)")
        else:
            ax.axhline(prevalence, color=INK3, lw=0.9, ls="--", label=f"prevalence = {prevalence:.3f} (random)")
            ax.set_xlabel("Recall (share of defaulters flagged)"); ax.set_ylabel("Precision (share of flagged that default)")
        ax.set_title(title); ax.legend(loc="lower right" if kind == "roc" else "upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    fig.suptitle(("ROC curves" if kind == "roc" else "Precision–recall curves") + " on the shared test set (n = 8,990)", fontsize=12)
    save_extra_figure(fig, f"all_{kind}.png")


plot_curves("roc"); plot_curves("pr")


def plot_auc_bar(df, fname, title, label_col="name", figsize=(12.5, 5.6), show_ap=True):
    d = df.sort_values("select_auc", ascending=True, na_position="first", kind="mergesort")    # top row = selected model
    ncol = 2 if show_ap else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    yy = np.arange(len(d))
    subs = [bool(cv_note_of(R[s])) for s in d["slug"]]                     # CV AUC measured on a tuning subsample
    labels = [SHORT.get(s, n) + (" †" if sb else "") for s, n, sb in zip(d["slug"], d[label_col], subs)]
    cols = [color_of(s) for s in d["slug"]]
    ax = axes[0]
    ax.barh(yy, d["test_auc"], color=cols, height=0.62, zorder=2)
    ok = d["cv_auc"].notna()
    cv_hi = (d["cv_auc"] + d["cv_auc_std"].fillna(0)).fillna(d["test_auc"])
    ax.errorbar(d.loc[ok, "cv_auc"], yy[ok.values], xerr=d.loc[ok, "cv_auc_std"].fillna(0), fmt="o", color=INK, ms=4,
                capsize=2.5, lw=1.1, zorder=4, label="5-fold CV AUC ± 1 sd (training set)")
    for y_, v, hi in zip(yy, d["test_auc"], cv_hi):              # value label to the right of both the bar and the error bar
        ax.text(max(v, hi) + 0.0015, y_, f"{v:.4f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(LOG["test_auc"], color=INK3, lw=1, ls="--", zorder=1, label=f"logistic baseline {LOG['test_auc']:.4f}")
    lo = max(0.5, float(np.nanmin(np.r_[d["test_auc"], d["cv_auc"].fillna(1)])) - 0.02)
    ax.set_xlim(lo, min(0.82, float(d["test_auc"].max()) + 0.03)); ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(d) - 0.3)
    for f_, c in FAMILY_COLOR.items():                             # family colour key (drawn off-canvas, legend only)
        if f_ in set(d["family"]): ax.barh([-5], [0], color=c, label=f_)
    ax.set_xlabel("Test AUC (bar) and CV AUC (marker ± sd)" + ("   † CV AUC measured on a 6,000-row tuning subsample" if any(subs) else ""))
    ax.set_title("ROC AUC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, columnspacing=1.2, handlelength=1.6)   # below the axes: never on a bar
    if show_ap:
        ax = axes[1]
        ax.barh(yy, d["test_ap"], color=cols, height=0.62, zorder=2)
        for y_, v in zip(yy, d["test_ap"]):
            ax.text(v + 0.002, y_, f"{v:.4f}", va="center", ha="left", fontsize=8, color=INK2)
        ax.axvline(LOG["test_ap"], color=INK3, lw=1, ls="--", zorder=1, label=f"logistic baseline {LOG['test_ap']:.4f}")
        ax.axvline(prevalence, color=INK3, lw=0.8, ls=":", zorder=1, label=f"random = prevalence {prevalence:.3f}")
        ax.set_xlim(max(0.2, float(d["test_ap"].min()) - 0.03), min(0.65, float(d["test_ap"].max()) + 0.04))
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.set_xlabel("Test average precision (area under the PR curve)"); ax.set_title("Average precision")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    fig.suptitle(title, fontsize=12)
    save_extra_figure(fig, fname)


plot_auc_bar(head, "all_auc_bar.png", "Headline models ranked by CV AUC (training set) — test AUC bars, CV AUC markers (colour = model family)")
plot_auc_bar(full, "all_auc_bar_full.png", f"All {len(full)} model results ranked by CV AUC — test AUC / AP (colour = model family)",
             figsize=(13.5, 10.5))

# recall / precision / F1 vs threshold for the top-4 models by CV AUC (test set), tuned thresholds marked
GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def thr_curves(y, p):
    P = p[None, :] >= GRID[:, None]
    tp = (P & (y == 1)).sum(1); fp = (P & (y == 0)).sum(1); fn = (~P & (y == 1)).sum(1); tn = (~P & (y == 0)).sum(1)
    rec = tp / np.maximum(tp + fn, 1); prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    f1 = np.where(np.isnan(prec), 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12))
    cost = (COST_FN * fn + COST_FP * fp) / len(y) * 1000.0            # cost per 1,000 clients
    fpr = fp / np.maximum(fp + tn, 1)
    return dict(recall=rec, precision=prec, f1=f1, cost=cost, fpr=fpr)


CURVES_TEST = {s: thr_curves(y_te, P_TEST[s]) for s in HEADLINE}
CURVES_OOF = {s: thr_curves(y_tr, P_OOF[s]) for s in HEADLINE if s in P_OOF and len(P_OOF[s]) == len(y_tr)}
TOP4_PLUS = TOP4 + (["logistic"] if "logistic" not in TOP4 else [])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for key, ax, ttl in zip(["recall", "precision", "f1"], axes, ["Recall of defaulters (class 1)", "Precision of class 1", "F1 of class 1"]):
    for s in TOP4_PLUS:
        ref = s not in TOP4; c = CURVES_TEST[s]
        ax.plot(GRID, c[key], color=(INK3 if ref else color_of(s)), ls=(":" if ref else ls_of(s)), lw=(1.4 if ref else 2.0),
                label=SHORT[s] + (" (reference)" if ref else ""))
        tf, tc = R[s]["at_tuned_f1"]["threshold"], R[s]["at_tuned_cost"]["threshold"]
        i_f, i_c = int(np.argmin(np.abs(GRID - tf))), int(np.argmin(np.abs(GRID - tc)))
        ax.plot([tf], [c[key][i_f]], marker="o", ms=6, color=(INK3 if ref else color_of(s)), mec="white", mew=1, zorder=5)
        ax.plot([tc], [c[key][i_c]], marker="D", ms=5.5, color=(INK3 if ref else color_of(s)), mec="white", mew=1, zorder=5)
    ax.axvline(0.5, color=INK3, lw=0.8, ls="--"); ax.text(0.505, 0.02, "default 0.5", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Decision threshold on P(default)"); ax.set_title(ttl)
axes[0].plot([], [], marker="o", ms=6, color=INK, ls="none", label="F1-tuned threshold (train OOF)")
axes[0].plot([], [], marker="D", ms=5.5, color=INK, ls="none", label="5:1 cost-tuned threshold (train OOF)")
axes[0].legend(loc="upper right"); axes[0].set_ylabel("Metric on the test set")
fig.suptitle("Threshold trade-off on the test set — top-4 models by CV AUC (markers = thresholds chosen on training OOF probabilities)", fontsize=11.5)
save_extra_figure(fig, "threshold_curves.png")

# cost vs threshold (5:1 assumption)
flag_none = COST_FN * prevalence * 1000; flag_all = COST_FP * (1 - prevalence) * 1000
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
ax = axes[0]
for s in TOP4_PLUS:
    ref = s not in TOP4; c = CURVES_TEST[s]
    ax.plot(GRID, c["cost"], color=(INK3 if ref else color_of(s)), ls=(":" if ref else ls_of(s)), lw=(1.4 if ref else 2.0),
            label=f"{SHORT[s]} — cost {R[s]['at_tuned_cost']['cost']:,.0f} at t = {R[s]['at_tuned_cost']['threshold']:.2f}")
    tc = R[s]["at_tuned_cost"]["threshold"]; i_c = int(np.argmin(np.abs(GRID - tc)))
    ax.plot([tc], [c["cost"][i_c]], marker="D", ms=6, color=(INK3 if ref else color_of(s)), mec="white", mew=1, zorder=5)
ax.axhline(flag_none, color=INK3, lw=0.9, ls="--"); ax.text(0.99, flag_none + 6, f"flag nobody: {flag_none:,.0f}", ha="right", fontsize=7.5, color=INK2)
ax.axhline(flag_all, color=INK3, lw=0.9, ls="--"); ax.text(0.99, flag_all + 6, f"flag everybody: {flag_all:,.0f}", ha="right", fontsize=7.5, color=INK2)
ax.axvline(0.5, color=INK3, lw=0.8, ls=":"); ax.text(0.505, 0.97, "default 0.5", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform(), va="top")
ax.set_xlim(0, 1); ax.set_ylim(450, 1200); ax.set_xlabel("Decision threshold on P(default)")
ax.set_ylabel(f"Expected cost per 1,000 clients  ({COST_FN:.0f}·FN + {COST_FP:.0f}·FP)"); ax.set_title("Test-set cost — top-4 by CV AUC (diamond = OOF-chosen threshold)")
ax.legend(loc="lower right")
ax = axes[1]
s = BEST; ct, co = CURVES_TEST[s], CURVES_OOF.get(s)
ax.plot(GRID, ct["cost"], color=color_of(s), lw=2.0, label=f"{SHORT[s]} — test set")
if co is not None:
    ax.plot(GRID, co["cost"], color=color_of(s), lw=1.6, ls="--", label=f"{SHORT[s]} — training out-of-fold (used to pick t)")
tc = R[s]["at_tuned_cost"]["threshold"]; i_t = int(np.argmin(ct["cost"]))
ax.axvline(tc, color=INK, lw=1, ls="-", alpha=0.6); ax.text(tc + 0.01, 0.62, f"chosen on OOF: t = {tc:.2f}", fontsize=8, color=INK2, transform=ax.get_xaxis_transform(), va="top")
ax.plot([GRID[i_t]], [ct["cost"][i_t]], marker="o", ms=7, color=INK, mec="white", ls="none", label=f"test-set minimum at t = {GRID[i_t]:.2f} ({ct['cost'][i_t]:.0f} / 1,000)")
ax.axhline(flag_none, color=INK3, lw=0.9, ls="--"); ax.axhline(flag_all, color=INK3, lw=0.9, ls="--")
ax.set_xlim(0, 1); ax.set_ylim(450, 1200); ax.set_xlabel("Decision threshold on P(default)"); ax.set_ylabel("Expected cost per 1,000 clients")
ax.set_title(f"Selected model ({SHORT[s]}): does the OOF threshold transfer to test?"); ax.legend(loc="lower right")
fig.suptitle(f"Cost-vs-threshold under the {COST_FN:.0f}:1 cost assumption (missing a defaulter costs {COST_FN:.0f}× a false alarm)", fontsize=11.5)
save_extra_figure(fig, "cost_curves.png")

# calibration (reliability) of the top-4 models by CV AUC
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
for s in TOP4_PLUS:
    ref = s not in TOP4
    pt, pp = calibration_curve(y_te, np.clip(P_TEST[s], 0, 1), n_bins=10, strategy="quantile")
    ax.plot(pp, pt, marker="o", ms=5, color=(INK3 if ref else color_of(s)), ls=(":" if ref else ls_of(s)), lw=(1.4 if ref else 2.0),
            mec="white", mew=0.8, label=f"{SHORT[s]} — Brier {R[s]['test_brier']:.4f}, mean p {P_TEST[s].mean():.3f}")
ax.plot([0, 1], [0, 1], color=INK3, lw=0.9, ls="--", label="perfect calibration")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted P(default) in decile bin"); ax.set_ylabel("Observed default rate in bin")
ax.set_title(f"Reliability diagram, 10 quantile bins (test set; prevalence {prevalence:.3f})"); ax.legend(loc="upper left")
ax = axes[1]
bins = np.linspace(0, 1, 41)
for s in TOP4_PLUS:
    ref = s not in TOP4
    ax.hist(P_TEST[s], bins=bins, histtype="step", lw=(1.4 if ref else 1.8), color=(INK3 if ref else color_of(s)),
            ls=(":" if ref else ls_of(s)), label=SHORT[s] + (" (reference)" if ref else ""))
ax.axvline(prevalence, color=INK3, lw=0.9, ls="--"); ax.text(prevalence + 0.01, 0.97, "prevalence", fontsize=8, color=INK2, transform=ax.get_xaxis_transform(), va="top")
ax.set_xlim(0, 1); ax.set_xlabel("Predicted P(default)"); ax.set_ylabel("Number of test clients"); ax.set_title("Distribution of predicted probabilities"); ax.legend(loc="upper right")
fig.suptitle("Calibration of the top-4 models by CV AUC (probabilities as saved by each script; nothing re-fitted)", fontsize=11.5)
save_extra_figure(fig, "calibration.png")
print(f"figures written ({time.time() - T0:.0f} s so far)")


# ----------------------------------------------------------------------------- 4. paired bootstrap of test-set AUC / AP differences
rng = np.random.default_rng(SEED)
idx_all = rng.integers(0, n_test, size=(B_BOOT, n_test))
AUC_B = {s: np.empty(B_BOOT) for s in HEADLINE}; AP_B = {s: np.empty(B_BOOT) for s in HEADLINE}
for b in range(B_BOOT):
    idx = idx_all[b]; yb = y_te[idx]
    for s in HEADLINE:
        pb = P_TEST[s][idx]
        AUC_B[s][b] = roc_auc_score(yb, pb); AP_B[s][b] = average_precision_score(yb, pb)
boot_rows = []
for s in HEADLINE:
    d_auc = AUC_B[BEST] - AUC_B[s]; d_ap = AP_B[BEST] - AP_B[s]
    boot_rows.append(dict(
        model=R[s]["name"], slug=s, best=BEST, select_auc=select_auc_of(R[s]), test_auc=R[s]["test_auc"],
        auc_ci_2_5=float(np.percentile(AUC_B[s], 2.5)), auc_ci_97_5=float(np.percentile(AUC_B[s], 97.5)),
        auc_diff_point=float(R[BEST]["test_auc"] - R[s]["test_auc"]), auc_diff_mean=float(d_auc.mean()),
        auc_diff_2_5=float(np.percentile(d_auc, 2.5)), auc_diff_97_5=float(np.percentile(d_auc, 97.5)),
        auc_share_replicates_best_not_better=float((d_auc <= 0).mean()),
        auc_diff_excludes_zero=bool(np.percentile(d_auc, 2.5) > 0 or np.percentile(d_auc, 97.5) < 0),
        test_ap=R[s]["test_ap"], ap_ci_2_5=float(np.percentile(AP_B[s], 2.5)), ap_ci_97_5=float(np.percentile(AP_B[s], 97.5)),
        ap_diff_point=float(R[BEST]["test_ap"] - R[s]["test_ap"]), ap_diff_mean=float(d_ap.mean()),
        ap_diff_2_5=float(np.percentile(d_ap, 2.5)), ap_diff_97_5=float(np.percentile(d_ap, 97.5)),
        ap_diff_excludes_zero=bool(np.percentile(d_ap, 2.5) > 0 or np.percentile(d_ap, 97.5) < 0)))
SEL_ORDER = {s_: i for i, s_ in enumerate(head["slug"])}           # display order = selection-score order, not test AUC
boot = pd.DataFrame(boot_rows).sort_values("slug", key=lambda c: c.map(SEL_ORDER)).reset_index(drop=True)
boot.to_csv(os.path.join(RESULTS, "auc_bootstrap_ci.csv"), index=False)
print(f"bootstrap done (B = {B_BOOT}, seed {SEED}, {time.time() - T0:.0f} s so far)")
print(boot[["slug", "select_auc", "test_auc", "auc_diff_mean", "auc_diff_2_5", "auc_diff_97_5", "auc_diff_excludes_zero"]].to_string(index=False))

# append the bootstrap table to the markdown comparison
md_boot = ["", f"## Paired bootstrap of test-set differences vs the headline model selected by CV AUC (`{BEST}`), B = {B_BOOT}, seed {SEED}", "",
           "Each replicate resamples the 8,990 test rows with replacement and re-scores every model on the same rows, so the difference "
           "is paired. `excl. 0` = the 95 % percentile interval of the difference excludes zero (the difference is larger than test-set "
           "sampling noise); `P(best ≤ model)` = share of replicates in which the selected model did not beat this model. Rows in "
           "selection-score order.", "",
           md_table(boot, ["model", "select_auc", "test_auc", "auc_ci_2_5", "auc_ci_97_5", "auc_diff_mean", "auc_diff_2_5", "auc_diff_97_5",
                           "auc_share_replicates_best_not_better", "auc_diff_excludes_zero", "test_ap", "ap_diff_mean", "ap_diff_2_5",
                           "ap_diff_97_5", "ap_diff_excludes_zero"],
                    ["model", "select AUC", "test AUC", "AUC 2.5 %", "AUC 97.5 %", "Δ AUC mean", "Δ 2.5 %", "Δ 97.5 %", "P(best ≤ model)", "excl. 0",
                     "test AP", "Δ AP mean", "Δ AP 2.5 %", "Δ AP 97.5 %", "excl. 0"],
                    {"auc_share_replicates_best_not_better": 3})]
open(os.path.join(RESULTS, "model_comparison.md"), "a").write("\n".join(md_boot) + "\n")


# ----------------------------------------------------------------------------- 5. summary JSON
def rec(s, keys=("name", "member", "cv_auc", "cv_auc_std", "oof_auc", "test_auc", "test_ap", "test_brier", "fit_seconds")):
    r = R[s]; out = {k: r.get(k) for k in keys}
    out.update(slug=s, family=family_of(s), select_auc=select_auc_of(r), thr_f1=r["at_tuned_f1"]["threshold"], recall_1_tuned=r["at_tuned_f1"]["recall_1"],
               precision_1_tuned=r["at_tuned_f1"]["precision_1"], f1_1_tuned=r["at_tuned_f1"]["f1_1"],
               accuracy_0_5=r["at_0_5"]["accuracy"], recall_1_0_5=r["at_0_5"]["recall_1"], precision_1_0_5=r["at_0_5"]["precision_1"],
               thr_cost=r["at_tuned_cost"]["threshold"], recall_1_cost=r["at_tuned_cost"]["recall_1"],
               fpr_cost=r["at_tuned_cost"]["fpr"], fnr_cost=r["at_tuned_cost"]["fnr"], cost_cost=r["at_tuned_cost"]["cost"],
               cost_0_5=r["at_0_5"]["cost"], cost_f1=r["at_tuned_f1"]["cost"])
    return out


def ranking(col, ascending=False, df=full):
    d = df.sort_values(col, ascending=ascending)
    return [dict(rank=i + 1, slug=s, value=float(v)) for i, (s, v) in enumerate(zip(d["slug"], d[col]))]


# imbalance study (LI Jingwen) and unsupervised study (RUAN Hangjing) — short machine-readable summaries for the report
imb_path = os.path.join(RESULTS, "imbalance_comparison.csv")
imb = None
if os.path.exists(imb_path):
    t = pd.read_csv(imb_path)
    imb = dict(source="results/imbalance_comparison.csv", strategies=list(t["strategy"].unique()),
               rows=[{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in row.items()}
                     for row in t[["learner", "strategy", "slug", "cv_auc", "test_auc", "test_ap", "test_brier", "mean_p_test",
                                   "tuned_threshold_f1", "recall_1_at_0_5", "recall_1_at_tuned", "precision_1_at_tuned",
                                   "cost_at_cost_threshold"]].to_dict(orient="records")])
    for learner in t["learner"].unique():
        tl = t[t["learner"] == learner].set_index("strategy")
        imb[f"{learner}_test_auc_range"] = [float(tl["test_auc"].min()), float(tl["test_auc"].max())]
        imb[f"{learner}_best_strategy_by_cv_auc"] = str(tl["cv_auc"].idxmax())            # selection: training CV only
        imb[f"{learner}_test_auc_leader_reported_only"] = str(tl["test_auc"].idxmax())
        imb[f"{learner}_brier_none_vs_balanced"] = [float(tl.loc["none", "test_brier"]), float(tl.loc["balanced", "test_brier"])]
    leak = os.path.join(RESULTS, "imbalance_leakage_demo.csv")
    if os.path.exists(leak):
        tl = pd.read_csv(leak).set_index("learner")
        imb["leakage_demo"] = {l: dict(leaky_test_auc=float(tl.loc[l, "leaky_test_auc"]), honest_test_auc_smote=float(tl.loc[l, "honest_test_auc_smote"]),
                                       inflation=float(tl.loc[l, "inflation_vs_honest_smote"])) for l in tl.index}
uns_path = os.path.join(RESULTS, "unsupervised_summary.json")
uns = None
if os.path.exists(uns_path):
    u = json.load(open(uns_path))
    prof = u.get("kmeans", {}).get("profiles", [])
    uns = dict(source="results/unsupervised_summary.json",
               pca=dict(pve_top3=u["pca"]["pve"][:3], n_pcs_for_90pct=u["pca"]["n_pcs_for_90pct"], n_pcs_for_95pct=u["pca"]["n_pcs_for_95pct"],
                        auc_of_single_pc_score=u["pca"]["auc_of_single_pc_score"]),
               kmeans=dict(K_chosen=u["kmeans"]["K_chosen"], K_max_silhouette=u["kmeans"].get("K_max_silhouette"),
                           segments=[{k: p.get(k) for k in p if k in ("segment", "n_train", "share_train", "default_rate_train", "default_rate_test", "lift_train", "lift_test", "descriptor")} for p in prof]),
               hierarchical=dict(ari_vs_kmeans=u["hierarchical"].get("ari_vs_kmeans_at_K")),
               supervised={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))} for k, v in u["supervised"].items()})

best_r = R[BEST]
summary = dict(
    component="aggregate", seed=SEED, generated_by="code/aggregate.py", n_model_results=int(len(full)),
    n_headline=len(HEADLINE), headline_slugs=HEADLINE, excluded=dropped, n_test=int(n_test), n_train=int(len(y_tr)),
    prevalence_test=prevalence, majority_class_accuracy=float(1 - prevalence),
    cost_assumption=dict(cost_fn=COST_FN, cost_fp=COST_FP, flag_nobody_cost=float(COST_FN * y_te.sum()),
                         flag_everybody_cost=float(COST_FP * (n_test - y_te.sum()))),
    logistic_baseline=rec("logistic"), logistic_reference_in_readme=LOGISTIC_REF,
    selection=dict(rule="highest training-set selection score among the headline models; test metrics never used",
                   score="cv_auc (5-fold stratified CV on the training set); oof_auc where the CV ran on a tuning subsample",
                   score_source={s: ("oof_auc" if cv_note_of(R[s]) else "cv_auc") for s in HEADLINE},
                   best=BEST, top4=TOP4, test_auc_leader_reported_only=TEST_LEADER, best_is_also_test_auc_leader=bool(BEST == TEST_LEADER)),
    best_headline=rec(BEST),
    best_by=dict(note="descriptive only: which model leads on each metric; selection uses the CV score (see 'selection')",
                 cv_auc=full.iloc[0]["slug"], headline_cv_auc=BEST,
                 test_auc=full.sort_values("test_auc", ascending=False).iloc[0]["slug"], test_ap=full.sort_values("test_ap", ascending=False).iloc[0]["slug"],
                 brier=full.sort_values("brier").iloc[0]["slug"], cost_at_cost_threshold=full.sort_values("cost_cost").iloc[0]["slug"],
                 f1_at_tuned_threshold=full.sort_values("f1_1_tuned", ascending=False).iloc[0]["slug"],
                 headline_test_auc=TEST_LEADER, headline_test_ap=head.sort_values("test_ap", ascending=False).iloc[0]["slug"],
                 headline_brier=head.sort_values("brier").iloc[0]["slug"], headline_cost=head.sort_values("cost_cost").iloc[0]["slug"]),
    gap_best_minus_logistic=dict(test_auc=float(best_r["test_auc"] - LOG["test_auc"]), test_ap=float(best_r["test_ap"] - LOG["test_ap"]),
                                 brier=float(best_r["test_brier"] - LOG["test_brier"]),
                                 cost_at_cost_threshold=float(best_r["at_tuned_cost"]["cost"] - LOG["at_tuned_cost"]["cost"])),
    headline=[rec(s) for s in HEADLINE],
    rankings=dict(all_by_cv_auc=ranking("select_auc"), headline_by_cv_auc=ranking("select_auc", df=head),
                  all_by_test_auc=ranking("test_auc"), all_by_test_ap=ranking("test_ap"), all_by_brier=ranking("brier", True),
                  all_by_cost_at_cost_threshold=ranking("cost_cost", True), headline_by_test_auc=ranking("test_auc", df=head),
                  headline_by_test_ap=ranking("test_ap", df=head), headline_by_fit_seconds=ranking("fit_seconds", True, df=head)),
    family_best=[dict(family=f_, slug=b_["slug"], select_auc=float(b_["select_auc"]), test_auc=float(b_["test_auc"]))
                 for f_, g in full.groupby("family")
                 for b_ in [g.sort_values("select_auc", ascending=False, na_position="last", kind="mergesort").iloc[0]]],
    top4_by_cv_auc=TOP4,
    bootstrap=dict(B=B_BOOT, seed=SEED, best=BEST, method="paired row bootstrap of the 8,990 test rows; percentile 95 % intervals",
                   rows=boot.to_dict(orient="records"),
                   auc_not_distinguishable_from_best=[s for s, e in zip(boot["slug"], boot["auc_diff_excludes_zero"]) if not e and s != BEST],
                   auc_significantly_below_best=[s for s, e in zip(boot["slug"], boot["auc_diff_excludes_zero"]) if e],
                   auc_borderline=[s for s, e, lo_ in zip(boot["slug"], boot["auc_diff_excludes_zero"], boot["auc_diff_2_5"]) if e and lo_ < 0.001],
                   ap_not_distinguishable_from_best=[s for s, e in zip(boot["slug"], boot["ap_diff_excludes_zero"]) if not e and s != BEST]),
    thresholds=dict(f1_tuned={s: R[s]["at_tuned_f1"]["threshold"] for s in HEADLINE},
                    cost_tuned={s: R[s]["at_tuned_cost"]["threshold"] for s in HEADLINE},
                    recall_1_at_cost={s: R[s]["at_tuned_cost"]["recall_1"] for s in HEADLINE},
                    cost_test={s: dict(at_0_5=R[s]["at_0_5"]["cost"], at_f1=R[s]["at_tuned_f1"]["cost"], at_cost=R[s]["at_tuned_cost"]["cost"]) for s in HEADLINE},
                    best_test_cost_curve_argmin=float(GRID[int(np.argmin(CURVES_TEST[BEST]["cost"]))])),
    calibration={s: dict(brier=R[s]["test_brier"], mean_p_test=float(P_TEST[s].mean()), logloss=R[s].get("test_logloss")) for s in TOP4_PLUS},
    runtime=dict(fit_seconds={s: R[s].get("fit_seconds") for s in HEADLINE}, tune_seconds={s: tune_seconds_of(R[s]) for s in HEADLINE},
                 predict_seconds_test=dict(knn=R["knn"].get("extra", {}).get("predict_seconds_test"),
                                           svm_rbf=R["svm_rbf"].get("extra", {}).get("predict_seconds_test_decision_function_after_evaluate")),
                 note="fit_seconds semantics differ by script (final refit only vs whole grid); wall-clock on a shared laptop"),
    cv_notes={s: cv_note_of(R[s]) for s in HEADLINE if cv_note_of(R[s])},
    imbalance_study=imb, unsupervised_study=uns,
    files=dict(tables=["results/model_comparison.csv", "results/model_comparison.md", "results/model_comparison_main.csv",
                       "results/model_comparison_main.md", "results/auc_bootstrap_ci.csv"],
               figures=["figures/all_roc.png", "figures/all_pr.png", "figures/all_auc_bar.png", "figures/all_auc_bar_full.png",
                        "figures/threshold_curves.png", "figures/cost_curves.png", "figures/calibration.png"]),
    seconds=round(time.time() - T0, 1))


def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, float): return None if np.isnan(o) else o
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o


json.dump(_clean(summary), open(os.path.join(RESULTS, "aggregate_summary.json"), "w"), indent=1, ensure_ascii=False)
print(f"results/aggregate_summary.json written; total {time.time() - T0:.0f} s")
print(f"selected headline (CV AUC): {BEST} test AUC {best_r['test_auc']:.4f} / AP {best_r['test_ap']:.4f} vs logistic {LOG['test_auc']:.4f} / {LOG['test_ap']:.4f}")
