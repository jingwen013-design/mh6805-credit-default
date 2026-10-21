"""
fold_stability.py - fold stability of the headline models, simple-rule benchmarks, the cost-preferred model by cost ratio,
top-k capture and the out-of-fold ROC / PR curves. TRAINING ROWS ONLY; nothing is fitted, tuned or re-scored.

Run from Project/code:  /opt/miniconda3/bin/python3 fold_stability.py        (about 5 seconds, single process)

Training-only contract
  * Reads data/processed/train.csv directly (20,975 rows, asserted) for the labels and the three arrears columns PAY_1,
    MAX_DELAY and N_MONTHS_DELAYED. data/processed/test.csv is never opened; common.load_split() and common.evaluate()
    are never called; no held-out prediction vector is loaded.
  * The model scores are the frozen training out-of-fold probabilities results/<slug>_p_oof.npy of the eleven headline
    models (results/model_comparison.csv, headline == True; row order = train.csv order). Nothing is refitted.
  * Folds: common.CV (StratifiedKFold(5, shuffle=True, random_state=2607526)) applied to the training labels, checked
    against data/processed/cv_folds.csv. These are the folds every OOF vector was produced on.
  * Seed 2607526 (common.SEED). The script has no random step: ties in a top-k list are resolved by their expected value
    under a random order, which is exact and needs no random draw.

Paired fold comparisons use the Nadeau-Bengio corrected t interval, mine_v2_utils.corrected_t_ci(diffs, n_train, n_held)
with J = 5 folds of one 5-fold CV, i.e. n_held / n_train = 4,195 / 16,780 = 1/4 (95%, t with 4 df). Every paired table
reports the five per-fold values, the mean difference, the number of folds with a positive difference and that interval.

Inputs
  data/processed/train.csv, data/processed/cv_folds.csv, results/model_comparison.csv, results/<slug>.json (CV AUC only,
  as a consistency check), results/<slug>_p_oof.npy for the eleven headline slugs.

Outputs
  (a) results/fold_stability.csv          per-fold ROC-AUC and AP, per-fold ranks, wins and XGBoost-minus-model paired
                                          differences (NB interval, Holm-adjusted p), OOF Brier and calibration slope
      figures/fold_stability.png          per-fold AUC by model | XGBoost minus model, per fold, with NB interval
  (b) results/simple_rule_benchmark.csv   no-parameter arrears rules and blanket policies vs logit / XGBoost at their
                                          OOF-chosen 5:1 cut-offs (TPR, FPR, precision, flag rate, cost per 1,000 clients)
  (c) results/preferred_model_by_ratio_oof.csv       cost-preferred model at r = 1, 2, 3, 5, 7, 10, 15, 20 (one row per r)
      results/preferred_model_by_ratio_oof_long.csv  every model x ratio (cut-off, flag rate, OOF and cross-fitted cost)
  (d) results/topk_capture_oof.csv        share of defaulters captured, and precision, in the top 10 / 20 / 30% of scores
  (e) figures/roc_pr_oof.png              OOF ROC and PR curves: logit, KNN, XGBoost, random forest (+ rules, 5:1 points)
      results/fold_stability_summary.json the headline numbers of (a)-(d) in one place

Cut-offs are chosen on the pooled training OOF probabilities with the project grid mine_v2_utils.THR_GRID (0.01-0.95,
step 0.01). Because the same OOF rows choose and score that cut-off, the pooled OOF cost is slightly optimistic; the
cross-fitted cost (cut-off for each fold chosen on the other four folds, mine_v2_utils.cross_fitted_cost) is reported next
to it as the honest version. The OOF of tuned models still carries some selection optimism (only KNN and the logit were
nested); it affects every tuned model in the same direction.
"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy import stats
from common import CV, SEED, ROOT, RESULTS, FIGURES, TARGET          # constants only; no data loader is used
from mine_v2_utils import THR_GRID, best_threshold, cost_at, cross_fitted_cost, corrected_t_ci, calibration_summary
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
np.random.seed(SEED)                                                   # no random step below; set for completeness

# ----------------------------------------------------------------------------- data (training rows only)
TRAIN_CSV = os.path.join(ROOT, "data", "processed", "train.csv")
tr = pd.read_csv(TRAIN_CSV)
assert len(tr) == 20975, f"expected 20,975 training rows, found {len(tr):,}"
y = tr[TARGET].astype(int).values
N = len(y); N_POS = int(y.sum()); N_NEG = N - N_POS; BASE = N_POS / N

folds = list(CV.split(np.zeros((N, 1)), y))                           # identical to GridSearchCV / cross_val_predict(cv=CV)
fold_id = np.zeros(N, dtype=int)
for k, (_, va) in enumerate(folds, start=1):
    fold_id[va] = k
cvf = pd.read_csv(os.path.join(ROOT, "data", "processed", "cv_folds.csv"))
assert (cvf["fold"].values == fold_id).all() and (cvf["DEFAULT"].values == y).all(), "cv_folds.csv disagrees with common.CV"
J = len(folds)
N_HELD = int(np.median([len(va) for _, va in folds])); N_FIT = N - N_HELD          # 4,195 / 16,780
assert N_HELD == 4195 and N_FIT == 16780

mc = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
head = mc[mc["headline"] == True].copy()
HEADLINE = list(head["slug"])
assert len(HEADLINE) == 11, HEADLINE
NAME = {"xgboost": "XGBoost", "gradient_boosting": "Gradient boosting", "random_forest": "Random forest", "mlp": "MLP",
        "knn": "KNN (K=101)", "tree_pruned": "Pruned tree (17 leaves)", "svm_rbf": "SVM (RBF)", "logistic": "Logistic (plain)",
        "logistic_l1": "Logistic L1", "lda": "LDA", "naive_bayes": "Gaussian NB"}
P = {}
for s in HEADLINE:
    p = np.load(os.path.join(RESULTS, f"{s}_p_oof.npy")).astype(float)
    assert p.shape == (N,), (s, p.shape)
    P[s] = p
REF = "xgboost"

# ----------------------------------------------------------------------------- palette / style (copied from aggregate.py;
# aggregate.py is not imported because importing it runs its scoring code)
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#4a3aa7", red="#e34948")
INK, INK2, INK3, GRID_C, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd", "#fcfcfb"
FAMILY = {"xgboost": "boosting", "gradient_boosting": "boosting", "random_forest": "forest / bagging", "mlp": "neural net",
          "knn": "KNN", "tree_pruned": "single tree", "svm_rbf": "SVM", "logistic": "linear", "logistic_l1": "linear",
          "lda": "linear", "naive_bayes": "naive Bayes"}
FAMILY_COLOR = {"boosting": PAL["blue"], "linear": PAL["orange"], "forest / bagging": PAL["aqua"],
                "single tree": PAL["yellow"], "naive Bayes": PAL["magenta"], "KNN": PAL["green"],
                "SVM": PAL["violet"], "neural net": PAL["red"]}
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.dpi": 140, "savefig.dpi": 140, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def col(s):
    return FAMILY_COLOR[FAMILY[s]]


def nb(diffs):
    """Paired fold comparison: per-fold values, mean, folds > 0, Nadeau-Bengio 95% interval and two-sided p."""
    d = np.asarray(diffs, float)
    ci = corrected_t_ci(d, N_FIT, N_HELD)
    se = math.sqrt((1.0 / J + N_HELD / N_FIT) * d.var(ddof=1))
    p = float(2 * stats.t.sf(abs(d.mean()) / se, J - 1)) if se > 0 else float("nan")
    return dict(mean=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=p, n_pos=int((d > 0).sum()), folds=[float(v) for v in d])


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def per1000(cost, n):
    return 1000.0 * cost / n


def topk_tp(yy, s, k):
    """Expected number of defaulters among the k highest scores; a tied block at the boundary is taken at random,
    i.e. contributes its defaulter share times the number of its rows that fit (exact, no random draw)."""
    thr = np.sort(s)[::-1][k - 1]
    above = s > thr; tie = s == thr
    a = int(above.sum()); b = int(tie.sum())
    tp = float(yy[above].sum()) + float(yy[tie].sum()) * (k - a) / b
    return tp, b


def confusion(yy, flag):
    flag = np.asarray(flag, bool)
    tp = int((flag & (yy == 1)).sum()); fp = int((flag & (yy == 0)).sum())
    fn = int((~flag & (yy == 1)).sum()); tn = int((~flag & (yy == 0)).sum())
    return tp, fp, fn, tn


def op_stats(yy, flag, ratio=5.0):
    tp, fp, fn, tn = confusion(yy, flag); n = len(yy)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn); fpr = fp / (fp + tn)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    cost = ratio * fn + fp
    return dict(n_flagged=tp + fp, flag_rate=(tp + fp) / n, tp=tp, fp=fp, fn=fn, tn=tn, tpr_recall=rec, fpr=fpr,
                precision=prec, f1=f1, false_alarms_per_defaulter_caught=(fp / tp if tp else float("nan")),
                single_point_auc=(rec + 1 - fpr) / 2, cost_5to1=float(cost), cost_5to1_per_1000=per1000(cost, n))


def xfit_fold_costs(p, ratio):
    """Cost of each fold at the cut-off chosen on the other four folds' OOF probabilities (sums to cross_fitted_cost)."""
    out = []
    for trn, va in folds:
        t = best_threshold(y[trn], p[trn], ratio)
        out.append(cost_at(y[va], p[va], t, ratio))
    return np.array(out, float)


summary = dict(script="code/fold_stability.py", training_rows=N, defaulters=N_POS, base_rate=BASE, folds=J,
               fold_sizes=[int(len(va)) for _, va in folds], nb_ratio_held_over_fit=N_HELD / N_FIT, seed=SEED,
               threshold_grid="mine_v2_utils.THR_GRID = 0.01..0.95 step 0.01", headline=HEADLINE)

# ============================================================================= (a) fold stability
auc = pd.DataFrame({s: [roc_auc_score(y[va], P[s][va]) for _, va in folds] for s in HEADLINE}, index=range(1, J + 1))
ap = pd.DataFrame({s: [average_precision_score(y[va], P[s][va]) for _, va in folds] for s in HEADLINE}, index=range(1, J + 1))
rank_auc = auc.rank(axis=1, ascending=False, method="min").astype(int)
rank_ap = ap.rank(axis=1, ascending=False, method="min").astype(int)

rows = []
for s in HEADLINE:
    js = json.load(open(os.path.join(RESULTS, f"{s}.json")))
    cal = calibration_summary(y, P[s])
    r = dict(model=NAME[s], slug=s, family=FAMILY[s], cv_auc_json=js.get("cv_auc"), cv_auc_std_json_ddof0=js.get("cv_auc_std"),
             pooled_oof_auc=roc_auc_score(y, P[s]))
    for k in auc.index: r[f"auc_fold{k}"] = auc.loc[k, s]
    r.update(auc_mean=auc[s].mean(), auc_sd_ddof1=auc[s].std(ddof=1), auc_sd_ddof0=auc[s].std(ddof=0),
             auc_min=auc[s].min(), auc_max=auc[s].max())
    for k in auc.index: r[f"rank_auc_fold{k}"] = int(rank_auc.loc[k, s])
    r.update(rank_auc_mean=rank_auc[s].mean(), rank_auc_best=int(rank_auc[s].min()), rank_auc_worst=int(rank_auc[s].max()))
    for k in ap.index: r[f"ap_fold{k}"] = ap.loc[k, s]
    r.update(ap_mean=ap[s].mean(), ap_sd_ddof1=ap[s].std(ddof=1), pooled_oof_ap=average_precision_score(y, P[s]))
    r["ap_lift_over_base_rate"] = r["pooled_oof_ap"] / BASE
    for k in ap.index: r[f"rank_ap_fold{k}"] = int(rank_ap.loc[k, s])
    r["rank_ap_mean"] = rank_ap[s].mean()
    r["folds_auc_above_logistic"] = int((auc[s] > auc["logistic"]).sum()) if s != "logistic" else None
    r["folds_auc_above_knn"] = int((auc[s] > auc["knn"]).sum()) if s != "knn" else None
    for met, tab in (("auc", auc), ("ap", ap)):
        if s == REF:
            continue
        c = nb(tab[REF].values - tab[s].values)
        for k, v in enumerate(c["folds"], start=1): r[f"xgb_minus_{met}_fold{k}"] = v
        r.update({f"xgb_minus_{met}_mean": c["mean"], f"xgb_{met}_wins_of_5": c["n_pos"], f"xgb_minus_{met}_nb_lo": c["lo"],
                  f"xgb_minus_{met}_nb_hi": c["hi"], f"xgb_minus_{met}_nb_p": c["p"]})
    r.update(brier_oof=cal["brier"], cal_slope_oof=cal["cal_slope"], cal_intercept_oof=cal["cal_intercept"],
             mean_pred_oof=cal["mean_pred"])
    rows.append(r)
fs = pd.DataFrame(rows)
for met in ("auc", "ap"):
    m = fs[f"xgb_minus_{met}_nb_p"].notna()
    fs.loc[m, f"xgb_minus_{met}_holm_p"] = holm(fs.loc[m, f"xgb_minus_{met}_nb_p"].values)
fs = fs.sort_values("auc_mean", ascending=False).reset_index(drop=True)
# consistency: the per-fold mean of the OOF AUC must equal the JSON CV AUC wherever that CV ran on the full training set
fs["auc_mean_minus_cv_auc_json"] = fs["auc_mean"] - fs["cv_auc_json"]
fs.to_csv(os.path.join(RESULTS, "fold_stability.csv"), index=False)

order_by_fold = {int(k): list(auc.loc[k].sort_values(ascending=False).index) for k in auc.index}
summary["a_fold_stability"] = dict(
    note="per-fold ROC-AUC/AP from the frozen OOF vectors on the common.CV folds; rank 1 = highest; XGBoost-minus-model paired "
         "differences with the Nadeau-Bengio interval (J=5, n_held/n_train=1/4); Holm over the ten XGBoost comparisons. "
         "svm_rbf's JSON cv_auc comes from a 6,000-row tuning subsample, so its auc_mean_minus_cv_auc_json is not a check.",
    ranking_by_fold=order_by_fold,
    per_model={r["slug"]: dict(auc_folds=[round(r[f"auc_fold{k}"], 4) for k in range(1, 6)], auc_mean=r["auc_mean"],
                               auc_sd_ddof1=r["auc_sd_ddof1"], ranks=[r[f"rank_auc_fold{k}"] for k in range(1, 6)],
                               ap_mean=r["ap_mean"], cal_slope_oof=r["cal_slope_oof"],
                               xgb_minus_auc=None if r["slug"] == REF else dict(
                                   mean=r["xgb_minus_auc_mean"], wins=r["xgb_auc_wins_of_5"], nb_lo=r["xgb_minus_auc_nb_lo"],
                                   nb_hi=r["xgb_minus_auc_nb_hi"], holm_p=r["xgb_minus_auc_holm_p"]))
               for r in fs.to_dict("records")})

# ---- figure (a)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]), sharey=True)
ordered = list(fs["slug"]); ypos = {s: i for i, s in enumerate(ordered[::-1])}; TOP = len(ordered) - 0.35
for s in ordered:
    yi = ypos[s]
    ax1.scatter(auc[s].values, [yi] * J, s=34, color=col(s), edgecolor=SURF, linewidth=1.2, zorder=3)
    ax1.plot([auc[s].mean()] * 2, [yi - 0.28, yi + 0.28], color=INK, lw=2, zorder=4, solid_capstyle="round")
    ax1.annotate(" ".join(f"{int(v):>2d}" for v in rank_auc[s].values), xy=(1.0, yi), xycoords=("axes fraction", "data"),
                 xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=7.6, color=INK2, family="monospace")
ax1.annotate("rank in fold\n 1  2  3  4  5", xy=(1.0, TOP), xycoords=("axes fraction", "data"), xytext=(6, 0),
             textcoords="offset points", va="bottom", ha="left", fontsize=7.2, color=INK3, family="monospace")
ax1.set_yticks(list(ypos.values())); ax1.set_yticklabels([NAME[s] for s in ordered[::-1]])
ax1.set_ylim(-0.6, len(ordered) + 0.55)
ax1.set_xlabel("ROC-AUC within each validation fold (4,195 rows)")
ax1.set_title("Per-fold AUC (dots = folds 1-5, bar = mean)", loc="left")
ax1.grid(axis="y", visible=False)
for s in ordered:
    yi = ypos[s]
    if s == REF:
        ax2.annotate("reference", xy=(0, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        continue
    d = auc[REF].values - auc[s].values; c = nb(d)
    ax2.plot([c["lo"], c["hi"]], [yi, yi], color=col(s), lw=2.2, zorder=2, solid_capstyle="round")
    ax2.scatter(d, [yi] * J, s=18, color=INK3, edgecolor=SURF, linewidth=0.8, zorder=3)
    ax2.scatter([c["mean"]], [yi], marker="D", s=40, color=col(s), edgecolor=SURF, linewidth=1.2, zorder=4)
    ax2.annotate(f"{c['n_pos']}/5", xy=(1.0, yi), xycoords=("axes fraction", "data"), xytext=(6, 0),
                 textcoords="offset points", va="center", ha="left", fontsize=7.6, color=INK2)
ax2.annotate("XGBoost\nahead in", xy=(1.0, TOP), xycoords=("axes fraction", "data"), xytext=(6, 0),
             textcoords="offset points", va="bottom", ha="left", fontsize=7.2, color=INK3)
ax2.axvline(0, color=INK3, lw=1, ls="--", zorder=1)
ax2.set_xlabel("XGBoost AUC minus model AUC, same fold")
ax2.set_title("Paired difference: folds (grey), mean and NB 95% interval", loc="left")
ax2.grid(axis="y", visible=False)
fig.suptitle("Fold stability of the eleven headline models - training out-of-fold scores, common.CV folds (seed 2607526)",
             fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 0.955, 0.96)); fig.subplots_adjust(wspace=0.36)
fig.savefig(os.path.join(FIGURES, "fold_stability.png")); plt.close(fig)

# ============================================================================= (b) simple rules (no fitted parameters)
RULES = [("PAY_1 >= 1", "late at least one month in the most recent month (September)", "PAY_1", 1),
         ("PAY_1 >= 2", "two or more months late in the most recent month", "PAY_1", 2),
         ("MAX_DELAY >= 2", "two or more months late in any of the six months", "MAX_DELAY", 2),
         ("N_MONTHS_DELAYED >= 1", "any payment delay in the six months", "N_MONTHS_DELAYED", 1)]
blank_all = np.ones(N, bool); blank_none = np.zeros(N, bool)
best_blanket = min(op_stats(y, blank_all)["cost_5to1"], op_stats(y, blank_none)["cost_5to1"])
fold_cost = lambda flag: np.array([per1000(5 * ((~flag[va]) & (y[va] == 1)).sum() + (flag[va] & (y[va] == 0)).sum(), len(va))
                                   for _, va in folds])
xfit5 = {s: xfit_fold_costs(P[s], 5) for s in ("logistic", "xgboost")}
for s in xfit5:
    assert abs(xfit5[s].sum() - cross_fitted_cost(y, P[s], folds, 5)) < 1e-9
xfit5_pk = {s: np.array([per1000(c, len(va)) for c, (_, va) in zip(xfit5[s], folds)]) for s in xfit5}

brows = []
def add_row(label, kind, definition, flag, score=None, threshold=None, xfit_pk=None):
    st = op_stats(y, flag)
    r = dict(rule_or_model=label, kind=kind, definition=definition, threshold=threshold, **st)
    r["score_auc"] = roc_auc_score(y, score) if score is not None else None
    r["saving_vs_best_blanket_per_1000"] = per1000(best_blanket - st["cost_5to1"], N)
    pk = fold_cost(flag) if xfit_pk is None else xfit_pk
    r["cost_5to1_crossfitted_per_1000"] = pk.mean() if xfit_pk is not None else st["cost_5to1_per_1000"]
    for k, v in enumerate(pk, start=1): r[f"cost_per_1000_fold{k}"] = v
    k_flag = st["n_flagged"]
    if 0 < k_flag < N:
        for s, nm in (("xgboost", "xgb"), ("logistic", "logit")):
            tp, _ = topk_tp(y, P[s], k_flag)
            r[f"{nm}_recall_same_flag_rate"] = tp / N_POS; r[f"{nm}_precision_same_flag_rate"] = tp / k_flag
    if kind != "model":
        for s, nm in (("xgboost", "xgb"), ("logistic", "logit")):
            c = nb(pk - xfit5_pk[s])                                    # rule cost minus model cost: > 0 = model cheaper
            r[f"{nm}_saving_vs_rule_per_1000_mean"] = c["mean"]; r[f"{nm}_cheaper_folds_of_5"] = c["n_pos"]
            r[f"{nm}_saving_vs_rule_nb_lo"] = c["lo"]; r[f"{nm}_saving_vs_rule_nb_hi"] = c["hi"]
    brows.append(r)

for label, definition, var, thr in RULES:
    add_row(label, "rule", definition, tr[var].values >= thr, score=tr[var].values.astype(float), threshold=thr)
add_row("flag everybody", "blanket", "every client is flagged", blank_all)
add_row("flag nobody", "blanket", "no client is flagged", blank_none)
for s in ("logistic", "xgboost"):
    t5 = best_threshold(y, P[s], 5)
    add_row(f"{NAME[s]} at OOF 5:1 cut-off", "model", f"{s}_p_oof >= {t5:.2f} (cut-off chosen on pooled training OOF, "
            "THR_GRID); cross-fitted columns choose it on the other four folds", P[s] >= t5, score=P[s], threshold=t5,
            xfit_pk=xfit5_pk[s])
rb = pd.DataFrame(brows)
rb.to_csv(os.path.join(RESULTS, "simple_rule_benchmark.csv"), index=False)
summary["b_simple_rules"] = dict(
    note="training rows; the rules have no fitted parameter, so the 20,975 rows are an honest evaluation for them; cost = 5*FN + FP; "
         "models' per-fold costs use the cut-off chosen on the other four folds; saving_vs_rule = rule minus model, > 0 = model cheaper",
    best_blanket_cost_5to1=best_blanket, best_blanket_per_1000=per1000(best_blanket, N),
    rows=rb[["rule_or_model", "threshold", "flag_rate", "tpr_recall", "fpr", "precision", "f1", "cost_5to1", "cost_5to1_per_1000",
             "cost_5to1_crossfitted_per_1000", "score_auc"]].to_dict("records"))

# ============================================================================= (c) preferred model by cost ratio (OOF only)
RATIOS = [1, 2, 3, 5, 7, 10, 15, 20]
long_rows, short_rows = [], []
for r_ in RATIOS:
    recs = {}
    for s in HEADLINE:
        t = best_threshold(y, P[s], r_); c = cost_at(y, P[s], t, r_)
        xf = xfit_fold_costs(P[s], r_)
        assert abs(xf.sum() - cross_fitted_cost(y, P[s], folds, r_)) < 1e-9
        recs[s] = dict(ratio=r_, model=NAME[s], slug=s, cutoff_oof=t, flag_rate=float((P[s] >= t).mean()),
                       recall=float(((P[s] >= t) & (y == 1)).sum() / N_POS), cost_oof=float(c), cost_oof_per_1000=per1000(c, N),
                       cost_crossfitted=float(xf.sum()), cost_crossfitted_per_1000=per1000(xf.sum(), N),
                       **{f"crossfitted_per_1000_fold{k}": per1000(v, len(va)) for k, (v, (_, va)) in enumerate(zip(xf, folds), 1)})
    df = pd.DataFrame(recs.values())
    df["rank_oof"] = df["cost_oof"].rank(method="min").astype(int)
    df["rank_crossfitted"] = df["cost_crossfitted"].rank(method="min").astype(int)
    df["gap_to_preferred_pct"] = 100 * (df["cost_oof"] / df["cost_oof"].min() - 1)
    blanket = min(N_NEG, r_ * N_POS)
    df["saving_vs_best_blanket_per_1000"] = per1000(blanket - df["cost_oof"], N)
    long_rows.append(df)
    srt = df.sort_values(["cost_oof", "slug"]); w, ru, th = srt.iloc[0], srt.iloc[1], srt.iloc[2]
    sx = df.sort_values(["cost_crossfitted", "slug"]); wx, rux = sx.iloc[0], sx.iloc[1]
    fw = np.array([w[f"crossfitted_per_1000_fold{k}"] for k in range(1, 6)])
    fr = np.array([ru[f"crossfitted_per_1000_fold{k}"] for k in range(1, 6)])
    c = nb(fr - fw)                                                     # runner-up minus preferred: > 0 = preferred cheaper
    lg = df[df.slug == "logistic"].iloc[0]; tp_ = df[df.slug == "tree_pruned"].iloc[0]
    top3 = srt.iloc[:3]
    short_rows.append(dict(
        ratio=r_, cutoff_theory=round(1 / (1 + r_), 3), preferred_oof=w["model"], preferred_slug=w["slug"],
        preferred_cutoff=w["cutoff_oof"], preferred_flag_rate=w["flag_rate"], preferred_recall=w["recall"],
        preferred_cost_oof=w["cost_oof"], preferred_cost_per_1000=w["cost_oof_per_1000"],
        runner_up=ru["model"], runner_up_cost_oof=ru["cost_oof"], margin_cost=ru["cost_oof"] - w["cost_oof"],
        margin_per_1000=ru["cost_oof_per_1000"] - w["cost_oof_per_1000"], margin_pct=100 * (ru["cost_oof"] / w["cost_oof"] - 1),
        third=th["model"], top3_spread_pct=100 * (top3["cost_oof"].max() / top3["cost_oof"].min() - 1),
        preferred_crossfitted=wx["model"], preferred_crossfitted_cost=wx["cost_crossfitted"],
        crossfitted_runner_up=rux["model"], crossfitted_margin=rux["cost_crossfitted"] - wx["cost_crossfitted"],
        same_preferred_both_ways=bool(w["slug"] == wx["slug"]),
        **{f"runner_up_minus_preferred_per_1000_fold{k}": v for k, v in enumerate(c["folds"], 1)},
        runner_up_minus_preferred_mean_per_1000=c["mean"], preferred_cheaper_folds_of_5=c["n_pos"],
        runner_up_minus_preferred_nb_lo=c["lo"], runner_up_minus_preferred_nb_hi=c["hi"],
        logistic_rank_oof=int(lg["rank_oof"]), logistic_cost_oof=lg["cost_oof"], logistic_gap_pct=lg["gap_to_preferred_pct"],
        tree_pruned_rank_oof=int(tp_["rank_oof"]), tree_pruned_gap_pct=tp_["gap_to_preferred_pct"],
        best_blanket=("flag everybody" if N_NEG <= r_ * N_POS else "flag nobody"), best_blanket_cost=blanket,
        preferred_saving_vs_blanket_per_1000=per1000(blanket - w["cost_oof"], N),
        preferred_saving_vs_blanket_pct=100 * (1 - w["cost_oof"] / blanket)))
pl = pd.concat(long_rows, ignore_index=True)
pl.to_csv(os.path.join(RESULTS, "preferred_model_by_ratio_oof_long.csv"), index=False)
ps = pd.DataFrame(short_rows)
ps.to_csv(os.path.join(RESULTS, "preferred_model_by_ratio_oof.csv"), index=False)
summary["c_preferred_by_ratio"] = dict(
    note="cut-off and cost on pooled training OOF (THR_GRID); cross-fitted = cut-off chosen on the other four folds; "
         "paired folds use the cross-fitted per-fold costs of the OOF-preferred model and its OOF runner-up",
    preferred_oof={int(r["ratio"]): r["preferred_oof"] for r in short_rows},
    preferred_crossfitted={int(r["ratio"]): r["preferred_crossfitted"] for r in short_rows},
    answer_changes_across_ratio=bool(ps["preferred_slug"].nunique() > 1),
    distinct_preferred=sorted(ps["preferred_oof"].unique()),
    margin_pct={int(r["ratio"]): r["margin_pct"] for r in short_rows},
    top3_spread_pct={int(r["ratio"]): r["top3_spread_pct"] for r in short_rows})

# ============================================================================= (d) top-k capture
SHARES = [0.10, 0.20, 0.30]
trows = []
scorers = [(s, NAME[s], "model", P[s]) for s in HEADLINE] + \
          [(v, f"{v} (ordinal score)", "rule score", tr[v].values.astype(float)) for v in ("N_MONTHS_DELAYED", "MAX_DELAY", "PAY_1")]
for q in SHARES:
    k = int(math.ceil(q * N - 1e-9))
    ref_f = []
    for _, va in folds:
        kf = int(math.ceil(q * len(va) - 1e-9)); ref_f.append(topk_tp(y[va], P[REF][va], kf)[0] / y[va].sum())
    ref_f = np.array(ref_f)
    for slug, nm, kind, sc in scorers:
        tp, tie = topk_tp(y, sc, k)
        rf = []
        for _, va in folds:
            kf = int(math.ceil(q * len(va) - 1e-9)); rf.append(topk_tp(y[va], sc[va], kf)[0] / y[va].sum())
        rf = np.array(rf)
        r = dict(top_share=q, model=nm, slug=slug, kind=kind, n_reviewed=k, defaulters_captured=tp, recall_capture=tp / N_POS,
                 precision=tp / k, lift_over_base=(tp / k) / BASE, boundary_tie_rows=tie,
                 **{f"recall_fold{j}": v for j, v in enumerate(rf, 1)}, recall_fold_mean=rf.mean(), recall_fold_sd_ddof1=rf.std(ddof=1))
        if slug != REF:
            c = nb(ref_f - rf)
            r.update(xgb_minus_recall_mean=c["mean"], xgb_recall_wins_of_5=c["n_pos"], xgb_minus_recall_nb_lo=c["lo"],
                     xgb_minus_recall_nb_hi=c["hi"])
        trows.append(r)
    trows.append(dict(top_share=q, model="random list", slug="random", kind="reference", n_reviewed=k, defaulters_captured=k * BASE,
                      recall_capture=k / N, precision=BASE, lift_over_base=1.0))
    trows.append(dict(top_share=q, model="perfect list", slug="perfect", kind="reference", n_reviewed=k,
                      defaulters_captured=min(k, N_POS), recall_capture=min(k, N_POS) / N_POS, precision=min(k, N_POS) / k,
                      lift_over_base=(min(k, N_POS) / k) / BASE))
tk = pd.DataFrame(trows)
tk.to_csv(os.path.join(RESULTS, "topk_capture_oof.csv"), index=False)
summary["d_topk"] = dict(
    note="pooled OOF scores; ties at the list boundary taken at their expected value under a random order; per-fold columns "
         "take the top share of each validation fold",
    rows={f"{int(100 * q)}%": {r["slug"]: dict(recall=r["recall_capture"], precision=r["precision"])
                                for r in trows if r["top_share"] == q} for q in SHARES})

# ============================================================================= (e) OOF ROC and PR curves
CURVES = [("xgboost", "-"), ("random_forest", (0, (6, 2))), ("knn", (0, (4, 1.6, 1.2, 1.6))), ("logistic", (0, (1.6, 1.6)))]
PR_MIN_TP = 20
fig, (axr, axp) = plt.subplots(1, 2, figsize=(11.6, 5.0))
for s, ls in CURVES:
    fpr_, tpr_, _ = roc_curve(y, P[s]); pr_, rc_, _ = precision_recall_curve(y, P[s])
    a_, ap_ = roc_auc_score(y, P[s]), average_precision_score(y, P[s])
    keep = rc_ * N_POS >= PR_MIN_TP                      # drop the noisy head of the PR curve (fewer than 20 defaulters caught)
    axr.plot(fpr_, tpr_, color=col(s), ls=ls, lw=2, label=f"{NAME[s]}  AUC {a_:.3f}")
    axp.plot(rc_[keep], pr_[keep], color=col(s), ls=ls, lw=2, label=f"{NAME[s]}  AP {ap_:.3f}")
    t5 = best_threshold(y, P[s], 5); st = op_stats(y, P[s] >= t5)
    for ax, xy in ((axr, (st["fpr"], st["tpr_recall"])), (axp, (st["tpr_recall"], st["precision"]))):
        ax.scatter([xy[0]], [xy[1]], s=46, color=col(s), edgecolor=SURF, linewidth=1.6, zorder=5)
axr.plot([0, 1], [0, 1], color=INK3, lw=1, ls="--", label="no skill")
axp.axhline(BASE, color=INK3, lw=1, ls="--", label=f"no skill (base rate {BASE:.3f})")
RULE_TAG = {"PAY_1 >= 1": "A", "PAY_1 >= 2": "B", "MAX_DELAY >= 2": "C", "N_MONTHS_DELAYED >= 1": "D"}
for r in brows:
    if r["kind"] != "rule":
        continue
    tag = RULE_TAG[r["rule_or_model"]]
    for ax, xy in ((axr, (r["fpr"], r["tpr_recall"])), (axp, (r["tpr_recall"], r["precision"]))):
        ax.scatter([xy[0]], [xy[1]], marker="s", s=40, facecolor=SURF, edgecolor=INK, linewidth=1.4, zorder=6)
        ax.annotate(tag, xy=xy, xytext=(6, -11), textcoords="offset points", fontsize=8, color=INK, fontweight="bold")
axr.set_xlabel("false-positive rate"); axr.set_ylabel("true-positive rate (recall)")
axr.set_title("ROC, training OOF (20,975 rows)", loc="left")
axp.set_xlabel("recall"); axp.set_ylabel("precision"); axp.set_ylim(0, 1.0)
axp.set_title("Precision-recall, training OOF", loc="left")
axr.legend(loc="lower right"); axp.legend(loc="upper right")
fig.text(0.01, 0.01, "Dots: each model's 5:1 cost-minimising cut-off chosen on the same OOF scores.   Squares: no-parameter "
         "arrears rules  A  PAY_1 >= 1   B  PAY_1 >= 2   C  MAX_DELAY >= 2   D  N_MONTHS_DELAYED >= 1.\n"
         f"PR curves are drawn from {PR_MIN_TP} defaulters caught onwards (the first few points are noise); AP uses the whole curve.",
         fontsize=7.8, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(FIGURES, "roc_pr_oof.png")); plt.close(fig)

summary["files"] = ["results/fold_stability.csv", "results/simple_rule_benchmark.csv", "results/preferred_model_by_ratio_oof.csv",
                    "results/preferred_model_by_ratio_oof_long.csv", "results/topk_capture_oof.csv",
                    "results/fold_stability_summary.json", "figures/fold_stability.png", "figures/roc_pr_oof.png"]
summary["run_seconds"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(RESULTS, "fold_stability_summary.json"), "w"), indent=1, default=float)

# ----------------------------------------------------------------------------- console report
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print(f"training rows {N:,}, defaulters {N_POS:,} (base rate {BASE:.4f}); folds {[int(len(va)) for _, va in folds]}")
print("\n(a) per-fold AUC, ranks and XGBoost-minus-model (NB 95%)")
print(fs[["slug", "auc_mean", "auc_sd_ddof1", "auc_mean_minus_cv_auc_json"] + [f"rank_auc_fold{k}" for k in range(1, 6)] +
         ["xgb_minus_auc_mean", "xgb_auc_wins_of_5", "xgb_minus_auc_nb_lo", "xgb_minus_auc_nb_hi", "xgb_minus_auc_holm_p",
          "ap_mean", "xgb_ap_wins_of_5", "cal_slope_oof"]].round(4).to_string(index=False))
print("\n(b) simple rules vs models at 5:1")
print(rb[["rule_or_model", "flag_rate", "tpr_recall", "fpr", "precision", "f1", "cost_5to1", "cost_5to1_per_1000",
          "cost_5to1_crossfitted_per_1000", "score_auc", "xgb_recall_same_flag_rate", "logit_recall_same_flag_rate",
          "xgb_cheaper_folds_of_5", "logit_cheaper_folds_of_5"]].round(4).to_string(index=False))
print("\n(c) preferred model by ratio")
print(ps[["ratio", "preferred_oof", "preferred_cutoff", "preferred_cost_oof", "runner_up", "margin_cost", "margin_pct", "third",
          "top3_spread_pct", "preferred_crossfitted", "preferred_cheaper_folds_of_5", "runner_up_minus_preferred_nb_lo",
          "runner_up_minus_preferred_nb_hi", "logistic_rank_oof", "tree_pruned_rank_oof", "preferred_saving_vs_blanket_pct"]]
      .round(3).to_string(index=False))
print("\n(d) top-k capture")
print(tk[["top_share", "slug", "recall_capture", "precision", "boundary_tie_rows", "xgb_minus_recall_mean", "xgb_recall_wins_of_5",
          "xgb_minus_recall_nb_lo", "xgb_minus_recall_nb_hi"]].round(4).to_string(index=False))
print(f"\ndone in {time.time() - T0:.1f}s")
