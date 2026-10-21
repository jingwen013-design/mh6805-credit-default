"""
cost_sensitivity.py — how much does the 5:1 false-negative : false-positive cost assumption matter?
(1) Data-based anchors for the ratio (exposure of a defaulter vs. margin on a good customer, training partition only).
(2) For cost ratios r in {1, 2, 3, 5, 8, 10, 20}: pick each headline model's cost-minimising threshold on the TRAINING
    out-of-fold probabilities, then report test cost, recall and FPR (reporting only: nothing is ranked or chosen on test cost).
Writes results/cost_sensitivity.csv, results/cost_anchors.json, figures/cost_sensitivity.png.

Threshold grid (changed 21 Oct 2026, record hygiene): the cut-off is searched on mine_v2_utils.THR_GRID (0.01-0.95, step
0.01), the grid threshold_vs_model_saving.py uses, so the two cost tables agree.  The previous grid started at 0.02.
The script was re-run on 21 Oct 2026 (late evening) after this change. It re-reads only the frozen training-OOF and
test probability vectors to recompute the reporting table; nothing is refitted or selected. Exactly one of the 56
cells changed: the logistic regression at 20:1, whose training-OOF cut-off moves from 0.02 to 0.01 (test cost 6,976 ->
6,993, as already recorded in results/threshold_vs_model_saving.csv). See results/TEST_SET_LOG.md.
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_split, RESULTS, FIGURES, TARGET
from mine_v2_utils import THR_GRID          # 0.01-0.95: the same grid as threshold_vs_model_saving.py
X_tr, X_te, y_tr, y_te, _ = load_split()
# ---------------- (1) anchors
d = X_tr.assign(y=y_tr.values)
bad, good = d[d.y == 1], d[d.y == 0]
anchors = dict(
    median_bill_defaulter=float(bad.BILL_AMT1.median()), mean_bill_defaulter=float(bad.BILL_AMT1.mean()),
    median_bill_good=float(good.BILL_AMT1.median()), mean_bill_good=float(good.BILL_AMT1.mean()),
    median_limit_defaulter=float(bad.LIMIT_BAL.median()), median_limit_good=float(good.LIMIT_BAL.median()),
    assumptions="loss given default 50–70% of the outstanding bill; a false alarm costs the monthly interest margin (about 1.5% of the bill per month, i.e. an annual rate near 18%) plus customer friction for 1–3 months",
)
lgd = np.array([0.5, 0.7]); months = np.array([1, 3])
fn_cost = lgd * anchors["median_bill_defaulter"]                    # NT$ lost on a missed defaulter
fp_cost = 0.015 * anchors["median_bill_good"] * months               # NT$ margin forgone on a wrongly flagged good client
ratios = np.array([[fn / fp for fp in fp_cost] for fn in fn_cost])
anchors["implied_ratio_range"] = [float(ratios.min()), float(ratios.max())]
anchors["german_credit_reference"] = "UCI Statlog German Credit cost matrix: misclassifying a bad customer as good costs 5, the reverse costs 1"
json.dump(anchors, open(os.path.join(RESULTS, "cost_anchors.json"), "w"), indent=1)
print("anchors:", {k: (round(v) if isinstance(v, float) else v) for k, v in anchors.items() if k != "assumptions"})
# ---------------- (2) sensitivity over the ratio
models = [("Logistic regression", "logistic"), ("KNN", "knn"), ("Pruned tree", "tree_pruned"), ("Random forest", "random_forest"),
          ("Gradient boosting", "gradient_boosting"), ("XGBoost", "xgboost"), ("MLP", "mlp"), ("SVM (RBF)", "svm_rbf")]
grid = THR_GRID                             # was np.arange(0.02, 0.96, 0.01) until 21 Oct 2026 (see the docstring)
def cost_curve(y, p, r):
    out = []
    for t in grid:
        yhat = (p >= t).astype(int); fn = int(((yhat == 0) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        out.append(r * fn + fp)
    return np.array(out)
rows = []
for r in [1, 2, 3, 5, 8, 10, 20]:
    for name, slug in models:
        p_oof = np.load(os.path.join(RESULTS, f"{slug}_p_oof.npy")); p_te = np.load(os.path.join(RESULTS, f"{slug}_p_test.npy"))
        t = grid[cost_curve(y_tr.values, p_oof, r).argmin()]
        yhat = (p_te >= t).astype(int); fn = int(((yhat == 0) & (y_te.values == 1)).sum()); fp = int(((yhat == 1) & (y_te.values == 0)).sum())
        tp = int(((yhat == 1) & (y_te.values == 1)).sum()); tn = int(((yhat == 0) & (y_te.values == 0)).sum())
        rows.append(dict(ratio=r, model=name, slug=slug, threshold=float(t), test_cost=r * fn + fp, recall=tp / (tp + fn), fpr=fp / (fp + tn),
                         flagged_share=(tp + fp) / len(y_te), cost_flag_nobody=r * int(y_te.sum()), cost_flag_everybody=int((y_te == 0).sum())))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "cost_sensitivity.csv"), index=False)
piv = df.pivot(index="ratio", columns="model", values="test_cost"); print((piv.div(piv.min(axis=1), axis=0) - 1).round(3).to_string())
xg = df[df.slug == "xgboost"][["ratio", "threshold", "recall", "fpr", "flagged_share"]]; print("XGBoost by ratio:\n", xg.round(3).to_string(index=False))
# ---------------- figure
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for name, slug in models:
    sub = df[df.slug == slug]
    ax[0].plot(sub.ratio, sub.test_cost / sub.cost_flag_nobody, marker="o", ms=3, label=name)
    ax[1].plot(sub.ratio, sub.recall, marker="o", ms=3, label=name); ax[2].plot(sub.ratio, sub.threshold, marker="o", ms=3, label=name)
for a in ax: a.set_xscale("log"); a.set_xticks([1, 2, 3, 5, 8, 10, 20]); a.set_xticklabels(["1", "2", "3", "5", "8", "10", "20"]); a.set_xlabel("cost ratio  FN : FP"); a.axvline(5, color="grey", ls="--", lw=1)
ax[0].set_ylabel("test cost / cost of flagging nobody"); ax[0].set_title("(a) Relative test cost at the OOF-tuned threshold")
ax[1].set_ylabel("recall of defaulters (test)"); ax[1].set_title("(b) Recall at the cost-optimal threshold")
ax[2].set_ylabel("cost-optimal threshold (OOF)"); ax[2].set_title("(c) Threshold chosen on training OOF probabilities"); ax[2].legend(fontsize=7.5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "cost_sensitivity.png"), dpi=150); print("saved figures/cost_sensitivity.png")
