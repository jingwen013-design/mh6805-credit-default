"""
threshold_vs_model_saving.py — the professor's "key practical point", priced with uncertainty over a range of cost ratios.
Owner: LI Jingwen (method: RUAN Hangjing's proposal-v2 Table 2, re-run on the frozen 16 Sep models).

For each cost ratio r = C_FN : C_FP in {1, 2, 3, 4, 5, 7, 10, 15, 20}:
  * the cut-off of the logistic baseline and of XGBoost is chosen on TRAINING out-of-fold probabilities (never on test);
  * threshold saving  = test cost of the logistic model at 0.5  minus  its test cost at the tuned cut-off;
  * model saving      = test cost of the tuned logistic         minus  the test cost of the tuned XGBoost;
  * both savings get a paired bootstrap 95% interval over the 8,990 test clients (B = 1,000, seed 2607526, same rows for
    both models in every replicate — mine_v2_utils.paired_bootstrap, as in the proposal).
Reporting only: the frozen models (results/logistic*.npy, results/xgboost*.npy) are read, nothing is fitted or selected.
The proposal (30 Sep) computed the same table from RUAN's separately refitted logistic model (results/logistic_plain_v2*);
that table is kept as results/threshold_vs_model_saving_v2.csv and reproduced here in the column block "proposal_*".

Writes results/threshold_vs_model_saving.csv, results/threshold_vs_model_saving.json, figures/threshold_vs_model_saving.png
"""
import json
import os
import time

import numpy as np
import pandas as pd

from common import RESULTS, SEED, load_split, plt, save_extra_figure
from mine_v2_utils import COST_RATIOS, best_threshold, cost_at, paired_bootstrap

T0 = time.time()
_, _, y_train, y_test, _ = load_split()
ytr, yte = y_train.values.astype(int), y_test.values.astype(int)
B = 1000


def load(slug):
    return (np.load(os.path.join(RESULTS, f"{slug}_p_oof.npy")).astype(float),
            np.load(os.path.join(RESULTS, f"{slug}_p_test.npy")).astype(float))


lr_oof, lr_te = load("logistic")
xgb_oof, xgb_te = load("xgboost")
rows = []
for r in COST_RATIOS:
    t_lr, t_x = best_threshold(ytr, lr_oof, r), best_threshold(ytr, xgb_oof, r)        # training OOF only
    c05, c_lr, c_x = cost_at(yte, lr_te, 0.5, r), cost_at(yte, lr_te, t_lr, r), cost_at(yte, xgb_te, t_x, r)
    thr = paired_bootstrap(yte, lambda i: cost_at(yte[i], lr_te[i], 0.5, r) - cost_at(yte[i], lr_te[i], t_lr, r), B=B, seed=SEED)
    mod = paired_bootstrap(yte, lambda i: cost_at(yte[i], lr_te[i], t_lr, r) - cost_at(yte[i], xgb_te[i], t_x, r), B=B, seed=SEED)
    flagged_lr = float((lr_te >= t_lr).mean()); flagged_x = float((xgb_te >= t_x).mean())
    rows.append(dict(ratio=r, t_closed_form=round(1 / (1 + r), 3), t_logistic=t_lr, t_xgb=t_x,
                     cost_logistic_at_0_5=c05, cost_logistic_tuned=c_lr, cost_xgb_tuned=c_x,
                     cost_flag_nobody=int(r * yte.sum()), cost_flag_everybody=int((yte == 0).sum()),
                     share_flagged_logistic=flagged_lr, share_flagged_xgb=flagged_x,
                     threshold_saving=thr["estimate"], threshold_saving_lo=thr["lo"], threshold_saving_hi=thr["hi"],
                     model_saving=mod["estimate"], model_saving_lo=mod["lo"], model_saving_hi=mod["hi"],
                     model_saving_ci_excludes_zero=bool(mod["lo"] > 0 or mod["hi"] < 0)))
    print(f"r={r:>2}: cut-offs logistic {t_lr:.2f} / XGBoost {t_x:.2f} (closed form {1/(1+r):.3f}) | threshold saving {thr['estimate']:.0f} "
          f"[{thr['lo']:.0f}, {thr['hi']:.0f}] | model saving {mod['estimate']:.0f} [{mod['lo']:.0f}, {mod['hi']:.0f}] | flagged "
          f"{flagged_lr:.0%} / {flagged_x:.0%}  [{time.time() - T0:.0f}s]")
tab = pd.DataFrame(rows)
money = [c for c in tab.columns if "saving" in c and "excludes" not in c]
tab[money] = tab[money].round(0).astype(int)
tab.to_csv(os.path.join(RESULTS, "threshold_vs_model_saving.csv"), index=False)

prop_path = os.path.join(RESULTS, "threshold_vs_model_saving_v2.csv")
prop = pd.read_csv(prop_path).set_index("ratio") if os.path.exists(prop_path) else None

fig, ax = plt.subplots(figsize=(7.6, 4.4))
for col, lab, c in [("threshold_saving", "move the logistic cut-off 0.5 → tuned on training OOF", "#2a78d6"),
                    ("model_saving", "switch logistic → XGBoost (both at tuned cut-offs)", "#e34948")]:
    ax.plot(tab["ratio"], tab[col], marker="o", color=c, label=lab)
    ax.fill_between(tab["ratio"], tab[col + "_lo"], tab[col + "_hi"], color=c, alpha=0.18)
ax.axhline(0, color="#0b0b0b", lw=0.8); ax.axvline(5, color="#8a8983", lw=0.8, ls="--")
ax.text(5.15, ax.get_ylim()[1] * 0.93, "5:1 (report's reference ratio)", fontsize=8, color="#52514e")
ax.set_xscale("log"); ax.set_xticks(COST_RATIOS); ax.set_xticklabels([f"{r}:1" for r in COST_RATIOS])
ax.set_xlabel("cost ratio  C_FN : C_FP"); ax.set_ylabel("test cost saved (C_FP = 1), 95% paired-bootstrap band")
ax.set_title("Threshold choice vs model choice across cost ratios (test set, n = 8,990; cut-offs chosen on training OOF)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")
save_extra_figure(fig, "threshold_vs_model_saving.png")

r5 = tab.set_index("ratio").loc[5]
summary = dict(generated_by="code/threshold_vs_model_saving.py", B=B, seed=SEED, ratios=COST_RATIOS,
               models=dict(baseline="logistic (16 Sep frozen)", challenger="xgboost (16 Sep frozen)"),
               at_5_to_1=dict(t_logistic=float(r5.t_logistic), t_xgb=float(r5.t_xgb),
                              threshold_saving=[int(r5.threshold_saving), int(r5.threshold_saving_lo), int(r5.threshold_saving_hi)],
                              model_saving=[int(r5.model_saving), int(r5.model_saving_lo), int(r5.model_saving_hi)]),
               ratios_where_model_saving_ci_includes_zero=[int(r) for r, e in zip(tab.ratio, tab.model_saving_ci_excludes_zero) if not e],
               degenerate_ratios=[int(r) for r, t in zip(tab.ratio, tab.t_logistic) if t <= 0.02],
               proposal_stage_table=(prop.loc[5, ["t_logistic", "threshold_saving", "threshold_saving_lo", "threshold_saving_hi",
                                                  "model_saving", "model_saving_lo", "model_saving_hi"]].to_dict() if prop is not None else None),
               reading=("Moving the cut-off saves far more than changing the model at every ratio above 1:1, and the model saving's "
                        "interval includes zero at some ratios; at 20:1 the optimal cut-off is ≈0.01, i.e. flag almost everyone, so the "
                        "threshold 'saving' there is the trivial rule, not a model result."))
json.dump(summary, open(os.path.join(RESULTS, "threshold_vs_model_saving.json"), "w"), indent=1)
print(f"\nwrote results/threshold_vs_model_saving.csv/.json, figures/threshold_vs_model_saving.png  [{time.time() - T0:.0f}s]")
