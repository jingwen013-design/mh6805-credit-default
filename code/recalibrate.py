"""
recalibrate.py — Platt recalibration of the nine class-weighted / resampled models.   Owner: LI Jingwen.

Why.  The professor's comment on the proposal: resampling stays inside the folds, AND probabilities must be recalibrated
after weighting or SMOTE.  Weighting and resampling shift the predicted default probabilities upwards (mean test p
0.34-0.44 against a prevalence of 0.221), so their Brier scores look bad and the closed-form cost threshold
t = C_FP / (C_FN + C_FP) cannot be applied to them, because it holds only for calibrated probabilities.

What.  Platt scaling — a one-variable logistic regression of the label on logit(p) — fitted on each model's TRAINING
out-of-fold probabilities (results/<slug>_p_oof.npy) and nothing else.  No model is refitted.
  * training side: the calibration is cross-fitted on the project's five folds (fit on four folds' OOF scores, apply to
    the fifth), and both decision thresholds (F1, 5:1 cost) are re-chosen on those cross-fitted calibrated scores with
    common.tune_threshold — so every choice is made on training data;
  * test side: the Platt map fitted on all training OOF scores is applied to results/<slug>_p_test.npy and the result
    is scored ONCE.  Each calibrated version is a new configuration (slug <slug>_platt) and is registered in
    results/frozen_manifest.json before any test label is read; a re-run must reproduce it.

Writes
  results/recalibration.csv      one row per model: raw vs Platt on training (cross-fitted) and on test, plus the
                                 unweighted twin for reference
  results/recalibration.json     the Platt parameters and the per-variant test metrics (read by freeze.py --check)
  results/recalibrated/<slug>_platt_p_test.npy, _p_oof_crossfit.npy
  figures/recalibration.png      reliability before / after on training out-of-fold scores
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from common import (COST_FN, COST_FP, CV, RESULTS, SEED, freeze_check, freeze_record, load_split, metrics_at,
                    save_extra_figure, tune_threshold)

T0 = time.time()
# (weighted / resampled model, its unweighted twin)
TARGETS = [("logistic_balanced", "logistic"),
           ("imb_logistic_balanced", "imb_logistic_none"), ("imb_logistic_smote", "imb_logistic_none"),
           ("imb_logistic_under", "imb_logistic_none"),
           ("imb_rf_balanced", "imb_rf_none"), ("imb_rf_smote", "imb_rf_none"), ("imb_rf_under", "imb_rf_none"),
           ("xgboost_weighted", "xgboost"), ("mlp_weighted", "mlp")]
T_CLOSED_FORM = COST_FP / (COST_FN + COST_FP)          # 1/6 at 5:1
OUT_DIR = os.path.join(RESULTS, "recalibrated")
ROUND_NOTE = ("19 October 2026 — Platt map fitted on training out-of-fold scores, thresholds re-chosen on cross-fitted "
              "training scores, then scored on the test set once (added after the professor's feedback, which asked for it)")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt(p_fit, y_fit):
    """Unpenalised logistic regression of y on logit(p): p_cal = sigmoid(a * logit(p) + b)."""
    m = LogisticRegression(C=1e12, max_iter=10000).fit(logit(p_fit), y_fit)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def apply(a, b, p):
    return 1.0 / (1.0 + np.exp(-(a * logit(p).ravel() + b)))


X_train, X_test, y_train, y_test, _ = load_split()
ytr, yte = y_train.values.astype(int), y_test.values.astype(int)
folds = list(CV.split(X_train, ytr))                    # the same five folds every model script used
os.makedirs(OUT_DIR, exist_ok=True)

rows, variants, curves = [], {}, {}
for slug, twin in TARGETS:
    R = json.load(open(os.path.join(RESULTS, f"{slug}.json")))
    RT = json.load(open(os.path.join(RESULTS, f"{twin}.json")))
    p_oof = np.load(os.path.join(RESULTS, f"{slug}_p_oof.npy")).astype(float)
    p_test = np.load(os.path.join(RESULTS, f"{slug}_p_test.npy")).astype(float)
    p_oof_twin = np.load(os.path.join(RESULTS, f"{twin}_p_oof.npy")).astype(float)

    # ---- training side only: cross-fitted calibration and thresholds
    p_cal_oof = np.empty_like(p_oof)
    for tr, va in folds:
        a_k, b_k = platt(p_oof[tr], ytr[tr])
        p_cal_oof[va] = apply(a_k, b_k, p_oof[va])
    t_f1_raw, _ = tune_threshold(ytr, p_oof, "f1"); t_cost_raw, _ = tune_threshold(ytr, p_oof, "cost")
    t_f1_cal, _ = tune_threshold(ytr, p_cal_oof, "f1"); t_cost_cal, _ = tune_threshold(ytr, p_cal_oof, "cost")
    a, b = platt(p_oof, ytr)                              # final map: all training OOF scores
    p_cal_test = apply(a, b, p_test)

    # ---- register the calibrated configuration BEFORE any test label is read
    cal_slug = f"{slug}_platt"
    cal_name = f"{R['name']} + Platt recalibration"
    params = dict(base_model=slug, method="Platt scaling fitted on training out-of-fold scores",
                  a=round(a, 8), b=round(b, 8))
    status = freeze_check(cal_slug, cal_name, params, p_cal_test)

    # ---- test side: scored once
    m_f1, m_cost = metrics_at(yte, p_cal_test, t_f1_cal), metrics_at(yte, p_cal_test, t_cost_cal)
    res = dict(name=cal_name, slug=cal_slug, member="LI Jingwen", best_params=params,
               at_tuned_f1=m_f1, at_tuned_cost=m_cost, cv_auc=None,
               oof_auc=float(roc_auc_score(ytr, p_cal_oof)),
               test_auc=float(roc_auc_score(yte, p_cal_test)), test_ap=float(average_precision_score(yte, p_cal_test)),
               test_brier=float(brier_score_loss(yte, p_cal_test)),
               test_logloss=float(log_loss(yte, np.clip(p_cal_test, 1e-6, 1 - 1e-6))))
    np.save(os.path.join(OUT_DIR, f"{cal_slug}_p_test.npy"), p_cal_test)
    np.save(os.path.join(OUT_DIR, f"{cal_slug}_p_oof_crossfit.npy"), p_cal_oof)
    freeze_record(res, p_cal_test, status, "recalibration.json", role="recalibrated_variant", round_note=ROUND_NOTE,
                  current_p_test=os.path.join("recalibrated", f"{cal_slug}_p_test.npy"))
    variants[cal_slug] = {k: res[k] for k in ("name", "member", "best_params", "oof_auc", "test_auc", "test_ap",
                                               "test_brier", "test_logloss")}
    variants[cal_slug].update(thresholds_chosen_on_training=dict(f1=t_f1_cal, cost=t_cost_cal), freeze_status=status)

    raw_cost_t = R["at_tuned_cost"]
    rows.append(dict(
        model=R["name"], slug=slug, twin=twin, platt_a=a, platt_b=b,
        oof_mean_p_raw=float(p_oof.mean()), oof_mean_p_platt=float(p_cal_oof.mean()), prevalence_train=float(ytr.mean()),
        oof_brier_raw=float(brier_score_loss(ytr, p_oof)), oof_brier_platt=float(brier_score_loss(ytr, p_cal_oof)),
        oof_brier_twin=float(brier_score_loss(ytr, p_oof_twin)),
        t_cost_raw=t_cost_raw, t_cost_platt=t_cost_cal, t_closed_form=round(T_CLOSED_FORM, 4),
        t_f1_raw=t_f1_raw, t_f1_platt=t_f1_cal,
        test_mean_p_raw=float(p_test.mean()), test_mean_p_platt=float(p_cal_test.mean()), prevalence_test=float(yte.mean()),
        test_brier_raw=float(R["test_brier"]), test_brier_platt=res["test_brier"], test_brier_twin=float(RT["test_brier"]),
        test_auc_raw=float(R["test_auc"]), test_auc_platt=res["test_auc"],
        test_cost_at_cost_threshold_raw=float(raw_cost_t["cost"]), test_cost_at_cost_threshold_platt=m_cost["cost"],
        test_recall_at_cost_threshold_platt=m_cost["recall_1"],
        test_cost_at_closed_form_raw=metrics_at(yte, p_test, T_CLOSED_FORM)["cost"],
        test_cost_at_closed_form_platt=metrics_at(yte, p_cal_test, T_CLOSED_FORM)["cost"],
        freeze_status=status))
    curves[slug] = (p_oof, p_cal_oof)
    print(f"{slug:22s} [{status:10s}] a={a:+.3f} b={b:+.3f} | OOF mean p {p_oof.mean():.3f} -> {p_cal_oof.mean():.3f}, "
          f"Brier {rows[-1]['oof_brier_raw']:.4f} -> {rows[-1]['oof_brier_platt']:.4f} (twin {rows[-1]['oof_brier_twin']:.4f}); "
          f"5:1 thr {t_cost_raw:.2f} -> {t_cost_cal:.2f} | test Brier {R['test_brier']:.4f} -> {res['test_brier']:.4f}, "
          f"cost@1/6 {rows[-1]['test_cost_at_closed_form_raw']:.0f} -> {rows[-1]['test_cost_at_closed_form_platt']:.0f}")

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(RESULTS, "recalibration.csv"), index=False)
json.dump(dict(generated_by="code/recalibrate.py", seed=SEED, method=__doc__.split("What.")[1].split("Writes")[0].strip(),
               variants=variants), open(os.path.join(RESULTS, "recalibration.json"), "w"), indent=1)

# reliability before / after, training out-of-fold scores only
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
groups = [("Logistic family", ["logistic_balanced", "imb_logistic_smote", "imb_logistic_under"]),
          ("Random forest", ["imb_rf_balanced", "imb_rf_smote", "imb_rf_under"]),
          ("XGBoost and MLP", ["xgboost_weighted", "mlp_weighted"])]
for ax, (title, slugs) in zip(axes, groups):
    ax.plot([0, 1], [0, 1], color="#8a8983", ls="--", lw=0.9)
    for i, s in enumerate(slugs):
        raw, cal = curves[s]
        fr, mr = calibration_curve(ytr, raw, n_bins=10, strategy="quantile")
        fc, mc = calibration_curve(ytr, cal, n_bins=10, strategy="quantile")
        ax.plot(mr, fr, ls=":", marker="o", ms=3.5, color=f"C{i}", label=f"{s} raw")
        ax.plot(mc, fc, ls="-", marker="o", ms=3.5, color=f"C{i}", label=f"{s} + Platt")
    ax.set_title(title); ax.set_xlabel("mean predicted P(default), decile"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("observed default rate (training)")
fig.suptitle("Platt recalibration of the weighted / resampled models — training out-of-fold scores (cross-fitted)", fontsize=11)
save_extra_figure(fig, "recalibration.png")

print(f"\nwrote results/recalibration.csv, results/recalibration.json, figures/recalibration.png [{time.time() - T0:.0f}s]")
