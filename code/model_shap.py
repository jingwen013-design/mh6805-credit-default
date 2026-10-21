"""
model_shap.py — TreeSHAP attribution for the tuned XGBoost.   Owner: LU Xinyu.

A method beyond the B0–B12 syllabus, promised in the submitted proposal.  Importance measures
(MDI, permutation) rank variables; Shapley values additionally say in which DIRECTION a variable
pushes each individual prediction, and they do it with an exact polynomial-time algorithm for
tree ensembles (Lundberg et al., TreeSHAP) rather than by sampling.

The model explained here is the tuned XGBoost of results/xgboost.json: the same hyper-parameters
are refit inside the same Pipeline(make_preprocessor(scale=False), clf) and the script asserts that
the refit reproduces the stored training AUC (results/xgboost.json "train_auc", to 1e-9) before
explaining anything.

Writes
  1. results/shap_summary.json     headline numbers, the local-accuracy check, one fully worked
                                   client, and the one-sentence definition of a Shapley value.
  2. results/shap_importance.csv   mean |SHAP| per raw feature, joined to the MDI / training-set
                                   permutation / out-of-bag permutation columns of the random
                                   forest (results/rf_importance.csv) so the report can compare
                                   four rankings of the same 29 features.
  3. figures/shap_beeswarm.png     the beeswarm over the training rows.
  4. figures/shap_dependence_pay1.png, figures/shap_dependence_util_mean.png
  5. figures/shap_waterfall.png    the single worked prediction, decomposed.

The test set is not read at all.  The model comes from the frozen tuned configuration and is
explained on the 20,975 training rows it was fitted on: a Shapley value describes the fitted
model, so the training rows are the natural population, and every number the report quotes from
this script (mean |SHAP|, the value-vs-contribution correlations, the worked client) is then a
training-set statistic.  Until 2026-10-19 the script explained the 8,990 test rows instead; it
selected nothing there, but the professor's rule is that the test set only reports the frozen
models' performance, so the diagnostics were moved to the training rows.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (FIGURES, RESULTS, SEED, feature_names_out, load_split,
                    make_preprocessor, save_extra_figure)

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

MUTED = "#6b7280"
N_BEESWARM = 2000          # rows drawn for the beeswarm; the numbers below use ALL training rows


def aggregate_to_raw(values, transformed_names, raw_names):
    """Sum a per-transformed-column vector back to the 29 raw features (one-hot dummies summed)."""
    out = dict.fromkeys(raw_names, 0.0)
    for v, n in zip(values, transformed_names):
        base = n.split("__", 1)[1]
        if n.startswith("cat__"):
            base = base.rsplit("_", 1)[0]
        out[base] += float(v)
    return np.array([out[f] for f in raw_names])


# ---------------------------------------------------------------- 1. refit the tuned model
X_train, _, y_train, _, feature_names = load_split()          # test rows deliberately unused
log(f"train {X_train.shape}; the test rows are not read")

tuned = json.load(open(os.path.join(RESULTS, "xgboost.json")))
params = dict(tuned["best_params"])
params.setdefault("objective", "binary:logistic")
params.setdefault("eval_metric", "auc")
params.setdefault("tree_method", "hist")
params.setdefault("n_jobs", 2)
pipe = Pipeline([("pre", make_preprocessor(scale=False)),
                 ("clf", XGBClassifier(**params))])
t0 = time.time()
pipe.fit(X_train, y_train)
refit_seconds = time.time() - t0
p_train = pipe.predict_proba(X_train)[:, 1]

# reproduction check on the training rows: the stored train_auc was computed by common.evaluate()
# from the same refit, so matching it to 1e-9 shows this is the tuned model and not a near neighbour
train_auc_refit = float(roc_auc_score(y_train, p_train))
auc_abs_diff = abs(train_auc_refit - float(tuned["train_auc"]))
log(f"refit in {refit_seconds:.1f}s; training AUC {train_auc_refit:.10f} vs stored {tuned['train_auc']:.10f} "
    f"(|diff| {auc_abs_diff:.1e})")
if auc_abs_diff > 1e-9:
    raise SystemExit(f"refit does not reproduce the tuned model (training AUC differs by {auc_abs_diff:.2e})")

pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
tnames = feature_names_out(pre)
Xt_train = pre.transform(X_train)
Xt_train = np.asarray(Xt_train.todense()) if hasattr(Xt_train, "todense") else np.asarray(Xt_train)

# ---------------------------------------------------------------- 2. TreeSHAP
log("running TreeExplainer on all training rows ...")
t0 = time.time()
explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(Xt_train)                   # (n_train, n_transformed), margin scale
base_value = float(np.ravel(explainer.expected_value)[0])
shap_seconds = time.time() - t0
log(f"SHAP values for {sv.shape[0]:,} x {sv.shape[1]} in {shap_seconds:.1f}s")

# LOCAL ACCURACY: base value + contributions must equal the model's raw margin, exactly.
margin = clf.get_booster().inplace_predict(Xt_train, predict_type="margin")
recon = base_value + sv.sum(axis=1)
local_err = float(np.max(np.abs(recon - margin)))
log(f"local accuracy: max |base + sum(shap) - margin| = {local_err:.3e}")
if local_err > 1e-4:
    raise SystemExit(f"local accuracy violated ({local_err:.2e}) — the explainer is misconfigured")

# ---------------------------------------------------------------- 3. importance, four rankings
mean_abs_t = np.abs(sv).mean(axis=0)
mean_abs = aggregate_to_raw(mean_abs_t, tnames, feature_names)
mean_signed = aggregate_to_raw(sv.mean(axis=0), tnames, feature_names)

# direction: sign of the correlation between a feature's value and its own SHAP value
direction = {}
for j, f in enumerate(feature_names):
    cols = [i for i, n in enumerate(tnames) if n.split("__", 1)[1].rsplit("_", 1)[0] == f
            or n.split("__", 1)[1] == f]
    if not cols:
        direction[f] = np.nan
        continue
    s = sv[:, cols].sum(axis=1)
    v = X_train[f].values.astype(float)
    direction[f] = float(np.corrcoef(v, s)[0, 1]) if np.std(v) > 0 and np.std(s) > 0 else np.nan

imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "corr_value_vs_shap": [direction[f] for f in feature_names]})
imp["shap_rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)

rf_path = os.path.join(RESULTS, "rf_importance.csv")
rho = {}
if os.path.exists(rf_path):
    rf = pd.read_csv(rf_path)
    keep = [c for c in ("feature", "mdi", "mdi_rank", "perm_auc_drop", "perm_rank",
                        "oob_perm_auc_drop", "oob_perm_rank") if c in rf.columns]
    imp = imp.merge(rf[keep], on="feature", how="left")
    for col, lbl in (("mdi", "mdi"), ("perm_auc_drop", "perm_train"), ("oob_perm_auc_drop", "perm_oob")):
        if col in imp.columns:
            rho[lbl] = float(spearmanr(imp["mean_abs_shap"], imp[col])[0])
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(os.path.join(RESULTS, "shap_importance.csv"), index=False)
log("Spearman(mean|SHAP|, other measures): " + ", ".join(f"{k} {v:.3f}" for k, v in rho.items()))
print(imp.head(10).round(4).to_string(index=False))

# ---------------------------------------------------------------- 4. figures
rng = np.random.default_rng(SEED)
idx = rng.choice(sv.shape[0], size=min(N_BEESWARM, sv.shape[0]), replace=False)
pretty = [n.split("__", 1)[1] for n in tnames]

plt.figure(figsize=(7.2, 5.6))
shap.summary_plot(sv[idx], Xt_train[idx], feature_names=pretty, show=False, max_display=15,
                  plot_size=None)
plt.title(f"TreeSHAP, tuned XGBoost — {len(idx):,} training clients", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "shap_beeswarm.png"), dpi=150)
plt.close()

for raw, fname in (("PAY_1", "shap_dependence_pay1.png"),
                   ("UTIL_MEAN", "shap_dependence_util_mean.png")):
    col = next((i for i, n in enumerate(tnames) if n.split("__", 1)[1] == raw), None)
    if col is None:
        log(f"WARNING: {raw} not found among transformed columns; dependence plot skipped")
        continue
    plt.figure(figsize=(6.0, 4.2))
    shap.dependence_plot(col, sv, Xt_train, feature_names=pretty, show=False,
                         interaction_index="auto")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, fname), dpi=150)
    plt.close()

# ---------------------------------------------------------------- 5. one worked client
hi = int(np.argmax(p_train))
row_sv = aggregate_to_raw(sv[hi], tnames, feature_names)
order = np.argsort(-np.abs(row_sv))[:8]
worked = [{"feature": feature_names[k],
           "value": float(X_train.iloc[hi][feature_names[k]]),
           "shap_margin": float(row_sv[k])} for k in order]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
lab = [w["feature"] for w in worked][::-1]
val = [w["shap_margin"] for w in worked][::-1]
ax.barh(lab, val, color=["#b45309" if v > 0 else "#1d4ed8" for v in val])
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("SHAP contribution (log-odds)")
ax.set_title(f"Highest-risk training client: base {base_value:.3f} + contributions\n"
             f"= {margin[hi]:.3f} log-odds  (p = {p_train[hi]:.3f})", fontsize=9)
save_extra_figure(fig, "shap_waterfall.png")

# ---------------------------------------------------------------- 6. json
top = imp.head(10)[["feature", "mean_abs_shap", "corr_value_vs_shap", "shap_rank"]].to_dict("records")
summary = dict(
    name="TreeSHAP attribution for the tuned XGBoost", analysis="shap", member="LU Xinyu", seed=SEED,
    model=dict(source="results/xgboost.json", hyper_parameters=params,
               refit_seconds=round(refit_seconds, 2),
               reproduction_check=dict(train_auc_refit=train_auc_refit, train_auc_stored=float(tuned["train_auc"]),
                                       abs_diff=auc_abs_diff, passed=bool(auc_abs_diff <= 1e-9))),
    explained=dict(rows=f"all {sv.shape[0]:,} training clients (the test set is not read)",
                   n_transformed_columns=int(sv.shape[1]),
                   scale="margin (log-odds); the probability scale is not additive",
                   beeswarm_subsample=int(len(idx)), seconds=round(shap_seconds, 1)),
    local_accuracy=dict(check="base_value + sum(shap values) == raw margin for every row",
                        max_abs_error=local_err, passed=bool(local_err <= 1e-4),
                        base_value=base_value,
                        base_value_as_probability=float(1 / (1 + np.exp(-base_value)))),
    top_features_by_mean_abs_shap=top,
    direction=dict(
        definition="Pearson correlation between a feature's value and its own SHAP value over the "
                   "training rows; positive means larger values push the prediction towards default.",
        PAY_1=direction.get("PAY_1"), UTIL_MEAN=direction.get("UTIL_MEAN"),
        LIMIT_BAL=direction.get("LIMIT_BAL"),
        N_MONTHS_DELAYED=direction.get("N_MONTHS_DELAYED")),
    spearman_vs_other_measures=rho,
    worked_example=dict(
        client_row_in_train=hi, predicted_probability=float(p_train[hi]),
        margin=float(margin[hi]), base_value=base_value,
        top_contributions=worked,
        reading="base value + every feature's contribution reproduces this client's log-odds exactly; "
                "that additivity is what an importance ranking cannot give."),
    one_sentence_definition=(
        "A feature's Shapley value is its average marginal contribution to the prediction over all "
        "orders in which the features could be added to the model, so the contributions of all "
        "features sum exactly to the difference between this prediction and the average prediction."),
    limitations=[
        "TreeSHAP is exact for this model, but the attribution is to the FITTED model, not to the "
        "world: a variable can receive a large contribution because it proxies for another.",
        "Contributions live on the log-odds scale, where they are additive; they are not additive "
        "after the logistic transform, so they must not be read as probability shares.",
        "The interventional/tree-path convention TreeExplainer uses splits credit between correlated "
        "features in a way that depends on the trees actually grown, which matters here because the "
        "six bill amounts correlate at r = 0.80-0.95.",
    ],
    files=dict(results=["shap_summary.json", "shap_importance.csv"],
               figures=["shap_beeswarm.png", "shap_dependence_pay1.png",
                        "shap_dependence_util_mean.png", "shap_waterfall.png"]),
    seconds=round(time.time() - T0, 1),
)
json.dump(summary, open(os.path.join(RESULTS, "shap_summary.json"), "w"), indent=1)

log("done -> results/shap_summary.json, results/shap_importance.csv, "
    "figures/shap_beeswarm.png, figures/shap_dependence_*.png, figures/shap_waterfall.png")
print(f"\nbase value {base_value:.4f} log-odds (p = {1/(1+np.exp(-base_value)):.4f}); "
      f"local accuracy max error {local_err:.2e}")
print(f"top 5 by mean |SHAP|: " + ", ".join(f"{r.feature} {r.mean_abs_shap:.4f}"
                                            for r in imp.head(5).itertuples()))
