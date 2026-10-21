"""
make_slide_shap.py — projector-size TreeSHAP beeswarm for deck v2, slide S11 (22 Oct 2026).   Owner: shared.

Same model, rows and subsample as model_shap.py's figures/shap_beeswarm.png; only the size and text change
(15 features, labels at about 24 px on a 900 x 700 px slide figure).

TRAINING ROWS ONLY. Reads data/processed/train.csv directly (the held-out file is never opened) and
results/xgboost.json (the frozen tuned settings). Refits the tuned XGBoost on the 20,975 training rows,
checks that the refit reproduces the stored training AUC to 1e-9 (so it is the frozen configuration),
runs TreeSHAP on all training rows and draws the same 2,000-row subsample (seed 2607526).
Nothing is tuned, selected or scored; no test file is read.

Writes figures/slides/s_shap_beeswarm.png.
Run from Project/code:  /opt/miniconda3/bin/python3 make_slide_shap.py      (a few seconds)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import RESULTS, ROOT, SEED, TARGET, feature_names_out, make_preprocessor

OUT = os.path.join(ROOT, "figures", "slides")
N_BEESWARM = 2000                                              # as in model_shap.py

tr = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
assert len(tr) == 20975, len(tr)
X_train, y_train = tr.drop(columns=[TARGET]), tr[TARGET].astype(int)

tuned = json.load(open(os.path.join(RESULTS, "xgboost.json")))
params = dict(tuned["best_params"])
params.setdefault("objective", "binary:logistic")
params.setdefault("eval_metric", "auc")
params.setdefault("tree_method", "hist")
params.setdefault("n_jobs", 2)
pipe = Pipeline([("pre", make_preprocessor(scale=False)), ("clf", XGBClassifier(**params))])
pipe.fit(X_train, y_train)
diff = abs(float(roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])) - float(tuned["train_auc"]))
if diff > 1e-9:
    raise SystemExit(f"refit does not reproduce the tuned model (training AUC differs by {diff:.2e})")
print(f"refit reproduces the frozen XGBoost (training AUC |diff| {diff:.1e})")

pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
tnames = feature_names_out(pre)
Xt = pre.transform(X_train)
Xt = np.asarray(Xt.todense()) if hasattr(Xt, "todense") else np.asarray(Xt)
sv = shap.TreeExplainer(clf).shap_values(Xt)
rng = np.random.default_rng(SEED)
idx = rng.choice(sv.shape[0], size=min(N_BEESWARM, sv.shape[0]), replace=False)
pretty = [n.split("__", 1)[1] for n in tnames]

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.figure(figsize=(9.0, 7.0))                                 # 1350 x 1050 px at dpi 150; shown at 900 x 700
shap.summary_plot(sv[idx], Xt[idx], feature_names=pretty, show=False, max_display=15, plot_size=None)
fig = plt.gcf()
for ax in fig.axes:
    ax.tick_params(axis="both", labelsize=17)
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
main = fig.axes[0]
main.set_xlabel("SHAP value (push on the log-odds of default)", fontsize=17)
fig.suptitle(f"TreeSHAP, tuned XGBoost: {len(idx):,} training clients", x=0.012, y=0.99, ha="left", va="top", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT, "s_shap_beeswarm.png"), dpi=150, facecolor="white")
plt.close()
print("wrote s_shap_beeswarm.png")
