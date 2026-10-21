#!/bin/zsh
# Reproduce every result of the project from the raw UCI file (about 40–60 minutes on a laptop; seed 2607526 everywhere).
# Requires: python3 with numpy, pandas, scikit-learn >= 1.5, statsmodels, xgboost, imbalanced-learn, matplotlib, scipy, xlrd.
set -e
cd "$(dirname "$0")"
python3 fetch_data.py             # UCI .xls (downloaded if absent, SHA-256 checked) -> data/raw/default_of_credit_card_clients.csv
python3 common.py                 # cleaning, feature engineering, stratified 70/30 split -> data/processed/
python3 eda.py                    # exploratory figures + results/eda_summary.json
python3 make_proposal_figures.py  # composite EDA figures used in the proposal
python3 model_knn.py              # RUAN Hangjing
python3 model_logistic.py         # RUAN Hangjing
python3 model_discriminant.py     # RUAN Hangjing
python3 model_unsupervised.py     # RUAN Hangjing
python3 model_tree.py             # LU Xinyu
python3 model_rf.py               # LU Xinyu
python3 model_boosting.py         # LU Xinyu
# RUAN Hangjing — second pass of 30 Sep (nested CV on training only, freeze-before-test, calibration, cost-ratio sensitivity).
# Her 30 Sep result files (results/*_v2.*, results/frozen_choices_ruan_v2.json) are adopted into results/frozen_manifest.json
# and are NOT re-generated here: each script ends with a test-set evaluation, and a re-run on another machine would re-score
# the frozen configurations (evaluate() would refuse any prediction drift > 1e-6). The training-only parts of her analysis
# are reproduced by nested_cv_check.py and cluster_stability.py below. Set MH6805_RUN_V2=1 to run her scripts anyway.
if [ "${MH6805_RUN_V2:-0}" = "1" ]; then
  python3 model_knn_v2.py         # RUAN Hangjing
  python3 model_logistic_v2.py    # RUAN Hangjing — reads xgboost_p_oof.npy / xgboost_p_test.npy (read-only)
  python3 model_unsupervised_v2.py  # RUAN Hangjing
fi
python3 nested_cv_check.py        # RUAN's nested-CV second pass, training rows only (~15 min); confirms the frozen KNN / logistic choices
python3 cluster_stability.py      # RUAN's PCA + K-means stability and incremental-value analysis, training rows only (~10 min)
python3 model_pdp.py              # LU Xinyu — partial dependence / ICE of the tuned XGBoost (needs results/xgboost.json)
python3 model_svm.py              # LI Jingwen
python3 model_mlp.py              # LI Jingwen
python3 model_imbalance.py        # LI Jingwen
python3 rf_drop_column.py      # LU Xinyu — refit-without-a-column, the counterpart to permutation importance
python3 model_monotone.py     # LU Xinyu — beyond syllabus: monotone-constrained XGBoost
python3 model_shap.py         # LU Xinyu — beyond syllabus: TreeSHAP attribution
python3 recalibrate.py            # LI Jingwen — Platt recalibration of the 9 weighted / resampled models (seconds)
python3 threshold_vs_model_saving.py  # LI Jingwen — threshold saving vs model saving over cost ratios, paired bootstrap (reporting only)
python3 cost_sensitivity.py       # test cost of every headline model at its own training-OOF cut-off, 1:1–20:1 (reporting only)
python3 aggregate.py              # comparison tables, overlaid ROC/PR, AUC bars, threshold/cost/calibration figures, bootstrap CIs (~20 s)
python3 freeze.py --check         # every configuration scored on the test set still matches results/frozen_manifest.json
python3 make_report_appendix.py   # report/sec_appendix.tex from results/
python3 make_slide_figures.py     # figures/slides/*.png (slide-sized copies of the report figures)
echo "done: see ../results and ../figures"
