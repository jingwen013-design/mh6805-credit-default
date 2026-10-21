# R cross-check of the model comparison

## What it checks

The project's models are fitted in Python (scikit-learn, XGBoost). This folder refits the same model families with the R functions used in the MH6805 lab notes (`glm`, `cv.glmnet`, `MASS::lda` / `qda`, `e1071::naiveBayes`, `class::knn`, `tree` + cost-complexity pruning, `randomForest`, `gbm`, `xgboost`, `e1071::svm`, `nnet`). It then compares their five-fold cross-validated ROC-AUC (and average precision) with the Python CV AUC in `results/model_comparison.csv`.

- **Same rows, same folds.** `code/export_folds.py` rebuilds the Python `StratifiedKFold(5, shuffle=True, random_state=2607526)` assignment of the 20,975 training rows and writes `data/processed/cv_folds.csv`. The Rmd proves that these are the Python folds: the fold AUCs it recomputes from every Python model's saved out-of-fold probabilities (`results/<slug>_p_oof.npy`) reproduce that model's `cv_auc`.
- **Same features.** The 29 predictors of `train.csv`, coded with `model.matrix()` into the same 32 columns as the Python preprocessor (first level of SEX / EDUCATION / MARRIAGE as reference). The 26 numeric columns are standardised inside each fold for glm / glmnet / LDA / QDA / naive Bayes / KNN / SVM / nnet. Trees, forests and boosting use the raw columns.
- **Frozen settings.** KNN K = 101, random forest 500 trees with mtry 5 and nodesize 20, gbm with shrinkage 0.03 / depth 3 / bag fraction 0.8 / 200 and 300 trees, XGBoost with the frozen Python parameters, RBF SVM with cost 1, gamma 0.01 and balanced class weights fitted on the full folds. Ridge / lasso λ, the tree size and the nnet hidden-layer size are chosen by an inner CV inside each training fold.
- **Output:** `results/r_crosscheck.csv` (model, r_cv_auc, r_cv_auc_sd, r_ap, python_cv_auc, diff, r_seconds, notes, plus python_slug, python_ap, diff_fold_sd, folds_r_higher), `figures/r_crosscheck.png` (R vs Python CV AUC with the 45-degree line; paired differences per model) and `r_crosscheck.html` (the knitted notes, with the explanation of every difference above 0.005).

## The test set is never read

The Rmd reads only `data/processed/train.csv`, `data/processed/cv_folds.csv`, `results/model_comparison.csv` and the training out-of-fold vectors `results/<slug>_p_oof.npy`. It never builds a path to the test file and calls no Python code, so it cannot score anything on the test set or change the frozen manifest. `export_folds.py` imports `common.py` only for `CV`, `PROC`, `TARGET` and `SEED`. It reads the training file directly and does not call `load_split()`, which would also open the test file. Nothing here feeds a modelling choice in the report: the cross-check is descriptive.

## How to run

```bash
cd code && python3 export_folds.py && cd ..          # writes data/processed/cv_folds.csv (seconds)
Rscript code/r_crosscheck/run_r_crosscheck.R          # knits r_crosscheck.Rmd -> r_crosscheck.html, CSV, figure
```

- R 4.6.1 with knitr, rmarkdown, pROC, glmnet, MASS, e1071, class, tree, randomForest, gbm, xgboost and nnet. The runner installs any missing package from CRAN. Pandoc is needed for the HTML; without it the runner extracts the chunks with `knitr::purl()` and runs them with `source()` (same CSV and figure, no HTML).
- The repository path contains non-ASCII folder names on our machine. R must run in a UTF-8 locale, so the runner switches to `en_US.UTF-8` if the session is not UTF-8 (or start it with `LC_ALL=en_US.UTF-8`).
- Seed 2607526 is set before every random step (glmnet inner folds, knn tie-breaking, tree CV groups, randomForest, gbm, xgboost, nnet starts).

## Run time

About 12.5 minutes for the whole knit on our 8-core Apple-silicon laptop (single-threaded R, xgboost on 2 threads; measured 21 Oct 2026 while other jobs ran, load average about 9). Most of it is the nnet inner CV (about 4.7 min), the random forests (the `nodesize = 20` forest, its soft-vote recomputation and the `nodesize = 80` diagnostic forest: about 4.1 min) and the five full-fold RBF SVM fits (about 2.1 min, 25 s each). Everything else takes under 20 seconds per model. `results/r_crosscheck.csv` lists the seconds for each row. Two clean runs gave identical numbers.

## Result in one paragraph

On the same five folds, 13 of the 15 R fits that mirror a Python configuration land within 0.005 of the Python CV AUC (Spearman 0.99 between the two columns). `glm`, `lda`, the libsvm RBF SVM and KNN with K = 101 agree to four decimals, and XGBoost agrees to 0.0001. The two larger gaps are explained and confirmed by diagnostic refits. `MASS::qda` has no regularisation: it reproduces the Python `reg_param = 0` value, and a hand-written QDA with `reg_param = 0.5` reproduces the frozen Python value exactly. `randomForest` uses hard votes and lets leaves shrink below 20 rows: soft votes plus `nodesize = 80` bring it within 0.002 of scikit-learn. The lab-default tree (`mindev = 0.01`, `prune.misclass`) is a deliberate contrast. The ranking of the model families is the same in both languages, apart from the hard-vote random forest.
