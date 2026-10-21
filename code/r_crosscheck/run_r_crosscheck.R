# run_r_crosscheck.R - knit the R cross-check of the model comparison (training rows only).
#
# From the repository root:   Rscript code/r_crosscheck/run_r_crosscheck.R
# (or from code/r_crosscheck: Rscript run_r_crosscheck.R)
#
# Prerequisite: data/processed/cv_folds.csv, written by `cd code && python3 export_folds.py`.
# Writes code/r_crosscheck/r_crosscheck.html, results/r_crosscheck.csv and figures/r_crosscheck.png.
# Reads only data/processed/train.csv, data/processed/cv_folds.csv, results/model_comparison.csv and
# results/<slug>_p_oof.npy (training out-of-fold probabilities). The test file is never opened.
# If pandoc is not available, the R chunks are extracted with knitr::purl() and run with source() instead
# (same CSV and figure, no HTML).

# The repository path contains non-ASCII characters (Chinese folder names), so R must run in a UTF-8 locale.
if (!isTRUE(l10n_info()[["UTF-8"]])) {
  for (loc in c("en_US.UTF-8", "en_GB.UTF-8", "C.UTF-8")) if (nzchar(suppressWarnings(Sys.setlocale("LC_ALL", loc)))) break
}
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
rmd <- file.path(here, "r_crosscheck.Rmd")
stopifnot(file.exists(rmd))

pkgs <- c("knitr", "rmarkdown", "pROC", "glmnet", "MASS", "e1071", "class", "tree", "randomForest", "gbm", "xgboost", "nnet")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")

stopifnot(file.exists(file.path(here, "..", "..", "data", "processed", "cv_folds.csv")))

run.start <- Sys.time()
if (rmarkdown::pandoc_available()) {
  # knit in the global environment (as RStudio's Knit button does): tree::cv.tree() re-evaluates the tree() call there
  rmarkdown::render(rmd, output_file = "r_crosscheck.html", output_dir = here, knit_root_dir = here, quiet = TRUE)
  message("Knitted ", file.path(here, "r_crosscheck.html"))
} else {
  message("pandoc not found: running the R chunks with knitr::purl() + source() (no HTML)")
  r.file <- tempfile(fileext = ".R")
  knitr::purl(rmd, output = r.file, quiet = TRUE)
  owd <- setwd(here); on.exit(setwd(owd))
  source(r.file, echo = FALSE)
}
message(sprintf("Run time: %.1f minutes", as.numeric(difftime(Sys.time(), run.start, units = "mins"))))
