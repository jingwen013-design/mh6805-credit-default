"""
make_report_appendix.py — generate report/sec_appendix.tex from results/.   Owner: shared.

The course guideline exempts appendices from the report's 7-page limit, so everything the body
cannot hold lives here, and it is GENERATED rather than typed: every table is read out of
results/ at build time, so the appendix cannot drift away from the artefacts it describes.

Writes report/sec_appendix.tex (LaTeX, \\input from report.tex after \\appendix).
"""
import json
import os

import pandas as pd

from common import RESULTS, ROOT

REPORT = os.path.join(ROOT, "report")
FIGDIR = os.path.join(ROOT, "figures")
OUT = os.path.join(REPORT, "sec_appendix.tex")


def esc(s):
    """Escape the LaTeX specials that appear in feature names and model names."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def tt(s):
    # \allowbreak after each underscore lets long file and column names break inside a
    # caption or a narrow cell instead of forcing an underfull line.
    return r"\texttt{" + esc(s).replace(r"\_", r"\_\allowbreak{}") + "}"


def num(x, nd=3):
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if pd.isna(v):
        return "---"
    s = f"{int(v):,}" if v.is_integer() and abs(v) >= 1000 else f"{v:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s      # a real minus sign, not a hyphen


def snum(x, nd=4):
    """Signed number in math mode: +0.0001 / -0.0003 with proper signs."""
    return f"${float(x):+.{nd}f}$"


_FLOATS = [0]            # floats emitted so far; every 6th is followed by \clearpage so the
                         # 18 tables + 41 figures never overflow LaTeX's float queue


def _flush():
    _FLOATS[0] += 1
    return "\n\\clearpage" if _FLOATS[0] % 6 == 0 else ""


def table(caption, label, colspec, header, rows, small=True, note=None, tabcolsep=None, footnotesize=False,
          placement="!htbp", note_ragged=False):
    # "!" lets consecutive tables share a page (the appendix is tables only, so the usual
    # float-fraction limits only produce half-empty pages and float-only pages).
    out = [f"\\begin{{table}}[{placement}]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           r"  \centering"]
    if footnotesize:
        out.append(r"  \footnotesize")
    elif small:
        out.append(r"  \small")
    if tabcolsep is not None:
        out.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    out.append(f"  \\begin{{tabular}}{{@{{}}{colspec}@{{}}}}")
    out.append(r"    \toprule")
    out.append("    " + " & ".join(header) + r" \\")
    out.append(r"    \midrule")
    for r_ in rows:
        out.append("    " + " & ".join(r_) + r" \\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    if note:
        # A justified minipage rather than centred lines: a multi-line note set with \centering
        # produces underfull (badness 10000) lines at every forced break.
        rag = "\\raggedright " if note_ragged else ""
        out.append(f"  \\\\[3pt]\\begin{{minipage}}{{\\linewidth}}\\footnotesize {rag}{note}\\end{{minipage}}")
    out.append(r"\end{table}")
    return "\n".join(out) + _flush()


FULLPAGE_FIGS = {"eda_correlation_heatmap.png", "pdp_main.png", "pdp_interaction.png", "pca_loadings.png"}
# Multi-panel figures whose panel titles and tick labels are illegible at 0.92\linewidth: set at
# full width on a float page of their own (the appendix is exempt from the page limit).


def figure(path, caption, label, width="0.92", placement="htbp"):
    if not os.path.exists(os.path.join(FIGDIR, path)):
        return f"% MISSING FIGURE: {path}"
    if path in FULLPAGE_FIGS:
        width, placement = "1.0", "p"
    return "\n".join([
        f"\\begin{{figure}}[{placement}]", r"  \centering",
        f"  \\includegraphics[width={width}\\linewidth]{{{path}}}",
        f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"\end{figure}"]) + _flush()


L = []
A = L.append

A(r"\section{Appendix}")
A("")
A("Everything in this appendix is generated from \\texttt{results/} by "
  "\\texttt{code/make\\_report\\_appendix.py}, so the tables cannot drift from the artefacts "
  "submitted with the code. The course guideline exempts appendices from the seven-page limit "
  "on the body.")
A("")

# ------------------------------------------------------------------ A1 variables
meta = json.load(open(os.path.join(ROOT, "data", "processed", "split_meta.json")))
groups = [
    ("Credit line", ["LIMIT_BAL"], "numeric", "credit limit granted (NT\\$), individual plus family"),
    ("Demographics", ["SEX", "EDUCATION", "MARRIAGE", "AGE"], "3 categorical, 1 numeric",
     "gender; graduate / university / high school / others; married / single / others; age in years"),
    ("Repayment status", ["PAY_1", "\\dots", "PAY_6"], "ordinal ($-2 \\dots 8$)",
     "monthly status Sep $\\rightarrow$ Apr 2005: $-2$ no consumption, $-1$ paid in full, "
     "$0$ revolving credit, $k \\ge 1$ = $k$ months of delay"),
    ("Bill statements", ["BILL_AMT1", "\\dots", "BILL_AMT6"], "numeric",
     "monthly bill amount Sep $\\rightarrow$ Apr (NT\\$); may be negative"),
    ("Payments", ["PAY_AMT1", "\\dots", "PAY_AMT6"], "numeric",
     "amount paid in each month Sep $\\rightarrow$ Apr (NT\\$)"),
    ("Engineered", ["UTIL_1", "UTIL_MEAN", "PAY_RATIO_MEAN", "N_MONTHS_DELAYED", "MAX_DELAY",
                    "BILL_TREND"], "numeric",
     "utilisation of the limit (latest, mean); mean payment-to-bill ratio; months in arrears; "
     "worst delay; six-month balance trend relative to the limit"),
    ("Target", ["DEFAULT"], "binary",
     f"1 = default on the October 2005 payment ({100 * meta['default_rate_train']:.1f}\\% in training)"),
]
rows = [[esc(g), ",\\allowbreak\\ ".join(tt(v) if not v.startswith("\\") else v for v in vs), t, m]
        for g, vs, t, m in groups]
A(table(f"Variables in the cleaned data set: {len(meta['features'])} predictors and the target, "
        f"{meta['n_after_dedup']:,} clients after removing {meta['n_duplicates_dropped']} exact "
        f"duplicate rows.",
        "tab:a-variables", "L{0.14\\linewidth}L{0.25\\linewidth}L{0.12\\linewidth}L{0.35\\linewidth}",
        ["Group", "Variables", "Type", "Meaning"], rows))
A("")

# ------------------------------------------------------------------ A1b exploratory gradients (moved out of the body on 21 Oct)
eda = json.load(open(os.path.join(RESULTS, "eda_summary.json")))
pay1 = eda["default_rate_by"]["PAY_1"]
quint = eda["default_rate_by"]["LIMIT_BAL_quintile"]
PAY1_LABELS = {"-2": "No consumption ($-2$)", "-1": "Paid in full ($-1$)", "0": "Revolving credit (0)",
               "1": "1 month late", "2": "2 months late", "3": "3 months late"}
left = [(PAY1_LABELS[k], pay1[k]) for k in ["-2", "-1", "0", "1", "2", "3"]]
right = [("NT\\$" + esc(k).replace("\u2013", "--").replace("1000k", "1M"), v) for k, v in quint.items()]
rows = []
for i in range(max(len(left), len(right))):
    l = left[i] if i < len(left) else ("", None)
    r_ = right[i] if i < len(right) else ("", None)
    rows.append([l[0], f"{l[1]['n']:,}" if l[1] else "", f"{100 * l[1]['rate']:.1f}\\%" if l[1] else "",
                 r_[0], f"{r_[1]['n']:,}" if r_[1] else "", f"{100 * r_[1]['rate']:.1f}\\%" if r_[1] else ""])
A(table(f"Default rate along the two strongest credit-risk gradients, training partition only "
        f"($n = {eda['class_balance']['train']['n']:,}$; overall rate "
        f"{100 * eda['default_rate_by']['overall_train']:.1f}\\%). Source: \\texttt{{results/eda\\_summary.json}}.",
        "tab:a-gradients", "lrrlrr",
        ["September status", "Clients", "Default", "Credit-limit quintile", "Clients", "Default"], rows))
A("")

# ------------------------------------------------------------------ A1c data-quality checks (verbatim from the EDA)
dq_path = os.path.join(RESULTS, "eda_data_quality.csv")
if os.path.exists(dq_path):
    dq = pd.read_csv(dq_path)
    rows = [[esc(r_.check), esc(r_.finding), "---" if pd.isna(r_.action) else esc(r_.action)]
            for r_ in dq.itertuples()]
    A(table("Data-quality checks run on the raw download before modelling, and what was done about each "
            "finding. Reproduced verbatim from \\texttt{results/eda\\_data\\_quality.csv}; the three cleaning "
            "steps that precede the split (duplicate removal, the fixed recoding of undocumented codes, the "
            "row-wise engineered features) estimate nothing from the data, and everything else is left to "
            "the in-fold pipeline. The counts are over the full file (30,000 raw or 29,965 cleaned rows, test "
            "partition included); labels are used only for the class balance, the split check and the "
            "default rates marked as training-partition.",
            "tab:a-quality", "L{0.19\\linewidth}L{0.36\\linewidth}L{0.39\\linewidth}",
            ["Check", "Finding", "Action"], rows, tabcolsep="3pt", footnotesize=True))
    A("")

# ------------------------------------------------------------------ A2 workload
# The instructor asks for per-section attribution, not a flat "all contributed equally".
# Shared sections genuinely are equal thirds; the model sections are not, and saying so is
# the point of the table. Every model row is checkable against the member field of results/.
WORKLOAD = [
    # Draft of 21 October 2026 (percentages of the writing and analysis behind each section);
    # typeset in red until all three members have confirmed them.
    ("Abstract, \\S1 Introduction", 30, 20, 50, "LI drafted; RUAN and LU revised"),
    ("\\S2 Data", 25, 15, 60, "LI wrote the cleaning and split contract and the exploratory analysis"),
    ("\\S3 Methods", 30, 25, 45, "shared protocol; RUAN the nested cross-validation, LI the freeze and "
                                   "recalibration protocol"),
    ("\\S4 Results", 30, 35, 35, "RUAN: KNN, logistic family, second pass; LU: tree, forest, GBM, XGBoost; "
                                   "LI: SVM, MLP, LDA/QDA/NB, imbalance study, aggregation and bootstrap"),
    ("\\S5 Interpretation", 15, 60, 25, "LU owns TreeSHAP, the monotone constraints, partial dependence and "
                                          "the forest importances"),
    ("\\S6 Choosing the operating point", 40, 15, 45, "RUAN: cost-ratio table and nested CV; LI: recalibration "
                                                        "and threshold analysis"),
    ("\\S7 Conclusion and limitations", 34, 33, 33, "drafted jointly"),
    ("Code and reproducibility", 20, 25, 55, "LI: \\texttt{common.py}, aggregation, freeze, recalibration"),
    ("Presentation", 33, 33, 34, "three presenters, 10 minutes"),
]
MODELS = [
    ("KNN benchmark; logistic family (plain, ridge, lasso, class-weighted); PCA and K-means",
     "RUAN Hangjing"),
    ("Pruned tree; random forest and bagging; gradient boosting and XGBoost; TreeSHAP; "
     "monotone-constrained XGBoost; partial dependence", "LU Xinyu"),
    ("Support-vector machines; feed-forward neural network; LDA, QDA and naive Bayes; "
     "class-imbalance study and the leakage demonstration", "LI Jingwen"),
]
rows = [[sec, f"\\textcolor{{red}}{{{r}\\%}}", f"\\textcolor{{red}}{{{l}\\%}}",
         f"\\textcolor{{red}}{{{j}\\%}}", basis] for sec, r, l, j, basis in WORKLOAD]
A(table("Contribution by section. Percentages are of the writing and analysis behind each "
        "section; the model families below are owned end to end --- data, tuning, "
        "interpretation and presentation --- by the member named. The percentages are the "
        "group's draft of 21 October 2026 and are typeset in red until all three members have "
        "confirmed them.",
        "tab:a-workload", "L{0.20\\linewidth}cccL{0.34\\linewidth}",
        ["Section", "RUAN", "LU", "LI", "Basis"], rows,
        note="Each model family is run end to end by its owner: "
             + "; ".join(f"\\textbf{{{who}}} --- {what}" for what, who in MODELS)
             + ". The owner of every file in \\texttt{results/} is recorded in its "
               "\\texttt{member} field, so this table is checkable against the submitted code. "
               "This project is not shared with any other module."))
A("")

# ------------------------------------------------------------------ A6 method families
# Moved out of the body when the 7-page limit bit; kept verbatim.
mt = os.path.join(REPORT, "_methods_table_for_appendix.tex")
if os.path.exists(mt):
    A(open(mt).read().rstrip())
    A("")


def load_json(*parts):
    p = os.path.join(RESULTS, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt_val(v):
    """One grid value, compact: 0.03 -> 0.03, 31.622776 -> 31.6, 1e-09 -> 1e-09, 'sqrt' -> sqrt."""
    if v is None or (isinstance(v, str) and v.lower() in ("none", "null")):
        return "None"
    if isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    if f == 0:
        return "0"
    if abs(f) < 1e-3:
        return f"{f:.3g}"
    if f.is_integer():
        return f"{int(f):,}" if abs(f) >= 1e4 else str(int(f))
    return f"{f:.4g}"


def fmt_set(values, max_show=6):
    """A grid axis as {a, b, c}; long numeric axes as {min … max} (n values)."""
    vals = list(values)
    if len(vals) > max_show:
        try:
            nums = sorted(float(v) for v in vals)
            return f"\\{{{fmt_val(nums[0])} \\dots\\ {fmt_val(nums[-1])}\\}} ({len(vals)} values)"
        except (TypeError, ValueError):
            vals = vals[:max_show] + ["\\dots"]
    return "\\{" + ", ".join(fmt_val(v) if v != "\\dots" else v for v in vals) + "\\}"


METRIC_KEYS = {"cv_auc", "cv_auc_std", "train_auc", "rank", "fit_seconds", "fit_seconds_500",
               "fold_aucs", "mean_fit_time", "cv_auc_200trees", "cv_auc_500trees"}


def grid_axes_from_list(grid):
    """extra['grid'] as a list of dicts (one per setting) -> ordered {param: [unique values]}."""
    axes = {}
    for g in grid:
        for k, v in g.items():
            if k in METRIC_KEYS:
                continue
            axes.setdefault(k, [])
            if v not in axes[k]:
                axes[k].append(v)
    return axes


def axes_text(axes):
    return " $\\times$ ".join(f"{tt(k)} {fmt_set(v)}" for k, v in axes.items())


def chosen_text(bp, keys):
    return ", ".join(f"{tt(k)} = {fmt_val(bp.get(k))}" for k in keys if k in bp)


def tuning_row(slug):
    """(grid searched, chosen setting) for one headline model, read from results/<slug>.json."""
    d = load_json(f"{slug}.json")
    if d is None:
        return None
    bp, ex = d.get("best_params", {}) or {}, d.get("extra", {}) or {}
    grid = ex.get("grid")
    if slug == "xgboost":
        axes = grid_axes_from_list(grid)
        n = len(grid)
        g = f"{axes_text(axes)} ({n} settings); {tt('subsample')} = {fmt_val(bp.get('subsample'))}, " \
            f"{tt('colsample_bytree')} = {fmt_val(bp.get('colsample_bytree'))} fixed; " \
            f"rounds then checked on a learning curve to {ex.get('learning_curve_max_rounds')}"
        return g, chosen_text(bp, list(axes))
    if slug == "gradient_boosting":
        axes = grid_axes_from_list(grid)
        g = f"{axes_text(axes)} ({len(grid)} settings) on a {ex.get('grid_subsample_rows'):,}-row " \
            f"training subsample; staged CV curve on the full training set"
        return g, chosen_text(bp, list(axes)) + f", {tt('subsample')} = {fmt_val(bp.get('subsample'))}"
    if slug == "random_forest":
        axes = grid_axes_from_list(grid)
        g = f"{axes_text(axes)} ({len(grid)} settings) at 200 trees; top two re-run at " \
            f"{fmt_val(bp.get('n_estimators'))} trees"
        return g, chosen_text(bp, list(axes)) + f", {tt('n_estimators')} = {fmt_val(bp.get('n_estimators'))}"
    if slug == "mlp":
        keys = ["hidden_layer_sizes", "alpha", "learning_rate_init"]
        g = " $\\times$ ".join(f"{tt(k)} {fmt_set(grid[k])}" for k in keys) + f" ({grid.get('n_settings')} settings); " \
            f"{tt('relu')}, {tt('adam')}, early stopping fixed"
        bp2 = dict(bp, hidden_layer_sizes="x".join(str(h) for h in bp["hidden_layer_sizes"]))
        return g, chosen_text(bp2, keys)
    if slug == "knn":
        g = f"{tt('K')} {fmt_set(grid['K'])} $\\times$ {tt('weights')} {fmt_set(grid['weights'])} ({grid.get('n_settings')} settings)"
        return g, chosen_text(bp, ["n_neighbors", "weights"])
    if slug == "tree_pruned":
        g = f"{tt('ccp_alpha')} {fmt_set(ex['alpha_grid'])} (cost-complexity path) $\\times$ " \
            f"{tt('max_depth')} {fmt_set(ex['max_depth_grid'])}"
        return g, chosen_text(bp, ["criterion", "ccp_alpha", "max_depth"])
    if slug == "svm_rbf":
        t = ex.get("tuning", {})
        g = f"{tt('C')} {fmt_set(t['grid_C'])} $\\times$ {tt('gamma')} {fmt_set(t['grid_gamma'])} $\\times$ " \
            f"{tt('class_weight')} {fmt_set(t['grid_class_weight'])} on a {t.get('subsample_rows'):,}-row " \
            f"training subsample; one-standard-error rule"
        return g, chosen_text(bp, ["C", "gamma", "class_weight"]) + "; Platt-scaled"
    if slug == "logistic":
        return "none: unpenalised reference fit", chosen_text(bp, ["C", "solver"])
    if slug in ("logistic_l1", "logistic_l2"):
        g = f"{tt('C')} {fmt_set(ex['grid_C'])}"
        return g, chosen_text(bp, ["C", "penalty"])
    if slug in ("lda", "naive_bayes"):
        cg = ex.get("cv_grid", [])
        param = cg[0]["param"].split("__")[-1] if cg else "?"
        g = f"{tt(param)} {fmt_set([x['value'] for x in cg])}"
        return g, chosen_text(bp, list(bp))
    # Fallback: whatever grid information the file carries.
    if isinstance(grid, list):
        return axes_text(grid_axes_from_list(grid)), chosen_text(bp, list(bp))
    return "---", chosen_text(bp, list(bp))


# ------------------------------------------------------------------ tuning grids (headline models)
comp_path = os.path.join(RESULTS, "model_comparison.csv")
if os.path.exists(comp_path):
    hc = pd.read_csv(comp_path)
    hc = hc[hc["headline"].astype(str).str.lower() == "true"].sort_values("rank_cv_auc")
    rows = []
    for r_ in hc.itertuples():
        tr = tuning_row(r_.slug)
        if tr is None:
            continue
        rows.append([esc(r_.name), tr[0], tr[1], num(r_.select_auc, 4)])
    A(table("Hyper-parameter grids searched for the headline models and the setting chosen, read from "
            "\\texttt{results/<slug>.json}. Every grid is scored by 5-fold stratified cross-validation on the "
            "training partition (seed 2607526); the test set plays no part. The last column is the "
            "training-set selection score of the chosen setting (CV AUC; out-of-fold AUC for the "
            "subsample-tuned RBF SVM).",
            "tab:a-tuning", "L{0.15\\linewidth}L{0.40\\linewidth}L{0.29\\linewidth}c",
            ["Model", "Grid searched", "Chosen", "Sel.\\ AUC"], rows, tabcolsep="3pt"))
    A("")

# ------------------------------------------------------------------ software versions and seed
# Versions recorded by the model scripts themselves (xgboost.json, pdp_summary.json); the two packages
# whose version no result file records (shap, imbalanced-learn) are the ones installed in the
# environment that produced results/ (checked 2026-10-21: shap 0.52.0, imbalanced-learn 0.14.2).
xj = load_json("xgboost.json") or {}
pj = load_json("pdp_summary.json") or {}
fm = load_json("frozen_manifest.json") or {}
vers = dict(xj.get("extra", {}).get("versions", {}))
vers.update({k: v for k, v in (pj.get("versions") or {}).items() if k not in vers})
SW_FALLBACK = {"shap": "0.52.0", "imbalanced-learn": "0.14.2"}
SW_ROWS = [("Python", "python"), ("scikit-learn", "sklearn"), ("XGBoost", "xgboost"), ("SHAP", "shap"),
           ("NumPy", "numpy"), ("pandas", "pandas"), ("imbalanced-learn", "imbalanced-learn"), ("SciPy", "scipy")]
rows = [[name, esc(vers.get(key, SW_FALLBACK.get(key, "---")))] for name, key in SW_ROWS]
rows.append(["Random seed (split, folds, samplers, every model)", esc(fm.get("seed", meta.get("seed", "---")))])
rows.append(["Cross-validation", "5 stratified folds of the training partition, shuffled with the seed"])
rows.append(["Split", f"stratified {int(round(100 * (1 - meta['test_size'])))}/{int(round(100 * meta['test_size']))}, "
                      f"{meta['n_train']:,} training / {meta['n_test']:,} test rows"])
A(table("Software versions and the single seed behind every result in this report. Versions are the "
        "ones recorded in the result files by the scripts that produced them.",
        "tab:a-software", "L{0.42\\linewidth}L{0.50\\linewidth}",
        ["Component", "Version / value"], rows))
A("")

# ------------------------------------------------------------------ monotone constraint vector
ms = load_json("monotone_summary.json")
if ms:
    cv_ = ms["constraint_vector"]
    zero_raw = []
    for col in cv_["columns"]:
        if col["constraint"] == 0 and col["raw"] not in zero_raw:
            zero_raw.append(col["raw"])
    feats = lambda xs: ",\\allowbreak\\ ".join(tt(x) for x in xs)
    rows = [["$+1$ (more may only raise the predicted risk)", feats(cv_["plus_features"]), str(cv_["n_plus"])],
            ["$-1$ (more may only lower it)", feats(cv_["minus_features"]), str(cv_["n_minus"])],
            ["$0$ (unconstrained)", feats(zero_raw), str(cv_["n_zero"])]]
    viol = ms["violations"]
    rp = viol["pay1_random_profiles"]
    allc = viol["all_constrained_features"]
    pb = ms["cost_of_the_constraint"]["paired_bootstrap"]
    A(table("The monotone constraint vector of the constrained XGBoost, fixed a priori from credit logic "
            f"and read back from the fitted booster: {tt(cv_['string']).replace(',', ',\\allowbreak ')} over the "
            f"{len(cv_['columns'])} preprocessed columns (one-hot categories count once per raw variable here). "
            "Hyper-parameters are inherited unchanged from the tuned unconstrained model.",
            "tab:a-monotone", "L{0.30\\linewidth}L{0.54\\linewidth}c",
            ["Sign", "Variables", "$n$"], rows,
            note=f"Sweeping each constrained variable with the others at their training median, the unconstrained "
                 f"booster moves against the required direction on {allc['total_violations_unconstrained']} grid steps "
                 f"in total ({viol['pay1_median_profile']['unconstrained']['violations']} of them on "
                 f"\\texttt{{PAY\\_1}}); over {rp['unconstrained']['n_profiles']} random training profiles it violates monotonicity in "
                 f"\\texttt{{PAY\\_1}} on {rp['unconstrained']['violating_steps']} of {rp['unconstrained']['total_steps']:,} "
                 f"steps (largest fall {rp['unconstrained']['largest_fall']:.3f}); the constrained booster on none. "
                 f"Price of the constraint: CV AUC {snum(ms['cost_of_the_constraint']['delta_cv_auc'])}, test AUC "
                 f"{snum(ms['cost_of_the_constraint']['delta_test_auc'])} (paired bootstrap, $B={pb['B']}$: mean "
                 f"{snum(pb['auc_diff_mean'])}, 95\\,\\% interval [{snum(pb['auc_diff_2_5'])}, {snum(pb['auc_diff_97_5'])}])."))
    A("")

# ------------------------------------------------------------------ freeze timeline
if fm.get("models"):
    groups_ = {"r1": [], "ruan": [], "platt": [], "neg": []}
    for slug, e in fm["models"].items():
        role, rnd = e.get("role", ""), str(e.get("round", ""))
        if role == "negative_control":
            groups_["neg"].append(e)
        elif role == "recalibrated_variant":
            groups_["platt"].append(e)
        elif slug.endswith("_v2") or "adopted" in rnd:
            groups_["ruan"].append(e)
        else:
            groups_["r1"].append(e)
    assert sum(len(v) for v in groups_.values()) == len(fm["models"])

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def when(es):
        ds = sorted(str(e.get("first_scored", ""))[:10] for e in es if e.get("first_scored"))
        if not ds:
            return "---"

        def f(s):
            return f"{int(s[8:10])} {MONTHS[int(s[5:7]) - 1]}"
        return f(ds[0]) if ds[0] == ds[-1] else f"{f(ds[0])}--{f(ds[-1])}"

    r1_members = sorted({e["member"] for e in groups_["r1"]})
    rows = [
        ["1", when(groups_["r1"]), str(len(groups_["r1"])),
         "every model variant of Table~\\ref{tab:a-allmodels}: each configuration tuned on training CV, then scored on "
         "the test set (before the professor's 30 September comment) and registered on 19 October; the forest and "
         "bagging were scored twice with the same configuration (16 and 18 September); dates are result-file times",
         "5-fold CV AUC on the training partition; both thresholds on training out-of-fold probabilities"],
        ["2", when(groups_["ruan"]), str(len(groups_["ruan"])),
         "RUAN Hangjing's second pass under her freeze-before-test protocol (\\texttt{frozen\\_choices\\_ruan\\_v2.json}): "
         "the KNN ($K=101$, uniform) and plain-logistic configurations re-fitted and re-scored; adopted on 21 October "
         "unchanged; the headline tables keep the round-1 scoring (predictions differ by $\\le 10^{-2}$, test AUC by $5\\times10^{-5}$)",
         "nested CV with the pre-registered 1-SE and 1\\,\\% cost rules (Table~\\ref{tab:a-nestedcv}); training rows only"],
        ["3", when(groups_["platt"]), str(len(groups_["platt"])),
         "Platt-recalibrated versions of the nine class-weighted or resampled models (Table~\\ref{tab:a-recal}), each "
         "registered before its 19 October scoring (a 6 October review had already applied the same maps to the test scores)",
         "Platt map fitted on training out-of-fold scores; thresholds re-chosen on cross-fitted training scores"],
        ["---", when(groups_["neg"]), str(len(groups_["neg"])),
         "negative control: SMOTE applied before the split (the leakage demonstration); it pools the training and test "
         "rows, so its re-split `test' rows are not the real test set, and it is not a model result",
         "none --- deliberately leaky"],
    ]
    A(table(f"Freeze timeline: the {len(fm['models'])} entries of \\texttt{{results/frozen\\_manifest.json}} "
            "grouped by round. A configuration is frozen the first time it is scored on the test set, and "
            "\\texttt{common.evaluate()} refuses to score a changed configuration under a frozen slug "
            "(\\texttt{code/freeze.py -{}-check} verifies every stored prediction vector). The manifest lists "
            "configurations only; every read of the test set, including descriptive reads outside it, is dated in "
            "\\texttt{results/TEST\\_SET\\_LOG.md}.",
            "tab:a-freeze", "cL{0.12\\linewidth}cL{0.40\\linewidth}L{0.26\\linewidth}",
            ["Round", "Scored", "$n$", "What", "Selection basis"], rows, tabcolsep="3pt"))
    A("")

# ------------------------------------------------------------------ nested CV (RUAN, 30 Sep; reproduced 21 Oct)
ncv_path = os.path.join(RESULTS, "logistic_nested_cv_v2.csv")
knn2 = load_json("knn_v2.json")
if os.path.exists(ncv_path):
    ncv = pd.read_csv(ncv_path)
    NCV_LABEL = {"plain": "Logistic, plain (unpenalised)", "ridge": "Logistic, ridge ($C$ tuned inside each fold)",
                 "lasso": "Logistic, lasso ($C$ tuned inside each fold)", "weighted": "Logistic, class-weighted",
                 "weighted_cal": "Logistic, class-weighted + Platt"}
    rows = [[NCV_LABEL.get(r_.variant, esc(r_.variant)),
             f"{num(r_.nested_auc, 4)} $\\pm$ {num(r_.nested_auc_se, 4)}", num(r_.nested_ap, 4),
             num(r_.brier, 4), num(r_.cal_slope, 3), num(r_.t_cost5_oof, 2), num(r_.crossfit_cost5, 0)]
            for r_ in ncv.itertuples()]
    if knn2:
        ex2 = knn2.get("extra", {})
        bp2 = knn2.get("best_params", {})
        oc = ex2.get("oof_calibration", {})
        rows.append([f"KNN ($K$, weights tuned in-fold; $K={bp2.get('n_neighbors')}$ {esc(bp2.get('weights'))} chosen)",
                     f"{num(ex2.get('nested_cv_auc'), 4)} $\\pm$ {num(ex2.get('nested_cv_auc_std'), 4)}", "---",
                     num(oc.get("brier"), 4), num(oc.get("cal_slope"), 3), "---", "---"])
    rep = load_json("nested_cv_summary_repro.json") or {}
    lg = rep.get("logistic", {})
    note = ("Nested CV: outer loop = the project's five folds, inner loop = a 5-fold grid search inside each outer "
            "training fold, so the AUC is that of the whole tuning procedure. Columns: nested AUC $\\pm$ its "
            "standard error over the outer folds (for KNN, the standard deviation), nested average precision, "
            "Brier score and calibration slope of the nested out-of-fold scores, the 5:1 cut-off chosen on those "
            f"scores (closed form $1/6 = 0.167$) and the cross-fitted 5:1 cost on the {meta['n_train']:,} training rows. ")
    if lg:
        _cost = dict(zip(ncv["variant"].astype(str), ncv["crossfit_cost5"].astype(float)))
        _pc, _wc = _cost.get("plain"), _cost.get("weighted_cal")
        if _pc and _wc:
            _rel = 100 * (_wc - _pc) / _pc        # +0.43: the weighted + Platt model costs MORE
            _word = "costlier" if _rel > 0 else "cheaper"
            note += (f"Pre-registered rule: {esc(lg.get('rule'))}. Outcome: {esc(lg.get('selected'))} kept; the "
                     f"Platt-calibrated weighting is {abs(_rel):.2f}\\,\\% {_word} ({_wc:,.0f} against {_pc:,.0f}), "
                     f"so the rule (adopt only if $>1\\,\\%$ cheaper) is not met and it is not adopted. ")
    oof_p = os.path.join(RESULTS, "logistic_p_oof.npy")
    tr_p = os.path.join(ROOT, "data", "processed", "train.csv")
    if os.path.exists(oof_p) and os.path.exists(tr_p):
        import numpy as np
        _p = np.load(oof_p)
        _y = pd.read_csv(tr_p, usecols=["DEFAULT"])["DEFAULT"].to_numpy()
        if len(_p) == len(_y):
            def _c5(t):            # 5:1 cost with the p >= t convention of common.evaluate
                f = _p >= t
                return int(5 * int((~f & (_y == 1)).sum()) + int((f & (_y == 0)).sum()))
            c15, c17 = _c5(0.15), _c5(0.17)
            note += (f" For comparison, the plain 5-fold out-of-fold 5:1 cost of the frozen 16 September logit "
                     f"({tt('results/logistic_p_oof.npy')} against the {len(_y):,} training labels) is {c15:,} at 0.15 "
                     f"and {c17:,} at 0.17 ({100 * (c17 - c15) / c15:+.2f}\\,\\%): a different cross-fit from the nested one, "
                     "which is why the two costs at 0.17 differ.")
    note += " The test set is not read anywhere in this table."
    A(table("Nested cross-validation of the KNN and logistic families, RUAN Hangjing's second pass of 30 September "
            f"2026 ({tt('logistic_nested_cv_v2.csv')}, {tt('knn_v2.json')}). The 21 October reproduction "
            f"on the training rows ({tt('nested_cv_summary_repro.json')}, {tt('logistic_nested_cv_repro.csv')}) "
            "matched the nested AUCs and standard errors, the logistic rows and the selected configurations; "
            "the KNN Brier and calibration-slope columns are RUAN's 30 September out-of-fold figures "
            f"({tt('knn_v2.json')}), which the reproduction ({tt('nested_cv_knn_repro.json')}) puts at "
            f"{num(((load_json('nested_cv_knn_repro.json') or {}).get('oof_calibration') or {}).get('brier'), 4)} and "
            f"{num(((load_json('nested_cv_knn_repro.json') or {}).get('oof_calibration') or {}).get('cal_slope'), 3)}. "
            "The pre-registered rules re-select the configurations frozen on 16 September.",
            "tab:a-nestedcv", "L{0.30\\linewidth}cccccc",
            ["Model", "Nested AUC", "Nested AP", "Brier", "Cal.\\ slope", "5:1 cut-off", "5:1 cost"],
            rows, note=note, tabcolsep="3pt", footnotesize=True))
    A("")

# ------------------------------------------------------------------ A3 full comparison
comp_path = os.path.join(RESULTS, "model_comparison.csv")
if os.path.exists(comp_path):
    c = pd.read_csv(comp_path)
    sel = "select_auc" if "select_auc" in c.columns else "cv_auc"      # training-set selection score (see aggregate.py)
    c = c.sort_values(sel, ascending=False, na_position="last", kind="mergesort")
    cols = [x for x in ("name", "slug", "cv_auc", "select_auc", "select_source", "test_auc", "test_ap", "brier")
            if x in c.columns]
    # RUAN's 30 Sep second pass re-scored two configurations that were already frozen on 16 Sep
    # (knn / logistic). They are documented in the freeze-timeline and nested-CV tables, not listed
    # as separate models: aggregate.py excludes the *_v2 slugs, and this guard keeps it that way.
    v2_dupes = [s for s in c["slug"].astype(str) if s.endswith("_v2")]
    if v2_dupes:
        raise SystemExit(f"model_comparison.csv lists *_v2 duplicates as separate models: {v2_dupes}")
    # The round-1 cluster-label logit stays in the list (it was scored on the test set in round 1),
    # but the pre-registered rule of the 30 Sep / 21 Oct analysis did not adopt the cluster label.
    NAME_OVERRIDE = {"logistic_plus_clusters":
                     "Logistic + K-means segment dummies (round-1 scoring; not adopted by the pre-registered rule)"}
    rows, subsample_notes = [], []
    for r_ in c.itertuples():
        d = {k: getattr(r_, k, None) for k in cols}
        slug = str(d.get("slug"))
        label = NAME_OVERRIDE.get(slug) or str(d.get("name", slug))      # the column wraps; no truncation
        # The printed score is the selection score the rows are sorted by: CV AUC, or the full-training
        # out-of-fold AUC for the two SVMs tuned on a subsample (their subsample CV AUC is not comparable
        # and goes to the footnote), so the table never shows one number and sorts by another.
        sel_v = d.get("select_auc") if d.get("select_auc") is not None and not pd.isna(d.get("select_auc")) else d.get("cv_auc")
        dag = ""
        if str(d.get("select_source", "cv_auc")) != "cv_auc":
            dag = "$^{\\dagger}$"
            subsample_notes.append(f"{esc(label)} {num(d.get('cv_auc'))}")
        rows.append([esc(label) + dag,
                     num(sel_v), num(d.get("test_auc")), num(d.get("test_ap")),
                     num(d.get("brier"))])
    note = None
    if subsample_notes:
        note = ("$^{\\dagger}$Tuned on a 6,000-row training subsample; the CV AUC shown is the out-of-fold AUC of the "
                "chosen setting refitted on the full training partition (the same figure as Table~\\ref{tab:headline}). "
                "The CV AUC on the tuning subsample, not comparable with the other rows, was " + "; ".join(subsample_notes) + ".")
    A(table(f"Every model variant fitted in this project ({len(c)} in total), ordered by CV AUC, the "
            "training-set selection score (for the two SVMs, tuned on a subsample, the out-of-fold AUC on the full "
            "training partition). CV AUC is the mean 5-fold out-of-fold AUC on the training partition; the remaining "
            "columns are on the held-out test set, reported once and not used to rank. RUAN Hangjing's "
            "30 September re-scoring of the KNN and plain-logistic configurations (Table~\\ref{tab:a-freeze}) "
            "is not a separate model and is not listed.",
            "tab:a-allmodels", "L{0.40\\linewidth}cccc",
            ["Model", "CV AUC", "Test AUC", "Test AP", "Brier"], rows, note=note))
    A("")

# ------------------------------------------------------------------ A4 operating points
agg_path = os.path.join(RESULTS, "aggregate_summary.json")
if os.path.exists(agg_path):
    agg = json.load(open(agg_path))
    ct = agg.get("thresholds", {}).get("cost_test", {})
    thr = agg.get("thresholds", {}).get("cost_tuned", {})
    rec = agg.get("thresholds", {}).get("recall_1_at_cost", {})
    # Readable model names (the name column of model_comparison.csv) rather than code slugs, so the
    # rows can be matched against Table 1 of the body.
    names, thr_f1 = {}, {}
    if os.path.exists(comp_path):
        _c = pd.read_csv(comp_path)
        names = dict(zip(_c["slug"].astype(str), _c["name"].astype(str)))
        if "thr_f1" in _c.columns:
            thr_f1 = dict(zip(_c["slug"].astype(str), _c["thr_f1"]))
    rows = []
    for slug in ct:
        rows.append([esc(names.get(slug, slug)), num(thr_f1.get(slug), 2), num(thr.get(slug), 2), num(rec.get(slug), 2),
                     num(ct[slug].get("at_0_5")), num(ct[slug].get("at_f1")),
                     num(ct[slug].get("at_cost"))])
    ca = agg.get("cost_assumption", {})
    A(table(f"Operating points of the {len(ct)} headline configurations of \\texttt{{aggregate\\_summary.json}} "
            "(ten model families: the lasso logit duplicates the plain logit to four decimals and is omitted from "
            "Table~\\ref{tab:headline}). Both thresholds are chosen on training "
            "out-of-fold probabilities (maximum $F_1$; minimum 5:1 cost); recall is at the cost threshold; "
            "the cost columns are the expected test cost at a 5:1 false-negative to false-positive ratio.",
            "tab:a-operating", "L{0.27\\linewidth}cccccc",
            ["Model", "$F_1$ thr.", "Cost thr.", "Recall", "Cost @0.5", "Cost @$F_1$", "Cost @cost thr."], rows, tabcolsep="4pt",
            note=f"Trivial rules for reference: flagging every client costs "
                 f"{num(ca.get('flag_everybody_cost'))}, flagging none costs "
                 f"{num(ca.get('flag_nobody_cost'))}."))
    A("")

# ------------------------------------------------------------------ A5 cost sensitivity
cs_path = os.path.join(RESULTS, "cost_sensitivity.csv")
if os.path.exists(cs_path):
    cs = pd.read_csv(cs_path)
    piv = cs.pivot_table(index="model", columns="ratio", values="test_cost", aggfunc="first")
    ratios = sorted(piv.columns)
    rows = [[esc(m)] + [num(piv.loc[m, r_]) for r_ in ratios] for m in piv.index]
    # The CSV on disk (21 Oct 15:46) was made on a cut-off grid starting at 0.02; cost_sensitivity.py now uses the
    # 0.01 grid of threshold_vs_model_saving.py but was not re-run (it reads the test set). Say so where the floor binds.
    cs_note = []
    sv_path_ = os.path.join(RESULTS, "threshold_vs_model_saving.csv")
    if os.path.exists(sv_path_):
        sv_ = pd.read_csv(sv_path_)
        for r_ in sv_.itertuples():
            row_ = cs[(cs.slug == "logistic") & (cs.ratio == r_.ratio)]
            if len(row_) and abs(float(row_.threshold.iloc[0]) - float(r_.t_logistic)) > 1e-9:
                cs_note.append(f"Logistic regression at {int(r_.ratio)}:1: computed on a cut-off grid starting at "
                               f"{float(row_.threshold.iloc[0]):.2f}, whose floor binds; on the 0.01 grid of "
                               f"Table~\\ref{{tab:a-saving}} the cut-off is {float(r_.t_logistic):.2f} and the test cost "
                               f"{num(r_.cost_logistic_tuned)}.")
    for r_ in cs.itertuples():
        if abs(float(r_.flagged_share) - 1.0) < 1e-9:      # every test client flagged (cost = flag-everyone rule)
            cs_note.append(f"{esc(r_.model)} at {int(r_.ratio)}:1: the cut-off flags every client, so its cost is the "
                           f"flag-everyone rule ({num(r_.test_cost)}).")
    A(table("Cost-ratio sensitivity: total test cost of each model at its own cut-off, chosen on training "
            "out-of-fold probabilities, as the false-negative to false-positive ratio varies (reporting only; "
            "nothing is selected on these numbers).",
            "tab:a-costsens", "L{0.24\\linewidth}" + "c" * len(ratios),
            ["Model"] + [f"{int(r_)}:1" for r_ in ratios], rows, tabcolsep="4pt",
            note=" ".join(cs_note) or None))
    A("")

# ------------------------------------------------------------------ A5a threshold saving vs model saving, nine ratios
sv_path = os.path.join(RESULTS, "threshold_vs_model_saving.csv")
if os.path.exists(sv_path):
    sv = pd.read_csv(sv_path)
    svj = load_json("threshold_vs_model_saving.json") or {}
    ci = lambda m, lo, hi: f"{num(m, 0)} [{num(lo, 0)}, {num(hi, 0)}]"
    rows = []
    for r_ in sv.itertuples():
        deg = r_.share_flagged_logistic >= 0.99
        rows.append([f"{int(r_.ratio)}:1" + ("$^{\\dagger}$" if deg else ""), num(r_.t_closed_form, 3),
                     num(r_.t_logistic, 2), num(r_.t_xgb, 2), num(r_.cost_logistic_at_0_5, 0),
                     ci(r_.threshold_saving, r_.threshold_saving_lo, r_.threshold_saving_hi),
                     ci(r_.model_saving, r_.model_saving_lo, r_.model_saving_hi)])
    inc0 = svj.get("ratios_where_model_saving_ci_includes_zero", [])
    ps = svj.get("proposal_stage_table", {})
    note = (f"Paired bootstrap of the {meta['n_test']:,} test rows, $B={svj.get('B', '---')}$, seed {svj.get('seed', '---')}; "
            "the test set is read for these intervals but selects nothing (both cut-offs are chosen on training out-of-fold "
            "scores; the closed form is $1/(1+r)$). The model-saving interval includes zero at "
            + ", ".join(f"{int(x)}:1" for x in inc0) + ". $^{\\dagger}$At 20:1 the logistic cut-off "
            f"flags {100 * float(sv.loc[sv.ratio == 20, 'share_flagged_logistic'].iloc[0]):.1f}\\,\\% of clients, "
            "i.e.\\ the trivial flag-everyone rule, so the threshold `saving' there is not a model result. ")
    if ps:
        note += (f"Proposal-stage table (RUAN Hangjing's 30 September refit of the same plain logit, cut-off "
                 f"{num(ps.get('t_logistic'), 2)}): threshold saving {ci(ps.get('threshold_saving'), ps.get('threshold_saving_lo'), ps.get('threshold_saving_hi'))}, "
                 f"model saving {ci(ps.get('model_saving'), ps.get('model_saving_lo'), ps.get('model_saving_hi'))} at 5:1.")
    A(table("Threshold saving against model saving at nine false-negative to false-positive cost ratios $r$. "
            "Threshold saving: expected test cost of the frozen plain logistic at the 0.5 cut-off minus its cost at "
            "the cut-off chosen on training out-of-fold scores. Model saving: the logistic at its own cut-off minus "
            "the frozen XGBoost at its own cut-off. Brackets are 95\\,\\% paired-bootstrap intervals. "
            f"Source: {tt('results/threshold_vs_model_saving.csv')} (Table~\\ref{{tab:saving}} in the body is the "
            "six-ratio excerpt).",
            "tab:a-saving", "lcccccc",
            ["$r$", "$1/(1+r)$", "Logit cut-off", "XGB cut-off", "Logit cost @0.5", "Threshold saving", "Model saving"],
            rows, note=note, tabcolsep="3.5pt"))
    A("")

# ------------------------------------------------------------------ A5b Platt recalibration of the weighted / resampled models
rc_path = os.path.join(RESULTS, "recalibration.csv")
if os.path.exists(rc_path):
    rc = pd.read_csv(rc_path)
    arrow = lambda x, y, nd=3: f"{num(x, nd)} $\\to$ {num(y, nd)}"
    RC_LABEL = {"logistic_balanced": "Logistic, class-weighted", "imb_logistic_balanced": "Logistic (C=1), class-weighted",
                "imb_logistic_smote": "Logistic (C=1), SMOTE in folds", "imb_logistic_under": "Logistic (C=1), under-sampled",
                "imb_rf_balanced": "Forest, class-weighted", "imb_rf_smote": "Forest, SMOTE in folds",
                "imb_rf_under": "Forest, under-sampled", "xgboost_weighted": "XGBoost, class-weighted",
                "mlp_weighted": "MLP, balanced sample weights"}
    rows = [[esc(RC_LABEL.get(r_.slug, r_.model)), arrow(r_.test_mean_p_raw, r_.test_mean_p_platt),
             arrow(r_.test_brier_raw, r_.test_brier_platt, 4) + f" ({num(r_.test_brier_twin, 4)})",
             arrow(r_.t_cost_raw, r_.t_cost_platt, 2),
             f"{num(r_.test_cost_at_closed_form_raw, 0)} $\\to$ {num(r_.test_cost_at_closed_form_platt, 0)}"]
            for r_ in rc.itertuples()]
    A(table("Platt recalibration of the nine class-weighted or resampled models. The Platt map is fitted on each "
            "model's training out-of-fold scores; the 5:1 cut-off is re-chosen on cross-fitted training scores; each "
            "recalibrated version was then registered, scored on the test set (19 October) and frozen; a 6 October "
            "review had already applied the same maps to the test scores, but Platt was the method committed in the "
            "30 September proposal. Columns: mean predicted "
            "probability on test (prevalence 0.221); test Brier score, with the unweighted twin in brackets; 5:1 cut-off; "
            "test cost at the closed-form threshold 1/6.",
            "tab:a-recal", "L{0.28\\linewidth}cccc",
            ["Model", "Mean $p$", "Brier (twin)", "5:1 cut-off", "Cost at $1/6$"], rows, tabcolsep="4pt"))
    A("")

# ------------------------------------------------------------------ A5c clustering: K selection, profiles, incremental value
ks_path = os.path.join(RESULTS, "kmeans_selection_v2.csv")
kp_path = os.path.join(RESULTS, "kmeans_profiles_v2.csv")
us = load_json("unsupervised_summary_v2.json")
if os.path.exists(ks_path) and us:
    ks = pd.read_csv(ks_path)
    k_final = int(us.get("k_final", 0))
    bold = lambda s, on: f"\\textbf{{{s}}}" if on else s
    rows = [[bold(str(int(r_.K)), int(r_.K) == k_final), bold(num(r_.silhouette, 3), int(r_.K) == k_final),
             num(r_.boot_ari_mean, 3), num(r_.boot_ari_p5, 3), num(r_.min_cluster_jaccard, 3)]
            for r_ in ks.itertuples()]
    pcv = us.get("pca_cum_var", {})
    ba = us.get("bootstrap_ari", {})
    ward = us.get("ward_vs_kmeans_ari", [])
    A(table(f"Choice of $K$ for K-means on the first 10 principal components of the {xj.get('extra', {}).get('n_features_after_onehot', 32)} "
            f"preprocessed columns ({100 * pcv.get('10', 0):.1f}\\,\\% of the variance; 2, 5 and 15 components carry "
            f"{100 * pcv.get('2', 0):.0f}, {100 * pcv.get('5', 0):.0f} and {100 * pcv.get('15', 0):.0f}\\,\\%), training "
            "partition only. RUAN Hangjing's analysis of 30 September 2026 (\\texttt{kmeans\\_selection\\_v2.csv}), "
            "reproduced on 21 October (\\texttt{kmeans\\_selection\\_repro.csv}, identical). Bold: the $K$ with the highest "
            "silhouette, chosen by the rule fixed in advance.",
            "tab:a-kselect", "ccccc",
            ["$K$", "Silhouette", "Bootstrap ARI (mean)", "ARI 5th pct.", "Min.\\ cluster Jaccard"], rows,
            note=(f"Two bootstrap runs, not one. The three right-hand columns come from the selection sweep over $K$ "
                  f"(for $K={k_final}$: ARI {num(ks.loc[ks.K == k_final, 'boot_ari_mean'].iloc[0], 3)}, minimum cluster "
                  f"Jaccard {num(ks.loc[ks.K == k_final, 'min_cluster_jaccard'].iloc[0], 3)}). The stability figures quoted "
                  f"in the body and in Table~\\ref{{tab:a-kprofiles}} come from the separate 100-refit stability run of the "
                  f"chosen $K={k_final}$ ({tt('unsupervised_summary_v2.json')}): ARI {num(ba.get('mean'), 3)} "
                  f"[{num(ba.get('lo'), 3)}, {num(ba.get('hi'), 3)}], per-cluster Jaccard "
                  + "/".join(num(j, 3) for j in us.get("cluster_jaccard", [])) +
                  f"; agreement with Ward's hierarchical clustering at the same $K$: ARI {num(min(ward), 2)}--{num(max(ward), 2)} "
                  "over five bootstrap samples (moderate).")))
    A("")
if os.path.exists(kp_path) and us:
    kp = pd.read_csv(kp_path)
    rows = [[str(int(r_.cluster)), f"{int(r_.n_clients):,}", f"{100 * r_.share:.1f}\\%", f"{100 * r_.default_rate:.1f}\\%",
             f"{r_.avg_limit:,.0f}", num(r_.avg_months_delayed, 2), num(r_.avg_util, 2), num(r_.jaccard_stability, 3)]
            for r_ in kp.itertuples()]
    inc = us.get("incremental", {})
    da = inc.get("auc: logistic + clusters - logistic", {})
    dap = inc.get("ap: logistic + clusters - logistic", {})
    ma = us.get("mean_auc", {})
    sgn = snum
    A(table(f"Profiles of the {len(kp)} K-means segments on the training partition (RUAN Hangjing, 30 September 2026, "
            "\\texttt{kmeans\\_profiles\\_v2.csv}; reproduced 21 October). Jaccard: per-cluster stability over the bootstrap "
            "refits. The default rates differ because the inputs contain the delay and limit variables; the segments are "
            "descriptive, not economically distinct.",
            "tab:a-kprofiles", "cccccccc",
            ["Cluster", "$n$", "Share", "Default rate", "Mean limit (NT\\$)", "Months delayed", "Utilisation", "Jaccard"],
            rows, tabcolsep="4pt",
            note=(f"Incremental value of the one-hot segment label as an extra predictor of the plain logit, $5\\times5$ repeated CV "
                  f"on the training rows with the Nadeau--Bengio corrected interval: $\\Delta$AUC {sgn(da.get('mean', 0))} "
                  f"[{sgn(da.get('lo', 0))}, {sgn(da.get('hi', 0))}], $\\Delta$AP {sgn(dap.get('mean', 0))} "
                  f"[{sgn(dap.get('lo', 0))}, {sgn(dap.get('hi', 0))}]; adoption rule (both lower bounds $> 0$) not met, "
                  f"adopted = {'True' if us.get('adopted') else 'False'}, so the test set was not consulted for it. "
                  f"The label used as a score alone has AUC {num(ma.get('auc|cluster label as score'), 3)} against "
                  f"{num(ma.get('auc|N_MONTHS_DELAYED alone'), 3)} for months-in-arrears alone; the logit on the 10 PCs alone "
                  f"changes AUC by {sgn(inc.get('auc: logistic on 10 PCs - logistic', {}).get('mean', 0), 3)} against the logit on the raw columns.")))
    A("")

# ------------------------------------------------------------------ A6 importance, four measures
shap_path = os.path.join(RESULTS, "shap_importance.csv")
if os.path.exists(shap_path):
    s = pd.read_csv(shap_path)
    want = ["feature", "mean_abs_shap", "shap_rank", "mdi", "mdi_rank",
            "perm_auc_drop", "perm_rank", "oob_perm_auc_drop", "oob_perm_rank"]
    have = [c for c in want if c in s.columns]
    s = s[have].sort_values("mean_abs_shap", ascending=False)
    rows = [[tt(r_.feature), num(r_.mean_abs_shap, 4), str(int(r_.shap_rank)),
             num(getattr(r_, "mdi", None), 4), str(int(getattr(r_, "mdi_rank", 0) or 0)),
             num(getattr(r_, "perm_auc_drop", None), 4), str(int(getattr(r_, "perm_rank", 0) or 0)),
             num(getattr(r_, "oob_perm_auc_drop", None), 4),
             str(int(getattr(r_, "oob_perm_rank", 0) or 0))] for r_ in s.itertuples()]
    rf_sp = (load_json("random_forest.json") or {}).get("extra", {}).get("spearman", {})
    sh_sp = (load_json("shap_summary.json") or {}).get("spearman_vs_other_measures", {})
    sp_note = None
    if rf_sp and sh_sp:
        sp_note = ("Spearman rank correlation over the 29 predictors (\\texttt{random\\_forest.json} "
                   "\\texttt{extra.spearman}, \\texttt{shap\\_summary.json} \\texttt{spearman\\_vs\\_other\\_measures}): "
                   f"MDI vs out-of-bag permutation {num(rf_sp.get('mdi_vs_oob_permutation'), 3)}, "
                   f"MDI vs training-set permutation {num(rf_sp.get('mdi_vs_train_permutation'), 3)}, "
                   f"training-set vs out-of-bag permutation {num(rf_sp.get('train_vs_oob_permutation'), 3)}; "
                   f"mean $|$SHAP$|$ vs MDI {num(sh_sp.get('mdi'), 3)}, vs training-set permutation "
                   f"{num(sh_sp.get('perm_train'), 3)}, vs out-of-bag permutation {num(sh_sp.get('perm_oob'), 3)}.")
    A(table("Four ways to rank the same 29 predictors: mean $|$SHAP$|$ on the tuned XGBoost over the "
            "20,975 training clients, and "
            "the random forest's impurity decrease, training-set permutation and out-of-bag "
            "permutation importances. Ranks, not magnitudes, are comparable across columns.",
            "tab:a-importance", "L{0.20\\linewidth}" + "cc" * 4,
            ["Feature", "mean $|$SHAP$|$", "rank", "MDI", "rank", "perm.\\ (train)", "rank",
             "perm.\\ (OOB)", "rank"], rows, tabcolsep="3.5pt", note=sp_note, note_ragged=True))
    A("")

# ------------------------------------------------------------------ A7 refit without a column (OOB only)
rd_path = os.path.join(RESULTS, "rf_refit_drop_column.csv")
if os.path.exists(rd_path):
    rd = pd.read_csv(rd_path, comment="#")
    if {"feature", "oob_auc", "oob_loss"} <= set(rd.columns):
        rows = [[esc(r_.feature) if str(r_.feature).startswith("(") else tt(r_.feature),
                 num(r_.oob_auc, 4), num(r_.oob_loss, 5),
                 num(getattr(r_, "oob_perm_drop", None), 4)] for r_ in rd.itertuples()]
        A(table("Refitting the tuned random forest without one column, for the six columns with the "
                "largest out-of-bag permutation drop. Permutation measures how much the fitted forest "
                "leans on a column; refitting measures what is lost if the column is never available. "
                "Out-of-bag AUC on the training rows; the test set is not used.",
                "tab:a-refit", "L{0.26\\linewidth}ccc",
                ["Column removed", "OOB AUC", "OOB AUC lost", "OOB perm.\\ drop"], rows))
        A("")

# ------------------------------------------------------------------ figures
FIGS = [
    ("Exploratory analysis", [
        ("eda_target_distribution.png", "Class balance of the cleaned sample and of the stratified training and test partitions."),
        ("eda_default_rate_by_pay1.png", "Default rate by the most recent repayment status, training partition."),
        ("eda_default_rate_by_groups.png", "Training-partition default rate by credit-limit quintile, age band, education, marriage and sex."),
        ("eda_default_rate_util_delay.png", "Default rate by utilisation decile and by number of months in arrears, training partition."),
        ("eda_correlation_heatmap.png", "Correlation heat-map of the 29 predictors and the target."),
        ("eda_univariate_strength.png", "Univariate strength of every predictor, as an oriented AUC."),
        ("eda_distributions_by_default.png", "Distributions of the main numeric features by class, training partition."),
    ]),
    ("Model comparison", [
        ("all_roc.png", "Test-set ROC curves of the headline models."),
        ("all_pr.png", "Test-set precision--recall curves; the horizontal line is the prevalence."),
        ("all_auc_bar_full.png", "Test AUC of every model variant fitted, ordered by CV AUC, with the benchmark marked."),
        ("calibration.png", "Calibration curves of the four models with the highest CV AUC: predicted against observed default rate by decile of score."),
        ("threshold_curves.png", "Recall, precision and F1 against the decision threshold."),
        ("cost_curves.png", "Expected cost against the decision threshold at a 5:1 ratio."),
        ("cost_sensitivity.png", "Cost-minimising threshold and cost as the cost ratio varies from 1:1 to 20:1."),
        ("recalibration.png", "Reliability of the nine weighted or resampled models before and after Platt recalibration, on training out-of-fold scores."),
    ]),
    ("Interpretation", [
        ("rf_importance.png", "Random forest importances: impurity decrease, training-set permutation and out-of-bag permutation, side by side."),
        ("shap_beeswarm.png", "TreeSHAP beeswarm for the tuned XGBoost over training clients."),
        ("shap_dependence_pay1.png", "TreeSHAP dependence plot for the most recent repayment status."),
        ("shap_dependence_util_mean.png", "TreeSHAP dependence plot for mean utilisation."),
        ("shap_waterfall.png", "One client decomposed: base value plus each feature's contribution."),
        ("pdp_main.png", "Partial dependence with ICE curves for the four features that matter."),
        ("pdp_interaction.png", "Two-way partial dependence; cells with no training client are crossed."),
        ("monotone_pay1.png", "Predicted probability against the most recent repayment status, with and without monotone constraints."),
        ("logistic_coefficients.png", "Standardised logistic coefficients."),
        ("logistic_l1_path.png", "Lasso path: order of entry of the predictors as the penalty is relaxed."),
        ("xgb_importance.png", "XGBoost gain-based importance."),
        ("tree_pruned.png", "The pruned decision tree, top levels."),
    ]),
    ("Class imbalance and the leakage demonstration", [
        ("imbalance_comparison.png", "Resampling strategies applied inside the folds: AUC is unchanged, the operating point moves."),
        ("imbalance_threshold_shift.png", "Where each resampling strategy puts the tuned threshold."),
        ("imbalance_leakage_demo.png", "Negative control: SMOTE applied before the split inflates the apparent test AUC."),
    ]),
    ("Model selection diagnostics", [
        ("knn_cv_auc_vs_k.png", "KNN: cross-validated AUC against $K$."),
        ("tree_cv_auc_vs_alpha.png", "Cost-complexity pruning path."),
        ("rf_auc_vs_ntrees.png", "Random forest and bagging: out-of-bag AUC on the training rows against the number "
                                 "of trees (the tree count, 500, was fixed in advance)."),
        ("gbm_learning_curve.png", "Gradient boosting: cross-validated AUC against the number of rounds."),
        ("xgb_learning_curve.png", "XGBoost learning curves at two learning rates."),
        ("svm_rbf_cv_heatmap.png", "RBF support-vector machine: cross-validated AUC over $C$ and $\\gamma$."),
        ("mlp_loss_curve.png", "Neural network: training loss and validation AUC by epoch."),
        ("mlp_overfit_epochs.png", "Over-parameterised network: the over-fitting diagnostic."),
        ("pca_scree.png", "PCA scree plot and cumulative variance explained."),
        ("pca_loadings.png", "Loadings of the first three principal components."),
        ("kmeans_selection_repro.png", "K-means on the first 10 principal components: silhouette, bootstrap ARI and "
         "minimum cluster Jaccard against $K$ (Table~\\ref{tab:a-kselect}); reproduction on 21 October of RUAN "
         "Hangjing's 30 September analysis, training partition only."),
    ]),
]
n = 0
for title, items in FIGS:
    A("\\clearpage")
    A(f"\\subsection{{{title}}}")
    A("")
    for path, cap in items:
        n += 1
        # Label from the file name, so the body can \ref a figure by what it is
        # (fig:pdp-main, fig:shap-beeswarm, ...) instead of by an appendix ordinal.
        A(figure(path, cap, "fig:" + os.path.splitext(path)[0].replace("_", "-")))
        A("")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(L)} blocks, {n} figures")
