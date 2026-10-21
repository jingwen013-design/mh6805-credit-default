"""
make_report_appendix.py — generate report/sec_appendix.tex from results/.   Owner: shared.

The course guideline exempts appendices from the report's 7-page limit, so everything the body
cannot hold lives here, and it is GENERATED rather than typed: every table is read out of
results/ at build time, so the appendix cannot drift away from the artefacts it describes.

Writes report/sec_appendix.tex (LaTeX, \\input from report.tex after \\appendix).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, ROOT
from mine_v2_utils import corrected_t_ci      # pure function: Nadeau-Bengio corrected t interval

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


def isnum(x):
    try:
        return x is not None and not pd.isna(float(x))
    except (TypeError, ValueError):
        return False


def pct(x, nd=1, sign=False):
    """A share as a percentage: 0.3394 -> 33.9\\%."""
    if not isnum(x):
        return "---"
    v = 100 * float(x)
    return (f"${v:+.{nd}f}$" if sign else f"{v:.{nd}f}") + "\\%"


def pnum(x, nd=1):
    """A share as a bare percentage number (the % sign is in the column header): 0.3394 -> 33.9."""
    return f"{100 * float(x):.{nd}f}" if isnum(x) else "---"


def mnum(x):
    """An integer with a thousands separator inside math mode: 20975 -> 20{,}975 (no space after the comma)."""
    return f"{int(x):,}".replace(",", "{,}")


def k5(k):
    """Number of folds (of five) in which a paired difference is positive."""
    return f"{int(round(float(k)))}/5" if isnum(k) else "---"


def ci(lo, hi, nd=4, scale=1.0):
    return f"[{snum(scale * float(lo), nd)}, {snum(scale * float(hi), nd)}]"


def stack(*lines, align="c", pos="t"):
    """Several short lines in one cell (a nested tabular), so wide intervals do not widen the table."""
    lines = [l for l in lines if l not in (None, "")]
    return f"\\begin{{tabular}}[{pos}]{{@{{}}{align}@{{}}}}" + " \\\\ {}".join(lines) + "\\end{tabular}"


def nb_cell(mean, lo, hi, k=None, nd=4, scale=1.0, extra=None):
    """Paired difference: mean (k/5) on the first line, the Nadeau-Bengio interval on the second."""
    if not isnum(mean):
        return "---"
    first = snum(scale * float(mean), nd) + (f", {k5(k)}" if k is not None else "")
    return stack(first, ci(lo, hi, nd, scale) if isnum(lo) else None, extra)


def nb_text(mean, lo, hi, k=None, nd=4, scale=1.0):
    """The same, inline for notes: +0.0082 [+0.0021, +0.0143], 5/5."""
    s = f"{snum(scale * float(mean), nd)} {ci(lo, hi, nd, scale)}"
    return s + (f", {k5(k)}" if k is not None else "")


_FLOATS = [0]            # floats emitted so far; every 6th is followed by \clearpage so the
                         # 18 tables + 41 figures never overflow LaTeX's float queue


def _flush():
    _FLOATS[0] += 1
    return "\n\\clearpage" if _FLOATS[0] % 6 == 0 else ""


def table(caption, label, colspec, header, rows, small=True, note=None, tabcolsep=None, footnotesize=False,
          placement="!htbp", note_ragged=False, pre_header=None, after=None):
    # "!" lets consecutive tables share a page (the appendix is tables only, so the usual
    # float-fraction limits only produce half-empty pages and float-only pages).
    # pre_header: raw header lines (e.g. grouped column titles with \cmidrule) set above `header`.
    # rows: a list of cells is joined with " & "; a plain string is emitted as is (\midrule, \multicolumn rows).
    # after: raw LaTeX set below the tabular and above the note (a second small block of the same table).
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
    for h in (pre_header or []):
        out.append("    " + h)
    out.append("    " + " & ".join(header) + r" \\")
    out.append(r"    \midrule")
    for r_ in rows:
        out.append("    " + (r_ if isinstance(r_, str) else " & ".join(r_) + r" \\"))
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    if after:
        out.append(after)
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
WIDE_FIGS = {"fold_stability.png", "ablation_summary.png", "fairness_flag_rates.png"}
# Wide, short multi-panel figures of the post-freeze checks: full text width, ordinary placement.


def figure(path, caption, label, width="0.92", placement="htbp"):
    if not os.path.exists(os.path.join(FIGDIR, path)):
        return f"% MISSING FIGURE: {path}"
    if path in FULLPAGE_FIGS:
        width, placement = "1.0", "p"
    elif path in WIDE_FIGS:
        width = "1.0"
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
  "on the body. Section~\\ref{sec:a-main} documents the data, the protocol and the frozen models; "
  "Section~\\ref{sec:a-postfreeze} collects post-freeze checks on the training rows, with their figures in "
  "Section~\\ref{sec:a-postfreeze-figs}; \\texttt{results/TEST\\_SET\\_LOG.md} dates every read of the test rows "
  "(Table~\\ref{tab:a-testlog}); the figures of the data and the frozen models follow from "
  "Section~\\ref{sec:a-figs}.")
A("")
A("\\subsection{Data, protocol and frozen models}\\label{sec:a-main}")
A("")

# ------------------------------------------------------------------ A1 variables
meta = json.load(open(os.path.join(ROOT, "data", "processed", "split_meta.json")))
groups = [
    ("Credit line", ["LIMIT_BAL"], "numeric", "credit limit granted (NT\\$), individual plus family"),
    ("Demographics", ["SEX", "EDUCATION", "MARRIAGE", "AGE"], "3 categorical, 1 numeric",
     "gender; graduate / university / high school / others; married / single / others; age in years. "
     "EDUCATION and MARRIAGE: an a-priori mapping of codes the UCI dictionary does not list "
     "($\\{0, 5, 6\\} \\rightarrow$ others; $0 \\rightarrow$ others)"),
    ("Repayment status", ["PAY_1", "\\dots", "PAY_6"], "ordinal ($-2 \\dots 8$)",
     "monthly status Sep $\\rightarrow$ Apr 2005: $-2$ no consumption$^{\\dagger}$, $-1$ paid in full, "
     "$0$ revolving credit$^{\\dagger}$, $k \\ge 1$ = $k$ months late; $^{\\dagger}$not defined in the UCI "
     "documentation (a widely used reading)"),
    ("Bill statements", ["BILL_AMT1", "\\dots", "BILL_AMT6"], "numeric",
     "monthly bill amount Sep $\\rightarrow$ Apr (NT\\$); may be negative"),
    ("Payments", ["PAY_AMT1", "\\dots", "PAY_AMT6"], "numeric",
     "amount paid in each month Sep $\\rightarrow$ Apr (NT\\$)"),
    ("Engineered", ["UTIL_1", "UTIL_MEAN", "PAY_RATIO_MEAN", "N_MONTHS_DELAYED", "MAX_DELAY",
                    "BILL_TREND"], "numeric",
     "utilisation = bill over limit (September; six-month mean); \\texttt{PAY\\_RATIO\\_MEAN} = mean over "
     "months of the payment in month $k$ over the bill of the same month $k$ (bills $\\le 0$ skipped), capped "
     "at 5, set to 0 when all six bills are non-positive (a payment usually settles the previous statement); "
     "months with status $\\ge 1$; worst status, floored at 0; (September bill $-$ April bill) over the limit"),
    ("Target", ["DEFAULT"], "binary",
     f"1 = default on the October 2005 payment ({100 * meta['default_rate_train']:.1f}\\% in training)"),
]
rows = [[esc(g), ",\\allowbreak\\ ".join(tt(v) if not v.startswith("\\") else v for v in vs), t, m]
        for g, vs, t, m in groups]
A(table(f"Variables in the cleaned data set: {len(meta['features'])} predictors and the target, "
        f"{meta['n_after_dedup']:,} clients after removing {meta['n_duplicates_dropped']} rows identical to "
        f"another row on every column except the ID (the key included the label). The monthly variables are "
        f"dated; the credit limit and the demographics are undated snapshots.",
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
    # Round from the implied integer count, not from the 4-dp stored rate (0.3205 x 5,429 = 1,740 defaulters
    # = 32.05%, which the float 0.3205 would print as 32.0%; the body quotes 32.1%).
    _r = lambda d: f"{100 * round(d['n'] * d['rate']) / d['n']:.1f}\\%"
    rows.append([l[0], f"{l[1]['n']:,}" if l[1] else "", _r(l[1]) if l[1] else "",
                 r_[0], f"{r_[1]['n']:,}" if r_[1] else "", _r(r_[1]) if r_[1] else ""])
A(table(f"Default rate along the two strongest credit-risk gradients, training partition only "
        f"($n = {mnum(eda['class_balance']['train']['n'])}$; overall rate "
        f"{100 * eda['default_rate_by']['overall_train']:.1f}\\%). Source: \\texttt{{results/eda\\_summary.json}}. "
        f"A status is compared within a month, not across months: code 1 (one month late) is recorded almost only "
        f"in September and mixes clients with no positive September bill and clients rolling back from two months "
        f"late (Table~\\ref{{tab:a-paycodes-rates}}).",
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


def tuning_budget(slug):
    """(rows the grid was scored on, number of settings searched, selection rule) from results/<slug>.json."""
    d = load_json(f"{slug}.json") or {}
    ex = d.get("extra", {}) or {}
    n_rows = meta["n_train"]
    grid = ex.get("grid")
    if isinstance(grid, list):
        n_set = len(grid)
    elif isinstance(grid, dict):
        n_set = grid.get("n_settings")
    elif slug == "tree_pruned":
        n_set = len(ex.get("alpha_grid", [])) * len(ex.get("max_depth_grid", []))
    elif "tuning" in ex:
        t = ex["tuning"]
        n_set = len(t.get("grid_C", [])) * len(t.get("grid_gamma", [1])) * len(t.get("grid_class_weight", [1]))
    elif "grid_C" in ex:
        n_set = len(ex["grid_C"])
    elif "cv_grid" in ex:
        n_set = len(ex["cv_grid"])
    else:
        n_set = 1
    if slug == "gradient_boosting":
        n_rows = ex.get("grid_subsample_rows", n_rows)
    if "tuning" in ex and ex["tuning"].get("subsample_rows"):
        n_rows = ex["tuning"]["subsample_rows"]
    rule = "1-SE" if "one-standard-error" in str(ex.get("selection_rule", "")) else ("none" if n_set == 1 else "argmax")
    return n_rows, n_set, rule


SVM_SEL = load_json("svm_selection_statistics.json") or {}


def svm_per_fold_mean(slug):
    """Mean of the five shared-fold out-of-fold AUCs of a subsample-tuned SVM (svm_selection_statistics.json)."""
    return (SVM_SEL.get(slug) or {}).get("per_fold_mean_oof_auc")


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
        n_rows, n_set, rule = tuning_budget(r_.slug)
        sel = svm_per_fold_mean(r_.slug)
        sel_txt = num(sel, 4) + "$^{\\dagger}$" if sel is not None else num(r_.select_auc, 4)
        rows.append([esc(r_.name), tr[0], tr[1], f"{int(n_rows):,}", str(n_set), rule, sel_txt])
    A(table("Hyper-parameter grids searched for the headline models and the setting chosen, read from "
            "\\texttt{results/<slug>.json}. Every grid is scored by 5-fold stratified cross-validation on the "
            "training rows shown (seed 2607526); the test set plays no part. Rule: argmax of CV AUC, or the "
            "one-standard-error rule (1-SE). The last column is the training-set selection score of the chosen "
            "setting. The reported CV is the maximum over the grid on the same folds, an upper estimate for close "
            "winners. Gradient boosting was tuned on a training subsample, not fixed a priori; only the monotone "
            "constraint vector and bagging's inherited leaf size were fixed a priori.",
            "tab:a-tuning", "L{0.12\\linewidth}L{0.30\\linewidth}L{0.21\\linewidth}cccc",
            ["Model", "Grid searched", "Chosen", "Rows", "Settings", "Rule", "Sel.\\ AUC"], rows,
            tabcolsep="2.5pt", footnotesize=True,
            note="$^{\\dagger}$Tuned on a 6,000-row training subsample; the score shown is the mean of the five "
                 "shared-fold out-of-fold AUCs of the chosen setting refitted on the full training partition "
                 f"(pooled out-of-fold {num((SVM_SEL.get('svm_rbf') or {}).get('pooled_oof_auc'), 3)}, subsample CV "
                 f"{num(((SVM_SEL.get('svm_rbf') or {}).get('subsample_cv_auc') or {}).get('value'), 3)}; "
                 "\\texttt{results/svm\\_selection\\_statistics.json}). Post-freeze checks of these grids: "
                 "Table~\\ref{tab:a-tuningchecks}."))
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
    pb = ms["cost_of_the_constraint"]["paired_bootstrap"]
    mrc = load_json("monotone_respec_cv.json") or {}
    price = ((mrc.get("ladder") or {}).get("mono_frozen_numeric - free_numeric (price of the frozen constraint)") or {}).get("auc")
    price_txt = (f"{snum(price['mean'])} CV AUC ({k5(price['folds_positive'])}, {ci(price['nb_lo'], price['nb_hi'])})"
                 if price else f"{snum(ms['cost_of_the_constraint']['delta_cv_auc'])} CV AUC")
    A(table("The monotone constraint vector of the constrained XGBoost, fixed a priori from credit logic "
            f"and read back from the fitted booster: {tt(cv_['string']).replace(',', ',\\allowbreak ')} over the "
            f"{len(cv_['columns'])} preprocessed columns (one-hot categories count once per raw variable here). "
            "Hyper-parameters are inherited unchanged from the tuned unconstrained model.",
            "tab:a-monotone", "L{0.30\\linewidth}L{0.54\\linewidth}c",
            ["Sign", "Variables", "$n$"], rows,
            note="The constraint imposes an order on the non-delinquent codes $-2$/$-1$/$0$ and a rising utilisation "
                 "that the training data contradict (Table~\\ref{tab:a-monotone-respec}, which also gives the "
                 "wrong-direction steps of the unconstrained booster under a consistent sweep). Its price is "
                 f"{price_txt}; test AUC {snum(ms['cost_of_the_constraint']['delta_test_auc'])} as a check (paired "
                 f"bootstrap, $B={pb['B']}$: mean {snum(pb['auc_diff_mean'])}, 95\\,\\% interval "
                 f"[{snum(pb['auc_diff_2_5'])}, {snum(pb['auc_diff_97_5'])}])."))
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

    # -------------------------------------------------------------- every read of the test rows (TEST_SET_LOG.md)
    # A record, not a check, so it sits right after the freeze timeline. The rows condense the sections of
    # results/TEST_SET_LOG.md (section ids in the first column); the counts are read from the manifest.
    n_r1, n_ruan, n_platt, n_neg = (len(groups_[k]) for k in ("r1", "ruan", "platt", "neg"))
    log_path = os.path.join(RESULTS, "TEST_SET_LOG.md")
    log_txt = open(log_path, encoding="utf-8").read() if os.path.exists(log_path) else ""
    for sec_id in ("A1", "A2", "A3", "B1", "B12", "## C", "## D", "## F"):
        if sec_id not in log_txt:
            raise SystemExit(f"TEST_SET_LOG.md has no section {sec_id}: update the tab:a-testlog rows")
    rows = [
        ["A1", "16--18 Sep", f"Round 1: {n_r1} configurations and {n_neg} negative control (SMOTE before the split, "
                             "which pools the real test rows), scored before any freeze existed and registered on "
                             "19 October; the forest and bagging were scored twice (16 and 18 September) with the same "
                             "configuration",
         "Modelling choices: no (hyper-parameters and cut-offs from training CV / out-of-fold scores). Until 19 October "
         "test AUC set two presentation choices of the aggregation: the bootstrap's reference model and the four "
         "models drawn in three figures"],
        ["A2", "30 Sep", f"RUAN Hangjing's second pass: {n_ruan} re-fits of frozen configurations (\\texttt{{knn\\_v2}}, "
                         "\\texttt{logistic\\_plain\\_v2}), with reporting reads (cost sensitivity, saving bootstrap)",
         "No: selections by nested CV on training rows"],
        ["A3", "19 Oct", f"{n_platt} Platt-recalibrated variants, each registered before its scoring; the same maps had "
                         "been applied to the test scores in the 6 October review (B8)",
         "No: Platt was pre-committed in the 30 September proposal"],
        ["B1--B12", "16 Sep--6 Oct", "Reads outside the manifest: a plain-logit reference with $C = 1$ (B1); the "
                                     "hinge-loss linear SVM (B2); test AUC against the number of trees (B4); the "
                                     "monotone model's twin and bootstrap (B5); six drop-one-column forests and a third "
                                     "forest scoring (B6); RUAN's proposal-stage logit (B7); the 6 October review's "
                                     "recalibration maps and bootstraps (B8); descriptive reads in four scripts (B9--B12)",
         "Once: the linear SVM's hinge-to-squared-hinge loss swap (B2) came after the hinge model's test score; its "
         "stated basis is training CV. Otherwise no"],
        ["C", "16 Sep--21 Oct", "Reporting reads of frozen predictions: aggregation (curves, bootstrap), "
                                "threshold against model saving, cost sensitivity (re-run on the 0.01 cut-off grid on "
                                "21 October), the split check, slide figures and the 21 October fresh-clone reproduction",
         "No: reporting only"],
        ["D", "16 Sep--21 Oct", "Test features without labels: reproduction checks of frozen predictions, feature counts "
                                "in Table~\\ref{tab:a-quality}, calls that open the test file and discard it",
         "No"],
        ["F", "21 Oct", "Activity that did not read the test set: the post-freeze checks of "
                        "Section~\\ref{sec:a-postfreeze} and the R cross-check (Table~\\ref{tab:a-rcheck})",
         "---"],
    ]
    A(table("Every read of the test rows, condensed from \\texttt{results/TEST\\_SET\\_LOG.md} (section ids in the first "
            "column). Each configuration was frozen at its first scoring; the manifest of Table~\\ref{tab:a-freeze} lists "
            "configurations, this log lists reads. The only choice made after a test read is the linear SVM's loss "
            "swap, whose stated basis is training CV; until 19 October test AUC set two presentation choices of the "
            "aggregation.",
            "tab:a-testlog", "L{0.08\\linewidth}L{0.11\\linewidth}L{0.46\\linewidth}L{0.30\\linewidth}",
            ["Section", "When", "What", "Influenced a choice?"], rows, tabcolsep="3pt", footnotesize=True))
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
        rows.append([f"KNN, RUAN's 16-setting grid ($K$, weights tuned in-fold; $K={bp2.get('n_neighbors')}$ "
                     f"{esc(bp2.get('weights'))} on the full training set)",
                     f"{num(ex2.get('nested_cv_auc'), 4)} $\\pm$ {num(ex2.get('nested_cv_auc_std'), 4)}", "---",
                     num(oc.get("brier"), 4), num(oc.get("cal_slope"), 3), "---", "---"])
    kfg = load_json("nested_cv_knn_frozen_grid.json")
    if kfg:
        of = kfg.get("outer_folds", [])
        ks = pd.Series([o["chosen_K"] for o in of]).value_counts()
        n_dist = sum(1 for o in of if o["chosen_weights"] == "distance")
        choice = "; ".join(f"$K = {int(k)}$ in {int(c)}" for k, c in ks.items()) + f"; distance weights in {n_dist}"
        rows.append([f"KNN, the frozen {kfg['grid']['n_settings']}-setting grid (outer-fold choices: {choice} of "
                     f"{len(of)})",
                     f"{num(kfg['nested_cv_auc'], 4)} $\\pm$ {num(kfg['nested_cv_auc_std'], 4)}",
                     "---", "---", "---", "---", "---"])
    rep = load_json("nested_cv_summary_repro.json") or {}
    lg = rep.get("logistic", {})
    note = ("Nested CV: outer loop = the project's five folds, inner loop = a 5-fold grid search inside each outer "
            "training fold, so the AUC is that of the whole tuning procedure. Columns: nested AUC $\\pm$ its "
            "standard error over the outer folds (for KNN, the standard deviation over the outer folds, ddof 0"
            + (f"; with ddof 1, as in Table~\\ref{{tab:headline}}, the 24-setting run gives "
               f"{num(kfg['nested_cv_auc_sd_ddof1'], 4)}" if kfg and 'nested_cv_auc_sd_ddof1' in kfg else "")
            + "), nested average precision, "
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
                     f"and {c17:,} at 0.17 ({100 * (c17 - c15) / c15:+.2f}\\,\\%): a flat stretch of the cost curve, and a "
                     "different cross-fit from the nested one, which is why the two costs at 0.17 differ.")
    # Moved here from the sec_methods.tex footnote (POST_REVIEW 9.5): RUAN's 30 Sep refit of the same logit.
    lv2, lv1 = load_json("logistic_plain_v2.json"), load_json("logistic.json")
    if lv2 and lv1:
        note += (f" RUAN's 30 September refit of the same plain logit ({tt('logistic_plain_v2.json')}) gave a test cost of "
                 f"{num(lv2['at_0_5']['cost'])} at 0.5 and a training out-of-fold cut-off of "
                 f"{num(lv2['at_tuned_cost']['threshold'], 2)} costing {num(lv2['at_tuned_cost']['cost'])}, against "
                 f"{num(lv1['at_0_5']['cost'])} / {num(lv1['at_tuned_cost']['threshold'], 2)} / "
                 f"{num(lv1['at_tuned_cost']['cost'])} for the 16 September scoring reported throughout; the proposal's "
                 "saving table is that refit (note to Table~\\ref{tab:a-saving}).")
    note += (" The nested-CV columns use training rows only; the test costs in this note are those of the frozen 16 and "
             "30 September scorings (Table~\\ref{tab:a-testlog}, A1--A2).")
    A(table("Nested cross-validation of the KNN and logistic families, RUAN Hangjing's second pass of 30 September "
            f"2026 ({tt('logistic_nested_cv_v2.csv')}, {tt('knn_v2.json')}), plus the KNN re-tuned inside each outer "
            f"fold over the same 24-setting grid as the frozen model ({tt('nested_cv_knn_frozen_grid.json')}, 21 October). "
            "The earlier 16-setting KNN run is RUAN's 30 September method, reproduced by \\texttt{nested\\_cv\\_check.py}. "
            f"The 21 October reproduction on the training rows ({tt('nested_cv_summary_repro.json')}, "
            f"{tt('logistic_nested_cv_repro.csv')}) "
            "matched the nested AUCs and standard errors, the logistic rows and the selected configurations; "
            "the KNN Brier and calibration-slope columns are RUAN's 30 September out-of-fold figures "
            f"({tt('knn_v2.json')}), which the reproduction ({tt('nested_cv_knn_repro.json')}) puts at "
            f"{num(((load_json('nested_cv_knn_repro.json') or {}).get('oof_calibration') or {}).get('brier'), 4)} and "
            f"{num(((load_json('nested_cv_knn_repro.json') or {}).get('oof_calibration') or {}).get('cal_slope'), 3)}. "
            "The pre-registered rules re-select the configurations frozen on 16 September. Only KNN and the logistic "
            "family were nested; the ensembles were tuned by plain 5-fold CV.",
            "tab:a-nestedcv", "L{0.30\\linewidth}cccccc",
            ["Model", "Nested AUC", "Nested AP", "Brier", "Cal.\\ slope", "5:1 cut-off", "5:1 cost"],
            rows, note=note, tabcolsep="3pt", footnotesize=True))
    A("")

# ------------------------------------------------------------------ A3 full comparison
comp_path = os.path.join(RESULTS, "model_comparison.csv")
if os.path.exists(comp_path):
    c = pd.read_csv(comp_path)
    sel = "select_auc" if "select_auc" in c.columns else "cv_auc"      # training-set selection score (see aggregate.py)
    # The two subsample-tuned SVMs are scored by the mean of their five shared-fold out-of-fold AUCs
    # (svm_selection_statistics.json), the same statistic as every other row's CV AUC (POST_REVIEW 3.5);
    # aggregate.py's pooled figure goes to the footnote. Sort by the printed score.
    c["shown_score"] = [svm_per_fold_mean(s) if svm_per_fold_mean(s) is not None else v
                   for s, v in zip(c["slug"].astype(str), c[sel])]
    c = c.sort_values("shown_score", ascending=False, na_position="last", kind="mergesort")
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
                     "Logistic + K-means segment dummies (round-1 scoring, $K = 4$ first pass; not adopted by the "
                     "pre-registered rule; the $K = 5$ decision came from repeated CV and was never test-scored)"}
    rows, subsample_notes = [], []
    for r_ in c.itertuples():
        d = {k: getattr(r_, k, None) for k in cols}
        slug = str(d.get("slug"))
        label = NAME_OVERRIDE.get(slug) or esc(str(d.get("name", slug)))      # the column wraps; no truncation
        # The printed score is the score the rows are sorted by, so the table never shows one number and
        # sorts by another.
        sel_v = r_.shown_score
        dag = ""
        if str(d.get("select_source", "cv_auc")) != "cv_auc":
            dag = "$^{\\dagger}$"
            subsample_notes.append(f"{esc(d.get('name', slug))}: pooled out-of-fold {num(d.get('select_auc'))}, "
                                   f"subsample CV {num(d.get('cv_auc'))}")
        rows.append([label + dag,
                     num(sel_v), num(d.get("test_auc")), num(d.get("test_ap")),
                     num(d.get("brier"))])
    floor = sorted(str(s) for s in c.loc[c["thr_cost"].round(2) == 0.05, "slug"]) if "thr_cost" in c.columns else []
    note = None
    if subsample_notes:
        note = ("$^{\\dagger}$Tuned on a 6,000-row training subsample (changed on 21 October): the CV AUC shown is the "
                "mean of the five shared-fold out-of-fold AUCs of the chosen setting refitted on the full training "
                "partition, the same statistic as every other row (\\texttt{results/svm\\_selection\\_statistics.json}); "
                "the earlier table showed the pooled out-of-fold AUC. " + "; ".join(subsample_notes) + ".")
    A(table(f"Every model variant fitted in this project ({len(c)} in total), ordered by CV AUC, the "
            "training-set selection score: the mean of the five 5-fold out-of-fold AUCs on the training partition. "
            "The remaining columns are the frozen configurations' test-set scores, reporting only and not used to rank "
            "(every read is dated in Table~\\ref{tab:a-testlog}). The imbalance variants use fixed, untuned learners "
            "(logit $C = 1$, forest minimum leaf 5). "
            + (f"Four round-1 variants ({', '.join(tt(s) for s in floor)}) sit at the 0.05 floor of "
               f"\\texttt{{common.tune\\_threshold}}. " if len(floor) == 4 else
               (f"Variants at the 0.05 floor of \\texttt{{common.tune\\_threshold}}: {', '.join(tt(s) for s in floor)}. "
                if floor else ""))
            + "RUAN Hangjing's 30 September re-scoring of the KNN and plain-logistic configurations "
            "(Table~\\ref{tab:a-freeze}) is not a separate model and is not listed.",
            "tab:a-allmodels", "L{0.44\\linewidth}cccc",
            ["Model", "CV AUC", "Test AUC", "Test AP", "Test Brier"], rows, note=note))
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
        atc = (load_json(f"{slug}.json") or {}).get("at_tuned_cost") or {}
        n_te = sum(atc.get(k, 0) for k in ("tp", "fp", "tn", "fn"))
        flagged = (atc.get("tp", 0) + atc.get("fp", 0)) / n_te if n_te else None
        if atc and abs(float(atc.get("threshold", -1)) - float(thr.get(slug, -2))) > 1e-9:
            raise SystemExit(f"{slug}.json at_tuned_cost threshold differs from aggregate_summary.json")
        rows.append([esc(names.get(slug, slug)), num(thr_f1.get(slug), 2), num(thr.get(slug), 2), num(rec.get(slug), 2),
                     num(atc.get("precision_1"), 2), pnum(flagged),
                     num(ct[slug].get("at_0_5")), num(ct[slug].get("at_f1")),
                     num(ct[slug].get("at_cost"))])
    ca = agg.get("cost_assumption", {})
    A(table(f"Operating points of the {len(ct)} headline configurations of \\texttt{{aggregate\\_summary.json}} "
            "(the lasso logit duplicates the plain logit to four decimals). Both thresholds are chosen on training "
            "out-of-fold probabilities (maximum $F_1$; minimum 5:1 cost); recall, precision and the share of test "
            "clients flagged are at the cost threshold (\\texttt{at\\_tuned\\_cost} of each frozen "
            "\\texttt{<slug>.json}); the cost columns are the expected test cost at a 5:1 false-negative to "
            "false-positive ratio. Frozen configurations only; reporting, nothing is selected here.",
            "tab:a-operating", "L{0.22\\linewidth}cccccccc",
            ["Model", stack("$F_1$", "thr."), stack("Cost", "thr."), "Recall", stack("Preci-", "sion"),
             stack("Flagged", "(\\%)"), stack("Cost", "@0.5"), stack("Cost", "@$F_1$"), stack("Cost @", "cost thr.")],
            rows, tabcolsep="3pt", footnotesize=True,
            note=f"Trivial rules for reference: flagging every client costs "
                 f"{num(ca.get('flag_everybody_cost'))}, flagging none costs "
                 f"{num(ca.get('flag_nobody_cost'))}."))
    A("")

# ------------------------------------------------------------------ short model names (narrow cost columns)
# Labels only: the slugs of the eleven headline models and the abbreviations printed in narrow cells.
SHORT_NAME = {"xgboost": "XGB", "gradient_boosting": "GBM", "random_forest": "RF", "mlp": "MLP", "knn": "KNN",
              "tree_pruned": "Tree", "svm_rbf": "SVM", "logistic": "Logit", "logistic_l1": "Logit L1", "lda": "LDA",
              "naive_bayes": "NB"}
# Readable names of the same models, keyed by slug and by the display name used in the 21 Oct CSVs.
LONG_NAME = {"xgboost": "XGBoost", "gradient_boosting": "Gradient boosting", "random_forest": "Random forest",
             "mlp": "MLP", "knn": "KNN ($K = 101$)", "tree_pruned": "Pruned tree (17 leaves)", "svm_rbf": "SVM (RBF)",
             "logistic": "Logistic (plain)", "logistic_l1": "Logistic L1", "lda": "LDA", "naive_bayes": "Gaussian NB"}
NAME_TO_SLUG = {"XGBoost": "xgboost", "Gradient boosting": "gradient_boosting", "Random forest": "random_forest",
                "MLP": "mlp", "KNN (K=101)": "knn", "Pruned tree (17 leaves)": "tree_pruned", "SVM (RBF)": "svm_rbf",
                "Logistic (plain)": "logistic", "Logistic L1": "logistic_l1", "LDA": "lda", "Gaussian NB": "naive_bayes"}

# ------------------------------------------------------------------ A5 cost sensitivity
cs_path = os.path.join(RESULTS, "cost_sensitivity.csv")
if os.path.exists(cs_path):
    cs = pd.read_csv(cs_path)
    piv = cs.pivot_table(index="model", columns="ratio", values="test_cost", aggfunc="first")
    ratios = sorted(piv.columns)
    rows = [[esc(m)] + [num(piv.loc[m, r_]) for r_ in ratios] for m in piv.index]
    # Reporting row (POST_REVIEW 6.2): the cheapest model on pooled training out-of-fold scores at the same
    # ratio, from preferred_model_by_ratio_oof.csv (no test read); "---" where that file has no such ratio.
    pref_path = os.path.join(RESULTS, "preferred_model_by_ratio_oof.csv")
    if os.path.exists(pref_path):
        pref = pd.read_csv(pref_path).set_index("ratio")
        rows.append(["\\textit{Cheapest on training OOF}"] +
                    [esc(SHORT_NAME.get(str(pref.loc[int(r_), "preferred_slug"]), pref.loc[int(r_), "preferred_oof"]))
                     if int(r_) in pref.index else "---" for r_ in ratios])
    # cost_sensitivity.csv was re-run on the shared 0.01 cut-off grid on 21 Oct (reporting only, TEST_SET_LOG.md C),
    # so this note is normally empty; it stays as a guard in case the CSV and the saving table drift apart again.
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
    ca = load_json("cost_anchors.json")
    if ca and ca.get("implied_ratio_range"):
        # the body quotes 9.6:1 to 40.5:1 and names the base of each side; the arithmetic is spelled out here
        _lo, _hi = ca["implied_ratio_range"]
        _bd = f"{ca['median_bill_defaulter']:,.0f}".replace(",", "{,}")
        _bg = f"{ca['median_bill_good']:,.1f}".replace(",", "{,}")
        cs_note.append("Cost anchors for the ratio (\\texttt{results/cost\\_anchors.json}): a missed defaulter loses "
                       "50--70\\% of the median defaulter's bill (NT\\$" + _bd.replace("{,}", ",") + "); a false alarm "
                       "forgoes one to three months of a 1.5\\% monthly margin on the median good client's bill (NT\\$"
                       + _bg.replace("{,}", ",") + f"), so the ratio runs from $0.5 \\times {_bd} / (3 \\times 0.015 "
                       f"\\times {_bg}) = {_lo:.1f}$ to $0.7 \\times {_bd} / (0.015 \\times {_bg}) = {_hi:.1f}$: an "
                       "upper bound, as the label is one missed payment and some defaulters later pay.")
    if os.path.exists(pref_path):
        cs_note.append("Last row: the cheapest of the eleven headline models on pooled training out-of-fold scores at "
                       "the same ratio (\\texttt{results/preferred\\_model\\_by\\_ratio\\_oof.csv}; no test read; "
                       "Table~\\ref{tab:a-ratio-pref}; --- where that file has no such ratio); XGB = XGBoost, GBM = gradient "
                       "boosting, RF = random forest.")
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
    # (named ci0, not ci: the module-level ci() helper is used by the post-freeze tables below)
    ci0 = lambda m, lo, hi: f"{num(m, 0)} [{num(lo, 0)}, {num(hi, 0)}]"
    ci0s = lambda m, lo, hi: stack(num(m, 0), f"[{num(lo, 0)}, {num(hi, 0)}]")
    rows = []
    for r_ in sv.itertuples():
        deg = r_.share_flagged_logistic >= 0.99
        # XGBoost against the cheaper of the two blanket rules at this ratio (test costs of frozen predictions).
        blanket = min(r_.cost_flag_nobody, r_.cost_flag_everybody)
        xb = blanket - r_.cost_xgb_tuned
        rows.append([f"{int(r_.ratio)}:1" + ("$^{\\dagger}$" if deg else ""), num(r_.t_closed_form, 3),
                     num(r_.t_logistic, 2), num(r_.t_xgb, 2), num(r_.cost_logistic_at_0_5, 0),
                     pnum(r_.share_flagged_logistic),
                     ci0s(r_.threshold_saving, r_.threshold_saving_lo, r_.threshold_saving_hi),
                     ci0s(r_.model_saving, r_.model_saving_lo, r_.model_saving_hi),
                     stack(num(xb, 0), f"({pnum(xb / blanket)}\\,\\%)")])
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
                 f"{num(ps.get('t_logistic'), 2)}): threshold saving {ci0(ps.get('threshold_saving'), ps.get('threshold_saving_lo'), ps.get('threshold_saving_hi'))}, "
                 f"model saving {ci0(ps.get('model_saving'), ps.get('model_saving_lo'), ps.get('model_saving_hi'))} at 5:1. ")
    note += ("Last column: the cheaper blanket rule at each ratio (flag nobody up to "
             + ", ".join(f"{int(x)}:1" for x in sv.loc[sv.cost_flag_nobody <= sv.cost_flag_everybody, "ratio"].tail(1))
             + ", flag everybody above) minus the frozen XGBoost at its own cut-off, with the saving as a share of the "
             "blanket cost; the same frozen test predictions, nothing re-scored. Training-row version against "
             "no-parameter arrears rules: Table~\\ref{tab:a-simplerules}.")
    A(table("Threshold saving against model saving at nine false-negative to false-positive cost ratios $r$. "
            "Threshold saving: expected test cost of the frozen plain logistic at the 0.5 cut-off minus its cost at "
            "the cut-off chosen on training out-of-fold scores. Model saving: the logistic at its own cut-off minus "
            "the frozen XGBoost at its own cut-off. Brackets are 95\\,\\% paired-bootstrap intervals. Flagged: share of "
            "the test clients the logistic flags at its cut-off. "
            f"Source: {tt('results/threshold_vs_model_saving.csv')} (Table~\\ref{{tab:saving}} in the body is the "
            "excerpt). Frozen configurations only; reporting, nothing is selected here.",
            "tab:a-saving", "lcccccccc",
            ["$r$", "$1/(1+r)$", stack("Logit", "cut-off"), stack("XGB", "cut-off"), stack("Logit cost", "@0.5"),
             stack("Flagged", "(logit, \\%)"), stack("Threshold", "saving"), stack("Model", "saving"),
             stack("XGB saving vs", "best blanket")],
            rows, note=note, tabcolsep="3pt", footnotesize=True))
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
                  f"Jaccard {num(ks.loc[ks.K == k_final, 'min_cluster_jaccard'].iloc[0], 3)}). The stability figures in "
                  f"Table~\\ref{{tab:a-kprofiles}} come from the separate 100-refit stability run of the "
                  f"chosen $K={k_final}$ ({tt('unsupervised_summary_v2.json')}): ARI {num(ba.get('mean'), 3)} "
                  f"[{num(ba.get('lo'), 3)}, {num(ba.get('hi'), 3)}] under bootstrap resampling, per-cluster Jaccard "
                  + "/".join(num(j, 3) for j in us.get("cluster_jaccard", [])) +
                  f". Across random starts (single-start refits with other seeds on the same rows, against the reference "
                  f"partition) the groups shift more: ARI mean {num(us.get('seed_ari', {}).get('mean'), 2)}, minimum "
                  f"{num(us.get('seed_ari', {}).get('min'), 2)}"
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
                   f"{num(sh_sp.get('perm_train'), 3)}, vs out-of-bag permutation {num(sh_sp.get('perm_oob'), 3)}. "
                   "The Spearman values between SHAP (XGBoost) and MDI or permutation (random forest) are cross-model: "
                   "they compare two different fitted models, not two measures of one model.")
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

# ================================================================== post-freeze checks (21 Oct 2026; training rows only)
# Every table below reads a file written by one of the six training-only scripts of 21 October
# (pay_coding_check.py, fold_stability.py, tuning_checks.py, ablation_checks.py, fairness_check.py,
# monotone_respec_check.py) or the R cross-check. None of them opens the test file; nothing here is
# scored on the test set and nothing here selects a model, a feature or a cut-off (POST_REVIEW_CHECKS.md).


def sd1(xs):
    """Standard deviation of the five fold values, ddof 1."""
    return float(np.std(np.asarray(xs, float), ddof=1))


def mean_sd(m, s, nd=4):
    return f"{num(m, nd)} $\\pm$ {num(s, nd)}"


def nb_d(d, nd=4, scale=1.0, mean_key="mean", k_key="folds_positive", lo_key="nb_lo", hi_key="nb_hi"):
    """nb_cell from a dict of a paired comparison."""
    return nb_cell(d[mean_key], d[lo_key], d[hi_key], d[k_key], nd=nd, scale=scale)


def nb_dt(d, nd=4, scale=1.0, mean_key="mean", k_key="folds_positive", lo_key="nb_lo", hi_key="nb_hi"):
    """nb_text from a dict of a paired comparison."""
    return nb_text(d[mean_key], d[lo_key], d[hi_key], d[k_key], nd=nd, scale=scale)


def nb_ci95(d, nd=4):
    """nb_text from the tuning_checks / svm_selection_statistics layout (mean_diff, nb_ci95, n_folds_positive)."""
    return nb_text(d["mean_diff"], d["nb_ci95"][0], d["nb_ci95"][1], d["n_folds_positive"], nd=nd)


def subtab(colspec, header, rows, pre_header=None, title=None, tabcolsep=None):
    """A second block of the same table (set through table(after=...)), with its own rules."""
    out = []
    if title:
        out.append(f"  \\\\[6pt]{{\\itshape {title}}}\\\\[2pt]")
    else:
        out.append("  \\\\[6pt]")
    if tabcolsep is not None:
        out.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    out.append(f"  \\begin{{tabular}}{{@{{}}{colspec}@{{}}}}")
    out.append(r"    \toprule")
    for h in (pre_header or []):
        out.append("    " + h)
    out.append("    " + " & ".join(header) + r" \\")
    out.append(r"    \midrule")
    for r_ in rows:
        out.append("    " + (r_ if isinstance(r_, str) else " & ".join(r_) + r" \\"))
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    return "\n".join(out)


def hdr(*lines):
    """A header cell on several lines."""
    return stack(*lines)


A("\\clearpage")
A("\\subsection{Post-freeze checks on the training rows (21 October 2026)}\\label{sec:a-postfreeze}")
A("")
A("Every table in this subsection is a post-freeze sensitivity check: training five-fold CV on the shared folds "
  "(\\texttt{data/processed/cv\\_folds.csv}, seed 2607526), run on 21 October 2026 after all configurations were "
  "frozen. Nothing here was scored on the test set, and nothing here selects a model, feature or cut-off. Paired "
  "comparisons give the mean of the five fold differences, a Nadeau--Bengio corrected 95\\,\\% interval "
  "($n_{\\mathrm{test}}/n_{\\mathrm{train}} = 1/4$) in brackets and the number of folds in which the difference is "
  "positive ($k$/5). Source: \\texttt{results/POST\\_REVIEW\\_CHECKS.md}. The figures of these checks are in "
  "Section~\\ref{sec:a-postfreeze-figs}.")
A("")

# ------------------------------------------------------------------ PF1 what the repayment codes mean
pcr = pd.read_csv(os.path.join(RESULTS, "pay_code_default_rates.csv"))
pcs = load_json("pay_coding_summary.json") or {}
c1 = pcs.get("pay1_code1", {})
PAY_VARS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
month_of = {v: str(pcr.loc[pcr.variable == v, "month"].iloc[0]).split()[0] for v in PAY_VARS}
CODE_MEANING = {-2: "no consumption$^{\\dagger}$", -1: "paid in full", 0: "revolving credit$^{\\dagger}$",
                1: "1 month late"}
rows = []
for code in range(-2, 9):
    sep = pcr[(pcr.variable == "PAY_1") & (pcr.code == code)]
    cells = [f"${code}$", CODE_MEANING.get(code, f"{code} months late")]
    cells += ([f"{int(sep.n.iloc[0]):,}", pnum(sep.default_rate.iloc[0])] if len(sep) else ["---", "---"])
    for v in PAY_VARS[1:]:
        r_ = pcr[(pcr.variable == v) & (pcr.code == code)]
        cells.append(f"{int(r_.n.iloc[0]):,}" if len(r_) else "---")
    rows.append(cells)
sep_ = pcr[pcr.variable == "PAY_1"].set_index("code")
n_tr = int(pcs.get("n_train", meta["n_train"]))
A(table("What the repayment codes mean and how often they occur, training rows only "
        f"($n = {mnum(n_tr)}$; \\texttt{{results/pay\\_code\\_default\\_rates.csv}}). Each cell counts the training clients "
        "with that code in that month; the September default rate is the share of them who defaulted on the October "
        "2005 payment. A status is compared within a month, not across months: the same code does not carry the same "
        "risk in different months.",
        "tab:a-paycodes-rates", "cL{0.22\\linewidth}rrrrrrr",
        ["Code", "Meaning", "$n$", "Default (\\%)"] + [month_of[v] for v in PAY_VARS[1:]], rows,
        pre_header=[" & & \\multicolumn{2}{c}{September (\\texttt{PAY\\_1})} & "
                    "\\multicolumn{5}{c}{Clients with the code in earlier months} \\\\",
                    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}"],
        tabcolsep="4pt", footnotesize=True,
        note=("$^{\\dagger}$Codes $-2$ and 0 are not defined in the UCI documentation; their readings are widely used, "
              "not documented. The delay counters (\\texttt{MAX\\_DELAY}, \\texttt{N\\_MONTHS\\_DELAYED}) treat $-2$, "
              "$-1$ and 0 all as 0. Code 1 is recorded almost only in September: of its "
              f"{int(c1['n']):,} September clients, {int(c1['n_sep_bill_nonpositive']):,} "
              f"({pnum(c1['share_sep_bill_nonpositive'])}\\,\\%) had no positive September bill, "
              f"{int(c1['n_pay2_eq_2']):,} ({pnum(c1['share_pay2_eq_2'])}\\,\\%) were two months late in August, "
              f"{int(c1['n_all_six_bills_zero']):,} had six zero bills "
              f"({pnum(c1['default_rate_all_six_bills_zero'])}\\,\\% default) and {int(c1['n_pay2_to_6_all_minus2']):,} "
              "followed five months of $-2$. Share of September $k$-months-late clients who were two months late in "
              "August (a backward share: where September's clients came from, not how many August clients moved on): "
              f"code 2 {pnum(sep_.loc[2, 'share_month_before_code_2'])}\\,\\% (arrears that persisted), code 3 "
              f"{pnum(sep_.loc[3, 'share_month_before_code_2'])}\\,\\% (arrears that deepened one step in a "
              "month).")))
A("")

# ------------------------------------------------------------------ PF2 reading the codes by meaning
pcv = pd.read_csv(os.path.join(RESULTS, "pay_coding_cv.csv"))
ms_ = pcs.get("model_settings", {})
MODEL_LBL = {"logit_plain": "Logit (plain)",
             "logit_l2": f"Logit, L2 ($C = {float(ms_.get('logit_l2', {}).get('C', float('nan'))):.1f}$)",
             "lda": "LDA",
             "knn": f"KNN ($K = {ms_.get('knn', {}).get('n_neighbors', '?')}$)",
             "mlp": "MLP (" + "--".join(str(h) for h in ms_.get("mlp", {}).get("hidden_layer_sizes", [])) + ")"}
CODING_LBL = {"numeric": "numeric (as frozen)", "delay_flags": "months late + flags", "onehot": "one-hot"}
rows, prev = [], None
for r_ in pcv.itertuples():
    if r_.model != prev:
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{MODEL_LBL.get(r_.model, esc(r_.model))}}}}} \\\\")
    prev = r_.model
    conv = str(r_.converged_all_folds).lower() == "true"
    ref = r_.coding == "numeric"
    rows.append(["\\quad " + CODING_LBL.get(r_.coding, esc(r_.coding)) + ("" if conv else "$^{\\dagger}$"),
                 str(int(r_.n_inputs)), mean_sd(r_.mean_auc, r_.sd_auc_ddof1),
                 "---" if ref else nb_cell(r_.diff_auc_vs_numeric_mean, r_.diff_auc_nb_lo, r_.diff_auc_nb_hi,
                                           r_.diff_auc_folds_positive),
                 num(r_.mean_ap, 4),
                 "---" if ref else nb_cell(r_.diff_ap_vs_numeric_mean, r_.diff_ap_nb_lo, r_.diff_ap_nb_hi,
                                           r_.diff_ap_folds_positive),
                 num(r_.mean_brier, 4)])
repro = pcs.get("reproduction_of_frozen_numeric_oof", {})
repro_ok = repro and all(float(v.get("max_abs_diff", 1)) == 0.0 for v in repro.values())
cm = pcs.get("cross_model", {})
x_num = cm["xgboost_frozen_minus_logit_plain_numeric"]
x_del = cm["xgboost_frozen_minus_logit_plain_delay_flags"]
gap_closed_xgb = 1 - float(x_del["mean"]) / float(x_num["mean"])
pco = pd.read_csv(os.path.join(RESULTS, "pay_coding_logit_coefficients.csv"))
pco = pco[(pco.coding == "delay_flags") & (pco.model == "logit_plain")].set_index("feature")
del_later = [float(pco.loc[f"PAY_{k}_DEL", "coef_per_unit"]) for k in range(2, 7)]
signs_ok = all(int(pco.loc[f, "fold_sign_agreement"]) == 5
               for f in ["PAY_1_PD", "PAY_1_NC", "PAY_1_DEL"] + [f"PAY_{k}_DEL" for k in range(2, 7)])
A(table("Reading the repayment codes by meaning (post-freeze, training CV only, never test-scored). Five model "
        "families refitted with their frozen settings on three codings of \\texttt{PAY\\_1}--\\texttt{PAY\\_6}: the "
        "frozen single number per month; months late, $\\max(\\mathrm{code}, 0)$, plus indicators for $-2$ and $-1$; and "
        "one dummy per code and month. $\\Delta$ = this coding minus the numeric coding of the same model, fold-paired. "
        "SD over the five fold AUCs, ddof 1. Sources: \\texttt{results/pay\\_coding\\_cv.csv}, "
        "\\texttt{pay\\_coding\\_summary.json}, \\texttt{pay\\_coding\\_logit\\_coefficients.csv}.",
        "tab:a-paycoding", "L{0.20\\linewidth}cccccc",
        ["Model and coding", "Inputs", "CV AUC $\\pm$ SD", "$\\Delta$AUC, $k$/5 [NB]", "CV AP",
         "$\\Delta$AP, $k$/5 [NB]", "Brier"], rows, tabcolsep="2.5pt", footnotesize=True,
        note=(("Numeric coding reproduces every frozen out-of-fold vector exactly. " if repro_ok else "")
              + "$^{\\dagger}$The unpenalised one-hot logit stopped at \\texttt{max\\_iter} in every fold (rare codes, "
              "some held by a single training client, make the maximum-likelihood estimate infinite); the L2 one-hot "
              "row is the converged fit of the same coding. Cross-model, fold-paired: recoded logit $-$ KNN "
              f"{nb_dt(cm['logit_plain_delay_flags_minus_knn_numeric'])}; XGBoost $-$ logit (numeric) {nb_dt(x_num)}; "
              f"XGBoost $-$ recoded logit {nb_dt(x_del)}; XGBoost $-$ one-hot logit "
              f"{nb_dt(cm['xgboost_frozen_minus_logit_plain_onehot'])}. The recoding closes "
              f"{pnum(cm['share_of_knn_logit_gap_closed_by_delay_flags'], 0)}\\,\\% of the KNN--logit gap but only "
              f"{pnum(gap_closed_xgb, 0)}\\,\\% of the XGBoost--logit gap. Recoded plain logit, per unit, against "
              "revolving credit in September: paid in full "
              f"{snum(pco.loc['PAY_1_PD', 'coef_per_unit'], 2)}, no consumption {snum(pco.loc['PAY_1_NC', 'coef_per_unit'], 2)} "
              f"log-odds, each month late {snum(pco.loc['PAY_1_DEL', 'coef_per_unit'], 2)}; one more month late in "
              f"August \\dots\\ April {snum(del_later[0], 2)} \\dots\\ {snum(del_later[-1], 2)} log-odds per month, "
              "conditional on the delay counters"
              + (" (all five folds agree in sign)." if signs_ok else ".")
              + " These coefficients describe associations in this vintage, not causes.")))
A("")

# ------------------------------------------------------------------ PF3 fold-by-fold stability
fs = pd.read_csv(os.path.join(RESULTS, "fold_stability.csv"))
fss = load_json("fold_stability_summary.json") or {}
fold_cols = [f"auc_fold{k}" for k in range(1, 6)]
rows_a, rows_b = [], []
for r_ in fs.itertuples():
    lbl = LONG_NAME.get(r_.slug, esc(r_.model)) + ("$^{\\dagger}$" if r_.slug in SVM_SEL else "")
    ranks = " ".join(str(int(getattr(r_, f"rank_auc_fold{k}"))) for k in range(1, 6))
    rows_a.append([lbl] + [num(getattr(r_, c), 4) for c in fold_cols]
                  + [mean_sd(r_.auc_mean, r_.auc_sd_ddof1), ranks])
    ref = r_.slug == "xgboost"
    rows_b.append([lbl, f"{num(r_.pooled_oof_ap, 3)} ({num(r_.ap_lift_over_base_rate, 2)}$\\times$)",
                   num(r_.brier_oof, 4), num(r_.cal_slope_oof, 2),
                   "reference" if ref else f"{snum(r_.xgb_minus_auc_mean)}, {k5(r_.xgb_auc_wins_of_5)}",
                   "---" if ref else ci(r_.xgb_minus_auc_nb_lo, r_.xgb_minus_auc_nb_hi),
                   "---" if ref else num(r_.xgb_minus_auc_holm_p, 3)])
fsi = fs.set_index("slug")


def fold_pair(a, b):
    """a minus b over the five folds, Nadeau-Bengio interval from corrected_t_ci (n_train/n_test = 4/1)."""
    d = fsi.loc[a, fold_cols].astype(float).values - fsi.loc[b, fold_cols].astype(float).values
    r = corrected_t_ci(d, 4, 1)
    return nb_text(r["mean"], r["lo"], r["hi"], int((d > 0).sum()))


# Blocks of ranks that hold in every fold (checked here, so the note cannot drift from the file).
RANK_BLOCKS = [(["xgboost", "gradient_boosting", "random_forest"], "XGBoost, GBM, forest"), (["mlp"], "MLP"),
               (["knn", "tree_pruned", "svm_rbf"], "KNN, tree, SVM"), (["logistic", "logistic_l1"], "logits"),
               (["lda"], "LDA"), (["naive_bayes"], "naive Bayes")]
block_txt, lo_ = [], 1
for slugs, name in RANK_BLOCKS:
    hi_ = lo_ + len(slugs) - 1
    got = {int(fsi.loc[s, f"rank_auc_fold{k}"]) for s in slugs for k in range(1, 6)}
    if got != set(range(lo_, hi_ + 1)):
        raise SystemExit(f"fold_stability.csv: ranks of {slugs} are {sorted(got)}, not {lo_}-{hi_} in every fold")
    block_txt.append(f"\\{{{name}\\}} {lo_}--{hi_}" if hi_ > lo_ else f"{name} {lo_}")
    lo_ = hi_ + 1
svm_pair = next(p for p in SVM_SEL.get("svm_rbf_paired_vs_neighbours", []) if p["comparison"] == "svm_rbf minus logistic")
base_rate = fss.get("base_rate")
A(table("Fold-by-fold stability of the eleven headline configurations: training out-of-fold scores of the frozen "
        "configurations on the five shared folds (\\texttt{results/fold\\_stability.csv}); nothing re-fitted, nothing "
        "test-scored. Top: AUC in each validation fold (4,195 clients), mean $\\pm$ SD (ddof 1) and the rank in each "
        "fold. Bottom: pooled out-of-fold average precision (AP) with its lift over the base rate"
        + (f" ({num(base_rate, 3)})" if base_rate else "") + ", Brier score and calibration slope, and the fold-paired "
        "difference XGBoost $-$ model with its Nadeau--Bengio interval and the Holm-adjusted $p$-value over the ten "
        "comparisons.",
        "tab:a-foldstab", "L{0.22\\linewidth}cccccccc",
        ["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean $\\pm$ SD", "Ranks"],
        rows_a, tabcolsep="3pt", footnotesize=True,
        after=subtab("L{0.22\\linewidth}cccccc",
                     ["Model", hdr("Pooled OOF AP", "(lift)"), hdr("OOF", "Brier"), hdr("Cal.", "slope"),
                      hdr("XGBoost $-$ model", "mean, $k$/5"), hdr("NB 95\\,\\%", "interval"), hdr("Holm", "$p$")],
                     rows_b, tabcolsep="3pt"),
        note=(f"$^{{\\dagger}}$SVM: mean of the five shared-fold out-of-fold AUCs (pooled "
              f"{num(SVM_SEL['svm_rbf']['pooled_oof_auc'], 3)}; \\texttt{{svm\\_selection\\_statistics.json}}). The "
              "frozen JSON \\texttt{cv\\_auc\\_std} uses ddof 0; this table uses ddof 1. Blocks that hold in every "
              "fold: " + ", ".join(block_txt) + ". Other neighbours, fold-paired: MLP $-$ KNN "
              f"{fold_pair('mlp', 'knn')}; pruned tree $-$ KNN {fold_pair('tree_pruned', 'knn')}; SVM $-$ plain logit "
              f"{nb_text(svm_pair['mean_diff'], svm_pair['nb_ci95'][0], svm_pair['nb_ci95'][1], svm_pair['n_folds_positive'])} "
              "(stored in \\texttt{svm\\_selection\\_statistics.json}).")))
A("")

# ------------------------------------------------------------------ PF4 tuning-budget and grid checks
tcs = load_json("tuning_checks_summary.json") or {}
g_ = tcs["a_gbm_grid_fulltrain"]
gw, gf = g_["winner_fulltrain"], g_["frozen_setting_fulltrain"]
gx = g_["xgboost_comparison"]["xgb_minus_fulltrain_gbm_winner"]
kfg_ = load_json("nested_cv_knn_frozen_grid.json")
of_ = kfg_["outer_folds"]
ks_ = pd.Series([o["chosen_K"] for o in of_]).value_counts()
k_choice = "; ".join(f"$K = {int(k)}$ in {int(c)}" for k, c in ks_.items())
n_dist_ = sum(1 for o in of_ if o["chosen_weights"] == "distance")
npf = kfg_["nested_minus_plain_frozen"]
kd = pd.read_csv(os.path.join(RESULTS, "knn_dimensionality_cv.csv")).set_index("variant")
xe = tcs["d_grid_edges"]["xgb_extension"]
xb_, xbv = xe["best_setting"], xe["best_vs_frozen"]
xbr = xe["beyond_requested_range"]["best"]
xd2 = max((b for b in xe["best_per_depth_lr"] if b["max_depth"] == 2), key=lambda b: b["cv_auc"])
ge = pd.read_csv(os.path.join(RESULTS, "grid_edge_check.csv"))
ge = ge[ge.section == "edge_check"]
EDGE_MODEL = {"random_forest": "forest", "gradient_boosting": "GBM", "mlp": "MLP", "xgboost": "XGBoost"}
unprobed = []
for r_ in ge.itertuples():
    ev = str(r_.evidence)
    if not any(s in ev for s in ("not tried", "not probed", "NOT bracketed")):
        continue
    txt = f"{EDGE_MODEL.get(r_.model, esc(r_.model))} {tt(r_.parameter)} {esc(r_.chosen)}"
    if "still rising" in ev:
        cvs = [float(x) for x in re.findall(r"0\.\d{4,}", ev)]
        txt += " (CV still rising: " + " / ".join(num(x, 4) for x in cvs) + ")"
    unprobed.append(txt)
qj = load_json("qda.json") or {}
qg = [(float(x["value"]), float(x["cv_auc"])) for x in qj.get("extra", {}).get("cv_grid", [])]
q_reg = float(qj.get("best_params", {}).get("reg_param", float("nan")))
if qg and q_reg == max(v for v, _ in qg):
    unprobed.append(f"QDA {tt('reg_param')} {fmt_val(q_reg)} (upper edge; CV "
                    + " / ".join(num(c, 4) for _, c in sorted(qg)) + ")")
sv_r, sv_l = SVM_SEL["svm_rbf"], SVM_SEL["svm_linear"]
ranks_ = {k: v["rank_among_11_headline"] for k, v in SVM_SEL.get("svm_rbf_rank_by_statistic", {}).items()}
mi_sel = (tcs.get("c_knn_dimensionality", {}).get("mi_selection_frequency", {}) or {}).get("mutual_info_top8", {})
mi_status = bool(mi_sel) and all(f.startswith("PAY_") or f in ("MAX_DELAY", "N_MONTHS_DELAYED") for f in mi_sel)
KD_READ = {"pca_15": "PCA fitted inside each fold;", "pca_10": "$K = 101$ uniform, not re-tuned",
           "mutual_info_top8": "selector fitted inside each fold" + ("; all eight columns are repayment-status codes "
                                                                     "or counters, so many clients coincide"
                                                                     if mi_status else "")}
KD_LBL = [("frozen_32_columns", "32 columns: 26 standardised, 6 one-hot (frozen)"), ("pca_15", "15 principal components"),
          ("pca_10", "10 principal components"), ("pca_5", "5 principal components"),
          ("mutual_info_top8", "top 8 mutual-information columns")]
rows = [
    ["GBM grid re-scored on the full training set",
     f"{len(g_['all_12'])} settings on {int(g_['tuning_rows']):,} rows (frozen grid was scored on a subsample); winner "
     f"depth {gw['max_depth']}, learning rate {fmt_val(gw['learning_rate'])}, {gw['n_estimators']} trees; the frozen "
     f"setting ranks {gf['rank_fulltrain']} ({num(gf['cv_auc'], 4)})",
     num(gw["cv_auc"], 4),
     nb_cell(g_["winner_vs_frozen"]["mean_diff"], *g_["winner_vs_frozen"]["nb_ci95"],
             g_["winner_vs_frozen"]["n_folds_positive"]),
     f"Spearman with the subsample order {num(g_['spearman_fulltrain_vs_subsample_cv'], 2)}; XGBoost $-$ full-set "
     f"winner {snum(gx['mean_diff'])}, {k5(gx['n_folds_positive'])}; rule fixed beforehand: report only, keep the frozen GBM"],
    "\\midrule",
    [f"KNN nested CV over the frozen {kfg_['grid']['n_settings']}-setting grid",
     f"outer folds = the five shared folds; choices: {k_choice}; distance weights in {n_dist_}",
     mean_sd(kfg_["nested_cv_auc"], kfg_["nested_cv_auc_std"]),
     nb_cell(npf["mean_diff"], *npf["nb_ci95"], npf["n_folds_positive"]),
     f"nested $-$ plain CV of the frozen $K = 101$ ({npf['n_folds_exactly_zero']} folds tie)"],
    "\\midrule",
]
for i, (v, lbl) in enumerate(KD_LBL):
    r_ = kd.loc[v]
    extra = ""
    if v.startswith("pca"):
        extra = f" ({pnum(r_.pca_variance_share_mean, 0)}\\,\\% of the variance)"
    elif v.startswith("mutual"):
        extra = f" (about {num(r_.distinct_fitting_vectors_mean, 0)} distinct points)"
    rows.append(["KNN in fewer dimensions" if i == 0 else "", lbl + extra, num(r_.cv_auc_mean, 4),
                 "---" if v == "frozen_32_columns" else nb_cell(r_.mean_diff_vs_frozen, r_.nb_ci95_lo, r_.nb_ci95_hi,
                                                                r_.n_folds_positive),
                 KD_READ.get(v, "")])
rows += [
    "\\midrule",
    ["XGBoost grid extension",
     f"{xe['n_settings']} settings: depth 2--3 $\\times$ learning rate 0.01--0.03 $\\times$ 150--600 rounds; best depth "
     f"{xb_['max_depth']}, rate {fmt_val(xb_['learning_rate'])}, {xb_['n_estimators']} rounds",
     num(xb_["cv_auc"], 5),
     stack(f"{snum(xbv['mean_diff'], 5)}, {k5(xbv['n_folds_positive'])}", ci(*xbv["nb_ci95"])),
     ("a setting beats" if xe.get("any_beats_by_more_than_one_fold_sd") else "no setting beats")
     + f" the frozen one by one fold SD ({num(xe['frozen_cv_auc_std_ddof0'], 4)})"
     f"; best at depth 2: {num(xd2['cv_auc'], 4)}; rate {fmt_val(xbr['learning_rate'])} at {xbr['n_estimators']:,} "
     f"rounds {num(xbr['cv_auc'], 5)} ({snum(xbr['diff_vs_frozen'])})"],
    "\\midrule",
    ["Grid edges not probed", "; ".join(unprobed), "---", "---",
     "the chosen value sits on the edge of its grid and nothing beyond it was tried"],
    "\\midrule",
    ["SVM selection statistic",
     f"RBF: subsample CV {num(sv_r['subsample_cv_auc']['value'], 3)}, pooled OOF {num(sv_r['pooled_oof_auc'], 3)}, mean "
     f"of fold AUCs {num(sv_r['per_fold_mean_oof_auc'], 3)}; linear: {num(sv_l['subsample_cv_auc']['value'], 3)}, "
     f"{num(sv_l['pooled_oof_auc'], 3)}, {num(sv_l['per_fold_mean_oof_auc'], 3)}",
     num(sv_r["per_fold_mean_oof_auc"], 4), "---",
     ("rank " + "/".join(str(v) for v in sorted(set(ranks_.values()))) + " of 11 under each statistic")
     if len(set(ranks_.values())) == 1 else
     "ranks " + ", ".join(f"{k} {v}" for k, v in ranks_.items())],
]
A(table("Tuning-budget and grid checks (post-freeze, training CV only, never test-scored). Nothing was selected: "
        "every frozen configuration stays. The reported CV is the maximum over each grid on the same folds, so "
        "winners' gaps are upper estimates. Differences are against the frozen configuration of the same model. "
        "Sources: " + ", ".join(tt(f) for f in ("results/gbm_grid_fulltrain.csv", "tuning_checks_summary.json",
                                                "nested_cv_knn_frozen_grid.json", "knn_dimensionality_cv.csv",
                                                "grid_edge_check.csv", "svm_selection_statistics.json")) + ".",
        "tab:a-tuningchecks", "L{0.15\\linewidth}L{0.27\\linewidth}L{0.10\\linewidth}cL{0.22\\linewidth}",
        ["Check", "Setting", "CV AUC", hdr("Difference", "$k$/5 [NB]"), "Reading"], rows,
        tabcolsep="3pt", footnotesize=True))
A("")

# ------------------------------------------------------------------ PF5-7 ablations (depth, history, demographics)
ACH = load_json("ablation_checks_summary.json") or {}


def abl(path):
    return pd.read_csv(os.path.join(RESULTS, path))


def acv(df, spec, metric="auc"):
    r = df[(df.kind == "cv") & (df.spec == spec) & (df.metric == metric)]
    if len(r) != 1:
        raise SystemExit(f"ablation file: no unique cv row for {spec!r} / {metric}")
    return r.iloc[0]


def apd(df, spec, ref, metric="auc", model=None):
    r = df[(df.kind == "paired_diff") & (df.spec == spec) & (df.spec_ref == ref) & (df.metric == metric)]
    if model is not None:
        r = r[r.model == model]
    if len(r) != 1:
        raise SystemExit(f"ablation file: no unique paired row for {spec!r} - {ref!r} / {metric}")
    return r.iloc[0]


def apd_t(row, nd=4):
    return nb_text(row["mean"], row["nb_lo"], row["nb_hi"], row["n_pos"], nd=nd)


def rounds_txt(s):
    return "\\scriptsize " + "/".join(f"{int(x):,}" for x in str(s).split("/"))


ad = abl("ablation_depth.csv")
E29, R23 = "29 predictors (engineered)", "23 raw predictors"
l29, l23 = acv(ad, f"logit | {E29}"), acv(ad, f"logit | {R23}")
rows = [f"Plain logit & \\multicolumn{{2}}{{c}}{{{num(l29['mean'], 4)}}} & "
        f"\\multicolumn{{2}}{{c}}{{{num(l23['mean'], 4)}}} \\\\"]
for d in (1, 2, 3, 4):
    cells = [f"XGBoost depth {d}" + {1: " (additive)", 3: " (frozen)"}.get(d, "")]
    for fsname in (E29, R23):
        a300, atun = acv(ad, f"depth {d} | {fsname}"), acv(ad, f"depth {d}, rounds tuned | {fsname}")
        cells += [num(a300["mean"], 4), stack(num(atun["mean"], 4), rounds_txt(atun["n_estimators"]))]
    rows.append(cells)
d1t_l = apd(ad, f"depth 1, rounds tuned | {E29}", f"logit | {E29}")
d31t = apd(ad, f"depth 3, rounds tuned | {E29}", f"depth 1, rounds tuned | {E29}")
d31 = apd(ad, f"depth 3 | {E29}", f"depth 1 | {E29}")
d31t_raw = apd(ad, f"depth 3, rounds tuned | {R23}", f"depth 1, rounds tuned | {R23}")
did = apd(ad, f"(depth 3 - depth 1), rounds tuned | {R23}", f"(depth 3 - depth 1), rounds tuned | {E29}")
eng_l = apd(ad, f"logit | {E29}", f"logit | {R23}")
eng_x = apd(ad, f"depth 3 | {E29}", f"depth 3 | {R23}")
tot = apd(ad, f"depth 3 | {E29}", f"logit | {E29}")
A(table("Interaction depth (post-freeze, training CV only, never test-scored): CV AUC of XGBoost with the frozen "
        "settings and only the tree depth changed (in the re-tuned columns also the number of rounds), against the "
        "plain logit, on the frozen 29 predictors and on the 23 raw ones. Depth 3 was selected on these folds without "
        "nesting, so its gain over depth 1 is if anything an upper estimate; rounds re-tuned by inner 3-fold CV inside "
        "each outer fold (rounds per fold under the AUC). Source: \\texttt{results/ablation\\_depth.csv}.",
        "tab:a-ablation-depth", "L{0.20\\linewidth}cccc",
        ["Model", "300 rounds", hdr("Rounds re-tuned", "(rounds per fold)"), "300 rounds",
         hdr("Rounds re-tuned", "(rounds per fold)")], rows,
        pre_header=[f" & \\multicolumn{{2}}{{c}}{{29 predictors (23 raw + 6 engineered)}} & "
                    f"\\multicolumn{{2}}{{c}}{{23 raw predictors}} \\\\",
                    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"],
        tabcolsep="3pt", footnotesize=True,
        note=("Fold-paired differences in CV AUC (29 predictors unless stated): total gap, depth 3 $-$ logit "
              f"{apd_t(tot)}; additive booster, depth 1 (re-tuned) $-$ logit {apd_t(d1t_l)}; interactions, depth 3 $-$ "
              f"depth 1, both re-tuned {apd_t(d31t)}; the same at 300 rounds {apd_t(d31)} (depth 1 is under-fitted at 300 "
              f"rounds); on the 23 raw columns, depth 3 $-$ depth 1 re-tuned {apd_t(d31t_raw)}; difference-in-differences "
              f"(raw minus engineered interaction gain) {apd_t(did)}. Value of the six engineered features: to the logit "
              f"{apd_t(eng_l)}, to XGBoost at depth 3 {apd_t(eng_x)}. The additive booster recovers "
              f"{pnum(float(d1t_l['mean']) / float(tot['mean']), 0)}\\,\\% of the gap; the partial-dependence curves "
              "show where its one-variable shape lies, but no share per variable was measured.")))
A("")

ah = abl("ablation_history.csv")
fsets = ACH.get("feature_sets", {})
s6 = [v for v in fsets.get("H0+S6", {}).get("numeric", []) if v not in fsets.get("H0", {}).get("numeric", [])]
H_LBL = {"H0": "H0: limit + demographics", "H1": "H1: + September", "H3": "H3: + July--September",
         "H6": "H6: + April--September"}
rows = []
for h in ("H0", "H1", "H3", "H6"):
    cells = [H_LBL[h]]
    for model in ("XGBoost", "logit"):
        for spec in (f"{h} raw months only", f"{h} + six-month summaries", f"{h} + window summaries (months 1-k)"):
            r = ah[(ah.kind == "cv") & (ah.spec == spec) & (ah.metric == "auc") & (ah.model == model)]
            cells.append(f"{num(r['mean'].iloc[0], 4)} ({int(r['n_predictors'].iloc[0])})" if len(r) else "---")
    rows.append(cells)
X = "XGBoost"
h10 = apd(ah, "H1 raw months only", "H0 raw months only", model=X)
h31 = apd(ah, "H3 raw months only", "H1 raw months only", model=X)
h63 = apd(ah, "H6 raw months only", "H3 raw months only", model=X)
h61 = apd(ah, "H6 raw months only", "H1 raw months only", model=X)
h61s = apd(ah, "H6 + six-month summaries", "H1 + six-month summaries", model=X)
s_w1 = apd(ah, "H1 + six-month summaries", "H1 + window summaries (months 1-k)", model=X)
h61s_l = apd(ah, "H6 + six-month summaries", "H1 + six-month summaries", model="logit")
A(table("Recent versus older repayment history (post-freeze, training CV only, never test-scored): CV AUC of "
        "XGBoost (frozen settings) and the plain logit as months of history are added, number of predictors in "
        "brackets. Raw months: the monthly status, bill and payment of each month in the window. Six-month summaries: "
        + ", ".join(tt(v) for v in s6) + ", always computed over all six months; window summaries recompute them over "
        "the months in the window only. H6 with either kind of summary is the frozen 29-predictor model. Source: "
        "\\texttt{results/ablation\\_history.csv}.",
        "tab:a-ablation-history", "L{0.22\\linewidth}cccccc",
        ["History window", "Raw months", hdr("+ six-month", "summaries"), hdr("+ window", "summaries"),
         "Raw months", hdr("+ six-month", "summaries"), hdr("+ window", "summaries")], rows,
        pre_header=[" & \\multicolumn{3}{c}{XGBoost} & \\multicolumn{3}{c}{Plain logit} \\\\",
                    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"],
        tabcolsep="3pt", footnotesize=True,
        note=(f"Fold-paired differences, XGBoost, raw months: H1 $-$ H0 {apd_t(h10)}; H3 $-$ H1 {apd_t(h31)}; "
              f"H6 $-$ H3 {apd_t(h63)}; all older months together, H6 $-$ H1 {apd_t(h61)}. With the six-month summaries, H6 $-$ H1 {apd_t(h61s)}: once the summaries are in, "
              f"the older raw months add nothing measurable for XGBoost (for the logit they still add {apd_t(h61s_l)}). Six-month "
              f"minus September-window summaries at H1 {apd_t(s_w1)}: the older history enters through the summaries. "
              "H1 adds September's status, bill and payment together, not the status alone. Code 1 occurs almost only "
              "in September (Table~\\ref{tab:a-paycodes-rates}), so part of what H1 adds is that month's coding.")))
A("")

adm = abl("ablation_demographics.csv")
W29, W25 = "with demographics (29)", "without demographics (25)"
fsum = load_json("fairness_summary.json") or {}
same_oof = all((fsum.get("ablation_file", {}).get(m, {}) or {}).get("identical") for m in ("xgb", "logit"))
DEMO_ROWS = [("auc", "$\\Delta$AUC", 4, 1.0), ("ap", "$\\Delta$AP", 4, 1.0), ("brier", "$\\Delta$Brier", 4, 1.0),
             ("cost_5to1", "$\\Delta$ cross-fitted 5:1 cost per fold", 1, 1.0),
             ("recall_5to1", "$\\Delta$ recall at the 5:1 cut-off", 3, 1.0)]
rows = []
for lbl, spec in (("CV AUC with SEX, EDUCATION, MARRIAGE, AGE (29 predictors)", W29),
                  ("CV AUC without them (25 predictors)", W25)):
    rows.append([lbl] + [num(acv(adm[adm.model == m], spec)["mean"], 4) for m in ("XGBoost", "logit")])
for metric, lbl, nd, sc in DEMO_ROWS:
    cells = [lbl + (", with $-$ without" if metric == "auc" else "")]
    for m in ("XGBoost", "logit"):
        p = apd(adm, W29, W25, metric=metric, model=m)
        cells.append(nb_cell(p["mean"], p["nb_lo"], p["nb_hi"], p["n_pos"], nd=nd))
    rows.append(cells)
    if metric == "auc":
        rows.append(["\\quad pooled-OOF bootstrap 95\\,\\% interval"]
                    + [ci(apd(adm, W29, W25, metric="auc", model=m)["boot_lo"],
                          apd(adm, W29, W25, metric="auc", model=m)["boot_hi"]) for m in ("XGBoost", "logit")])
A(table("Incremental value of SEX, AGE, MARRIAGE and EDUCATION (post-freeze, training CV only, never test-scored): the "
        "frozen XGBoost and plain logit refitted without the four columns on identical folds and settings. Each "
        "difference is with minus without (mean, $k$/5 with the difference positive, Nadeau--Bengio interval); for "
        "Brier and cost a negative difference favours the model with the demographics. The 5:1 cut-off is chosen on "
        "the other four folds (cross-fitted); a fold holds about 4,195 clients. Source: "
        "\\texttt{results/ablation\\_demographics.csv}.",
        "tab:a-ablation-demo", "L{0.46\\linewidth}cc",
        ["Quantity", "XGBoost", "Plain logit"], rows, footnotesize=True,
        note=("The no-demographics out-of-fold vectors (\\texttt{ablation\\_oof\\_*\\_nodemo.npy}) are identical to "
              "those of the fairness check (Table~\\ref{tab:a-fairness})." if same_oof else None)))
A("")

# ------------------------------------------------------------------ PF8 fairness diagnostic
fg = pd.read_csv(os.path.join(RESULTS, "fairness_groups.csv"))
fpx = pd.read_csv(os.path.join(RESULTS, "fairness_proxies.csv")).set_index("target")


def fgr(variant, attr, group):
    r = fg[(fg.variant == variant) & (fg.attribute == attr) & (fg.group == group)]
    if len(r) != 1:
        raise SystemExit(f"fairness_groups.csv: no unique row {variant}/{attr}/{group}")
    return r.iloc[0]


ATTR_LBL = {"SEX": "Sex", "AGE": "Age band", "MARRIAGE": "Marital status", "EDUCATION": "Education (recoded)"}
rows = []
for attr in ("ALL", "SEX", "AGE", "MARRIAGE", "EDUCATION"):
    groups_attr = fg[(fg.variant == "xgb_frozen") & (fg.attribute == attr)]["group"].tolist()
    if attr != "ALL":
        rows.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{ATTR_LBL[attr]}}}}} \\\\")
    for grp in groups_attr:
        x, lg, xn = fgr("xgb_frozen", attr, grp), fgr("logit_frozen", attr, grp), fgr("xgb_nodemo", attr, grp)
        name = "All training clients" if attr == "ALL" else ("\\quad " + esc(grp).replace("-", "--")
                                                              + (" (ref.)" if bool(x.is_reference) else ""))
        rows.append([name, f"{int(x.n):,}", pnum(x.default_rate), snum(x.calib_gap_pp, 2), pnum(x.flag_rate),
                     pnum(x.fpr), pnum(lg.flag_rate), pnum(lg.fpr), pnum(xn.flag_rate), snum(xn.calib_gap_pp, 2)])
    if attr == "ALL":
        rows.append("\\midrule")
cut = fsum.get("cutoffs", {})
mech = fsum.get("mechanism_checks", {})
rr = lambda v, a, g, col="flag_rate_ratio_to_ref": num(fgr(v, a, g)[col], 2)
proxy_targets = [("female (SEX = 2)", "female"), ("age < 30", "age under 30"), ("single (MARRIAGE = 2)", "single"),
                 ("graduate school (EDUCATION = 1)", "graduate school")]
proxy_txt = ", ".join(f"{lbl} {num(fpx.loc[t, 'oof_auc'], 2)}" for t, lbl in proxy_targets)
lead_proxy = {str(fpx.loc[t, "top3_predictors"]).split(" (")[0] for t, _ in proxy_targets}
A(table("Group outcomes at the 5:1 cut-off (training out-of-fold scores, post-freeze diagnostic, never test-scored). "
        "This is a diagnostic on training out-of-fold scores, not a compliance assessment. PD $-$ obs.: mean predicted "
        "probability of default minus the observed default rate, in percentage points; flagged: share of the group at "
        "or above the cut-off; FPR: false-alarm rate, the share of the group's non-defaulters flagged. `No demogr.': "
        "XGBoost refitted without SEX, AGE, MARRIAGE and EDUCATION (Table~\\ref{tab:a-ablation-demo}). `Others/unknown' "
        "education pools documented `others' with the undocumented codes 0, 5 and 6 --- a coding artefact, not an "
        "education effect. Sources: \\texttt{results/fairness\\_groups.csv}, \\texttt{fairness\\_summary.json}, "
        "\\texttt{fairness\\_proxies.csv}.",
        "tab:a-fairness", "L{0.20\\linewidth}ccccccccc",
        ["Group", "$n$", hdr("Default", "(\\%)"), hdr("PD $-$ obs.", "(pp)"), hdr("Flagged", "(\\%)"),
         hdr("FPR", "(\\%)"), hdr("Flagged", "(\\%)"), hdr("FPR", "(\\%)"), hdr("Flagged", "(\\%)"),
         hdr("PD $-$ obs.", "(pp)")], rows,
        pre_header=[" & & & \\multicolumn{3}{c}{XGBoost (frozen)} & \\multicolumn{2}{c}{Logit (frozen)} & "
                    "\\multicolumn{2}{c}{XGBoost, no demogr.} \\\\",
                    "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}"],
        tabcolsep="2.5pt", footnotesize=True,
        note=(f"Cut-offs: the frozen 5:1 cut-off {num(cut.get('xgb_frozen'), 2)} for both frozen models (chosen on their "
              "own training out-of-fold scores); the demographics-free refits use the same cost rule on their own "
              f"out-of-fold scores ({num(cut.get('xgb_nodemo'), 2)} XGBoost, {num(cut.get('logit_nodemo'), 2)} logit). "
              f"Flag-rate ratios to the reference group: XGBoost men/women {rr('xgb_frozen', 'SEX', 'male')} against a "
              f"default-rate ratio of {rr('xgb_frozen', 'SEX', 'male', 'default_rate_ratio_to_ref')}; plain logit "
              f"{rr('logit_frozen', 'SEX', 'male')} (false-alarm ratio {rr('logit_frozen', 'SEX', 'male', 'fpr_ratio_to_ref')}) "
              f"and married/single {rr('logit_frozen', 'MARRIAGE', 'married')}. Scoring every client with the "
              "reference values of the four attributes (`neutralised') gives "
              f"{rr('logit_neutralised', 'SEX', 'male')}\\,/\\,{rr('logit_neutralised', 'MARRIAGE', 'married')} (sex\\,/\\,married) "
              f"for the logit and {rr('xgb_neutralised', 'SEX', 'male')}\\,/\\,{rr('xgb_neutralised', 'MARRIAGE', 'married')} "
              f"for XGBoost; the demographics-free XGBoost gives {rr('xgb_nodemo', 'SEX', 'male')}\\,/\\,"
              f"{rr('xgb_nodemo', 'MARRIAGE', 'married')}, but its under-30 ratio rises from {rr('xgb_frozen', 'AGE', '<30')} "
              f"to {rr('xgb_nodemo', 'AGE', '<30')}. Proxies: the 25 non-demographic predictors recover {proxy_txt} "
              "(out-of-fold AUC)"
              + (f", with {tt(next(iter(lead_proxy)))} the leading proxy each time" if len(lead_proxy) == 1 else "")
              + ". PD density near the cut-off: "
              f"{pnum(mech['pd_density_per_0_01_around_cutoff']['logit'])}\\,\\% of clients per 0.01 of PD for the logit "
              f"against {pnum(mech['pd_density_per_0_01_around_cutoff']['xgb'])}\\,\\% for XGBoost. Largest $|$mean PD "
              "$-$ observed$|$ of the frozen XGBoost in groups of 1,000 or more clients: "
              f"{num(mech['xgb_frozen_max_abs_calib_gap_pp_groups_n_ge_1000'], 2)} pp.")))
A("")

# ------------------------------------------------------------------ PF9 monotone constraints re-specified
mr = load_json("monotone_respec_cv.json") or {}
mv, mp = mr["variants"], mr["paired_vs_frozen_oof"]
MV_LBL = {"free_numeric": "Free (as frozen)", "mono_frozen_numeric": "Monotone (frozen)",
          "free_recoded": "Free", "mono_frozen_signs_recoded": "Frozen signs",
          "mono_no_util_recoded": "Utilisation freed", "mono_respec": "Respecified"}
rows, prev = [], None
CODING_GRP = {"numeric": "Numeric coding (as frozen)", "recoded": "Recoded: months late + flags for $-2$, $-1$"}
for v, lbl in MV_LBL.items():
    x = mv[v]
    if x["coding"] != prev:
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{CODING_GRP.get(x['coding'], esc(x['coding']))}}}}} \\\\")
    prev = x["coding"]
    nc = f"{len(x['plus'])} / {len(x['minus'])}" if (x["plus"] or x["minus"]) else "---"
    df_, dm_ = mp[v]["auc_minus_frozen_free"], mp[v]["auc_minus_frozen_mono"]
    rows.append(["\\quad " + lbl, nc, mean_sd(x["mean_auc"], sd1(x["fold_auc"])), num(x["mean_ap"], 4),
                 num(x["mean_brier"], 4), "---" if v == "free_numeric" else nb_d(df_),
                 "---" if v == "mono_frozen_numeric" else nb_d(dm_)])
sh = mr["sweep_headline"]
csc = mr["cross_state_check"]["counts"]
rows_b = []
for v in ("free_numeric", "mono_frozen_numeric", "free_recoded", "mono_respec"):
    af, cs_ = sh[f"{v}|random_200|as_frozen|PAY_1"], sh[f"{v}|random_200|consistent|PAY_1"]
    bt = cs_["between_delay_levels"]
    cc = csc[f"{v}|random_200|consistent"]
    rows_b.append([MV_LBL[v], f"{af['all_steps']['violating']:,}", f"{cs_['all_steps']['violating']:,}",
                   f"{cs_['non_delinquent']['violating']:,}",
                   f"{af['entry_into_arrears']['violating']} $\\to$ {cs_['entry_into_arrears']['violating']}",
                   f"{bt['violating']:,} ({bt['violating_thin']:,})",
                   f"{cc['violating']:,} ({num(cc['largest_fall'], 3)})"])
s0 = sh["free_numeric|random_200|consistent|PAY_1"]
lad = mr["ladder"]
price_f = lad["mono_frozen_numeric - free_numeric (price of the frozen constraint)"]["auc"]
price_r = lad["mono_respec - free_recoded (price of the respecified constraint)"]["auc"]
gain_r = lad["mono_respec - mono_frozen_numeric (all changes together)"]["auc"]
rs = mv["mono_respec"]
free_r = sorted({c for v in ("mono_frozen_signs_recoded",) for c in mv[v]["plus"] + mv[v]["minus"]}
                - set(rs["plus"] + rs["minus"]))
other_cs = "; ".join(f"{MV_LBL[v].lower()} {csc[f'{v}|random_200|consistent']['violating']} "
                     f"(largest {num(csc[f'{v}|random_200|consistent']['largest_fall'], 3)})"
                     for v in ("mono_frozen_signs_recoded", "mono_no_util_recoded"))
n_pairs = csc["free_numeric|random_200|consistent"]["pairs"]
A(table("Monotone constraints re-specified (post-freeze, training CV only). The respecified model is a training-CV "
        "check only and was never scored on the test set; its guarantee holds from revolving credit upwards. Part A: "
        "XGBoost with the frozen settings under six constraint sets, on the numeric coding of the repayment codes or "
        "recoded (months late plus indicators for $-2$ and $-1$, as in Table~\\ref{tab:a-paycoding}); $+$\\,/\\,$-$ = "
        "number of columns constrained to raise\\,/\\,lower the risk; `utilisation freed' keeps the frozen signs except on "
        "the two utilisation columns; SD over the five folds, ddof 1. Part B: "
        "wrong-direction steps of the predicted probability. Source: \\texttt{results/monotone\\_respec\\_cv.json}, "
        "\\texttt{monotone\\_check\\_consistent.csv}.",
        "tab:a-monotone-respec", "L{0.20\\linewidth}cccccc",
        ["Variant", "$+$ / $-$", "CV AUC $\\pm$ SD", "CV AP", "Brier", hdr("$\\Delta$ vs free", "(frozen)"),
         hdr("$\\Delta$ vs monotone", "(frozen)")], rows, tabcolsep="2.5pt", footnotesize=True,
        after=subtab("L{0.20\\linewidth}cccccc",
                     ["Variant", hdr("All steps,", "as frozen"), hdr("All steps,", "consistent"),
                      hdr("Non-", "delinquent"), hdr("Entry into", "arrears"), hdr("Between delay", "levels (thin)"),
                      hdr("Cross-state pairs", "(largest fall)")],
                     rows_b, tabcolsep="3pt",
                     title=f"Part B: wrong-direction steps over {s0['all_steps']['steps'] // 10} random training "
                           "profiles, consistent sweep unless stated"),
        note=(f"Respecified constraints: $+$ on " + ", ".join(tt(c) for c in rs["plus"]) + "; $-$ on "
              + ", ".join(tt(c) for c in rs["minus"]) + "; free: the $-2$\\,/\\,$-1$ indicators and "
              + ", ".join(tt(c) for c in free_r) + ". Ladder, fold-paired AUC: price of the frozen constraint "
              f"{nb_dt(price_f)}; price of the respecified one against the free recoded booster {nb_dt(price_r)}; "
              f"respecified $-$ frozen monotone {nb_dt(gain_r)}. Part B: each profile's \\texttt{{PAY\\_1}} is swept over the "
              f"codes seen in training, {s0['all_steps']['steps']:,} adjacent steps in all "
              f"({s0['non_delinquent']['steps']:,} between non-delinquent codes, {s0['entry_into_arrears']['steps']:,} "
              f"entries into arrears, {s0['between_delay_levels']['steps']:,} between delay levels of which "
              f"{s0['between_delay_levels']['steps_thin']:,} in thin cells, where either code has fewer than 100 training "
              "clients in that month; thin-cell violations in brackets). `As frozen' holds \\texttt{MAX\\_DELAY} and "
              "\\texttt{N\\_MONTHS\\_DELAYED} fixed while \\texttt{PAY\\_1} moves, as in Figure~\\ref{fig:monotone-pay1}; "
              "`consistent' recomputes them. The respecified constraint leaves the non-delinquent steps free by design. "
              f"Cross-state pairs: for every month, each code $-2$ or $-1$ against each observed $k \\ge 1$, {n_pairs:,} "
              f"pairs; a pair violates when the probability at $k$ is lower. Other variants: {other_cs}.")))
A("")

# ------------------------------------------------------------------ PF10 no-parameter arrears rules
srb = pd.read_csv(os.path.join(RESULTS, "simple_rule_benchmark.csv"))
RULE_LBL = {"PAY_1 >= 1": tt("PAY_1") + " $\\ge 1$", "PAY_1 >= 2": tt("PAY_1") + " $\\ge 2$",
            "MAX_DELAY >= 2": tt("MAX_DELAY") + " $\\ge 2$",
            "N_MONTHS_DELAYED >= 1": tt("N_MONTHS_DELAYED") + " $\\ge 1$ (any delay in six months)",
            "flag everybody": "Flag everybody", "flag nobody": "Flag nobody"}
rows = []
for r_ in srb.itertuples():
    is_model = r_.kind == "model"
    if is_model:
        slug = "xgboost" if r_.rule_or_model.startswith("XGBoost") else "logistic"
        lbl = f"{LONG_NAME[slug]} at {num(r_.threshold, 2)}"
    else:
        lbl = RULE_LBL.get(r_.rule_or_model, esc(r_.rule_or_model))
    cost = r_.cost_5to1_crossfitted_per_1000 if is_model else r_.cost_5to1_per_1000
    rows.append([lbl, pnum(r_.flag_rate), num(r_.tpr_recall, 3), num(r_.precision, 3),
                 num(r_.false_alarms_per_defaulter_caught, 2), num(cost, 1),
                 "---" if is_model else nb_cell(r_.xgb_saving_vs_rule_per_1000_mean, r_.xgb_saving_vs_rule_nb_lo,
                                                r_.xgb_saving_vs_rule_nb_hi, r_.xgb_cheaper_folds_of_5, nd=0),
                 "---" if is_model else nb_cell(r_.logit_saving_vs_rule_per_1000_mean, r_.logit_saving_vs_rule_nb_lo,
                                                r_.logit_saving_vs_rule_nb_hi, r_.logit_cheaper_folds_of_5, nd=0),
                 num(r_.xgb_recall_same_flag_rate, 3), num(r_.logit_recall_same_flag_rate, 3)])
cand = srb[srb.kind == "rule"]
best_rule = cand.loc[cand.cost_5to1_per_1000.idxmin()]
xgb_any = float(best_rule.xgb_saving_vs_rule_per_1000_mean) / float(best_rule.cost_5to1_per_1000)
A(table("No-parameter arrears rules against the models on the training rows at a 5:1 cost ratio (post-freeze, never "
        "test-scored). Cost is $5\\,\\mathrm{FN} + \\mathrm{FP}$ per 1,000 training clients; for the two models it is "
        "cross-fitted (the cut-off chosen on the other four folds). Savings are rule minus model per 1,000 clients over "
        "the five folds ($k$/5 = folds in which the model is cheaper). The last two columns give the recall of XGBoost "
        "and of the logit when they flag the same share of clients as the rule. Source: "
        "\\texttt{results/simple\\_rule\\_benchmark.csv}.",
        "tab:a-simplerules", "L{0.18\\linewidth}ccccccccc",
        ["Rule or model", hdr("Flagged", "(\\%)"), "Recall", hdr("Preci-", "sion"), hdr("False", "alarms", "per catch"),
         hdr("Cost", "per", "1,000"), hdr("XGBoost", "saving", "$k$/5 [NB]"), hdr("Logit", "saving", "$k$/5 [NB]"),
         "XGB", "Logit"], rows, tabcolsep="2.5pt", footnotesize=True,
        pre_header=[" & & & & & & & & \\multicolumn{2}{c}{Recall at the} \\\\",
                    " & & & & & & & & \\multicolumn{2}{c}{same share} \\\\", "\\cmidrule(lr){9-10}"],
        note=("The rules have no fitted parameter, so the training rows are a fair test for them; they were not scored "
              f"on the test set. {esc(best_rule.definition).capitalize()} ({RULE_LBL[best_rule.rule_or_model].split(' (')[0]}) "
              "was the cheapest of the four candidate rules on these same rows, a mild selection effect in its favour; "
              f"XGBoost still costs {pnum(xgb_any)}\\,\\% less than it. The models' cut-off "
              f"({num(srb.loc[srb.kind == 'model', 'threshold'].iloc[0], 2)}) was chosen on pooled training out-of-fold "
              "scores.")))
A("")

# ------------------------------------------------------------------ PF11 cheapest model at each cost ratio
pr = pd.read_csv(os.path.join(RESULTS, "preferred_model_by_ratio_oof.csv"))
nm = lambda name: SHORT_NAME.get(NAME_TO_SLUG.get(str(name), ""), esc(name))
rows_a, rows_b = [], []
for r_ in pr.itertuples():
    rows_a.append([f"{int(r_.ratio)}:1", num(r_.cutoff_theory, 3), f"{nm(r_.preferred_oof)} ({num(r_.preferred_cutoff, 2)})",
                   pnum(r_.preferred_flag_rate), num(r_.preferred_recall, 3), num(r_.preferred_cost_per_1000, 1),
                   f"{nm(r_.runner_up)} ({num(r_.margin_pct, 2)})", num(r_.top3_spread_pct, 2)])
    same = str(r_.same_preferred_both_ways).lower() == "true"
    rows_b.append([f"{int(r_.ratio)}:1", nm(r_.preferred_crossfitted) + ("" if same else " (differs)"),
                   nb_cell(r_.runner_up_minus_preferred_mean_per_1000, r_.runner_up_minus_preferred_nb_lo,
                           r_.runner_up_minus_preferred_nb_hi, None, nd=1),
                   k5(r_.preferred_cheaper_folds_of_5),
                   f"{int(r_.logistic_rank_oof)} ({num(r_.logistic_gap_pct, 1)})",
                   f"{int(r_.tree_pruned_rank_oof)} ({num(r_.tree_pruned_gap_pct, 1)})",
                   f"{esc(r_.best_blanket)} ({num(r_.preferred_saving_vs_blanket_pct, 1)})"])
A(table("Cheapest of the eleven headline models at each false-negative to false-positive cost ratio $r$ on pooled "
        "training out-of-fold scores (reporting only; no cost winner is named, and nothing here was scored on the test "
        "set). Top: the cheapest model with its cut-off (chosen on the same scores), the share of clients it flags, its "
        "recall and cost per 1,000 clients, the runner-up with its cost margin in per cent, and the cost spread of the "
        "top three in per cent. Bottom: the cheapest model when every fold's cut-off is chosen on the other four "
        "(cross-fitted), the runner-up minus cheapest cost per 1,000 over the folds with the folds in which the "
        "cheapest is cheaper, the ranks of the plain logit and the pruned tree (gap to the cheapest, \\%), and the "
        "best blanket rule with the cheapest model's saving against it (\\%). Source: "
        "\\texttt{results/preferred\\_model\\_by\\_ratio\\_oof.csv}.",
        "tab:a-ratio-pref", "cccccccc",
        ["$r$", "$1/(1+r)$", hdr("Cheapest", "(cut-off)"), hdr("Flagged", "(\\%)"), "Recall", hdr("Cost per", "1,000"),
         hdr("Runner-up", "(margin, \\%)"), hdr("Top-3", "spread (\\%)")], rows_a, tabcolsep="3pt", footnotesize=True,
        after=subtab("ccccccc",
                     ["$r$", hdr("Cheapest,", "cross-fitted"), hdr("Runner-up $-$ cheapest", "per 1,000 [NB]"),
                      hdr("Cheapest in", "$k$/5 folds"), hdr("Logit rank", "(gap, \\%)"), hdr("Tree rank", "(gap, \\%)"),
                      hdr("Best blanket rule", "(saving, \\%)")],
                     rows_b, tabcolsep="3pt"),
        note="XGB = XGBoost, GBM = gradient boosting, RF = random forest, KNN = $K$-nearest neighbours ($K = 101$)."))
A("")

# ------------------------------------------------------------------ PF12 who is caught in the riskiest 10/20/30 %
tk = pd.read_csv(os.path.join(RESULTS, "topk_capture_oof.csv"))
tk = tk[tk.slug != "perfect"]
order = list(tk[tk.top_share == 0.1]["slug"])
TK_LBL = {"N_MONTHS_DELAYED": "Rule score: \\texttt{N\\_MONTHS\\_DELAYED}", "MAX_DELAY": "Rule score: \\texttt{MAX\\_DELAY}",
          "PAY_1": "Rule score: \\texttt{PAY\\_1}", "random": "Random list"}
rows = []
for s in order:
    cells = [LONG_NAME.get(s) or TK_LBL.get(s, esc(s))]
    for share in (0.1, 0.2, 0.3):
        r_ = tk[(tk.top_share == share) & (tk.slug == s)].iloc[0]
        cells += [pnum(r_.recall_capture), pnum(r_.precision)]
        if share == 0.1:
            cells.append("---" if not isnum(r_.xgb_minus_recall_mean) else
                         nb_cell(r_.xgb_minus_recall_mean, r_.xgb_minus_recall_nb_lo, r_.xgb_minus_recall_nb_hi,
                                 r_.xgb_recall_wins_of_5, nd=1, scale=100))
    if s == "naive_bayes" or s == "PAY_1":
        rows.append(cells)
        rows.append("\\midrule")
    else:
        rows.append(cells)
perfect = pd.read_csv(os.path.join(RESULTS, "topk_capture_oof.csv"))
perfect = perfect[perfect.slug == "perfect"].set_index("top_share")["recall_capture"]
ties = tk[tk.kind == "rule score"].pivot_table(index="slug", columns="top_share", values="boundary_tie_rows")
A(table("Who is caught in the riskiest 10, 20 and 30\\,\\% of the training clients (training out-of-fold scores of the "
        "frozen configurations; post-freeze, never test-scored): the share of all defaulters captured and the "
        "precision of the list, for the eleven models, three arrears counts used as scores and a random list. "
        "XGB $-$ row: XGBoost's capture minus the row's at 10\\,\\%, in percentage points, over the five folds. "
        "Source: \\texttt{results/topk\\_capture\\_oof.csv}.",
        "tab:a-topk", "L{0.25\\linewidth}ccccccc",
        ["List ranked by", hdr("Captured", "(\\%)"), hdr("Precision", "(\\%)"), hdr("XGB $-$ row", "$k$/5 [NB]"),
         hdr("Captured", "(\\%)"), hdr("Precision", "(\\%)"), hdr("Captured", "(\\%)"), hdr("Precision", "(\\%)")],
        rows,
        pre_header=[" & \\multicolumn{3}{c}{Riskiest 10\\,\\%} & \\multicolumn{2}{c}{Riskiest 20\\,\\%} & "
                    "\\multicolumn{2}{c}{Riskiest 30\\,\\%} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"],
        tabcolsep="3pt", footnotesize=True,
        note=("Scores with ties at the list boundary (the rule scores, KNN and the tree) are counted at their expected "
              "value over the tied clients; tied clients at the boundary for the rule scores, 10/20/30\\,\\%: "
              + "; ".join(f"{tt(s)} " + "/".join(f"{int(ties.loc[s, c]):,}" for c in sorted(ties.columns))
                          for s in ties.index)
              + f". A perfect list would capture {pnum(perfect.loc[0.1])}, {pnum(perfect.loc[0.2])} and "
              f"{pnum(perfect.loc[0.3])}\\,\\% of the defaulters.")))
A("")

# ------------------------------------------------------------------ PF13 R cross-check
rc_ = pd.read_csv(os.path.join(RESULTS, "r_crosscheck.csv"))
R_NOTE = {  # shortened notes (labels and settings only); the full notes are in results/r_crosscheck.csv
    "glm (logistic)": "same likelihood; IRLS against lbfgs",
    "cv.glmnet ridge": "$\\lambda$ by inner 10-fold deviance; glmnet also scales the dummies",
    "cv.glmnet lasso": "as ridge; both penalties end up tiny",
    "MASS::lda": "same model",
    "MASS::qda": "no regularisation: reproduces the Python \\texttt{reg\\_param} = 0 grid point",
    "[diagnostic] QDA reg 0.5 (by hand)": "QDA with $r = 0.5$ written by hand: matches the frozen Python model",
    "e1071::naiveBayes": "Python adds \\texttt{var\\_smoothing} to every variance",
    "class::knn K=101": "same $K$ and inputs; \\texttt{class::knn} keeps all distance ties",
    "class::knn K=5": "mirrors the round-1 $K = 5$ variant",
    "tree + CV pruning": "deviance tree, pruned by inner CV; Python prunes by \\texttt{ccp\\_alpha}",
    "tree lab defaults (prune.misclass)": "lab defaults stop at a few leaves",
    "randomForest": "hard votes and smaller leaves than scikit-learn's soft votes, leaf $\\ge 20$",
    "[diagnostic] randomForest soft votes": "the same R trees averaged as leaf shares",
    "[diagnostic] randomForest soft votes, nodesize 80": "soft votes with leaves about as large as scikit-learn's",
    "gbm 200 trees": "\\texttt{interaction.depth} 3 = 4 leaves; scikit-learn depth 3 allows 8",
    "gbm 300 trees": "as above, read at 300 trees",
    "xgboost": "same library and parameters; only the subsampling draws differ",
    "e1071::svm radial": "libsvm in both, same $C$, $\\gamma$ and weights",
    "nnet": "one sigmoid layer (BFGS) against two ReLU layers (Adam)",
}
is_diag = rc_.model.str.startswith("[diagnostic]")
is_contrast = rc_.notes.str.startswith("contrast only")
mirror = rc_[~is_diag & ~is_contrast]
n_within = int((mirror["diff"].abs() <= 0.005).sum())
rho = spearmanr(mirror["r_cv_auc"], mirror["python_cv_auc"]).correlation
rows = []
for r_, d_, c_ in zip(rc_.itertuples(), is_diag, is_contrast):
    name = esc(r_.model.replace("[diagnostic] ", ""))
    name = f"\\textit{{{name}}}$^{{d}}$" if d_ else (f"{name}$^{{c}}$" if c_ else name)
    rows.append([name, mean_sd(r_.r_cv_auc, r_.r_cv_auc_sd), num(r_.python_cv_auc, 4), snum(r_.diff),
                 f"{int(r_.folds_r_higher)}/5", R_NOTE.get(r_.model, esc(r_.notes))])
A(table("R cross-check on the same five training folds: the course's R models refitted on "
        f"\\texttt{{data/processed/cv\\_folds.csv}} with the frozen settings; {n_within} of {len(mirror)} mirrors within "
        f"0.005 of the Python CV AUC, Spearman {num(rho, 2)} between the two columns over the mirrors "
        f"({tt('code/r_crosscheck/README.md')}); the test file is never opened. $^{{d}}$Diagnostic refit with the "
        "Python setting; $^{c}$contrast with the lab defaults (neither counts as a mirror). Source: "
        "\\texttt{results/r\\_crosscheck.csv}, whose \\texttt{notes} column gives each explanation in full.",
        "tab:a-rcheck", "L{0.23\\linewidth}ccccL{0.28\\linewidth}",
        ["R model", "R CV AUC $\\pm$ SD", hdr("Python", "CV AUC"), hdr("R $-$", "Python"), hdr("Folds", "R higher"),
         "Note"], rows, tabcolsep="3pt", footnotesize=True))
A("")

# ------------------------------------------------------------------ figures
# Caption facts read from results/ (no typed numbers): the share of empty PAY_1 x LIMIT_BAL cells.
_pdp = load_json("pdp_summary.json") or {}
_empty = (((_pdp.get("interaction") or {}).get("candidates") or {}).get("PAY_1 x LIMIT_BAL") or {}).get("share_cells_empty")
PDP_HOLD = ("The partial dependence holds the derived delay and utilisation features fixed while one column moves, so "
            "some swept rows cannot exist in the data.")
SHAP_BLOCK = (" TreeSHAP divides the credit among the collinear \\texttt{PAY\\_1}, \\texttt{MAX\\_DELAY} and "
              "\\texttt{N\\_MONTHS\\_DELAYED}; read them as a block.")
TEST_FROZEN = " Test set, frozen configurations; reporting only."
# Items are (file, caption) or (file, caption, label); the default label is "fig:" + the file stem.
FIGS = [
    ("Post-freeze checks (training rows only)", [
        ("fold_stability.png", "Fold stability of the eleven headline configurations (post-freeze check on the training "
         "out-of-fold scores of the frozen configurations; nothing re-fitted, nothing test-scored). Left: AUC in each of "
         "the five validation folds (dots), the mean (bar) and the rank in each fold. Right: XGBoost minus each model "
         "in the same fold (grey dots), mean and Nadeau--Bengio 95\\,\\% interval, with the number of folds XGBoost is "
         "ahead (Table~\\ref{tab:a-foldstab}).", "fig:a-foldstab"),
        ("ablation_summary.png", "Where XGBoost's gain over the plain logit comes from: post-freeze ablations, training "
         "five-fold CV with the frozen settings, never test-scored. (a) Interaction depth "
         "(Table~\\ref{tab:a-ablation-depth}); (b) recent against older repayment history "
         "(Table~\\ref{tab:a-ablation-history}); (c) the incremental value of SEX, EDUCATION, MARRIAGE and AGE "
         "(Table~\\ref{tab:a-ablation-demo}).", "fig:a-ablation"),
        ("fairness_flag_rates.png", "Share of each group flagged at the 5:1 cut-off by the frozen XGBoost and by its "
         "demographics-free refit, with the observed default rate (diamonds). Training out-of-fold scores; a post-freeze "
         "diagnostic, not a compliance assessment (Table~\\ref{tab:a-fairness}).", "fig:a-fairness"),
        ("roc_pr_oof.png", "ROC and precision--recall curves on the training out-of-fold scores of four frozen "
         "configurations, with each model's 5:1 cut-off (dots) and the four no-parameter arrears rules A--D (squares; "
         "Table~\\ref{tab:a-simplerules}). Training rows only; the test-set curves of the frozen configurations are "
         "Figures~\\ref{fig:all-roc} and~\\ref{fig:all-pr}.", "fig:a-rocpr-oof"),
        ("r_crosscheck.png", "R cross-check on the same five training folds: (a) R against Python CV AUC, one dot per "
         "model, band = 45-degree line $\\pm$ 0.005; (b) R minus Python, mean and range over the five folds; open "
         "circles are diagnostic R refits with the Python setting (Table~\\ref{tab:a-rcheck}). The test file is never "
         "opened.", "fig:a-rcheck"),
    ]),
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
        ("all_roc.png", "ROC curves of the headline models." + TEST_FROZEN),
        ("all_pr.png", "Precision--recall curves of the headline models; the horizontal line is the prevalence." + TEST_FROZEN),
        ("all_auc_bar_full.png", "Test AUC of every model variant fitted, ordered by CV AUC, with the benchmark marked."),
        ("calibration.png", "Calibration curves of the four models with the highest CV AUC: predicted against observed default rate by decile of score."),
        ("threshold_curves.png", "Recall, precision and F1 against the decision threshold."),
        ("cost_curves.png", "Expected cost against the decision threshold at a 5:1 ratio."),
        ("cost_sensitivity.png", "Cost-minimising threshold and cost as the cost ratio varies from 1:1 to 20:1."),
        ("recalibration.png", "Reliability of the nine weighted or resampled models before and after Platt recalibration, on training out-of-fold scores."),
    ]),
    ("Interpretation", [
        ("rf_importance.png", "Random forest importances: impurity decrease (MDI), training-set permutation and "
         "out-of-bag permutation, side by side. MDI favours columns with many split points."),
        ("shap_beeswarm.png", "TreeSHAP beeswarm for the tuned XGBoost over training clients." + SHAP_BLOCK),
        ("shap_dependence_pay1.png", "TreeSHAP dependence plot for the most recent repayment status." + SHAP_BLOCK),
        ("shap_dependence_util_mean.png", "TreeSHAP dependence plot for mean utilisation."),
        ("shap_waterfall.png", "One client decomposed: base value plus each feature's contribution."),
        ("pdp_main.png", "Partial dependence with ICE curves for the four features that matter. " + PDP_HOLD
         + " The partial dependence of \\texttt{PAY\\_1} rises far less than its raw default rate "
         "(Table~\\ref{tab:a-paycodes-rates})."),
        ("pdp_interaction.png", "Two-way partial dependence; cells with no training client are crossed"
         + (f" ({pnum(_empty, 0)}\\,\\% of the \\texttt{{PAY\\_1}} $\\times$ \\texttt{{LIMIT\\_BAL}} cells contain no "
            "training client)" if isnum(_empty) else "") + ". " + PDP_HOLD),
        ("monotone_pay1.png", "Predicted probability against the most recent repayment status, with and without "
         "monotone constraints. The sweep holds \\texttt{MAX\\_DELAY} and \\texttt{N\\_MONTHS\\_DELAYED} fixed "
         "(`as frozen'); Table~\\ref{tab:a-monotone-respec} gives the consistent sweep."),
        ("logistic_coefficients.png", "Standardised logistic coefficients: dummy coefficients are per level, numeric "
         "ones per standard deviation. Read the bill and utilisation block, and \\texttt{PAY\\_2}--\\texttt{PAY\\_6} "
         "(conditional on the delay counters), as blocks; CV chose almost no shrinkage. The `others' education "
         "coefficient reflects a coding artefact (documented `others' merged with the undocumented codes 0, 5 and 6), "
         "not an education effect."),
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
SEC_LABEL = {"Post-freeze checks (training rows only)": "sec:a-postfreeze-figs", "Exploratory analysis": "sec:a-figs"}
n = 0
for title, items in FIGS:
    A("\\clearpage")
    A(f"\\subsection{{{title}}}" + (f"\\label{{{SEC_LABEL[title]}}}" if title in SEC_LABEL else ""))
    A("")
    for item in items:
        path, cap = item[0], item[1]
        n += 1
        # Label from the file name, so the body can \ref a figure by what it is
        # (fig:pdp-main, fig:shap-beeswarm, ...) instead of by an appendix ordinal; an explicit
        # third element overrides it (the post-freeze figures: fig:a-foldstab, fig:a-ablation, ...).
        label = item[2] if len(item) > 2 else "fig:" + os.path.splitext(path)[0].replace("_", "-")
        A(figure(path, cap, label))
        A("")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(L)} blocks, {n} figures")
