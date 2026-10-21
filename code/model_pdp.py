"""
model_pdp.py — Partial dependence (PDP) and individual conditional expectation (ICE) for the tuned XGBoost.  Owner: LU Xinyu.

Run from Project/code:  python3 model_pdp.py            (about 1 minute)

WHY.  Feature importance says WHICH variable the model uses; it does not say in which DIRECTION or with what SHAPE.
The proposal promises that the ranking "should be explainable by the interaction of repayment delays with limit and
utilisation, which feature-importance and partial-dependence analysis will test".  This script is that test, and it is
also the evidence for the report's other structural claim — that a monotone linear model cannot represent the U-shaped
utilisation effect the EDA found, which is one of the reasons tree-based methods were expected to win.

What it writes
  1. results/pdp_values.csv     — every number behind both figures, in long format: the four one-way PD curves (with the
     ICE decile bands, the per-bin training count and the per-bin empirical default rate) and the two two-way surfaces.
  2. results/pdp_summary.json   — shapes and verdicts (monotone / U-shaped), Friedman H for both candidate pairs on two
     scales, the quantified correlation caveat, and the U-shape verdict for utilisation.
  3. figures/pdp_main.png       — the four one-way PDPs: ICE lines, ICE p10-p90 band, the training histogram and the
     observed default rate per bin, with a second row that repeats each PD alone on its own scale (UTIL_MEAN moves by
     0.040 and would be a flat line on a panel that has to span 0.0-0.85 to hold the ICE curves).
  4. figures/pdp_interaction.png— the chosen two-way surface, its interaction component, and the delay curve by strata.

What it does
  1. Re-reads results/xgboost.json and refits the SAME tuned model (learning_rate 0.03, max_depth 3, min_child_weight 5,
     n_estimators 300, subsample 0.8, colsample_bytree 0.8, tree_method hist, scale_pos_weight 1.0, random_state 2607526)
     inside Pipeline(make_preprocessor(scale=False), XGBClassifier).  This is NOT a new model, so it does NOT go through
     evaluate(): a second entry for the same model would pollute the comparison tables.  Instead the refit is proved to be
     the same model by reproducing results/xgboost_p_test.npy exactly (max |Δp| is printed).
  2. Computes, for PAY_1, N_MONTHS_DELAYED, UTIL_MEAN and LIMIT_BAL, the full (n_grid x 20,975) matrix of predictions with
     that feature forced to each grid value.  The column means are the ICE curves, the row means are the partial dependence:
     ONE computation, so the PD and the ICE curves cannot drift apart.  The PD is cross-checked against
     sklearn.inspection.partial_dependence (method="brute", custom_values) and the max abs difference is printed.
  3. Two-way PD for BOTH candidate pairs, (PAY_1, LIMIT_BAL) and (N_MONTHS_DELAYED, UTIL_MEAN), and picks the one the data
     support better by Friedman's H statistic — the share of the joint PD variance that additivity cannot explain, weighted
     by how many training clients actually fall in each cell of the grid.  H is computed on two scales: the probability
     scale (what is plotted, but where the logistic link creates curvature even for a perfectly additive model) and the
     raw margin / log-odds scale (where an additive booster is exactly additive, so H there is genuine interaction).
  4. States and QUANTIFIES the standard PDP objection: partial dependence averages over the marginal distribution, so it
     evaluates the model on feature combinations that the data never contain.  Reported per PDP feature: its correlation
     with every other feature, the pairwise correlations inside the BILL_AMT1..6 block, the share of two-way grid cells
     with zero training clients, and the algebraic identities that the sweep breaks by construction.

PARTITION.  Everything here is computed on the TRAINING partition (20,975 clients).  Partial dependence is a property of
the FITTED MODEL, not of a data set, so the held-out test set is not needed for it, and using the training rows keeps the
test set untouched.  Nothing in this script is tuned, selected or thresholded; it only describes a model that was already
fitted and already evaluated in results/xgboost.json.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence
import sklearn, xgboost, scipy
from xgboost import XGBClassifier
from common import *

MEMBER = "LU Xinyu"
T0 = time.time()

BASELINE_SLUG = "xgboost"                                   # the tuned model whose response surface we are describing
PDP_FEATURES = ["PAY_1", "N_MONTHS_DELAYED", "UTIL_MEAN", "LIMIT_BAL"]
CANDIDATE_PAIRS = [("PAY_1", "LIMIT_BAL"), ("N_MONTHS_DELAYED", "UTIL_MEAN")]
GRID_N = 21                                                 # grid points for a continuous feature (discrete ones use their own values)
TRIM = (0.01, 0.99)                                         # continuous grids stay inside these training percentiles
N_ICE_LINES = 200                                           # ICE curves DRAWN (the bands use all 20,975 rows)
U_TOL = 0.002                                               # a "rise" in the U-shape test must exceed 0.2 percentage points
N_JOBS = 2

FEATURE_LABEL = {
    "PAY_1": "PAY_1 — repayment status last month\n(−2 no consumption / −1 paid in full / 0 revolving, k = k months in arrears)",
    "N_MONTHS_DELAYED": "N_MONTHS_DELAYED — months with a payment delay ≥ 1 (of 6)",
    "UTIL_MEAN": "UTIL_MEAN — mean of the six bills divided by the credit limit",
    "LIMIT_BAL": "LIMIT_BAL — credit limit (NT$)",
}
SHORT_LABEL = {"PAY_1": "PAY_1 (months in arrears last month)", "N_MONTHS_DELAYED": "N_MONTHS_DELAYED (of 6 months)",
               "UTIL_MEAN": "UTIL_MEAN (mean bill / limit)", "LIMIT_BAL": "LIMIT_BAL (NT$)"}
EXPECTED_SIGN = {"PAY_1": +1, "N_MONTHS_DELAYED": +1, "UTIL_MEAN": +1, "LIMIT_BAL": -1}   # the economic prior, only reported against

# validated categorical palette (same slots as model_boosting.py / model_rf.py): blue = PD, orange = highlight, aqua = empirical
C_PD, C_HL, C_EMP, C_ICE, C_TXT, C_TXT2, C_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b87", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "xtick.color": C_TXT2, "ytick.color": C_TXT2,
                     "text.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def log(msg):
    print(f"[pdp] {msg}   [t={time.time() - T0:,.0f}s]", flush=True)


# --------------------------------------------------------------------------------------------- prediction / PD machinery
def _predict(pipe, X, margin=False):
    """P(default) for every row, or the raw margin (log-odds) when margin=True."""
    if not margin:
        return pipe.predict_proba(X)[:, 1]
    Xt = pipe.named_steps["pre"].transform(X)
    return np.asarray(pipe.named_steps["clf"].predict(Xt, output_margin=True), dtype=float)


def _set(X, feature, value):
    """Copy of X with one feature forced to `value`, keeping the column's original dtype where that is exact."""
    Xg = X.copy()
    dt = X[feature].dtype
    if np.issubdtype(dt, np.integer) and float(value).is_integer():
        Xg[feature] = dt.type(value)
    else:
        Xg[feature] = float(value)
    return Xg


def ice_matrix(pipe, X, feature, grid, margin=False):
    """
    Matrix (n_grid, n_rows): entry [g, i] is the prediction for training client i with `feature` forced to grid[g],
    every other feature left at that client's own value.  The COLUMNS are the ICE curves; the ROW MEANS are the
    partial dependence.  Computing both from one matrix is deliberate — the plotted PD is by construction the average
    of the plotted ICE curves, so the figure cannot be internally inconsistent.
    """
    return np.stack([_predict(pipe, _set(X, feature, v), margin) for v in grid])


def pd_surface(pipe, X, fa, ga, fb, gb, margin=False):
    """Two-way partial dependence: entry [i, j] averages the prediction over all rows with fa=ga[i] and fb=gb[j]."""
    out = np.empty((len(ga), len(gb)))
    for i, a in enumerate(ga):
        Xa = _set(X, fa, a)
        for j, b in enumerate(gb):
            out[i, j] = _predict(pipe, _set(Xa, fb, b), margin).mean()
    return out


def make_grid(values, n=GRID_N, trim=TRIM):
    """Distinct values for a discrete feature; otherwise quantiles inside the trimmed range (no grid point in the tails)."""
    u = np.unique(values)
    if len(u) <= n:
        return u.astype(float), f"all {len(u)} distinct training values"
    lo, hi = np.quantile(values, trim)
    inner = values[(values >= lo) & (values <= hi)]
    g = np.unique(np.quantile(inner, np.linspace(0, 1, n)))
    return g, (f"{len(g)} quantiles of the training values between the {trim[0]:.0%} and {trim[1]:.0%} percentile "
               f"({lo:,.4g} to {hi:,.4g}); the tails are excluded so no grid point sits where the model saw ~no data")


def bin_index(values, grid):
    """Nearest grid point for every training value (cut points at the midpoints); the end bins absorb the trimmed tails."""
    edges = (grid[:-1] + grid[1:]) / 2.0
    return np.searchsorted(edges, np.asarray(values, dtype=float))


def bin_stats(values, y, grid):
    """Training count and empirical default rate per grid bin (NaN where the bin is empty)."""
    idx = bin_index(values, grid)
    n = np.bincount(idx, minlength=len(grid)).astype(float)
    s = np.bincount(idx, weights=np.asarray(y, dtype=float), minlength=len(grid))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return n, rate


def shape_stats(grid, pdv, expected_sign):
    """Direction, range and wrong-direction steps of one PD curve (the economic prior is reported against, never imposed)."""
    d = np.diff(pdv)
    i_min, i_max = int(np.argmin(pdv)), int(np.argmax(pdv))
    rho = float(spearmanr(grid, pdv).statistic)
    return dict(n_grid=len(grid), value_min=float(grid[0]), value_max=float(grid[-1]),
                pd_min=float(pdv.min()), pd_max=float(pdv.max()), pd_range=float(pdv.max() - pdv.min()),
                pd_at_left_end=float(pdv[0]), pd_at_right_end=float(pdv[-1]),
                value_at_pd_min=float(grid[i_min]), value_at_pd_max=float(grid[i_max]),
                monotone_increasing=bool((d >= 0).all()), monotone_decreasing=bool((d <= 0).all()),
                n_steps=int(len(d)), n_steps_up=int((d > 0).sum()), n_steps_down=int((d < 0).sum()),
                spearman_grid_vs_pd=rho, expected_sign=expected_sign,
                steps_against_expected_sign=int((d * expected_sign < 0).sum()),
                largest_step_against_expected_sign=float(max(0.0, float((-d * expected_sign).max()))),
                direction_matches_expected_sign=bool(np.sign(rho) == expected_sign))


def u_shape_test(grid, pdv, tol=U_TOL):
    """A U-shape needs an INTERIOR minimum with a real fall into it and a real rise out of it (both > tol)."""
    i = int(np.argmin(pdv))
    fall, rise = float(pdv[0] - pdv[i]), float(pdv[-1] - pdv[i])
    interior = bool(0 < i < len(pdv) - 1)
    return dict(index_of_minimum=i, value_at_minimum=float(grid[i]), pd_at_minimum=float(pdv[i]),
                pd_at_left_end=float(pdv[0]), pd_at_right_end=float(pdv[-1]),
                fall_from_left_end_into_minimum=fall, rise_from_minimum_to_right_end=rise,
                minimum_is_interior=interior, tolerance=float(tol),
                u_shape_survives=bool(interior and fall > tol and rise > tol))


def h_statistic(surface, pd_a, pd_b, weights):
    """
    Friedman's H for one pair: the share of the (weighted) variance of the joint PD that an ADDITIVE model cannot explain.

        H^2 = sum_w [ PD_ab - PD_a - PD_b ]^2 / sum_w [ PD_ab ]^2        (all three centred with the same weights)

    H = 0 means the joint surface is exactly the sum of the two one-way curves (no interaction); H = 1 means none of its
    variation is additive.  `weights` is how many training clients fall in each cell, so sparse corners of the grid —
    where partial dependence is pure extrapolation — cannot drive the statistic.  Returns (H, residual surface).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    wa, wb = w.sum(axis=1), w.sum(axis=0)
    f2 = surface - float((w * surface).sum())
    fa = pd_a - float((wa * pd_a).sum())
    fb = pd_b - float((wb * pd_b).sum())
    resid = f2 - fa[:, None] - fb[None, :]
    num, den = float((w * resid ** 2).sum()), float((w * f2 ** 2).sum())
    return (float(np.sqrt(num / den)) if den > 0 else float("nan")), resid


# --------------------------------------------------------------------------------------------- figures
def _axis_cosmetics(ax, f, grid):
    ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
    if f == "LIMIT_BAL":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    if f in ("PAY_1", "N_MONTHS_DELAYED"):
        ax.set_xticks(grid)


def plot_main(grids, pds, ice, hist, emp_n, emp_rate, shapes, ushape, base, n_train):
    """
    figures/pdp_main.png — the four one-way PDPs, each as a pair of stacked panels.
      top    : every ICE curve's p10-p90 band, 200 drawn ICE curves, the ICE median, the PD, the training histogram of
               the feature (grey bars), and the OBSERVED default rate of the clients in each bin of the same grid.  The
               last two are different objects on purpose: the PD is the model's average response when the feature is
               MOVED with everything else held, the observed rate is the raw rate of the clients who HAVE that value and
               who differ in everything else too — the gap between them IS the marginal-vs-conditional caveat, drawn.
      bottom : the SAME PD alone, zoomed to its own range, because UTIL_MEAN moves by 0.040 and LIMIT_BAL by 0.056 on a
               panel that has to span 0.0-0.85 to hold the ICE curves, where any shape would be invisible.
    """
    fig = plt.figure(figsize=(14.2, 11.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.15, 1.12, 2.15, 1.12], hspace=0.62, wspace=0.22)
    rng = np.random.default_rng(SEED)
    for k, f in enumerate(PDP_FEATURES):
        r, c = (k // 2) * 2, k % 2
        ax, bx = fig.add_subplot(gs[r, c]), fig.add_subplot(gs[r + 1, c])
        g, M, p, s = grids[f], ice[f], pds[f], shapes[f]

        cols = rng.choice(M.shape[1], size=min(N_ICE_LINES, M.shape[1]), replace=False)
        lo, mid, hi = np.percentile(M, [10, 50, 90], axis=1)
        ax.fill_between(g, lo, hi, color=C_PD, alpha=0.15, lw=0, label=f"ICE p10–p90 (all {n_train:,} clients)")
        ax.plot(g, M[:, cols], color=C_ICE, lw=0.6, alpha=0.11)
        ax.plot(g, mid, color=C_PD, lw=1.3, ls="--", alpha=0.8, label=f"ICE median ({N_ICE_LINES} curves drawn)")
        ax.plot(g, p, color=C_HL, lw=2.6, marker="o", ms=4.5, mec="white", mew=0.8, label="Partial dependence")
        ok = emp_n[f] >= 50                                # bins too thin for a rate are not drawn
        ax.plot(g[ok], emp_rate[f][ok], color=C_EMP, lw=1.7, ls=":", marker="s", ms=4, mec="white", mew=0.7,
                label="Observed default rate of the clients in that bin (≥ 50 clients)")
        ax2 = ax.twinx()                                  # training histogram, hidden axis, bottom fifth of the panel
        wid = np.diff(g); wid = np.append(wid, wid[-1]) * 0.85
        ax2.bar(g, hist[f], width=wid, color=C_TXT2, alpha=0.16, lw=0)
        ax2.set_ylim(0, hist[f].max() * 5.0); ax2.set_yticks([]); ax2.spines[:].set_visible(False)
        verdict = ("monotone ↑" if s["monotone_increasing"] else ("monotone ↓" if s["monotone_decreasing"] else
                   f"non-monotone, {s['n_steps_down'] if s['spearman_grid_vs_pd'] > 0 else s['n_steps_up']}/"
                   f"{s['n_steps']} steps reverse"))
        ax.set_title(f"{f}   PD {s['pd_min']:.3f}→{s['pd_max']:.3f} (range {s['pd_range']:.3f}), {verdict}",
                     loc="left", fontsize=10.5)
        ax.set_ylabel("Predicted P(default)", fontsize=9)
        _axis_cosmetics(ax, f, g); ax.tick_params(labelbottom=False)

        bx.plot(g, p, color=C_HL, lw=2.4, marker="o", ms=4.5, mec="white", mew=0.8)
        pad = 0.10 * (p.max() - p.min() + 1e-9)
        bx.set_ylim(p.min() - pad, p.max() + pad)
        if f == "UTIL_MEAN":
            bx.plot([ushape["value_at_minimum"]], [ushape["pd_at_minimum"]], "v", color=C_EMP, ms=11, mec="white", mew=1)
            bx.text(0.40, 0.05, f"PD minimum {ushape['pd_at_minimum']:.3f} at UTIL_MEAN = "
                    f"{ushape['value_at_minimum']:.2f}\nfalls {ushape['fall_from_left_end_into_minimum']:.3f} into it, "
                    f"rises {ushape['rise_from_minimum_to_right_end']:.3f} out of it\n→ U-shape survives: "
                    f"{ushape['u_shape_survives']}", transform=bx.transAxes, fontsize=8.2, color=C_EMP, va="bottom")
        bx.set_xlim(ax.get_xlim())
        bx.set_xlabel(FEATURE_LABEL[f], fontsize=8.5)
        bx.set_ylabel("PD, P(default)", fontsize=8.5)
        bx.set_title("the same PD alone, zoomed to its own range", loc="left", fontsize=8.8, color=C_TXT2)
        _axis_cosmetics(bx, f, g)
        if k == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8.2)
    fig.suptitle(f"Partial dependence and ICE, tuned XGBoost (test AUC {base['test_auc']:.4f}) — TRAINING partition, "
                 f"{n_train:,} clients.  Grey lines are individual clients (ICE), orange is their average (PD).",
                 fontsize=11.5, x=0.01, ha="left")
    return save_extra_figure(fig, "pdp_main.png")


def plot_interaction(ch, base):
    """figures/pdp_interaction.png — the chosen two-way surface, its interaction component, and the strata curves."""
    fa, fb, ga, gb = ch["feature_a"], ch["feature_b"], ch["grid_a"], ch["grid_b"]
    S, R, N = ch["surface"], ch["residual"], ch["counts"]
    fig, ax = plt.subplots(1, 3, figsize=(17.4, 5.8))

    im = ax[0].pcolormesh(ga, gb, S.T, shading="nearest", cmap="YlOrBr")
    cs = ax[0].contour(ga, gb, S.T, levels=7, colors="white", linewidths=0.7, alpha=0.85)
    ax[0].clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    fig.colorbar(im, ax=ax[0], label="Partial dependence, P(default)")
    ax[0].set_title(f"Two-way partial dependence, {fa} × {fb}\n"
                    f"surface spans {S.min():.3f} – {S.max():.3f}", loc="left", fontsize=9.5)

    v = float(np.abs(R).max())
    im2 = ax[1].pcolormesh(ga, gb, R.T, shading="nearest", cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im2, ax=ax[1], label="PD(a,b) − PD(a) − PD(b), centred")
    empty = np.argwhere(N == 0)
    for i, j in empty:
        ax[1].plot(ga[i], gb[j], "x", color=C_TXT2, ms=5, mew=1.0)
    ax[1].set_title(f"What additivity CANNOT explain (largest |gap| {v:.4f})\n"
                    f"H {ch['h_probability_weighted']:.3f} probability / {ch['h_margin_weighted']:.3f} log-odds;  "
                    f"× = empty cell ({len(empty)}/{N.size})", loc="left", fontsize=9.5)

    picks = sorted({int(np.argmin(np.abs(gb - np.quantile(gb, t)))) for t in (0.1, 0.3, 0.5, 0.7, 0.9)})
    shades = plt.cm.viridis(np.linspace(0.10, 0.86, len(picks)))
    for c, j in zip(shades, picks):
        lab = f"{fb} = {gb[j]:,.0f}" if fb == "LIMIT_BAL" else f"{fb} = {gb[j]:,.2f}"
        ax[2].plot(ga, S[:, j], "-o", color=c, lw=2, ms=4, mec="white", mew=0.7, label=lab)
    spread = S[:, picks].max(axis=1) - S[:, picks].min(axis=1)
    i_hi = int(np.argmax(spread))
    ax[2].set_title(f"{fa} curve at five fixed levels of {fb}\n"
                    f"gap between the curves {spread[0]:.3f} at {fa} = {ga[0]:g},\nwidest {spread.max():.3f} "
                    f"at {fa} = {ga[i_hi]:g}", loc="left", fontsize=9.5)
    ax[2].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[2].set_ylabel("Partial dependence, P(default)", fontsize=9)

    for a in ax:
        a.set_xlabel(SHORT_LABEL[fa], fontsize=9)
        a.grid(color=C_GRID, lw=0.5, alpha=0.5); a.set_axisbelow(True)
        if fa == "PAY_1":
            a.set_xticks(ga)
    for a in (ax[0], ax[1]):
        a.set_ylabel(SHORT_LABEL[fb], fontsize=9)
        if fb == "LIMIT_BAL":
            a.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    fig.suptitle(f"Does repayment delay interact with {'the credit limit' if fb == 'LIMIT_BAL' else 'utilisation'}?  "
                 f"{fa} × {fb}, chosen over {ch['rival_a']} × {ch['rival_b']} by Friedman's H "
                 f"({ch['h_probability_weighted']:.3f} vs {ch['rival_h_probability_weighted']:.3f}).\n"
                 f"H is weighted by how many training clients fall in each cell, so the empty corners of the grid cannot "
                 f"manufacture an interaction.  Tuned XGBoost, TRAINING partition.",
                 fontsize=11, x=0.01, ha="left")
    return save_extra_figure(fig, "pdp_interaction.png")


# --------------------------------------------------------------------------------------------- main
def main():
    X_train, X_test, y_train, y_test, feature_names = load_split()
    log(f"train {X_train.shape}, test {X_test.shape}; partial dependence is computed on the TRAINING rows only")
    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)

    # ============================================================ 1. the model whose surface we describe (not a new model)
    base_path = os.path.join(RESULTS, f"{BASELINE_SLUG}.json")
    if not os.path.exists(base_path):
        raise SystemExit(f"{base_path} not found — run model_boosting.py first; this script describes ITS tuned model.")
    base = json.load(open(base_path))
    bp = base["best_params"]
    params = dict(objective=bp["objective"], eval_metric="auc", tree_method=bp["tree_method"], n_jobs=N_JOBS,
                  random_state=bp["random_state"], learning_rate=bp["learning_rate"], max_depth=bp["max_depth"],
                  min_child_weight=bp["min_child_weight"], n_estimators=bp["n_estimators"], subsample=bp["subsample"],
                  colsample_bytree=bp["colsample_bytree"], scale_pos_weight=bp["scale_pos_weight"])
    assert params["random_state"] == SEED, "the baseline was not fitted with the project seed"
    t0 = time.time()
    pipe = Pipeline([("pre", make_preprocessor(scale=False)), ("clf", XGBClassifier(**params))]).fit(X_train, y_train)
    fit_seconds = time.time() - t0
    p_test_here = pipe.predict_proba(X_test)[:, 1]
    p_test_ref = np.load(os.path.join(RESULTS, f"{BASELINE_SLUG}_p_test.npy"))
    max_dp = float(np.abs(p_test_here - p_test_ref).max())
    log(f"refitted the tuned model in {fit_seconds:.1f}s and reproduced results/{BASELINE_SLUG}_p_test.npy "
        f"(max |Δp| = {max_dp:.2e}); it is the same model, so no second evaluate() entry is written")
    assert max_dp < 1e-9, "the refit does not reproduce the tuned model's test probabilities"

    # log-odds scale, used later so that 'interaction' is not confused with the curvature of the logistic link.
    # sigmoid(margin) must be the predicted probability; xgboost returns float32, so the agreement is float32-exact.
    m_chk = _predict(pipe, X_train, margin=True)
    p_chk = _predict(pipe, X_train)
    link_err = float(np.abs(1.0 / (1.0 + np.exp(-m_chk)) - p_chk).max())
    assert link_err < 1e-6, f"margin and probability disagree by {link_err:.2e}"
    log(f"log-odds scale available: max |sigmoid(margin) − predict_proba| = {link_err:.2e} over the training rows "
        f"(float32 rounding)")

    # ============================================================ 2. grids, one-way ICE matrices, partial dependence
    grids, grid_notes, pds, pds_margin, ice, hist, emp_rate, emp_n, shapes, ice_stats = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    y_tr = y_train.values
    for f in PDP_FEATURES:
        g, note = make_grid(X_train[f].values)
        grids[f], grid_notes[f] = g, note
        M = ice_matrix(pipe, X_train, f, g)
        ice[f] = M
        pds[f] = M.mean(axis=1)
        pds_margin[f] = ice_matrix(pipe, X_train, f, g, margin=True).mean(axis=1)
        n, rate = bin_stats(X_train[f].values, y_tr, g)
        emp_n[f], emp_rate[f], hist[f] = n, rate, n
        shapes[f] = shape_stats(g, pds[f], EXPECTED_SIGN[f])
        d = np.diff(M, axis=0)
        rng_per_client = M.max(axis=0) - M.min(axis=0)
        ice_stats[f] = dict(n_clients=int(M.shape[1]),
                            share_ice_monotone_increasing=float((d >= 0).all(axis=0).mean()),
                            share_ice_monotone_decreasing=float((d <= 0).all(axis=0).mean()),
                            ice_range_mean=float(rng_per_client.mean()), ice_range_median=float(np.median(rng_per_client)),
                            ice_range_p90=float(np.percentile(rng_per_client, 90)), ice_range_max=float(rng_per_client.max()),
                            pd_range=float(pds[f].max() - pds[f].min()),
                            ice_range_p90_over_pd_range=float(np.percentile(rng_per_client, 90) /
                                                              max(pds[f].max() - pds[f].min(), 1e-12)))
        log(f"PD {f:18s} grid {len(g):2d} ({note.split(';')[0]}) -> PD {pds[f].min():.4f}..{pds[f].max():.4f} "
            f"(range {pds[f].max() - pds[f].min():.4f})")

    # ---- cross-check against sklearn's own implementation (integer columns are cast to float only for this check) -------
    Xf = X_train.astype(float)
    cast_err = float(np.abs(_predict(pipe, Xf) - _predict(pipe, X_train)).max())
    sk_diff = {}
    for f in PDP_FEATURES:
        r = partial_dependence(pipe, Xf, features=[f], custom_values={f: grids[f]},
                               response_method="predict_proba", method="brute", kind="average")
        sk_diff[f] = float(np.abs(np.asarray(r["average"]).ravel() - pds[f]).max())
    log(f"cross-check vs sklearn.inspection.partial_dependence (brute): max abs difference "
        f"{max(sk_diff.values()):.2e} over the four curves (sklearn refuses integer columns, so it was given "
        f"X_train.astype(float), which changes no prediction: max |Δp| = {cast_err:.2e})")
    assert max(sk_diff.values()) < 1e-9 and cast_err == 0.0

    # ============================================================ 3. two-way PD for both candidate pairs, Friedman's H
    pairs = {}
    for fa, fb in CANDIDATE_PAIRS:
        ga, gb = grids[fa], grids[fb]
        t0 = time.time()
        S = pd_surface(pipe, X_train, fa, ga, fb, gb)
        Sm = pd_surface(pipe, X_train, fa, ga, fb, gb, margin=True)
        ia, ib = bin_index(X_train[fa].values, ga), bin_index(X_train[fb].values, gb)
        N = np.zeros((len(ga), len(gb))); D = np.zeros((len(ga), len(gb)))
        np.add.at(N, (ia, ib), 1.0); np.add.at(D, (ia, ib), y_tr.astype(float))
        with np.errstate(invalid="ignore"):
            E = np.where(N > 0, D / np.maximum(N, 1), np.nan)
        h_p_w, R = h_statistic(S, pds[fa], pds[fb], N)
        h_p_u, _ = h_statistic(S, pds[fa], pds[fb], np.ones_like(N))
        h_m_w, _ = h_statistic(Sm, pds_margin[fa], pds_margin[fb], N)
        h_m_u, _ = h_statistic(Sm, pds_margin[fa], pds_margin[fb], np.ones_like(N))
        # the interaction in plain terms: how much the FULL sweep of one feature moves the prediction, measured at the
        # lowest and at the highest level of the other one. Equal numbers = additive; a ratio far from 1 = interaction.
        eff = dict(effect_of_a_at_lowest_b=float(S[-1, 0] - S[0, 0]), effect_of_a_at_highest_b=float(S[-1, -1] - S[0, -1]),
                   effect_of_b_at_lowest_a=float(S[0, -1] - S[0, 0]), effect_of_b_at_highest_a=float(S[-1, -1] - S[-1, 0]))
        eff["ratio_effect_of_a_low_b_over_high_b"] = float(eff["effect_of_a_at_lowest_b"] /
                                                           (eff["effect_of_a_at_highest_b"] or np.nan))
        eff["ratio_effect_of_b_low_a_over_high_a"] = float(eff["effect_of_b_at_lowest_a"] /
                                                           (eff["effect_of_b_at_highest_a"] or np.nan))
        eff["sweep_a"] = [float(ga[0]), float(ga[-1])]; eff["sweep_b"] = [float(gb[0]), float(gb[-1])]
        pairs[(fa, fb)] = dict(feature_a=fa, feature_b=fb, grid_a=ga, grid_b=gb, surface=S, surface_margin=Sm,
                               range_effects=eff,
                               residual=R, counts=N, empirical=E,
                               h_probability_weighted=h_p_w, h_probability_uniform=h_p_u,
                               h_margin_weighted=h_m_w, h_margin_uniform=h_m_u,
                               n_cells=int(N.size), n_cells_empty=int((N == 0).sum()),
                               share_cells_empty=float((N == 0).mean()),
                               share_clients_in_the_10pct_densest_cells=float(
                                   np.sort(N.ravel())[::-1][:max(1, N.size // 10)].sum() / N.sum()),
                               surface_range=float(S.max() - S.min()), residual_max_abs=float(np.abs(R).max()),
                               seconds=round(time.time() - t0, 1))
        log(f"   full {fa} sweep ({ga[0]:g} -> {ga[-1]:g}) moves PD by {eff['effect_of_a_at_lowest_b']:+.4f} at "
            f"{fb} = {gb[0]:,.0f} and {eff['effect_of_a_at_highest_b']:+.4f} at {fb} = {gb[-1]:,.0f} "
            f"(ratio {eff['ratio_effect_of_a_low_b_over_high_b']:.2f}); full {fb} sweep moves PD by "
            f"{eff['effect_of_b_at_lowest_a']:+.4f} at {fa} = {ga[0]:g} and {eff['effect_of_b_at_highest_a']:+.4f} at "
            f"{fa} = {ga[-1]:g} (ratio {eff['ratio_effect_of_b_low_a_over_high_a']:.2f})")
        log(f"two-way PD {fa} x {fb}: {len(ga)}x{len(gb)} cells in {pairs[(fa, fb)]['seconds']}s | "
            f"H (probability, density-weighted) {h_p_w:.4f}, uniform {h_p_u:.4f} | "
            f"H (log-odds, density-weighted) {h_m_w:.4f}, uniform {h_m_u:.4f} | "
            f"{pairs[(fa, fb)]['n_cells_empty']}/{N.size} cells hold no training client")

    key_ch = max(pairs, key=lambda k: pairs[k]["h_probability_weighted"])
    key_rv = [k for k in pairs if k != key_ch][0]
    ch, rv = pairs[key_ch], pairs[key_rv]
    ch["rival_h_probability_weighted"] = rv["h_probability_weighted"]
    ch["rival_a"], ch["rival_b"] = rv["feature_a"], rv["feature_b"]
    margin_agrees = bool(ch["h_margin_weighted"] > rv["h_margin_weighted"])
    log(f"chosen pair: {key_ch[0]} x {key_ch[1]} (H_prob {ch['h_probability_weighted']:.4f} vs "
        f"{rv['h_probability_weighted']:.4f} for {key_rv[0]} x {key_rv[1]}); the log-odds-scale ranking agrees: {margin_agrees}")

    # ============================================================ 4. the correlation caveat, quantified
    num_df = X_train[NUM_COLS + CAT_COLS]
    C = num_df.corr(method="pearson")
    corr_block = {}
    for f in PDP_FEATURES:
        r = C[f].drop(index=f)
        top = r.reindex(r.abs().sort_values(ascending=False).index)[:6]
        corr_block[f] = dict(
            top_absolute_correlations=[dict(feature=k, r=float(v)) for k, v in top.items()],
            max_abs_r=float(r.abs().max()), max_abs_r_with=str(r.abs().idxmax()),
            n_features_with_abs_r_above_0_5=int((r.abs() > 0.5).sum()),
            n_features_with_abs_r_above_0_8=int((r.abs() > 0.8).sum()),
            max_abs_r_with_a_bill_amount=float(r[BILL_COLS].abs().max()),
            max_abs_r_with_a_bill_amount_col=str(r[BILL_COLS].abs().idxmax()))
    B = C.loc[BILL_COLS, BILL_COLS].values
    off = B[~np.eye(len(BILL_COLS), dtype=bool)]
    bills = dict(columns=BILL_COLS, n_pairs=int(len(off) // 2), min_abs_r=float(np.abs(off).min()),
                 mean_abs_r=float(np.abs(off).mean()), max_abs_r=float(np.abs(off).max()))
    # the identities the sweep breaks by construction (they hold to machine precision in the data)
    id_util = float(np.abs(X_train["UTIL_MEAN"] - X_train[BILL_COLS].mean(axis=1) / X_train["LIMIT_BAL"]).max())
    id_del = int((X_train["N_MONTHS_DELAYED"] != (X_train[PAY_COLS] > 0).sum(axis=1)).sum())
    log(f"correlation caveat: BILL_AMT1..6 pairwise |r| {bills['min_abs_r']:.2f}–{bills['max_abs_r']:.2f} "
        f"(mean {bills['mean_abs_r']:.2f}); UTIL_MEAN's strongest partner is {corr_block['UTIL_MEAN']['max_abs_r_with']} "
        f"at r = {C.loc['UTIL_MEAN', corr_block['UTIL_MEAN']['max_abs_r_with']]:+.2f}; "
        f"N_MONTHS_DELAYED's is {corr_block['N_MONTHS_DELAYED']['max_abs_r_with']} at "
        f"r = {C.loc['N_MONTHS_DELAYED', corr_block['N_MONTHS_DELAYED']['max_abs_r_with']]:+.2f}")

    # ============================================================ 5. does the EDA's U-shape survive in the PDP?
    ush = u_shape_test(grids["UTIL_MEAN"], pds["UTIL_MEAN"])
    dec = np.quantile(X_train["UTIL_MEAN"].values, np.linspace(0, 1, 11))
    di = np.clip(np.searchsorted(dec[1:-1], X_train["UTIL_MEAN"].values), 0, 9)
    dn = np.bincount(di, minlength=10).astype(float)
    dr = np.bincount(di, weights=y_tr.astype(float), minlength=10) / np.maximum(dn, 1)
    emp_u = dict(edges=[float(v) for v in dec], n=[int(v) for v in dn], default_rate=[float(v) for v in dr],
                 index_of_minimum=int(np.argmin(dr)), min_rate=float(dr.min()),
                 rate_first_decile=float(dr[0]), rate_last_decile=float(dr[-1]),
                 u_shaped=bool(0 < int(np.argmin(dr)) < 9 and dr[0] - dr.min() > 0.01 and dr[-1] - dr.min() > 0.01))
    r_u1_um = float(C.loc["UTIL_1", "UTIL_MEAN"])
    lg = None
    lg_path = os.path.join(RESULTS, "logistic_coefficients.csv")
    if os.path.exists(lg_path):
        lc = pd.read_csv(lg_path).set_index("feature")
        if "UTIL_MEAN" in lc.index:
            lg = dict(coef_per_sd=float(lc.loc["UTIL_MEAN", "coef"]), odds_ratio=float(lc.loc["UTIL_MEAN", "odds_ratio"]),
                      p_value=float(lc.loc["UTIL_MEAN", "p_value"]), source="results/logistic_coefficients.csv")
    log(f"U-shape test on the UTIL_MEAN PD curve: minimum {ush['pd_at_minimum']:.4f} at UTIL_MEAN = "
        f"{ush['value_at_minimum']:.3f} (interior: {ush['minimum_is_interior']}), fall from the left end "
        f"{ush['fall_from_left_end_into_minimum']:+.4f}, rise to the right end {ush['rise_from_minimum_to_right_end']:+.4f} "
        f"-> survives: {ush['u_shape_survives']}")

    # ============================================================ 6. results/pdp_values.csv
    rows = []
    for f in PDP_FEATURES:
        g, M = grids[f], ice[f]
        q = np.percentile(M, [10, 25, 50, 75, 90], axis=1)
        for k, v in enumerate(g):
            rows.append(dict(kind="one_way", feature_1=f, value_1=float(v), feature_2="", value_2=np.nan,
                             pd=float(pds[f][k]), pd_centered=float(pds[f][k] - pds[f].mean()),
                             pd_margin=float(pds_margin[f][k]), interaction_residual=np.nan,
                             n_train_rows=int(emp_n[f][k]), empirical_default_rate=float(emp_rate[f][k]),
                             ice_p10=float(q[0, k]), ice_p25=float(q[1, k]), ice_p50=float(q[2, k]),
                             ice_p75=float(q[3, k]), ice_p90=float(q[4, k])))
    for (fa, fb), pr in pairs.items():
        for i, a in enumerate(pr["grid_a"]):
            for j, b in enumerate(pr["grid_b"]):
                rows.append(dict(kind="two_way", feature_1=fa, value_1=float(a), feature_2=fb, value_2=float(b),
                                 pd=float(pr["surface"][i, j]), pd_centered=float(pr["surface"][i, j] - pr["surface"].mean()),
                                 pd_margin=float(pr["surface_margin"][i, j]), interaction_residual=float(pr["residual"][i, j]),
                                 n_train_rows=int(pr["counts"][i, j]), empirical_default_rate=float(pr["empirical"][i, j]),
                                 ice_p10=np.nan, ice_p25=np.nan, ice_p50=np.nan, ice_p75=np.nan, ice_p90=np.nan))
    vals = pd.DataFrame(rows)
    vals.to_csv(os.path.join(RESULTS, "pdp_values.csv"), index=False)
    log(f"results/pdp_values.csv written: {len(vals):,} rows "
        f"({int((vals['kind'] == 'one_way').sum())} one-way grid points, {int((vals['kind'] == 'two_way').sum())} two-way cells)")

    # ============================================================ 7. figures
    plot_main(grids, pds, ice, hist, emp_n, emp_rate, shapes, ush, base, len(X_train))
    plot_interaction(ch, base)
    log("figures/pdp_main.png and figures/pdp_interaction.png written")

    # ============================================================ 8. results/pdp_summary.json
    limitation = (
        "Partial dependence averages the model's prediction over the MARGINAL distribution of the other features: to draw "
        f"the curve at UTIL_MEAN = x it takes all {len(X_train):,} training clients, overwrites that one column with x and "
        "leaves every other column untouched. When the swept feature is strongly correlated with the ones held fixed, those "
        "synthetic clients are combinations the data never contain, and the curve is then partly extrapolation rather than a "
        "description of real borrowers. This dataset is exactly that case, and here is how far it goes: the six bill columns "
        f"BILL_AMT1..6 have pairwise |r| between {bills['min_abs_r']:.2f} and {bills['max_abs_r']:.2f} (mean "
        f"{bills['mean_abs_r']:.2f}); " +
        "; ".join(f"{f} has |r| = {corr_block[f]['max_abs_r']:.2f} with {corr_block[f]['max_abs_r_with']} and "
                  f"{corr_block[f]['n_features_with_abs_r_above_0_5']} features above 0.5" for f in PDP_FEATURES) +
        ". Worse than correlation, two of the four are DETERMINISTIC functions of columns the sweep holds fixed: "
        f"UTIL_MEAN = mean(BILL_AMT1..6)/LIMIT_BAL holds in the training data to {id_util:.2e}, and "
        f"N_MONTHS_DELAYED = #(PAY_k > 0) holds in all but {id_del} of {len(X_train):,} rows, so moving either one alone "
        "produces rows that are arithmetically impossible. The two-way surface makes the same point visibly: "
        f"{ch['n_cells_empty']} of {ch['n_cells']} cells of the {ch['feature_a']} x {ch['feature_b']} grid "
        f"({100 * ch['share_cells_empty']:.1f}%) contain no training client at all, and those cells are averaged into the "
        "picture with the same weight as the dense ones. Three things are done about it rather than said about it: the "
        "Friedman H statistics are weighted by the number of training clients per cell, so empty corners cannot create an "
        "interaction; the cells with no client are marked with a cross in figures/pdp_interaction.png; and the ICE curves "
        "are shown, because a PD curve is only a mean and the ICE spread (p90 of the per-client range is "
        f"{max(ice_stats[f]['ice_range_p90'] for f in PDP_FEATURES):.3f}) shows how much that mean hides. The honest reading "
        "is that these curves describe the MODEL's response surface, which is what the report claims about, and not the "
        "conditional expectation of default given the feature, which would need accumulated local effects (ALE) instead.")

    verdict_u = (
        f"The EDA found utilisation U-shaped rather than monotone. In the PD curve of UTIL_MEAN the minimum is "
        f"{ush['pd_at_minimum']:.4f} at UTIL_MEAN = {ush['value_at_minimum']:.3f}, it is "
        f"{'interior' if ush['minimum_is_interior'] else 'at an end point'}, the curve falls "
        f"{ush['fall_from_left_end_into_minimum']:+.4f} into it from the left end and rises "
        f"{ush['rise_from_minimum_to_right_end']:+.4f} out of it to the right end, so the U-shape "
        f"{'SURVIVES' if ush['u_shape_survives'] else 'does NOT survive'} in the fitted model and not only in the raw "
        f"default rates. The empirical training deciles of UTIL_MEAN move {emp_u['rate_first_decile']:.3f} -> minimum "
        f"{emp_u['min_rate']:.3f} (decile {emp_u['index_of_minimum'] + 1}) -> {emp_u['rate_last_decile']:.3f}, i.e. the same "
        f"shape. This matters for the report's argument for tree-based methods: a linear model in UTIL_MEAN can only be "
        f"monotone" + (f" — the logistic regression is forced into a single coefficient of {lg['coef_per_sd']:+.3f} per SD "
                       f"(odds ratio {lg['odds_ratio']:.3f}), which has to average the falling and the rising half of the "
                       f"curve into one number" if lg else "") + ", while the booster fits the fall and the rise separately. "
        f"Note that the EDA measured the U-shape on UTIL_1, which correlates r = {r_u1_um:.2f} with UTIL_MEAN.")

    summary = dict(
        name="Partial dependence and ICE for the tuned XGBoost", analysis="pdp", member=MEMBER, seed=SEED, versions=versions,
        partition=dict(computed_on="TRAINING partition", n_train=int(len(X_train)), n_test_untouched=int(len(X_test)),
                       why=("Partial dependence is a property of the FITTED MODEL, not of a data set: it is an average of the "
                            "model's own predictions, and no label is used anywhere in it. Using the training rows therefore "
                            "costs nothing and keeps the held-out test set untouched. Nothing here is tuned, selected or "
                            "thresholded; the model was tuned and evaluated in results/xgboost.json.")),
        model=dict(source=f"results/{BASELINE_SLUG}.json", name=base["name"], hyper_parameters=params,
                   refit_seconds=round(fit_seconds, 1), reproduces_baseline_p_test_max_abs_diff=max_dp,
                   baseline_test_auc=base["test_auc"], baseline_test_ap=base["test_ap"], baseline_cv_auc=base["cv_auc"],
                   evaluate_not_called="this is the same model as results/xgboost.json, so a second comparison-table row "
                                       "would be a duplicate; identity is proved by the p_test reproduction above"),
        method=dict(kind="brute force", rows_used=int(len(X_train)),
                    definition=("PD(x) = mean over all training clients of the model's predicted probability when the swept "
                                "feature is set to x and every other feature keeps that client's own value; the individual "
                                "curves before averaging are the ICE curves."),
                    grids={f: grid_notes[f] for f in PDP_FEATURES},
                    grid_values={f: [float(v) for v in grids[f]] for f in PDP_FEATURES},
                    cross_check_vs_sklearn=dict(function="sklearn.inspection.partial_dependence(method='brute', custom_values=...)",
                                                max_abs_difference_per_feature=sk_diff,
                                                note=("sklearn refuses integer columns for PDP, so it was given "
                                                      "X_train.astype(float); that cast changes no prediction of this tree "
                                                      f"model (max |Δp| = {cast_err:.2e})")),
                    link_check_margin_vs_probability=link_err),
        one_way={f: dict(feature=f, grid=[float(v) for v in grids[f]],
                         partial_dependence=[float(v) for v in pds[f]],
                         partial_dependence_log_odds=[float(v) for v in pds_margin[f]],
                         n_train_rows_per_bin=[int(v) for v in emp_n[f]],
                         empirical_default_rate_per_bin=[None if np.isnan(v) else float(v) for v in emp_rate[f]],
                         shape=shapes[f], ice=ice_stats[f]) for f in PDP_FEATURES},
        utilisation_u_shape=dict(pd_curve_test=ush, empirical_deciles_training=emp_u,
                                 corr_UTIL_1_UTIL_MEAN=r_u1_um, logistic_coefficient_UTIL_MEAN=lg, verdict=verdict_u),
        interaction=dict(
            claim_tested=("the proposal's hypothesis that the ranking is explainable by the interaction of repayment delay "
                          "with credit limit and utilisation"),
            statistic=("Friedman's H: the share of the variance of the two-way partial dependence that the sum of the two "
                       "one-way curves cannot reproduce. 0 = perfectly additive, 1 = nothing additive about it. Reported "
                       "both weighted by the number of training clients per grid cell (the number used to choose the pair, "
                       "so that empty corners of the grid cannot manufacture an interaction) and uniform over the grid; and "
                       "on both the probability scale (what the figures show) and the raw log-odds scale (where a boosted "
                       "tree ensemble is additive by construction, so H there is interaction rather than the curvature of "
                       "the logistic link)."),
            chosen_pair=[ch["feature_a"], ch["feature_b"]], rival_pair=[rv["feature_a"], rv["feature_b"]],
            margin_scale_ranking_agrees=margin_agrees,
            candidates={f"{k[0]} x {k[1]}": {kk: v for kk, v in p.items()
                                             if kk not in ("grid_a", "grid_b", "surface", "surface_margin", "residual",
                                                           "counts", "empirical", "feature_a", "feature_b")}
                        for k, p in pairs.items()},
            reading=(f"On the probability scale the density-weighted H is {ch['h_probability_weighted']:.4f} for "
                     f"{ch['feature_a']} x {ch['feature_b']} against {rv['h_probability_weighted']:.4f} for "
                     f"{rv['feature_a']} x {rv['feature_b']}; on the log-odds scale {ch['h_margin_weighted']:.4f} against "
                     f"{rv['h_margin_weighted']:.4f}. The largest departure from additivity anywhere on the chosen grid is "
                     f"{ch['residual_max_abs']:.4f} in probability, against a total surface range of "
                     f"{ch['surface_range']:.4f} — i.e. the interaction is small next to the two main effects, which is the "
                     f"honest version of the proposal's hypothesis: the effects mostly ADD, they do not mostly INTERACT. "
                     f"In plain terms: sweeping {ch['feature_a']} over its whole range moves the partial dependence by "
                     f"{ch['range_effects']['effect_of_a_at_lowest_b']:+.4f} for clients at the lowest {ch['feature_b']} "
                     f"and by {ch['range_effects']['effect_of_a_at_highest_b']:+.4f} at the highest "
                     f"(ratio {ch['range_effects']['ratio_effect_of_a_low_b_over_high_b']:.2f}); sweeping "
                     f"{ch['feature_b']} moves it by {ch['range_effects']['effect_of_b_at_lowest_a']:+.4f} at the lowest "
                     f"{ch['feature_a']} and {ch['range_effects']['effect_of_b_at_highest_a']:+.4f} at the highest "
                     f"(ratio {ch['range_effects']['ratio_effect_of_b_low_a_over_high_a']:.2f}).")),
        correlation_caveat=dict(per_pdp_feature=corr_block, bill_amount_block=bills,
                                identity_util_mean_max_abs_error=id_util,
                                identity_n_months_delayed_n_rows_violating=id_del,
                                two_way_cells_without_any_training_client=dict(
                                    pair=[ch["feature_a"], ch["feature_b"]], empty=ch["n_cells_empty"],
                                    total=ch["n_cells"], share=ch["share_cells_empty"]),
                                limitation=limitation),
        files=dict(results=["pdp_values.csv", "pdp_summary.json"], figures=["pdp_main.png", "pdp_interaction.png"]),
        seconds=round(time.time() - T0, 1))
    json.dump(summary, open(os.path.join(RESULTS, "pdp_summary.json"), "w"), indent=1, ensure_ascii=False)

    # ============================================================ 9. the headline
    print("\n=== one-way partial dependence (training partition, tuned XGBoost) ===")
    tab = pd.DataFrame([dict(feature=f, grid=len(grids[f]), pd_low=pds[f].min(), pd_high=pds[f].max(),
                             pd_range=pds[f].max() - pds[f].min(), at_left_end=pds[f][0], at_right_end=pds[f][-1],
                             monotone=("up" if shapes[f]["monotone_increasing"] else
                                       ("down" if shapes[f]["monotone_decreasing"] else "no")),
                             steps_reversing=shapes[f]["steps_against_expected_sign"], n_steps=shapes[f]["n_steps"],
                             spearman=shapes[f]["spearman_grid_vs_pd"],
                             ice_range_p90=ice_stats[f]["ice_range_p90"],
                             ice_share_respecting_sign=(ice_stats[f]["share_ice_monotone_increasing"]
                                                        if EXPECTED_SIGN[f] > 0 else
                                                        ice_stats[f]["share_ice_monotone_decreasing"]))
                            for f in PDP_FEATURES])
    print(tab.round(4).to_string(index=False))
    print(f"\nice_share_respecting_sign is the share of the {len(ice[PDP_FEATURES[0]][0]):,} individual ICE curves that "
          f"never move against the economic prior ({', '.join(f'{f} {s:+d}' for f, s in EXPECTED_SIGN.items())}). "
          f"Only {100 * tab.loc[tab.feature == 'PAY_1', 'ice_share_respecting_sign'].iloc[0]:.1f}% of clients have a "
          f"PAY_1 curve that never dips, which is the same defect results/monotone_summary.json prices in AUC.")
    print(f"\nRanked by the width of the PD curve: " +
          ", ".join(f"{r.feature} {r.pd_range:.3f}" for r in tab.sort_values("pd_range", ascending=False).itertuples()))
    print(f"\n=== the U-shape in utilisation ===\n{verdict_u}")
    print(f"\n=== interaction ===\n{summary['interaction']['reading']}")
    print(f"\n=== the caveat, quantified ===\n{limitation}")
    log(f"done in {time.time() - T0:.0f}s. results/pdp_values.csv, results/pdp_summary.json, "
        f"figures/pdp_main.png, figures/pdp_interaction.png")


if __name__ == "__main__":
    main()
