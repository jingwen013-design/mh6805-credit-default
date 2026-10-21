"""
model_unsupervised.py — Unsupervised learning (lectures B2/B3) used as customer segmentation for the MH6805 credit-default
project.  Owner: RUAN Hangjing.  Slug "unsupervised".

Everything unsupervised is fitted on the TRAINING set only: the StandardScaler is fitted on the 20,975 training rows and the
8,990 test rows are only ever *assigned* (transform / predict) with the fitted scaler + k-means.  The two supervised models
at the end are evaluated exactly once each with common.evaluate(); their unsupervised steps (scaler, PCA, k-means) live
INSIDE the sklearn Pipeline so they are refitted in every CV fold (no leakage of validation statistics).

  (1) PCA on the 26 standardised numeric features
        figures/pca_scree.png           proportion of variance explained (PVE) per component + cumulative PVE
        results/pca_variance.csv        PVE table
        results/pca_loadings.csv        loadings of all 26 PCs (PC1..PC3 plotted in figures/pca_loadings.png)
        figures/pca_scatter.png         PC1 vs PC2 of a fixed 5,000-row training sample coloured by DEFAULT + biplot arrows
  (2) K-means on the same standardised features
        results/kmeans_selection.csv    K = 2..10: within-cluster SS (inertia), silhouette on the fixed 5,000-row sample, sizes
        figures/kmeans_elbow.png        elbow + silhouette curves; K chosen by the kneedle elbow rule (documented below)
        results/kmeans_profiles.csv     per-cluster size / share / DEFAULT rate on TRAINING and on TEST (test rows assigned with
                                        the training scaler + k-means), mean & median of key variables, centroid z-scores
        figures/kmeans_default_rate.png default rate per segment (train vs test) + centroid z-score heat-map
  (3) Hierarchical clustering (complete linkage and Ward) on a 1,000-row stratified training sample
        figures/hclust_dendrogram.png   dendrograms with the K-cluster cut;  results/hclust_agreement.csv  ARI vs k-means
  (4) Supervised use of the unsupervised outputs (two evaluate() calls)
        logistic_plus_clusters   all 29 features (make_preprocessor) + (K-1) one-hot cluster dummies -> plain logistic
        logistic_on_pcs          first n PCs only (n in {5, 10, 15} chosen by 5-fold CV AUC, GridSearchCV) -> plain logistic
        figures/pca_logistic_vs_ncomp.png   CV AUC vs number of PCs (1..26) for context
  results/unsupervised_summary.json     everything above in one place

Contract (code/README.md): load_split() only, preprocessing inside Pipelines, cv=CV, scoring=SCORING, seed 2607526
everywhere, n_jobs<=2, nothing fitted or tuned on the test set.   Run:  python3 model_unsupervised.py   (~2-3 min).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")                       # keep BLAS/OpenMP threads modest: several scripts run concurrently
import time, textwrap
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from common import *                                     # load_split, make_preprocessor, CV, SCORING, SEED, evaluate, NUM_COLS, ...
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import FeatureUnion
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

MEMBER = "RUAN Hangjing"
N_JOBS = 2
MAX_ITER, LBFGS_TOL = 5000, 1e-8                         # identical solver settings to the plain logistic baseline (model_logistic.py)
K_RANGE = list(range(2, 11))                             # candidate numbers of clusters
N_INIT = 10                                              # k-means++ restarts per K
SIL_SAMPLE = 5000                                        # fixed training sample used for the silhouette AND the PCA scatter
HC_SAMPLE = 1000                                         # stratified training sample for hierarchical clustering
PC_GRID = [5, 10, 15]                                    # candidate numbers of PCs for logistic_on_pcs (chosen by CV)
N_PC_LOAD = 26                                           # loadings of all PCs written to CSV (PC1..PC3 plotted)
KEY_VARS = ["LIMIT_BAL", "AGE", "PAY_1", "N_MONTHS_DELAYED", "MAX_DELAY", "BILL_AMT1", "UTIL_MEAN",
            "PAY_AMT1", "PAY_RATIO_MEAN", "BILL_TREND"]  # profiled per cluster (raw units)

# colours: validated reference palette shared with the other scripts (diverging pair red/blue, categorical slots in order)
RED, BLUE, GREY, INK, INK2, GRID = "#e34948", "#2a78d6", "#c9c8c3", "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, "#f0efec", RED])

# hand-written interpretation of the first three PCs (signs are fixed by the convention below: largest |loading| positive)
PC_LABELS = {1: "overall indebtedness: bill size, utilisation and delay all load positively",
             2: "repayment delay (+) vs. affluent high-limit / large-bill / large-payment clients (−)",
             3: "recent balance build-up: rising bill & latest utilisation (+) vs. pay-ratio, delay, limit (−)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)


def logit():
    return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=MAX_ITER, tol=LBFGS_TOL, random_state=SEED)


def num_scaler():
    """StandardScaler on the 26 numeric columns only (categoricals dropped) — fitted inside the pipeline."""
    return ColumnTransformer([("num", StandardScaler(), NUM_COLS)], remainder="drop")


def fold_auc(model, X, y):
    s = cross_val_score(clone(model), X, y, cv=CV, scoring=SCORING, n_jobs=N_JOBS)
    return float(s.mean()), float(s.std())


def oof_probs(model, X, y):
    return cross_val_predict(clone(model), X, y, cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]


def kneedle(ks, vals):
    """Elbow of a convex decreasing curve (Satopaa et al. 2011): normalise both axes to [0, 1] and take the point with the
    largest vertical gap below the chord joining the first and last point."""
    ks, vals = np.asarray(ks, float), np.asarray(vals, float)
    x = (ks - ks[0]) / (ks[-1] - ks[0]); y = (vals - vals[-1]) / (vals[0] - vals[-1])
    gap = (1.0 - x) - y
    return int(ks[int(np.argmax(gap))]), gap


class ClusterDummies(BaseEstimator, TransformerMixin):
    """StandardScaler + KMeans fitted on the numeric columns of the rows it is fitted on (i.e. the training fold);
    transform() returns one-hot cluster membership without the first dummy (reference cluster) so the design stays full rank."""
    def __init__(self, n_clusters=4, n_init=N_INIT, random_state=SEED):
        self.n_clusters = n_clusters; self.n_init = n_init; self.random_state = random_state

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X[NUM_COLS])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
        self.kmeans_.fit(self.scaler_.transform(X[NUM_COLS]))
        return self

    def transform(self, X):
        lab = self.kmeans_.predict(self.scaler_.transform(X[NUM_COLS]))
        return np.eye(self.n_clusters)[lab][:, 1:]

    def get_feature_names_out(self, input_features=None):
        return np.array([f"cluster_{k}" for k in range(1, self.n_clusters)], dtype=object)


def pcs_pipeline(n):
    """Scaler -> PCA(n) -> plain logistic; scaler and PCA are refitted in every CV fold."""
    return Pipeline([("num", num_scaler()), ("pca", PCA(n_components=n, svd_solver="full", random_state=SEED)), ("clf", logit())])


def main():
    t_all = time.time()
    X_train, X_test, y_train, y_test, feature_names = load_split()
    n_tr = len(X_train)
    log(f"train {X_train.shape}, test {X_test.shape}; default rate train {y_train.mean():.4f} / test {y_test.mean():.4f}")
    summary = dict(member=MEMBER, slug="unsupervised", seed=SEED, n_train=int(n_tr), n_test=int(len(X_test)), numeric_features=NUM_COLS)

    # ------------------------------------------------------------------ standardise (fit on TRAINING only)
    scaler = StandardScaler().fit(X_train[NUM_COLS])
    Z_tr = scaler.transform(X_train[NUM_COLS]); Z_te = scaler.transform(X_test[NUM_COLS])
    rng = np.random.RandomState(SEED)
    idx_s = np.sort(rng.choice(n_tr, SIL_SAMPLE, replace=False))          # fixed 5,000-row training sample (silhouette + scatter)
    ytr = y_train.values; yte = y_test.values

    # ================================================================== (1) PCA
    log("(1) PCA on the 26 standardised numeric training features")
    pca = PCA(svd_solver="full", random_state=SEED).fit(Z_tr)
    comps = pca.components_.copy()
    for j in range(comps.shape[0]):                                         # sign convention: largest |loading| positive
        if comps[j, np.argmax(np.abs(comps[j]))] < 0:
            comps[j] *= -1
    pca.components_ = comps
    pve = pca.explained_variance_ratio_; cum = np.cumsum(pve); n_comp = len(pve)
    n90 = int(np.searchsorted(cum, 0.90) + 1); n95 = int(np.searchsorted(cum, 0.95) + 1)
    kaiser = int((pca.explained_variance_ > 1.0).sum())
    var_tab = pd.DataFrame(dict(component=np.arange(1, n_comp + 1), eigenvalue=pca.explained_variance_, pve=pve, cumulative_pve=cum))
    var_tab.to_csv(os.path.join(RESULTS, "pca_variance.csv"), index=False)
    load = pd.DataFrame(comps[:N_PC_LOAD].T, index=NUM_COLS, columns=[f"PC{j}" for j in range(1, N_PC_LOAD + 1)])
    load.index.name = "feature"; load.round(4).to_csv(os.path.join(RESULTS, "pca_loadings.csv"))
    S_tr = pca.transform(Z_tr)                                              # scores (all 26 PCs)
    auc_pc = {f"PC{j}": float(max(roc_auc_score(ytr, S_tr[:, j - 1]), 1 - roc_auc_score(ytr, S_tr[:, j - 1]))) for j in range(1, 6)}
    corr_pc = {f"PC{j}": float(np.corrcoef(S_tr[:, j - 1], ytr)[0, 1]) for j in range(1, 6)}
    log(f"    PVE PC1-5 = {np.round(pve[:5], 4).tolist()}; cumulative 5/10/15 PCs = {cum[4]:.3f}/{cum[9]:.3f}/{cum[14]:.3f}; "
        f"{n90} PCs for 90%, {n95} for 95%; Kaiser (eigenvalue > 1): {kaiser}")
    top_load = {}
    for j in range(1, 4):
        col = load[f"PC{j}"].sort_values()
        top_load[f"PC{j}"] = dict(positive=col.tail(4)[::-1].round(3).to_dict(), negative=col.head(4).round(3).to_dict())
        log(f"    PC{j} ({pve[j-1]*100:.1f}%): + {list(col.tail(4)[::-1].index)}  − {list(col.head(4).index)}  |  AUC of score alone {auc_pc[f'PC{j}']:.3f}")
    summary["pca"] = dict(pve=[float(v) for v in pve], cumulative_pve=[float(v) for v in cum], n_pcs_for_90pct=n90, n_pcs_for_95pct=n95,
                          n_eigenvalues_above_1=kaiser, top_loadings=top_load, interpretation=PC_LABELS,
                          auc_of_single_pc_score=auc_pc, corr_pc_with_default=corr_pc)

    # --- figure: scree + cumulative PVE
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    cc = np.arange(1, n_comp + 1)
    ax[0].bar(cc, pve * 100, color=BLUE, width=0.72, edgecolor="none")
    for j in range(3):
        ax[0].text(cc[j], pve[j] * 100 + 0.6, f"{pve[j]*100:.1f}%", ha="center", fontsize=8.5, color=INK2)
    ax[0].set_xlabel("Principal component", color=INK2); ax[0].set_ylabel("Proportion of variance explained (%)", color=INK2)
    ax[0].set_title(f"Scree plot — PCA of 26 standardised numeric features\n(training set, n = {n_tr:,}; {kaiser} eigenvalues > 1)",
                    fontsize=10.5, loc="left", color=INK); ax[0].set_xticks(cc[::1]); tidy(ax[0]); ax[0].grid(axis="x", visible=False)
    ax[1].plot(cc, cum * 100, color=BLUE, lw=2, marker="o", ms=4)
    for n in PC_GRID:
        ax[1].scatter([n], [cum[n - 1] * 100], color=RED, s=42, zorder=5)
        ax[1].annotate(f"{n} PCs: {cum[n-1]*100:.1f}%", (n, cum[n - 1] * 100), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK2)
    ax[1].axhline(90, color=GREY, ls="--", lw=1); ax[1].text(n_comp, 90.6, "90%", ha="right", fontsize=8, color=INK2)
    ax[1].set_xlabel("Number of components", color=INK2); ax[1].set_ylabel("Cumulative PVE (%)", color=INK2); ax[1].set_ylim(0, 102)
    ax[1].set_title(f"Cumulative PVE — {n90} PCs reach 90%, {n95} reach 95%\n(red = candidate sizes for the logistic-on-PCs model)",
                    fontsize=10.5, loc="left", color=INK); tidy(ax[1])
    save_extra_figure(fig, "pca_scree.png")

    # --- figure: loadings PC1..PC3 (horizontal bars, red = positive, blue = negative)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 6.6), sharey=True)
    ylab = NUM_COLS[::-1]
    for j in range(3):
        v = load[f"PC{j+1}"].values[::-1]
        ax[j].barh(ylab, v, color=[RED if c > 0 else BLUE for c in v], height=0.66, edgecolor="none")
        ax[j].axvline(0, color=INK2, lw=0.8); tidy(ax[j]); ax[j].grid(axis="y", visible=False)
        lim = float(np.abs(v).max()) * 1.18; ax[j].set_xlim(-lim, lim)
        ax[j].set_xlabel("loading (eigenvector weight)", color=INK2, fontsize=9)
        ax[j].set_title(f"PC{j+1} — {pve[j]*100:.1f}% of variance\n" + textwrap.fill(PC_LABELS[j+1], 50), fontsize=9.5, loc="left", color=INK)
    for yy in (2.5, 8.5, 14.5, 20.5, 23.5):                                  # separators between feature families
        for a in ax: a.axhline(yy, color=GRID, lw=0.8)
    ax[0].legend(handles=[Patch(color=RED, label="positive loading"), Patch(color=BLUE, label="negative loading")], loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("PCA loadings of the first three components (training set; sign convention: largest |loading| positive)", fontsize=11, x=0.01, ha="left", color=INK)
    save_extra_figure(fig, "pca_loadings.png")

    # --- figure: PC1 vs PC2 scatter (5,000-row sample) coloured by default + biplot arrows
    P = S_tr[idx_s, :2]; ys = ytr[idx_s]
    lo = np.percentile(P, 0.5, axis=0); hi = np.percentile(P, 99.5, axis=0)
    n_out = int(((P < lo) | (P > hi)).any(axis=1).sum())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    ax[0].scatter(P[ys == 0, 0], P[ys == 0, 1], s=7, color=BLUE, alpha=0.35, lw=0, label=f"no default (n = {(ys==0).sum():,})")
    ax[0].scatter(P[ys == 1, 0], P[ys == 1, 1], s=8, color=RED, alpha=0.55, lw=0, label=f"default (n = {(ys==1).sum():,})")
    ax[0].set_xlim(lo[0], hi[0]); ax[0].set_ylim(lo[1], hi[1]); tidy(ax[0])
    ax[0].set_xlabel(f"PC1 ({pve[0]*100:.1f}%) — overall indebtedness →", color=INK2); ax[0].set_ylabel(f"PC2 ({pve[1]*100:.1f}%) — repayment delay ↑, affluence ↓", color=INK2)
    ax[0].set_title(f"Random 5,000-row training sample, coloured by DEFAULT\n(axes clipped to the 0.5–99.5 percentiles, {n_out} points outside;\n"
                    f"AUC of the PC1 score alone {auc_pc['PC1']:.3f}, PC2 alone {auc_pc['PC2']:.3f})", fontsize=9.5, loc="left", color=INK)
    ax[0].legend(loc="upper right", frameon=False, fontsize=8.5, markerscale=2.2)
    ax[1].scatter(P[:, 0], P[:, 1], s=5, color=GREY, alpha=0.35, lw=0)
    groups = [("LIMIT_BAL", ["LIMIT_BAL"]), ("AGE", ["AGE"]), ("PAY_1…6 (repayment status)", PAY_COLS), ("BILL_AMT1…6", BILL_COLS),
              ("PAY_AMT1…6", PAYAMT_COLS), ("UTIL_1 / UTIL_MEAN", ["UTIL_1", "UTIL_MEAN"]), ("PAY_RATIO_MEAN", ["PAY_RATIO_MEAN"]),
              ("N_MONTHS_DELAYED / MAX_DELAY", ["N_MONTHS_DELAYED", "MAX_DELAY"]), ("BILL_TREND", ["BILL_TREND"])]
    vec = {g: load.loc[cols, ["PC1", "PC2"]].mean().values for g, cols in groups}     # one arrow per feature family (mean loading)
    scale = 0.8 * min(hi[0] - lo[0], hi[1] - lo[1]) / 2 / max(np.linalg.norm(v) for v in vec.values())
    for g, v in vec.items():
        x, y = v * scale
        ax[1].annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        ax[1].text(x * 1.06 + (0.15 if x >= 0 else -0.15), y * 1.06, g, fontsize=8, color=INK, ha="left" if x >= 0 else "right", va="center")
    ax[1].set_xlim(lo[0], hi[0]); ax[1].set_ylim(lo[1], hi[1]); tidy(ax[1])
    ax[1].set_xlabel(f"PC1 ({pve[0]*100:.1f}%)", color=INK2); ax[1].set_ylabel(f"PC2 ({pve[1]*100:.1f}%)", color=INK2)
    ax[1].set_title("Biplot: loading vectors, one arrow per feature family\n(PAY_1…6, BILL_AMT1…6, PAY_AMT1…6 = mean loading of the family;\nsame sample in grey; arrow length ∝ loading)", fontsize=9.5, loc="left", color=INK)
    save_extra_figure(fig, "pca_scatter.png")

    # ================================================================== (2) K-means
    log(f"(2) K-means on the standardised training features, K = {K_RANGE[0]}..{K_RANGE[-1]}, n_init = {N_INIT}")
    km_models, sel_rows = {}, []
    for k in K_RANGE:
        t0 = time.time(); km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Z_tr); km_models[k] = km
        sil = float(silhouette_score(Z_tr[idx_s], km.labels_[idx_s]))
        sizes = np.bincount(km.labels_, minlength=k); dr = np.array([ytr[km.labels_ == c].mean() for c in range(k)])
        sel_rows.append(dict(K=k, inertia=float(km.inertia_), silhouette_5000=sil, min_cluster_size=int(sizes.min()), max_cluster_size=int(sizes.max()),
                             min_default_rate=float(dr.min()), max_default_rate=float(dr.max()), n_iter=int(km.n_iter_), seconds=round(time.time() - t0, 1)))
        log(f"    K={k:2d}  inertia {km.inertia_:>10,.0f}  silhouette {sil:.4f}  sizes {sizes.tolist()}  default rates {np.round(dr, 3).tolist()}")
    sel = pd.DataFrame(sel_rows)
    sel["inertia_drop"] = -sel["inertia"].diff()
    K, gap = kneedle(sel["K"], sel["inertia"]); sel["kneedle_gap"] = gap
    k_sil = int(sel.loc[sel["silhouette_5000"].idxmax(), "K"])
    sel.to_csv(os.path.join(RESULTS, "kmeans_selection.csv"), index=False)
    log(f"    elbow (kneedle) -> K = {K}; silhouette maximum at K = {k_sil} ({sel['silhouette_5000'].max():.4f}); silhouette at K = {K}: "
        f"{float(sel.loc[sel['K'] == K, 'silhouette_5000'].iloc[0]):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(sel["K"], sel["inertia"] / 1e3, color=BLUE, lw=2, marker="o", ms=5)
    ax[0].scatter([K], [sel.loc[sel["K"] == K, "inertia"].iloc[0] / 1e3], color=RED, s=70, zorder=5)
    ax[0].annotate(f"elbow K = {K}", (K, sel.loc[sel["K"] == K, "inertia"].iloc[0] / 1e3), xytext=(12, 10), textcoords="offset points", fontsize=9, color=INK2)
    for _, r in sel.iterrows():
        if r["K"] > K_RANGE[0]:
            ax[0].text(r["K"], r["inertia"] / 1e3 + 6, f"−{r['inertia_drop']/1e3:.0f}k", ha="center", fontsize=7.5, color=INK2)
    ax[0].set_xlabel("K (number of clusters)", color=INK2); ax[0].set_ylabel("Within-cluster sum of squares (thousands)", color=INK2)
    ax[0].set_title(f"Elbow plot — total within-cluster SS vs K\n(n = {n_tr:,} training rows, 26 standardised features, {N_INIT} k-means++ restarts;\nlabels = drop from K−1)",
                    fontsize=9.5, loc="left", color=INK); tidy(ax[0]); ax[0].set_xticks(K_RANGE)
    ax[1].plot(sel["K"], sel["silhouette_5000"], color=BLUE, lw=2, marker="o", ms=5)
    ax[1].scatter([K], [sel.loc[sel["K"] == K, "silhouette_5000"].iloc[0]], color=RED, s=70, zorder=5)
    for _, r in sel.iterrows():
        ax[1].text(r["K"], r["silhouette_5000"] + 0.004, f"{r['silhouette_5000']:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax[1].set_xlabel("K (number of clusters)", color=INK2); ax[1].set_ylabel("Mean silhouette (5,000-row training sample)", color=INK2)
    ax[1].set_ylim(0, max(0.3, sel["silhouette_5000"].max() * 1.25))
    ax[1].set_title(f"Silhouette vs K (5,000-row training sample)\nmaximum at K = {k_sil}, nearly flat for K ≤ 5;\nred = chosen K = {K}", fontsize=9.5, loc="left", color=INK)
    tidy(ax[1]); ax[1].set_xticks(K_RANGE)
    save_extra_figure(fig, "kmeans_elbow.png")

    # --- final K-means: relabel clusters by ascending training default rate (segment 1 = safest ... segment K = riskiest)
    km = km_models[K]
    raw_tr = km.labels_; raw_te = km.predict(Z_te)
    dr_raw = np.array([ytr[raw_tr == c].mean() for c in range(K)])
    order = np.argsort(dr_raw); remap = np.empty(K, int); remap[order] = np.arange(K)
    lab_tr = remap[raw_tr]; lab_te = remap[raw_te]
    cent = np.vstack([Z_tr[lab_tr == c].mean(axis=0) for c in range(K)])   # centroids in z-units, in the new order
    base_tr, base_te = float(ytr.mean()), float(yte.mean())
    chi2, p_chi2, _, _ = chi2_contingency(pd.crosstab(lab_tr, ytr).values)
    prof = []
    for c in range(K):
        m_tr, m_te = lab_tr == c, lab_te == c
        topf = np.argsort(-np.abs(cent[c]))[:3]
        desc = ", ".join(f"{NUM_COLS[j]} {'high' if cent[c, j] > 0 else 'low'} ({cent[c, j]:+.1f} SD)" for j in topf)
        row = dict(segment=c + 1, descriptor=desc, n_train=int(m_tr.sum()), share_train=float(m_tr.mean()), default_rate_train=float(ytr[m_tr].mean()),
                   lift_train=float(ytr[m_tr].mean() / base_tr), n_test=int(m_te.sum()), share_test=float(m_te.mean()),
                   default_rate_test=float(yte[m_te].mean()), lift_test=float(yte[m_te].mean() / base_te),
                   share_of_all_defaulters_train=float(ytr[m_tr].sum() / ytr.sum()),
                   share_female_train=float((X_train.loc[m_tr, "SEX"] == 2).mean()), share_single_train=float((X_train.loc[m_tr, "MARRIAGE"] == 2).mean()),
                   share_grad_or_univ_train=float(X_train.loc[m_tr, "EDUCATION"].isin([1, 2]).mean()))
        for v in KEY_VARS:
            row[f"mean_{v}"] = float(X_train.loc[m_tr, v].mean())
        for v in ("LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"):
            row[f"median_{v}"] = float(X_train.loc[m_tr, v].median())
        for v in KEY_VARS:
            row[f"z_{v}"] = float(cent[c, NUM_COLS.index(v)])
        prof.append(row)
    allrow = dict(segment="ALL (train)", descriptor="training set", n_train=int(n_tr), share_train=1.0, default_rate_train=base_tr, lift_train=1.0,
                  n_test=int(len(X_test)), share_test=1.0, default_rate_test=base_te, lift_test=1.0, share_of_all_defaulters_train=1.0,
                  share_female_train=float((X_train["SEX"] == 2).mean()), share_single_train=float((X_train["MARRIAGE"] == 2).mean()),
                  share_grad_or_univ_train=float(X_train["EDUCATION"].isin([1, 2]).mean()))
    for v in KEY_VARS:
        allrow[f"mean_{v}"] = float(X_train[v].mean()); allrow[f"z_{v}"] = 0.0
    for v in ("LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"):
        allrow[f"median_{v}"] = float(X_train[v].median())
    prof = pd.DataFrame(prof + [allrow]); prof.to_csv(os.path.join(RESULTS, "kmeans_profiles.csv"), index=False)
    for _, r in prof.iterrows():
        log(f"    segment {r['segment']}: n_train {r['n_train']:>6,} ({r['share_train']*100:4.1f}%), default train {r['default_rate_train']:.3f} / "
            f"test {r['default_rate_test']:.3f} (lift {r['lift_train']:.2f});  {r['descriptor']}")
    log(f"    chi-square test of default rate across segments (train): chi2 = {chi2:.0f}, p = {p_chi2:.2e}")
    summary["kmeans"] = dict(K_range=K_RANGE, K_chosen=K, selection_rule="kneedle elbow on within-cluster SS; silhouette reported as check",
                             K_max_silhouette=k_sil, selection_table=sel.round(5).to_dict(orient="records"),
                             profiles=prof.round(5).to_dict(orient="records"), chi2_default_by_segment_train=float(chi2), chi2_pvalue=float(p_chi2),
                             label_order="segments numbered by ascending training default rate", n_iter_final=int(km.n_iter_))

    # --- figure: default rate per segment (train vs test) + centroid z-score heat-map
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5), gridspec_kw=dict(width_ratios=[1.1, 1.3]))
    xx = np.arange(K); w = 0.38
    p_ = prof.iloc[:K]
    b1 = ax[0].bar(xx - w / 2, p_["default_rate_train"] * 100, w, color=BLUE, label="training set (fitted)", edgecolor="none")
    b2 = ax[0].bar(xx + w / 2, p_["default_rate_test"] * 100, w, color=CAT[1], label="test set (assigned)", edgecolor="none")
    for b, r in zip(b1, p_["default_rate_train"]): ax[0].text(b.get_x() + w / 2, r * 100 + 1, f"{r*100:.1f}%", ha="center", fontsize=8.5, color=INK2)
    for b, r in zip(b2, p_["default_rate_test"]): ax[0].text(b.get_x() + w / 2, r * 100 + 1, f"{r*100:.1f}%", ha="center", fontsize=8.5, color=INK2)
    ax[0].axhline(base_tr * 100, color=INK2, ls="--", lw=1); ax[0].text(K - 0.5, base_tr * 100 + 1, f"base rate {base_tr*100:.1f}%", ha="right", fontsize=8.5, color=INK2)
    ax[0].set_xticks(xx); ax[0].set_xticklabels([f"segment {c+1}\n(n = {n:,} / {m:,})" for c, n, m in zip(xx, p_["n_train"], p_["n_test"])], fontsize=8)
    ax[0].set_ylabel("Default rate (%)", color=INK2); ax[0].set_ylim(0, max(p_["default_rate_train"].max(), p_["default_rate_test"].max()) * 100 * 1.22)
    ax[0].set_title(f"Default rate by K-means segment (K = {K}) — training vs test\n(segments ordered by risk; n = train / test rows)", fontsize=10, loc="left", color=INK)
    ax[0].legend(loc="upper left", frameon=False, fontsize=8.5); tidy(ax[0]); ax[0].grid(axis="x", visible=False)
    H = np.vstack([p_[[f"z_{v}" for v in KEY_VARS]].values])
    vmax = float(np.abs(H).max())
    im = ax[1].imshow(H, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax[1].set_xticks(range(len(KEY_VARS))); ax[1].set_xticklabels(KEY_VARS, rotation=35, ha="right", fontsize=8.5)
    ax[1].set_yticks(range(K)); ax[1].set_yticklabels([f"segment {c+1}" for c in range(K)], fontsize=9)
    for i in range(K):
        for j in range(len(KEY_VARS)):
            ax[1].text(j, i, f"{H[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(H[i, j]) > 0.6 * vmax else INK)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02); cb.set_label("segment mean, in training SDs from the overall mean", color=INK2, fontsize=8.5)
    ax[1].set_title("Segment profiles: centroid of each key variable (z-score units)\nred = above the population mean, blue = below", fontsize=10, loc="left", color=INK)
    ax[1].tick_params(colors=INK2)
    save_extra_figure(fig, "kmeans_default_rate.png")

    # ================================================================== (3) hierarchical clustering on a 1,000-row stratified sample
    log(f"(3) hierarchical clustering (Ward, complete) on a stratified {HC_SAMPLE}-row training sample")
    idx_hc, _ = train_test_split(np.arange(n_tr), train_size=HC_SAMPLE, stratify=ytr, random_state=SEED)
    idx_hc = np.sort(idx_hc); Z_hc = Z_tr[idx_hc]; y_hc = ytr[idx_hc]
    links = {m: linkage(Z_hc, method=m, metric="euclidean") for m in ("ward", "complete")}
    hc_rows, hc_lab = [], {}
    for m, L in links.items():
        for k in K_RANGE:
            cl = fcluster(L, k, criterion="maxclust")
            km_k = km_models[k].labels_[idx_hc]
            sizes = np.bincount(cl)[1:]
            hc_rows.append(dict(method=m, K=k, ari_vs_kmeans=float(adjusted_rand_score(km_k, cl)),
                                silhouette_hc=float(silhouette_score(Z_hc, cl)) if len(set(cl)) > 1 else np.nan,
                                silhouette_kmeans_same_rows=float(silhouette_score(Z_hc, km_k)),
                                min_cluster_size=int(sizes.min()), max_cluster_size=int(sizes.max()),
                                sizes=" ".join(map(str, sizes.tolist())),
                                default_rates=" ".join(f"{y_hc[cl == c].mean():.3f}" for c in range(1, k + 1))))
        hc_lab[m] = fcluster(L, K, criterion="maxclust")
    hc = pd.DataFrame(hc_rows); hc.to_csv(os.path.join(RESULTS, "hclust_agreement.csv"), index=False)
    ari_ward_complete = float(adjusted_rand_score(hc_lab["ward"], hc_lab["complete"]))
    at_k = hc[hc["K"] == K].set_index("method")
    for m in ("ward", "complete"):
        log(f"    {m:8s} K={K}: ARI vs k-means {at_k.loc[m, 'ari_vs_kmeans']:.3f}, silhouette {at_k.loc[m, 'silhouette_hc']:.3f} "
            f"(k-means on same rows {at_k.loc[m, 'silhouette_kmeans_same_rows']:.3f}), sizes [{at_k.loc[m, 'sizes']}], default rates [{at_k.loc[m, 'default_rates']}]")
    log(f"    ARI Ward vs complete at K={K}: {ari_ward_complete:.3f}")
    summary["hierarchical"] = dict(sample_rows=HC_SAMPLE, stratified_by="DEFAULT", table=hc.round(4).to_dict(orient="records"),
                                   ari_ward_vs_complete_at_K=ari_ward_complete,
                                   ari_vs_kmeans_at_K={m: float(at_k.loc[m, "ari_vs_kmeans"]) for m in ("ward", "complete")})

    # --- figure: dendrograms with the K-cluster cut
    set_link_color_palette(CAT[:K] + CAT[:K])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, (m, L) in zip(ax, links.items()):
        n = L.shape[0] + 1
        cut = 0.5 * (L[n - K - 1, 2] + L[n - K, 2])                       # height between the merges that leave K and K-1 clusters
        dendrogram(L, truncate_mode="lastp", p=40, color_threshold=cut, above_threshold_color=GREY, ax=a, no_labels=True, leaf_rotation=0)
        a.axhline(cut, color=INK2, ls="--", lw=1); a.text(a.get_xlim()[1] * 0.99, cut, f" cut → {K} clusters", ha="right", va="bottom", fontsize=8.5, color=INK2)
        sizes = np.bincount(hc_lab[m])[1:]
        a.set_title(f"{'Ward' if m == 'ward' else 'Complete'} linkage (Euclidean), {HC_SAMPLE}-row stratified training sample\n"
                    f"cut at K = {K}: sizes {sizes.tolist()}; ARI vs k-means = {at_k.loc[m, 'ari_vs_kmeans']:.3f}", fontsize=10, loc="left", color=INK)
        a.set_xlabel("last 40 merges (leaf = a merged group of clients)", color=INK2, fontsize=9); a.set_ylabel("merge height (distance)", color=INK2, fontsize=9)
        tidy(a); a.grid(axis="x", visible=False)
    save_extra_figure(fig, "hclust_dendrogram.png")

    # ================================================================== (4) supervised use of the unsupervised outputs
    log("(4a) logistic regression on all 29 features + one-hot K-means segment dummies (segments refitted inside every CV fold)")
    base = Pipeline([("pre", make_preprocessor()), ("clf", logit())])
    cv_base, cvsd_base = fold_auc(base, X_train, y_train)
    plus = Pipeline([("features", FeatureUnion([("pre", make_preprocessor()), ("clusters", ClusterDummies(K))])), ("clf", logit())])
    t0 = time.time(); plus.fit(X_train, y_train); fit_plus = time.time() - t0
    cv_plus, cvsd_plus = fold_auc(plus, X_train, y_train)
    p_oof_plus = oof_probs(plus, X_train, y_train)
    cd = plus.named_steps["features"].transformer_list[1][1]
    same_clusters = bool(np.array_equal(cd.kmeans_.labels_, raw_tr))
    coef = plus.named_steps["clf"].coef_[0]; n_pre = coef.shape[0] - (K - 1)
    ref_seg = int(remap[0]) + 1
    seg_coef = {f"segment_{int(remap[c]) + 1}_vs_segment_{ref_seg}": float(coef[n_pre + c - 1]) for c in range(1, K)}
    clusters_only = Pipeline([("clusters", ClusterDummies(K)), ("clf", logit())])
    cv_clu, cvsd_clu = fold_auc(clusters_only, X_train, y_train)
    plus_pcs = Pipeline([("features", FeatureUnion([("pre", make_preprocessor()), ("clusters", ClusterDummies(K)),
                                                    ("pcs", Pipeline([("num", num_scaler()), ("pca", PCA(5, svd_solver="full", random_state=SEED))]))])),
                         ("clf", logit())])
    cv_plus_pcs, cvsd_plus_pcs = fold_auc(plus_pcs, X_train, y_train)
    log(f"    CV AUC: full logistic {cv_base:.4f} ± {cvsd_base:.4f} | + {K-1} segment dummies {cv_plus:.4f} ± {cvsd_plus:.4f} | "
        f"segment dummies only {cv_clu:.4f} ± {cvsd_clu:.4f} | + dummies + top-5 PCs (collinear) {cv_plus_pcs:.4f} ± {cvsd_plus_pcs:.4f}")
    log(f"    segment coefficients (log-odds vs segment {ref_seg}, given all features): {dict((k, round(v, 3)) for k, v in seg_coef.items())}; "
        f"full-train k-means identical to section (2): {same_clusters}")
    res_plus = evaluate("Logistic regression + K-means segment dummies", plus, X_train, y_train, X_test, y_test, member=MEMBER,
                        best_params={"C": "inf (unpenalised)", "solver": "lbfgs", "n_clusters": K, "cluster_features": f"{K-1} one-hot dummies (reference = segment {ref_seg})",
                                     "kmeans_n_init": N_INIT}, cv_auc=cv_plus, cv_auc_std=cvsd_plus, fit_seconds=round(fit_plus, 2), p_oof=p_oof_plus,
                        slug="logistic_plus_clusters",
                        notes=f"Same plain logistic as 'logistic' (C=inf, lbfgs) on the 32 preprocessed columns plus {K-1} one-hot K-means segment "
                              f"dummies (K={K}, chosen by the elbow rule on the training set). Scaler + k-means are fitted inside the pipeline on each "
                              f"training fold. Adding the top-5 PC scores on top is exactly collinear with the 26 standardised numeric columns and "
                              f"leaves the CV AUC unchanged ({cv_plus_pcs:.4f}), so the evaluated model contains the dummies only.",
                        extra=dict(cv_auc_full_logistic_same_run=cv_base, cv_auc_std_full_logistic=cvsd_base, cv_auc_clusters_only=cv_clu,
                                   cv_auc_std_clusters_only=cvsd_clu, cv_auc_plus_clusters_plus_top5_pcs=cv_plus_pcs, cv_auc_std_plus_clusters_plus_top5_pcs=cvsd_plus_pcs,
                                   segment_dummy_coefficients=seg_coef, reference_segment=ref_seg, kmeans_identical_to_section2=same_clusters,
                                   n_columns=int(coef.shape[0])))

    log(f"(4b) logistic regression on the first n PCs only, n in {PC_GRID} chosen by GridSearchCV (5-fold CV AUC)")
    gs = GridSearchCV(pcs_pipeline(PC_GRID[0]), {"pca__n_components": PC_GRID}, cv=CV, scoring=SCORING, n_jobs=N_JOBS, refit=True)
    t0 = time.time(); gs.fit(X_train, y_train); fit_pcs = time.time() - t0
    r, i = gs.cv_results_, gs.best_index_
    n_best = int(gs.best_params_["pca__n_components"])
    grid_auc = {int(n): float(v) for n, v in zip(r["param_pca__n_components"], r["mean_test_score"])}
    grid_sd = {int(n): float(v) for n, v in zip(r["param_pca__n_components"], r["std_test_score"])}
    log(f"    grid CV AUC: {dict((k, round(v, 4)) for k, v in grid_auc.items())} -> best n = {n_best} ({fit_pcs:.0f}s)")
    p_oof_pcs = oof_probs(gs.best_estimator_, X_train, y_train)
    # context curve: CV AUC for every n = 1..26 (cheap: 26 x 5 tiny logistic fits) + numeric-only logistic (rotation invariance check)
    curve_n = list(range(1, n_comp + 1)); curve = []
    for n in curve_n:
        s = cross_val_score(pcs_pipeline(n), X_train, y_train, cv=CV, scoring=SCORING, n_jobs=N_JOBS); curve.append((float(s.mean()), float(s.std())))
    cv_num_only, cvsd_num_only = fold_auc(Pipeline([("num", num_scaler()), ("clf", logit())]), X_train, y_train)
    n_within_1se = next(n for n, (m_, s_) in zip(curve_n, curve) if m_ >= curve[-1][0] - curve[-1][1])
    log(f"    CV AUC curve: n=1 {curve[0][0]:.4f}, n=2 {curve[1][0]:.4f}, n=3 {curve[2][0]:.4f}, n=5 {curve[4][0]:.4f}, n=10 {curve[9][0]:.4f}, "
        f"n=15 {curve[14][0]:.4f}, n=26 {curve[-1][0]:.4f} (= numeric-only logistic {cv_num_only:.4f}); smallest n within 1 SE of n=26: {n_within_1se}")
    res_pcs = evaluate("Logistic regression on principal components only", gs.best_estimator_, X_train, y_train, X_test, y_test, member=MEMBER,
                       best_params={"n_components": n_best, "grid": PC_GRID, "C": "inf (unpenalised)", "solver": "lbfgs", "inputs": "26 standardised numeric features -> PCA scores (no categoricals)"},
                       cv_auc=float(r["mean_test_score"][i]), cv_auc_std=float(r["std_test_score"][i]), fit_seconds=round(fit_pcs, 1), p_oof=p_oof_pcs,
                       slug="logistic_on_pcs",
                       notes=f"Pipeline(StandardScaler on the 26 numeric columns -> PCA(n) -> plain logistic); n chosen from {PC_GRID} by GridSearchCV "
                             f"(cv=CV, scoring=roc_auc). The 3 categorical variables are not used. Full curve n=1..26 in extra/figures/pca_logistic_vs_ncomp.png.",
                       extra=dict(grid_n=PC_GRID, grid_mean_cv_auc=grid_auc, grid_std_cv_auc=grid_sd, curve_n=curve_n, curve_mean_cv_auc=[c[0] for c in curve],
                                  curve_std_cv_auc=[c[1] for c in curve], cv_auc_numeric_only_no_pca=cv_num_only, cv_auc_std_numeric_only=cvsd_num_only,
                                  cv_auc_full_logistic_same_run=cv_base, smallest_n_within_1se_of_all_pcs=int(n_within_1se),
                                  cumulative_pve_at_best=float(cum[n_best - 1])))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    mu = np.array([c[0] for c in curve]); sd = np.array([c[1] for c in curve])
    ax.fill_between(curve_n, mu - sd, mu + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(curve_n, mu, color=BLUE, lw=2, marker="o", ms=4, label="logistic on the first n PCs (5-fold CV AUC ± 1 SD)")
    ax.scatter(PC_GRID, [grid_auc[n] for n in PC_GRID], color=RED, s=55, zorder=5, label=f"candidates {PC_GRID}; chosen n = {n_best}")
    ax.axhline(cv_base, color=INK2, ls=":", lw=1.2, label=f"full logistic, 29 features: {cv_base:.4f}")
    ax.axhline(cv_num_only, color=CAT[1], ls="--", lw=1.2, label=f"logistic on the 26 numeric features (no PCA): {cv_num_only:.4f}")
    for n in PC_GRID:
        ax.annotate(f"{grid_auc[n]:.4f}", (n, grid_auc[n]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("n = number of principal components used as inputs", color=INK2); ax.set_ylabel("5-fold CV AUC (training set)", color=INK2)
    ax.set_xticks(curve_n); ax.set_title("Dimension reduction before logistic regression: CV AUC vs number of PCs", fontsize=10.5, loc="left", color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5); tidy(ax)
    save_extra_figure(fig, "pca_logistic_vs_ncomp.png")

    # ================================================================== summary
    summary["supervised"] = dict(
        logistic_reference=dict(cv_auc_same_run=cv_base, cv_auc_std=cvsd_base, test_auc_from_results_logistic_json=0.7636, test_ap=0.5234),
        logistic_plus_clusters=dict(cv_auc=cv_plus, cv_auc_std=cvsd_plus, test_auc=res_plus["test_auc"], test_ap=res_plus["test_ap"],
                                    test_brier=res_plus["test_brier"], cv_auc_clusters_only=cv_clu, cv_auc_plus_top5_pcs=cv_plus_pcs, segment_coefficients=seg_coef),
        logistic_on_pcs=dict(n_components=n_best, grid_cv_auc=grid_auc, cv_auc=float(r["mean_test_score"][i]), cv_auc_std=float(r["std_test_score"][i]),
                             test_auc=res_pcs["test_auc"], test_ap=res_pcs["test_ap"], test_brier=res_pcs["test_brier"],
                             curve=dict(n=curve_n, cv_auc=[c[0] for c in curve]), cv_auc_numeric_only_no_pca=cv_num_only, smallest_n_within_1se=int(n_within_1se)))
    summary["seconds"] = round(time.time() - t_all, 1)
    json.dump(summary, open(os.path.join(RESULTS, "unsupervised_summary.json"), "w"), indent=1)
    rows = []
    for rr in (res_plus, res_pcs):
        rows.append(dict(slug=rr["slug"], cv_auc=rr["cv_auc"], test_auc=rr["test_auc"], test_ap=rr["test_ap"], test_brier=rr["test_brier"],
                         acc_0_5=rr["at_0_5"]["accuracy"], recall_0_5=rr["at_0_5"]["recall_1"], precision_0_5=rr["at_0_5"]["precision_1"],
                         thr_f1=rr["at_tuned_f1"]["threshold"], recall_f1=rr["at_tuned_f1"]["recall_1"], precision_f1=rr["at_tuned_f1"]["precision_1"],
                         f1_f1=rr["at_tuned_f1"]["f1_1"], thr_cost=rr["at_tuned_cost"]["threshold"], cost=rr["at_tuned_cost"]["cost"], fit_seconds=rr["fit_seconds"]))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    log(f"done in {time.time() - t_all:.0f}s -> results/unsupervised_summary.json, pca_*.csv, kmeans_*.csv, hclust_agreement.csv, "
        f"logistic_plus_clusters.json, logistic_on_pcs.json; figures/pca_*.png, kmeans_*.png, hclust_dendrogram.png")


if __name__ == "__main__":
    main()
