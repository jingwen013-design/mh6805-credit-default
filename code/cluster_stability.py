"""
cluster_stability.py — RUAN Hangjing's v2 segmentation analysis (PCA + K-means + stability + incremental value), TRAINING
SET ONLY.  Owner of the method: RUAN Hangjing (model_unsupervised_v2.py, 30 Sep 2026).  Assembled 2026-10-21 by
LI Jingwen with the logic unchanged; the only edit is at the end: if the pre-registered adoption rule were to pass, the
cluster-label model is NOT scored on the test set here (it would be a new configuration and must go through the freeze
guard in common.evaluate() deliberately), the outcome of the rule is just reported.

What it answers (the professor's comment): silhouette as an intrinsic result; cluster stability (bootstrap ARI, per-cluster
Jaccard, Ward agreement); the incremental out-of-sample ROC-AUC / PR-AUC of the cluster label as a predictor, with
uncertainty (5 x 5 repeated CV on the training set, Nadeau–Bengio corrected interval); and that differing default rates
across clusters do not make them economically distinct (the inputs already contain the delay variables).

Writes results/kmeans_selection_repro.csv, kmeans_profiles_repro.csv, pca_loadings_repro.csv, cluster_incremental_folds_repro.csv,
unsupervised_summary_repro.json and figures/pca_scree_repro.png, kmeans_selection_repro.png, hclust_dendrogram_repro.png.
RUAN's own 30 Sep outputs (results/*_v2.*, received 21 Oct) are the record; the *_repro files are the 21 Oct reproduction.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from common import RESULTS, SEED, load_split, make_preprocessor, feature_names_out, plt, roc_auc_score, average_precision_score, save_extra_figure
from mine_v2_utils import corrected_t_ci

T0 = time.time()
X_train, _, y_train, _, feature_names = load_split()          # the test rows are not used anywhere in this script
ytr = y_train.values
N_PC, K_FINAL, K_GRID, B_STAB = 10, 5, [2, 3, 4, 5, 6, 7, 8], 100

# 1. PCA (descriptive; training set only)
pre = make_preprocessor()
Z = pre.fit_transform(X_train)
Z = np.asarray(Z.todense()) if hasattr(Z, "todense") else np.asarray(Z)
pca_full = PCA(random_state=SEED).fit(Z)
ev = pca_full.explained_variance_ratio_; cum = np.cumsum(ev)
print("preprocessed training matrix:", Z.shape)
for n in [2, 5, 10, 15]:
    print(f"cumulative variance of the first {n} PCs: {cum[n - 1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, len(ev) + 1), ev, marker="o"); ax[0].set_title("Scree plot"); ax[0].set_xlabel("principal component"); ax[0].set_ylabel("explained variance ratio")
ax[1].plot(range(1, len(cum) + 1), cum, marker="o"); ax[1].axhline(0.8, color="red", ls="--", label="80%"); ax[1].axvline(N_PC, color="grey", lw=0.8)
ax[1].set_title("Cumulative variance"); ax[1].legend(); ax[1].set_xlabel("number of components"); ax[1].set_ylabel("cumulative explained variance")
save_extra_figure(fig, "pca_scree_repro.png")
pc_names = feature_names_out(pre)
load = pd.DataFrame(pca_full.components_[:2].T, columns=["PC1", "PC2"], index=pc_names)
load.round(4).to_csv(os.path.join(RESULTS, "pca_loadings_repro.csv"))
P = pca_full.transform(Z)[:, :N_PC]


# 2. choose K: elbow + silhouette + bootstrap stability
def fit_km(X, k, seed=SEED, n_init=10):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def bootstrap_stability(X, k, B=B_STAB, seed=SEED):
    ref = fit_km(X, k).labels_
    rng = np.random.default_rng(seed); aris, jac = [], np.zeros((B, k))
    ref_sets = [set(np.where(ref == c)[0]) for c in range(k)]
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))
        lab = fit_km(X[idx], k, seed=seed + b + 1).predict(X)
        aris.append(adjusted_rand_score(ref, lab))
        boot_sets = [set(np.where(lab == c)[0]) for c in range(k)]
        jac[b] = [max(jaccard(r, s) for s in boot_sets) for r in ref_sets]
    return ref, np.array(aris), jac.mean(axis=0)


sel_rows = []
for k in K_GRID:
    km = fit_km(P, k)
    sil = silhouette_score(P, km.labels_, sample_size=5000, random_state=SEED)
    _, aris, jac = bootstrap_stability(P, k, B=30)
    sel_rows.append(dict(K=k, inertia=float(km.inertia_), silhouette=float(sil), boot_ari_mean=float(aris.mean()),
                         boot_ari_p5=float(np.percentile(aris, 5)), min_cluster_jaccard=float(jac.min())))
    print(f"K={k}  inertia={km.inertia_:>10.0f}  silhouette={sil:.3f}  bootstrap ARI={aris.mean():.3f} (5th pct {np.percentile(aris, 5):.3f})  "
          f"min cluster Jaccard={jac.min():.3f}  [{time.time() - T0:.0f}s]")
sel = pd.DataFrame(sel_rows); sel.round(4).to_csv(os.path.join(RESULTS, "kmeans_selection_repro.csv"), index=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(sel["K"], sel["inertia"], marker="o"); ax[0].set_title("Elbow")
ax[1].plot(sel["K"], sel["silhouette"], marker="o", color="C1"); ax[1].set_title("Silhouette")
ax[2].plot(sel["K"], sel["boot_ari_mean"], marker="o", color="C2", label="mean bootstrap ARI")
ax[2].plot(sel["K"], sel["min_cluster_jaccard"], marker="s", color="C4", label="least stable cluster (Jaccard)")
ax[2].axhline(0.75, color="grey", ls="--", lw=0.8); ax[2].set_title("Stability"); ax[2].legend(fontsize=8)
for a in ax: a.set_xlabel("K"); a.axvline(K_FINAL, color="grey", lw=0.8)
save_extra_figure(fig, "kmeans_selection_repro.png")

# 3. K = 5 in detail
ref, aris, jac5 = bootstrap_stability(P, K_FINAL, B=B_STAB)
seed_ari = [adjusted_rand_score(ref, fit_km(P, K_FINAL, seed=SEED + 100 + s, n_init=1).labels_) for s in range(20)]
sil5 = silhouette_score(P, ref, sample_size=5000, random_state=SEED)
print(f"\nK={K_FINAL}: silhouette={sil5:.3f}; bootstrap ARI mean={aris.mean():.3f} [{np.percentile(aris, 2.5):.3f}, {np.percentile(aris, 97.5):.3f}]; "
      f"seed ARI (n_init=1) mean={np.mean(seed_ari):.3f} min={np.min(seed_ari):.3f}; per-cluster Jaccard {np.round(jac5, 3)}")

# 4. Ward vs K-means
rng = np.random.default_rng(SEED); h_ari = []
for s in range(5):
    idx = rng.choice(len(P), 3000, replace=False)
    L = linkage(P[idx], method="ward")
    h_lab = fcluster(L, t=K_FINAL, criterion="maxclust")
    h_ari.append(float(adjusted_rand_score(ref[idx], h_lab)))
    if s == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(L, truncate_mode="lastp", p=30, ax=ax, no_labels=True, color_threshold=L[-(K_FINAL - 1), 2])
        ax.set_title(f"Ward dendrogram (3,000-client training subsample, cut at K={K_FINAL})"); ax.set_ylabel("merge distance")
        save_extra_figure(fig, "hclust_dendrogram_repro.png")
print(f"Ward vs K-means ARI on 5 subsamples: {np.round(h_ari, 3)} (mean {np.mean(h_ari):.3f})")

# 5. profiles (descriptive only — the inputs contain the delay variables, so differing default rates are expected)
prof = X_train.copy(); prof["cluster"] = ref; prof["DEFAULT"] = ytr
summary_tab = prof.groupby("cluster").agg(n_clients=("DEFAULT", "size"), default_rate=("DEFAULT", "mean"),
                                          avg_limit=("LIMIT_BAL", "mean"), avg_age=("AGE", "mean"),
                                          avg_months_delayed=("N_MONTHS_DELAYED", "mean"), avg_util=("UTIL_MEAN", "mean"))
summary_tab["share"] = summary_tab["n_clients"] / len(prof); summary_tab["jaccard_stability"] = jac5
summary_tab = summary_tab.round(3); summary_tab.to_csv(os.path.join(RESULTS, "kmeans_profiles_repro.csv"))
print("\ncluster profiles (training set, descriptive):\n" + summary_tab.to_string())


# 6. incremental value of the cluster label as a predictor — PCA + K-means refitted inside every fold
class ClusterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=N_PC, n_clusters=K_FINAL, random_state=SEED):
        self.n_components, self.n_clusters, self.random_state = n_components, n_clusters, random_state

    def fit(self, X, y=None):
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        self.pca_ = PCA(self.n_components, random_state=self.random_state).fit(X)
        self.km_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=10).fit(self.pca_.transform(X))
        return self

    def labels(self, X):
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        return self.km_.predict(self.pca_.transform(X))

    def transform(self, X):
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        oh = np.eye(self.n_clusters)[self.labels(X)][:, 1:]
        return np.hstack([X, oh])


LR = LogisticRegression(C=np.inf, max_iter=5000)
MODELS = {
    "logistic": Pipeline([("pre", make_preprocessor()), ("clf", clone(LR))]),
    "logistic + clusters": Pipeline([("pre", make_preprocessor()), ("clu", ClusterFeatures()), ("clf", clone(LR))]),
    "logistic on 10 PCs": Pipeline([("pre", make_preprocessor()), ("pca", PCA(N_PC, random_state=SEED)), ("clf", clone(LR))]),
}
RCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
rows = []
for f, (tr, va) in enumerate(RCV.split(X_train, ytr)):
    row = dict(fold=f)
    for name, m in MODELS.items():
        p = clone(m).fit(X_train.iloc[tr], ytr[tr]).predict_proba(X_train.iloc[va])[:, 1]
        row[f"auc|{name}"] = float(roc_auc_score(ytr[va], p)); row[f"ap|{name}"] = float(average_precision_score(ytr[va], p))
    pre_f = make_preprocessor().fit(X_train.iloc[tr]); cf = ClusterFeatures().fit(pre_f.transform(X_train.iloc[tr]))
    lab_tr, lab_va = cf.labels(pre_f.transform(X_train.iloc[tr])), cf.labels(pre_f.transform(X_train.iloc[va]))
    rate = pd.Series(ytr[tr]).groupby(lab_tr).mean()
    row["auc|cluster label as score"] = float(roc_auc_score(ytr[va], rate.reindex(lab_va).values))
    row["auc|N_MONTHS_DELAYED alone"] = float(roc_auc_score(ytr[va], X_train.iloc[va]["N_MONTHS_DELAYED"].values))
    rows.append(row)
    if f % 5 == 4:
        print(f"repeated CV: {f + 1}/25 folds done  [{time.time() - T0:.0f}s]")
cvtab = pd.DataFrame(rows); cvtab.round(5).to_csv(os.path.join(RESULTS, "cluster_incremental_folds_repro.csv"), index=False)
n_tr, n_va = int(len(ytr) * 0.8), int(len(ytr) * 0.2)
inc = {}
for met in ["auc", "ap"]:
    for other in ["logistic + clusters", "logistic on 10 PCs"]:
        inc[f"{met}: {other} - logistic"] = corrected_t_ci(cvtab[f"{met}|{other}"] - cvtab[f"{met}|logistic"], n_tr, n_va)
print("\nincremental value (training set, 5 x 5 repeated CV, Nadeau–Bengio 95% CI)")
for k, v in inc.items():
    print(f"{k:<38} {v['mean']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
for c in [c for c in cvtab.columns if c.startswith("auc|")]:
    print(f"mean {c:<38} {cvtab[c].mean():.4f}")
adopt = bool(inc["auc: logistic + clusters - logistic"]["lo"] > 0 and inc["ap: logistic + clusters - logistic"]["lo"] > 0)
print(f"\npre-registered rule (adopt the cluster feature only if both CI lower bounds > 0): {adopt}"
      + ("" if not adopt else "  -> NOT scored here; scoring a new configuration on the test set must go through the freeze guard deliberately"))

json.dump(dict(generated_by="code/cluster_stability.py", method_owner="RUAN Hangjing (v2, 30 Sep 2026)", assembled_by="LI Jingwen, 2026-10-21",
               uses_test_set=False, configuration=dict(pca_input="make_preprocessor() output (26 standardised numeric + 6 one-hot = 32 columns)",
                                                        n_components=N_PC, K=K_FINAL, k_grid=K_GRID),
               pca_cum_var={n: float(cum[n - 1]) for n in [2, 5, 10, 15]}, silhouette=float(sil5),
               silhouette_by_k={int(r.K): float(r.silhouette) for r in sel.itertuples()},
               bootstrap_ari=dict(B=B_STAB, mean=float(aris.mean()), lo=float(np.percentile(aris, 2.5)), hi=float(np.percentile(aris, 97.5))),
               seed_ari=dict(mean=float(np.mean(seed_ari)), min=float(np.min(seed_ari))),
               cluster_jaccard=[float(j) for j in jac5], ward_vs_kmeans_ari=h_ari,
               profiles=summary_tab.reset_index().to_dict(orient="records"),
               incremental=inc, mean_auc={c: float(cvtab[c].mean()) for c in cvtab.columns if c.startswith("auc|")},
               adoption_rule="adopt the one-hot cluster label as a predictor only if the Nadeau–Bengio 95% CI lower bounds of ΔAUC and ΔAP are both > 0",
               adopted=adopt, seconds=round(time.time() - T0, 1)),
          open(os.path.join(RESULTS, "unsupervised_summary_repro.json"), "w"), indent=1)
print(f"\nwrote results/unsupervised_summary_repro.json and companions  [{time.time() - T0:.0f}s]")
