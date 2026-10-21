# PCA + K-means / hierarchical 第二版
# 老师说 cluster 当特征的话要报增量 AUC/AP 和稳定性,不能因为违约率不一样就说簇有经济意义
from mine_v2_utils import *
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

X_train, X_test, y_train, y_test, feature_names = load_split()
ytr = y_train.values
N_PC, K_FINAL, K_GRID, B_STAB = 10, 5, [2, 3, 4, 5, 6, 7, 8], 100   # 10个主成分和K=5沿用第一版

# 1. PCA(描述性;只在训练集拟合)
pre = make_preprocessor()
Z = pre.fit_transform(X_train)
pca_full = PCA(random_state=SEED).fit(Z)
ev = pca_full.explained_variance_ratio_; cum = np.cumsum(ev)
print("预处理后训练集维度:", Z.shape)
for n in [2, 5, 10, 15]:
    print(f"前{n}个主成分累计解释方差: {cum[n-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, len(ev) + 1), ev, marker="o"); ax[0].set_title("Scree plot")
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("explained variance ratio")
ax[1].plot(range(1, len(cum) + 1), cum, marker="o"); ax[1].axhline(0.8, color="red", ls="--", label="80%")
ax[1].axvline(N_PC, color="grey", lw=0.8); ax[1].set_title("Cumulative variance"); ax[1].legend()
ax[1].set_xlabel("number of components"); ax[1].set_ylabel("cumulative explained variance")
save_extra_figure(fig, "pca_scree_v2.png")

pc_names = feature_names_out(pre)
load = pd.DataFrame(pca_full.components_[:2].T, columns=["PC1", "PC2"], index=pc_names)
load.round(4).to_csv(os.path.join(RESULTS, "pca_loadings_v2.csv"))
for c in ["PC1", "PC2"]:
    print(f"\n{c} loadings(前6个):\n" + load[c].reindex(load[c].abs().sort_values(ascending=False).index).head(6).round(3).to_string())

P = pca_full.transform(Z)[:, :N_PC]

# 2. 选 K:elbow + silhouette + 稳定性
def fit_km(X, k, seed=SEED, n_init=10):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def bootstrap_stability(X, k, B=B_STAB, seed=SEED):
    # bootstrap 重抽样后重新跑 K-means,和原来的标签比 ARI / 每个簇的 Jaccard
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
    _, aris, jac = bootstrap_stability(P, k, B=30)          # 每个K跑30次
    sel_rows.append(dict(K=k, inertia=km.inertia_, silhouette=sil, boot_ari_mean=aris.mean(),
                         boot_ari_p5=np.percentile(aris, 5), min_cluster_jaccard=jac.min()))
    print(f"K={k}  inertia={km.inertia_:>10.0f}  silhouette={sil:.3f}  bootstrap ARI={aris.mean():.3f} "
          f"(5th pct {np.percentile(aris, 5):.3f})  min cluster Jaccard={jac.min():.3f}")
sel = pd.DataFrame(sel_rows); sel.round(4).to_csv(os.path.join(RESULTS, "kmeans_selection_v2.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(sel["K"], sel["inertia"], marker="o"); ax[0].set_title("Elbow")
ax[1].plot(sel["K"], sel["silhouette"], marker="o", color="C1"); ax[1].set_title("Silhouette")
ax[2].plot(sel["K"], sel["boot_ari_mean"], marker="o", color="C2", label="mean bootstrap ARI")
ax[2].plot(sel["K"], sel["min_cluster_jaccard"], marker="s", color="C4", label="least stable cluster (Jaccard)")
ax[2].axhline(0.75, color="grey", ls="--", lw=0.8); ax[2].set_title("Stability"); ax[2].legend(fontsize=8)
for a in ax: a.set_xlabel("K"); a.axvline(K_FINAL, color="grey", lw=0.8)
save_extra_figure(fig, "kmeans_selection_v2.png")

# 3. K=5 的详细稳定性
ref, aris, jac5 = bootstrap_stability(P, K_FINAL, B=B_STAB)
seed_ari = [adjusted_rand_score(ref, fit_km(P, K_FINAL, seed=SEED + 100 + s, n_init=1).labels_) for s in range(20)]
sil5 = silhouette_score(P, ref, sample_size=5000, random_state=SEED)
print(f"\nK={K_FINAL}: silhouette={sil5:.3f}; bootstrap ARI mean={aris.mean():.3f} [{np.percentile(aris, 2.5):.3f}, "
      f"{np.percentile(aris, 97.5):.3f}]; seed ARI (n_init=1) mean={np.mean(seed_ari):.3f} min={np.min(seed_ari):.3f}")
print("每个簇的 Jaccard 稳定性:", np.round(jac5, 3))

# 4. hierarchical (Ward) vs K-means
rng = np.random.default_rng(SEED); h_ari = []
for s in range(5):
    idx = rng.choice(len(P), 3000, replace=False)
    L = linkage(P[idx], method="ward")
    h_lab = fcluster(L, t=K_FINAL, criterion="maxclust")
    h_ari.append(adjusted_rand_score(ref[idx], h_lab))
    if s == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(L, truncate_mode="lastp", p=30, ax=ax, no_labels=True, color_threshold=L[-(K_FINAL - 1), 2])
        ax.set_title(f"Ward dendrogram (3,000-client training subsample, cut at K={K_FINAL})"); ax.set_ylabel("merge distance")
        save_extra_figure(fig, "hclust_dendrogram_v2.png")
print(f"Ward 和 K-means 在5个子样本上的一致性 ARI = {np.round(h_ari, 3)}  (平均 {np.mean(h_ari):.3f})")

# 5. 画像(只做描述)
prof = X_train.copy(); prof["cluster"] = ref; prof["DEFAULT"] = ytr
summary = prof.groupby("cluster").agg(n_clients=("DEFAULT", "size"), default_rate=("DEFAULT", "mean"),
                                      avg_limit=("LIMIT_BAL", "mean"), avg_age=("AGE", "mean"),
                                      avg_months_delayed=("N_MONTHS_DELAYED", "mean"), avg_util=("UTIL_MEAN", "mean"))
summary["share"] = summary["n_clients"] / len(prof); summary["jaccard_stability"] = jac5
summary = summary.round(3); summary.to_csv(os.path.join(RESULTS, "kmeans_profiles_v2.csv"))
print("\n===== 5个客户群体画像(只做描述,聚类输入里本来就有逾期变量) =====\n" + summary.to_string())

# 6. 增量价值:cluster 特征放进 Pipeline,折内拟合
class ClusterFeatures(BaseEstimator, TransformerMixin):
    # 在给定的数据上拟合 PCA + K-means,然后把 cluster 的 one-hot 拼到原特征后面
    def __init__(self, n_components=N_PC, n_clusters=K_FINAL, random_state=SEED):
        self.n_components, self.n_clusters, self.random_state = n_components, n_clusters, random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.pca_ = PCA(self.n_components, random_state=self.random_state).fit(X)
        self.km_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=10).fit(self.pca_.transform(X))
        return self

    def labels(self, X):
        return self.km_.predict(self.pca_.transform(np.asarray(X)))

    def transform(self, X):
        oh = np.eye(self.n_clusters)[self.labels(X)][:, 1:]
        return np.hstack([np.asarray(X), oh])


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
        row[f"auc|{name}"] = roc_auc_score(ytr[va], p); row[f"ap|{name}"] = average_precision_score(ytr[va], p)
    # 把"簇"本身当作一个评分(簇 -> 训练折违约率),与单一逾期变量对比
    pre_f = make_preprocessor().fit(X_train.iloc[tr]); cf = ClusterFeatures().fit(pre_f.transform(X_train.iloc[tr]))
    lab_tr, lab_va = cf.labels(pre_f.transform(X_train.iloc[tr])), cf.labels(pre_f.transform(X_train.iloc[va]))
    rate = pd.Series(ytr[tr]).groupby(lab_tr).mean()
    row["auc|cluster label as score"] = roc_auc_score(ytr[va], rate.reindex(lab_va).values)
    row["auc|N_MONTHS_DELAYED alone"] = roc_auc_score(ytr[va], X_train.iloc[va]["N_MONTHS_DELAYED"].values)
    rows.append(row)
cvtab = pd.DataFrame(rows); cvtab.round(5).to_csv(os.path.join(RESULTS, "cluster_incremental_folds_v2.csv"), index=False)

n_tr, n_va = int(len(ytr) * 0.8), int(len(ytr) * 0.2)
inc = {}
for met in ["auc", "ap"]:
    for other in ["logistic + clusters", "logistic on 10 PCs"]:
        inc[f"{met}: {other} - logistic"] = corrected_t_ci(cvtab[f"{met}|{other}"] - cvtab[f"{met}|logistic"], n_tr, n_va)
print("\n===== cluster 特征的增量(训练集 5x5 repeated CV, Nadeau-Bengio 95% CI) =====")
for k, v in inc.items():
    print(f"{k:<38} {v['mean']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
for c in [c for c in cvtab.columns if c.startswith("auc|")]:
    print(f"mean {c:<38} {cvtab[c].mean():.4f}")

# 事先写好的规则:ΔAUC 和 ΔAP 的 95% CI 下界都 > 0 才采用 cluster 特征;否则不碰 test
adopt = inc["auc: logistic + clusters - logistic"]["lo"] > 0 and inc["ap: logistic + clusters - logistic"]["lo"] > 0
print(f"\n要不要把 cluster 当特征: {adopt}")
if adopt:
    freeze_or_check("logistic_plus_clusters_v2", dict(n_components=N_PC, n_clusters=K_FINAL, base="plain logistic"))
    evaluate("Logistic + K-means clusters (v2)", MODELS["logistic + clusters"].fit(X_train, ytr), X_train, y_train,
             X_test, y_test, member=MEMBER, notes="Adopted by repeated-CV rule; frozen before test.", slug="logistic_plus_clusters_v2")
else:
    print("不用 cluster 特征,只做描述分析,不碰 test")

json.dump(dict(pca_cum_var={n: float(cum[n - 1]) for n in [2, 5, 10, 15]}, k_final=K_FINAL, silhouette=float(sil5),
               bootstrap_ari=dict(mean=float(aris.mean()), lo=float(np.percentile(aris, 2.5)), hi=float(np.percentile(aris, 97.5))),
               seed_ari=dict(mean=float(np.mean(seed_ari)), min=float(np.min(seed_ari))),
               cluster_jaccard=[float(j) for j in jac5], ward_vs_kmeans_ari=[float(a) for a in h_ari],
               incremental=inc, mean_auc={c: float(cvtab[c].mean()) for c in cvtab.columns if c.startswith("auc|")},
               adopted=bool(adopt)),
          open(os.path.join(RESULTS, "unsupervised_summary_v2.json"), "w"), indent=1)
