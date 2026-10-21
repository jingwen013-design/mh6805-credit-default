# 几个共用的小函数:冻结配置、成本、bootstrap 置信区间、校准图
# 放在 common.py 同一个文件夹里用
from common import *
from scipy import stats
from sklearn.calibration import calibration_curve

MEMBER = "RUAN Hangjing"
COST_RATIOS = [1, 2, 3, 4, 5, 7, 10, 15, 20]          # 漏掉违约 : 误报,误报成本记为1
THR_GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)
FREEZE_FILE = os.path.join(RESULTS, "frozen_choices_ruan_v2.json")
N_BOOT = 2000


def freeze_or_check(slug, config):
    # 跑 test 之前先把最终配置写进 json;同一个模型配置变了就报错,防止看完 test 再回去改
    reg = json.load(open(FREEZE_FILE)) if os.path.exists(FREEZE_FILE) else {}
    cfg = json.loads(json.dumps(config, sort_keys=True, default=str))
    if slug in reg and reg[slug]["config"] != cfg and os.environ.get("ALLOW_REFREEZE") != "1":
        raise RuntimeError(f"{slug} 之前冻结的是 {reg[slug]['config']},现在变成了 {cfg},看过 test 之后不能再改")
    reg[slug] = dict(config=cfg, frozen_at=reg.get(slug, {}).get("frozen_at", time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(reg, open(FREEZE_FILE, "w"), indent=1)
    print(f"[freeze] {slug}: {cfg}")


def cost_at(y, p, t, ratio):
    yhat = p >= t
    fn = int(np.sum(~yhat & (y == 1))); fp = int(np.sum(yhat & (y == 0)))
    return ratio * fn + fp


def best_threshold(y, p, ratio):
    costs = [cost_at(y, p, t, ratio) for t in THR_GRID]
    return float(THR_GRID[int(np.argmin(costs))])


def cross_fitted_cost(y, p_oof, folds, ratio):
    # 每一折的阈值用其他折的 OOF 概率来选,这样算出来的成本不会偏乐观
    total = 0.0
    for tr, va in folds:
        t = best_threshold(y[tr], p_oof[tr], ratio)
        total += cost_at(y[va], p_oof[va], t, ratio)
    return total


def cost_sensitivity(y_tr, p_oof, y_te=None, p_te=None, ratios=COST_RATIOS):
    # 每个成本比:理论阈值 1/(1+r)、训练 OOF 上选的阈值、test 上 0.5 和调过阈值的成本
    rows = []
    for r in ratios:
        t = best_threshold(y_tr, p_oof, r)
        row = dict(ratio=r, t_theory=round(1 / (1 + r), 3), t_oof=t)
        if p_te is not None:
            row.update(test_cost_05=cost_at(y_te, p_te, 0.5, r), test_cost_tuned=cost_at(y_te, p_te, t, r),
                       test_cost_flag_all=int(np.sum(y_te == 0)), test_cost_flag_none=int(r * np.sum(y_te == 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(y, stat_fn, B=N_BOOT, seed=SEED):
    # 对客户做 bootstrap,返回估计值和 95% 区间
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(stat_fn(idx))
    vals = np.asarray(vals)
    return dict(estimate=float(stat_fn(np.arange(n))), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)))


def delta_metric_ci(y, p_a, p_b, metric=roc_auc_score, **kw):
    return paired_bootstrap(y, lambda i: metric(y[i], p_b[i]) - metric(y[i], p_a[i]), **kw)


def corrected_t_ci(diffs, n_train, n_test, alpha=0.05):
    # repeated CV 各折共用训练数据,直接算标准误会偏小,用 Nadeau & Bengio (2003) 的修正
    d = np.asarray(diffs, float); J = len(d)
    var = (1.0 / J + n_test / n_train) * d.var(ddof=1)
    half = stats.t.ppf(1 - alpha / 2, J - 1) * np.sqrt(var)
    return dict(mean=float(d.mean()), lo=float(d.mean() - half), hi=float(d.mean() + half))


def calibration_summary(y, p):
    # Brier、平均预测概率 vs 实际违约率、校准斜率和截距
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=np.inf, max_iter=5000).fit(lp.reshape(-1, 1), y)
    return dict(brier=float(brier_score_loss(y, p)), mean_pred=float(p.mean()), obs_rate=float(y.mean()),
                cal_slope=float(lr.coef_[0, 0]), cal_intercept=float(lr.intercept_[0]))


def plot_reliability(curves, filename, title):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for lab, (y, p) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", lw=1.5, label=f"{lab} (Brier {brier_score_loss(y, p):.4f})")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default rate")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="upper left")
    return save_extra_figure(fig, filename)


def plot_cost_sensitivity(tables, filename, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    first = next(iter(tables.values()))
    ax[0].plot(first["ratio"], first["t_theory"], "k--", lw=1, label="theory 1/(1+r)")
    for i, (lab, tab) in enumerate(tables.items()):
        ax[0].plot(tab["ratio"], tab["t_oof"], marker="o", color=f"C{i}", label=f"{lab}: tuned on train OOF")
        ax[1].plot(tab["ratio"], tab["test_cost_tuned"], marker="o", color=f"C{i}", label=f"{lab}: tuned cut-off")
        ax[1].plot(tab["ratio"], tab["test_cost_05"], ls=":", color=f"C{i}", label=f"{lab}: cut-off 0.5")
    ax[1].plot(first["ratio"], np.minimum(first["test_cost_flag_all"], first["test_cost_flag_none"]), "k--", lw=1,
               label="best trivial rule")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(COST_RATIOS); a.set_xticklabels(COST_RATIOS); a.axvline(5, color="grey", lw=0.8)
        a.set_xlabel("cost ratio C_FN : C_FP")
    ax[0].set_ylabel("cost-minimising cut-off"); ax[1].set_ylabel("expected cost on test (C_FP = 1)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    return save_extra_figure(fig, filename)
