# logistic 第二版(plain / ridge / lasso / class-weighted)
# 第一版四个都看了 test 再比较,这里改成只在训练集上 nested CV 比较,选完冻结才碰 test
# class-weighted 加了折内校准,最后做成本比 1:1~20:1 的敏感性
from mine_v2_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV

X_train, X_test, y_train, y_test, feature_names = load_split()
ytr, yte = y_train.values, y_test.values
folds = list(CV.split(X_train, ytr))

C_GRID = list(np.round(np.logspace(-3, 2, 11), 5))                    # C 的范围
INNER = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)       # 内层调 C
CAL_CV = StratifiedKFold(5, shuffle=True, random_state=SEED + 2)      # 校准用的折


def lr_pipe(**kw):
    return Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=5000, **kw))])


def tuned(**kw):
    return GridSearchCV(lr_pipe(**kw), {"clf__C": C_GRID}, cv=INNER, scoring=SCORING, n_jobs=2)


# sklearn >= 1.8: 'penalty' 已弃用;C=inf 即不加惩罚,l1_ratio=0 是 ridge,l1_ratio=1 是 lasso
VARIANTS = {
    "plain":        lr_pipe(C=np.inf),
    "ridge":        tuned(l1_ratio=0.0),
    "lasso":        tuned(l1_ratio=1.0, solver="liblinear"),
    "weighted":     tuned(l1_ratio=0.0, class_weight="balanced"),
    "weighted_cal": CalibratedClassifierCV(tuned(l1_ratio=0.0, class_weight="balanced"), method="sigmoid", cv=CAL_CV),
}

# 1. nested CV,只用训练集
print("===== 4个变体的 nested CV(只用训练集) =====")
oof, rows = {}, []
for name, est in VARIANTS.items():
    t0 = time.time()
    p = cross_val_predict(est, X_train, ytr, cv=CV, method="predict_proba")[:, 1]
    oof[name] = p
    fold_auc = [roc_auc_score(ytr[va], p[va]) for _, va in folds]
    cal = calibration_summary(ytr, p)
    rows.append(dict(variant=name, nested_auc=np.mean(fold_auc), nested_auc_se=np.std(fold_auc, ddof=1) / np.sqrt(len(folds)),
                     nested_ap=average_precision_score(ytr, p), **cal,
                     t_cost5_oof=best_threshold(ytr, p, 5), t_theory5=round(1 / 6, 3),
                     crossfit_cost5=cross_fitted_cost(ytr, p, folds, 5), seconds=round(time.time() - t0, 1)))
    print(f"{name:<13} nested AUC={rows[-1]['nested_auc']:.4f}  AP={rows[-1]['nested_ap']:.4f}  "
          f"Brier={cal['brier']:.4f}  mean p={cal['mean_pred']:.3f}  t*(5:1)={rows[-1]['t_cost5_oof']:.2f}  "
          f"cross-fitted cost={rows[-1]['crossfit_cost5']:.0f}")
nested = pd.DataFrame(rows).set_index("variant")
nested.round(4).to_csv(os.path.join(RESULTS, "logistic_nested_cv_v2.csv"))

# 2. 事先写好的选择规则(不看 test)
# 第一步:plain/ridge/lasso 比 nested AUC,和最好的差不到1个标准误就选更简单的(plain > ridge > lasso)
rank_cands = ["plain", "ridge", "lasso"]
best_auc = nested.loc[rank_cands, "nested_auc"].max()
se = nested.loc[nested.loc[rank_cands, "nested_auc"].idxmax(), "nested_auc_se"]
step1 = next(v for v in rank_cands if nested.loc[v, "nested_auc"] >= best_auc - se)
# 第二步:没校准的 weighted 概率是偏的不能直接用;校准后的 weighted_cal 成本要比第一步低1%以上才用它
gain = 1 - nested.loc["weighted_cal", "crossfit_cost5"] / nested.loc[step1, "crossfit_cost5"]
selected = "weighted_cal" if gain > 0.01 else step1
print(f"\n第一步选出: {step1}   weighted_cal 成本变化: {gain:+.2%}   最终选: {selected}")

# 3. 校准:加权会扭曲概率,校准后恢复
plot_reliability({"plain": (ytr, oof["plain"]), "class-weighted": (ytr, oof["weighted"]),
                  "class-weighted + sigmoid calibration": (ytr, oof["weighted_cal"])},
                 "logistic_reliability_v2.png", "Logistic family: reliability (nested out-of-fold, training)")
thr = pd.DataFrame([dict(ratio=r, theory=round(1 / (1 + r), 3), **{v: best_threshold(ytr, oof[v], r) for v in
                   ["plain", "weighted", "weighted_cal"]}) for r in COST_RATIOS])
thr.to_csv(os.path.join(RESULTS, "logistic_threshold_vs_theory_v2.csv"), index=False)
print("\n训练集OOF上的最优阈值 vs 理论值 1/(1+r):\n" + thr.to_string(index=False))

# 4. 解释(训练集上拟合,与 test 无关)
lasso_fit = VARIANTS["lasso"].fit(X_train, ytr).best_estimator_
names = feature_names_out(lasso_fit.named_steps["pre"])
lasso_zero = [n for n, c in zip(names, lasso_fit.named_steps["clf"].coef_[0]) if c == 0]
print(f"\nLasso(C={lasso_fit.named_steps['clf'].C}) 把 {len(lasso_zero)}/{len(names)} 个系数压成了0: {lasso_zero}")

final_est = VARIANTS[selected] if selected != "plain" else lr_pipe(C=np.inf)
final_est.fit(X_train, ytr)
if selected in ("ridge", "lasso", "weighted"):
    final_pipe = final_est.best_estimator_; final_params = {"C": final_pipe.named_steps["clf"].C}
elif selected == "weighted_cal":
    final_pipe = final_est; final_params = {"C": "tuned inside each calibration fold", "calibration": "sigmoid"}
else:
    final_pipe = final_est; final_params = {"C": "inf (no penalty)"}

coef_src = final_pipe if selected != "weighted_cal" else lr_pipe(C=np.inf).fit(X_train, ytr)
coef = pd.DataFrame({"feature": feature_names_out(coef_src.named_steps["pre"]), "coef": coef_src.named_steps["clf"].coef_[0]})
coef["odds_ratio_per_sd"] = np.exp(coef["coef"])
coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
coef.round(4).to_csv(os.path.join(RESULTS, "logistic_odds_ratios_v2.csv"), index=False)
print("\n系数解读:Top 10 优势比(每1个标准差,BILL_AMT 共线性强,符号别太当真):\n" + coef.head(10).round(3).to_string(index=False))

# 5. 冻结 -> test
freeze_or_check("logistic_plain_v2", dict(variant="plain", C="inf"))
freeze_or_check("logistic_selected_v2", dict(variant=selected, params=final_params, c_grid=C_GRID,
                                             rule="1-SE on nested AUC (plain>ridge>lasso), weighting only if cross-fitted 5:1 cost -1%"))

plain_fit = lr_pipe(C=np.inf).fit(X_train, ytr)
res_plain = evaluate("Logistic regression (plain baseline, v2)", plain_fit, X_train, y_train, X_test, y_test,
                     member=MEMBER, best_params={"C": "inf"}, cv_auc=float(nested.loc["plain", "nested_auc"]),
                     notes="Frozen baseline; same model as reported in the proposal.", p_oof=oof["plain"],
                     slug="logistic_plain_v2", extra=dict(oof_calibration=calibration_summary(ytr, oof["plain"])))
if selected == "plain":
    res_sel, p_sel_test, p_sel_oof = res_plain, np.load(os.path.join(RESULTS, "logistic_plain_v2_p_test.npy")), oof["plain"]
else:
    res_sel = evaluate(f"Logistic regression ({selected}, v2)", final_pipe, X_train, y_train, X_test, y_test,
                       member=MEMBER, best_params=final_params, cv_auc=float(nested.loc[selected, "nested_auc"]),
                       notes="Selected by nested CV on training only; frozen before the test set was used.",
                       p_oof=oof[selected], slug="logistic_selected_v2",
                       extra=dict(oof_calibration=calibration_summary(ytr, oof[selected])))
    p_sel_test, p_sel_oof = np.load(os.path.join(RESULTS, "logistic_selected_v2_p_test.npy")), oof[selected]
print(f"\n测试集: plain AUC={res_plain['test_auc']:.4f}   selected [{selected}] AUC={res_sel['test_auc']:.4f}")

# 6. 成本比敏感性:阈值 vs 模型
sens = cost_sensitivity(ytr, p_sel_oof, yte, p_sel_test)
tables = {"logistic": sens}
xgb_oof, xgb_te = os.path.join(RESULTS, "xgboost_p_oof.npy"), os.path.join(RESULTS, "xgboost_p_test.npy")
if os.path.exists(xgb_oof) and os.path.exists(xgb_te):       # LU Xinyu 已冻结的 XGBoost 预测,只读取不改
    p_xgb_oof, p_xgb_te = np.load(xgb_oof), np.load(xgb_te)
    tables["XGBoost"] = cost_sensitivity(ytr, p_xgb_oof, yte, p_xgb_te)
    d_auc = delta_metric_ci(yte, p_sel_test, p_xgb_te, roc_auc_score)
    d_ap = delta_metric_ci(yte, p_sel_test, p_xgb_te, average_precision_score)
    print(f"XGBoost - logistic on test: dAUC={d_auc['estimate']:.4f} [{d_auc['lo']:.4f}, {d_auc['hi']:.4f}]  "
          f"dAP={d_ap['estimate']:.4f} [{d_ap['lo']:.4f}, {d_ap['hi']:.4f}]")
    rows = []
    for r in COST_RATIOS:
        t_lr, t_x = best_threshold(ytr, p_sel_oof, r), best_threshold(ytr, p_xgb_oof, r)
        thr_save = paired_bootstrap(yte, lambda i: cost_at(yte[i], p_sel_test[i], 0.5, r) - cost_at(yte[i], p_sel_test[i], t_lr, r), B=1000)
        mod_save = paired_bootstrap(yte, lambda i: cost_at(yte[i], p_sel_test[i], t_lr, r) - cost_at(yte[i], p_xgb_te[i], t_x, r), B=1000)
        rows.append(dict(ratio=r, t_logistic=t_lr, t_xgb=t_x,
                         threshold_saving=thr_save["estimate"], threshold_saving_lo=thr_save["lo"], threshold_saving_hi=thr_save["hi"],
                         model_saving=mod_save["estimate"], model_saving_lo=mod_save["lo"], model_saving_hi=mod_save["hi"]))
    save = pd.DataFrame(rows)
    money = [c for c in save.columns if "saving" in c]
    save[money] = save[money].round(0).astype(int)
    save.to_csv(os.path.join(RESULTS, "threshold_vs_model_saving_v2.csv"), index=False)
    print("\n调阈值(0.5 -> 调过的) 省的钱 vs 换成 XGBoost 省的钱(test, 95% bootstrap CI):")
    print(save.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for col, lab, c in [("threshold_saving", "move logistic cut-off 0.5 -> tuned", "C0"),
                        ("model_saving", "switch logistic -> XGBoost (both tuned)", "C3")]:
        ax.plot(save["ratio"], save[col], marker="o", color=c, label=lab)
        ax.fill_between(save["ratio"], save[col + "_lo"], save[col + "_hi"], color=c, alpha=0.2)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(5, color="grey", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(COST_RATIOS); ax.set_xticklabels(COST_RATIOS)
    ax.set_xlabel("cost ratio C_FN : C_FP"); ax.set_ylabel("cost saved on test (C_FP = 1), 95% CI")
    ax.set_title("Threshold choice vs model choice across cost ratios", fontsize=10); ax.legend(fontsize=8)
    save_extra_figure(fig, "threshold_vs_model_saving_v2.png")

pd.concat({k: v for k, v in tables.items()}, names=["model"]).to_csv(os.path.join(RESULTS, "logistic_cost_sensitivity_v2.csv"))
plot_cost_sensitivity(tables, "logistic_cost_sensitivity_v2.png", "Cost-ratio sensitivity (cut-offs tuned on training OOF only)")
print("\n" + sens.to_string(index=False))
