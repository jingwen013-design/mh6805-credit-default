# KNN benchmark 第二版
# 按老师意见改:K 和 weights 只用训练集的 nested CV 来选,选完先冻结再跑 test,另外加了校准和成本比 1:1~20:1
from mine_v2_utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict

X_train, X_test, y_train, y_test, feature_names = load_split()
ytr, yte = y_train.values, y_test.values

K_GRID = [5, 11, 21, 31, 51, 75, 101, 151]           # 和第一版一样
W_GRID = ["uniform", "distance"]
INNER = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)   # 内层用不同的切分

pipe = Pipeline([("pre", make_preprocessor()), ("clf", KNeighborsClassifier())])
grid = {"clf__n_neighbors": K_GRID, "clf__weights": W_GRID}

# 1. nested CV (只用训练集)
print("===== KNN nested CV(外层5折,内层调K和weights) =====")
folds = list(CV.split(X_train, ytr))
p_nested = np.zeros(len(ytr)); outer_auc, outer_choice = [], []
for k, (tr, va) in enumerate(folds):
    gs = GridSearchCV(pipe, grid, cv=INNER, scoring=SCORING, n_jobs=2)
    gs.fit(X_train.iloc[tr], ytr[tr])
    p_nested[va] = gs.predict_proba(X_train.iloc[va])[:, 1]
    outer_auc.append(roc_auc_score(ytr[va], p_nested[va])); outer_choice.append(gs.best_params_)
    print(f"外层第{k}折 选了 {gs.best_params_}  AUC = {outer_auc[-1]:.4f}")
print(f"Nested CV AUC = {np.mean(outer_auc):.4f} +/- {np.std(outer_auc):.4f}")

# 2. 在全部训练集上选最终配置
gs_full = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, n_jobs=2)
gs_full.fit(X_train, ytr)
cvres = pd.DataFrame(gs_full.cv_results_)
best = gs_full.best_params_
best_k, best_w = best["clf__n_neighbors"], best["clf__weights"]
print(f"最终选择: K={best_k}, weights={best_w}, CV AUC={gs_full.best_score_:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
for w in W_GRID:
    sub = cvres[cvres["param_clf__weights"] == w]
    ax.errorbar(sub["param_clf__n_neighbors"].astype(int), sub["mean_test_score"], yerr=sub["std_test_score"],
                marker="o", capsize=3, label=w)
ax.set_xlabel("K (neighbours)"); ax.set_ylabel("5-fold CV AUC (training)"); ax.set_title("KNN: CV AUC vs K")
ax.legend(title="weights")
save_extra_figure(fig, "knn_cv_auc_vs_k_v2.png")

# 3. 校准与成本敏感性(都在训练集 OOF 上决定)
final_pipe = gs_full.best_estimator_
p_oof = cross_val_predict(final_pipe, X_train, ytr, cv=CV, method="predict_proba", n_jobs=2)[:, 1]
cal = calibration_summary(ytr, p_oof)
print("OOF 校准情况:", {k: round(v, 4) for k, v in cal.items()})
plot_reliability({f"KNN K={best_k} ({best_w})": (ytr, p_oof)}, "knn_reliability_v2.png",
                 "KNN reliability (training out-of-fold)")

# 4. 冻结 -> 测试集只用一次
freeze_or_check("knn_v2", dict(n_neighbors=best_k, weights=best_w, k_grid=K_GRID, w_grid=W_GRID,
                               threshold_rule="tuned on training OOF", cost_ratios=COST_RATIOS))
res = evaluate(
    "KNN (benchmark, v2)", final_pipe, X_train, y_train, X_test, y_test,
    member=MEMBER, best_params={"n_neighbors": best_k, "weights": best_w},
    cv_auc=float(gs_full.best_score_), cv_auc_std=float(cvres.loc[gs_full.best_index_, "std_test_score"]),
    notes=f"K x weights chosen by 5-fold CV on training only over {K_GRID} x {W_GRID}; nested CV AUC "
          f"{np.mean(outer_auc):.4f}. Configuration frozen before the single test evaluation.",
    p_oof=p_oof, slug="knn_v2",
    extra=dict(nested_cv_auc=float(np.mean(outer_auc)), nested_cv_auc_std=float(np.std(outer_auc)),
               outer_choices=outer_choice, oof_calibration=cal))
p_test = np.load(os.path.join(RESULTS, "knn_v2_p_test.npy"))

sens = cost_sensitivity(ytr, p_oof, yte, p_test)
sens.to_csv(os.path.join(RESULTS, "knn_cost_sensitivity_v2.csv"), index=False)
plot_cost_sensitivity({"KNN": sens}, "knn_cost_sensitivity_v2.png", "KNN: cost-ratio sensitivity (thresholds tuned on training OOF)")

print("\n===== 测试集(只跑这一次) =====")
print(f"Test AUC = {res['test_auc']:.4f}   AP = {res['test_ap']:.4f}   Brier = {res['test_brier']:.4f}")
print(sens.to_string(index=False))
