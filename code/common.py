"""
common.py — shared contract for every model script in this project.

Usage inside a model script (run from the Project/code directory):
    from common import *
    X_train, X_test, y_train, y_test, feature_names = load_split()
    pre = make_preprocessor()                     # ColumnTransformer: scale numeric, one-hot categorical
    pipe = Pipeline([("pre", pre), ("clf", SomeClassifier(...))])
    ... tune with cv=CV, scoring=SCORING ...
    res = evaluate("Logistic regression", pipe_fitted, X_train, y_train, X_test, y_test,
                   member="LI Jingwen", best_params={...}, cv_auc=0.77, notes="...")
Everything (JSON with metrics + curves, figures) is written under Project/results and Project/figures.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
                             balanced_accuracy_score, brier_score_loss, log_loss)
warnings.filterwarnings("ignore")

MATRIC = 2607526                      # seed everywhere (matriculation number G2607526K)
SEED = MATRIC
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "default_of_credit_card_clients.csv")
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
for d in (PROC, RESULTS, FIGURES):
    os.makedirs(d, exist_ok=True)

TARGET = "DEFAULT"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
ENG_COLS = ["UTIL_1", "UTIL_MEAN", "PAY_RATIO_MEAN", "N_MONTHS_DELAYED", "MAX_DELAY", "BILL_TREND"]
NUM_COLS = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS + ENG_COLS
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "roc_auc"                   # primary CV criterion (threshold-free); AP is reported as well
COST_FN, COST_FP = 5.0, 1.0           # business cost assumption: missing a defaulter costs 5x a false alarm

# ----------------------------------------------------------------------------- data
def load_raw():
    df = pd.read_csv(RAW)
    return df.drop(columns=["ID"])

def clean(df):
    """Recode undocumented category levels, drop exact duplicates, add interpretable engineered features.

    WHY CLEANING BEFORE THE SPLIT IS NOT LEAKAGE
    --------------------------------------------
    Leakage means information from the held-out rows influences how the training rows are
    represented. Every operation below is one of three kinds, and the first two cannot leak
    by construction:

    (a) Fixed-codebook recoding — EDUCATION {0,5,6} -> 4 and MARRIAGE {0} -> 3. The mapping
        comes from the UCI data dictionary, not from the data: it is the same mapping whatever
        rows happen to be present, and it would be identical if we had only the training half.
        No statistic is estimated, so nothing about the test rows can reach the training rows.

    (b) Row-wise algebra — all six engineered features (UTIL_1, UTIL_MEAN, PAY_RATIO_MEAN,
        N_MONTHS_DELAYED, MAX_DELAY, BILL_TREND). Each is a function of one client's own
        columns in that one row; no mean, variance, quantile or category frequency is taken
        over the sample. Row i's value is unchanged if every other row is deleted.

    (c) Exact-duplicate removal — this is the honest exception: it is the ONLY pre-split step
        that compares rows with each other, so it does look across the whole file. The defence
        is threefold. It removes 35 byte-identical rows out of 30,000 (0.12%); it happens before
        any model, split or statistic exists, so no fitted quantity is involved; and it cannot
        transfer a label from a test row to a training row, because de-duplication keeps one
        surviving row carrying its own original label rather than copying or averaging labels
        across rows. The alternative — leaving the duplicates in — is the worse error, since it
        lets the SAME client land in both train and test and inflates the test score.

    Everything that estimates a statistic FROM the data is deliberately NOT here. Standardisation
    (StandardScaler means/SDs), the one-hot category set (OneHotEncoder categories_), resampling
    (SMOTE / class weights), PCA loadings, K-means centroids and the decision threshold all live
    inside the sklearn Pipeline and are therefore re-fitted on the training part of every CV fold.
    See make_preprocessor() and tune_threshold().
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})      # 1 grad school, 2 university, 3 high school, 4 others/unknown
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                    # 1 married, 2 single, 3 others
    n0 = len(df); df = df.drop_duplicates(); n_dup = n0 - len(df)
    lim = df["LIMIT_BAL"].astype(float)
    df["UTIL_1"] = df["BILL_AMT1"] / lim                                # credit utilisation, latest month
    df["UTIL_MEAN"] = df[BILL_COLS].mean(axis=1) / lim
    df["PAY_RATIO_MEAN"] = np.clip(np.nanmean(np.where(df[BILL_COLS].values > 0, df[PAYAMT_COLS].values / np.maximum(df[BILL_COLS].values, 1.0), np.nan), axis=1), 0, 5)
    # Clients whose six bills are ALL non-positive have no month with a positive balance, so the
    # ratio is undefined for every month and the row-mean above is NaN. We impute 0.0 (nothing owed,
    # nothing repaid). This is a self-inflicted missing value and must be reported as one.
    n_undef = int((df[BILL_COLS].values <= 0).all(axis=1).sum())
    df["PAY_RATIO_MEAN"] = df["PAY_RATIO_MEAN"].fillna(0.0)
    df["N_MONTHS_DELAYED"] = (df[PAY_COLS] > 0).sum(axis=1)             # months with payment delay ≥ 1
    df["MAX_DELAY"] = df[PAY_COLS].max(axis=1).clip(lower=0)
    df["BILL_TREND"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / lim        # rising balance relative to limit
    df.attrs["n_duplicates_dropped"] = n_dup
    df.attrs["n_all_bills_nonpositive"] = n_undef
    df.attrs["share_all_bills_nonpositive"] = float(n_undef / len(df))
    return df

def make_split(test_size=0.30):
    """Stratified 70/30 split with the matric seed; written once to data/processed so every member uses the same rows."""
    df = clean(load_raw())
    X = df.drop(columns=[TARGET]); y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    X_tr.assign(DEFAULT=y_tr).to_csv(os.path.join(PROC, "train.csv"), index=False)
    X_te.assign(DEFAULT=y_te).to_csv(os.path.join(PROC, "test.csv"), index=False)
    meta = dict(n_raw=30000, n_after_dedup=len(df), n_duplicates_dropped=int(df.attrs["n_duplicates_dropped"]),
                n_all_bills_nonpositive=int(df.attrs["n_all_bills_nonpositive"]),
                share_all_bills_nonpositive=float(df.attrs["share_all_bills_nonpositive"]),
                n_train=len(X_tr), n_test=len(X_te), default_rate_train=float(y_tr.mean()), default_rate_test=float(y_te.mean()),
                test_size=test_size, seed=SEED, features=list(X.columns), cat_cols=CAT_COLS, num_cols=NUM_COLS, engineered=ENG_COLS)
    json.dump(meta, open(os.path.join(PROC, "split_meta.json"), "w"), indent=1)
    return meta

def load_split():
    """Return X_train, X_test, y_train, y_test, feature_names (creates the split files if absent)."""
    if not os.path.exists(os.path.join(PROC, "train.csv")):
        make_split()
    tr = pd.read_csv(os.path.join(PROC, "train.csv")); te = pd.read_csv(os.path.join(PROC, "test.csv"))
    X_tr, y_tr = tr.drop(columns=[TARGET]), tr[TARGET].astype(int)
    X_te, y_te = te.drop(columns=[TARGET]), te[TARGET].astype(int)
    return X_tr, X_te, y_tr, y_te, list(X_tr.columns)

def make_preprocessor(scale=True, onehot=True):
    """ColumnTransformer fitted INSIDE the pipeline (so CV folds never see validation statistics)."""
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(drop="first", handle_unknown="ignore") if onehot else "passthrough"
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)], remainder="drop")

def feature_names_out(pre):
    """Names of the transformed columns (call after the pipeline is fitted)."""
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return None

# ----------------------------------------------------------------------------- evaluation
def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))

def threshold_table(y, p, grid=None):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        yhat = (p >= t).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
        rows.append(dict(threshold=float(t), precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat),
                         f1=f1_score(y, yhat), fpr=fp / max(tn + fp, 1), accuracy=accuracy_score(y, yhat),
                         cost=float(COST_FN * fn + COST_FP * fp)))
    return pd.DataFrame(rows)

def tune_threshold(y_oof, p_oof, criterion="f1"):
    """Pick a decision threshold on TRAINING out-of-fold probabilities only (never on the test set)."""
    tab = threshold_table(y_oof, p_oof, grid=np.round(np.arange(0.05, 0.96, 0.01), 2))
    row = tab.loc[tab["cost"].idxmin()] if criterion == "cost" else tab.loc[tab["f1"].idxmax()]
    return float(row["threshold"]), tab

def metrics_at(y, p, t):
    yhat = (p >= t).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1]); tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return dict(threshold=float(t), accuracy=float(accuracy_score(y, yhat)), balanced_accuracy=float(balanced_accuracy_score(y, yhat)),
                precision_1=float(precision_score(y, yhat, zero_division=0)), recall_1=float(recall_score(y, yhat)),
                f1_1=float(f1_score(y, yhat)), precision_0=float(precision_score(y, yhat, pos_label=0, zero_division=0)),
                recall_0=float(recall_score(y, yhat, pos_label=0)), f1_0=float(f1_score(y, yhat, pos_label=0)),
                fpr=float(fp / max(tn + fp, 1)), fnr=float(fn / max(tp + fn, 1)), tn=tn, fp=fp, fn=fn, tp=tp,
                cost=float(COST_FN * fn + COST_FP * fp))

# ---------------------------------------------------------------- freeze guard (added 2026-10-19)
# Every configuration scored on the test set is recorded in results/frozen_manifest.json (written by freeze.py), with a
# copy of its test predictions under results/frozen/. evaluate() checks a run against that record BEFORE any test label
# is used: re-running a frozen configuration is allowed only if it reproduces the frozen predictions; a changed
# configuration under a frozen slug is refused, so the test set can no longer be used to compare versions of a model.
# A slug that is not yet frozen is frozen at its first scoring. MH6805_ALLOW_RESCORE="<reason>" overrides a refusal and
# the override is written into the manifest's history, so it cannot happen silently.
FROZEN_MANIFEST = os.path.join(RESULTS, "frozen_manifest.json")
FROZEN_DIR = os.path.join(RESULTS, "frozen")
FROZEN_TOL = 1e-6          # max |p_test - frozen p_test| still counted as a reproduction (thread-order float noise)


def config_fingerprint(name, best_params):
    """sha256 of the model name and its hyper-parameters, normalised exactly as evaluate() writes them to JSON."""
    import hashlib
    canon = json.dumps({"name": name, "best_params": json.loads(json.dumps(best_params or {}))}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def p_test_fingerprint(p):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(p, float), 6)).tobytes()).hexdigest()


def _load_manifest():
    return json.load(open(FROZEN_MANIFEST)) if os.path.exists(FROZEN_MANIFEST) else None


def _write_manifest(man):
    tmp = FROZEN_MANIFEST + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1); os.replace(tmp, FROZEN_MANIFEST)


def freeze_check(slug, name, best_params, p_test):
    """Called by evaluate() after the test PREDICTIONS exist and before any test LABEL is read.
    Returns 'no manifest', 'new', 'reproduced' or 'override'; raises SystemExit on a refused re-score."""
    man = _load_manifest()
    if man is None:
        return "no manifest"
    entry = man["models"].get(slug)
    if entry is None:
        return "new"
    problems = []
    if entry["config_sha256"] != config_fingerprint(name, best_params):
        problems.append("its name / hyper-parameters differ from the frozen configuration")
    frozen_p = np.load(os.path.join(RESULTS, entry["frozen_p_test"]))
    if frozen_p.shape != np.shape(p_test):
        problems.append(f"its test predictions have shape {np.shape(p_test)}, the frozen ones {frozen_p.shape}")
    elif float(np.max(np.abs(frozen_p - p_test))) > FROZEN_TOL:
        problems.append(f"its test predictions differ from the frozen ones by up to {float(np.max(np.abs(frozen_p - p_test))):.2e}")
    if not problems:
        return "reproduced"
    reason = os.environ.get("MH6805_ALLOW_RESCORE", "").strip()
    msg = (f"'{slug}' is frozen (first scored {entry.get('first_scored', '?')}) and this run would score a DIFFERENT model on the "
           f"test set: " + "; ".join(problems) + ". The test set only reports frozen models. Restore the frozen configuration, "
           "or score it under a new slug, or set MH6805_ALLOW_RESCORE='<reason>' to override (the override is logged).")
    if not reason:
        raise SystemExit(msg)
    man.setdefault("history", []).append(dict(event="override_rescore", slug=slug, reason=reason, problems=problems,
                                              at=time.strftime("%Y-%m-%d %H:%M:%S")))
    _write_manifest(man)
    print(f"WARNING: {msg}\n  -> overridden: {reason} (logged in {FROZEN_MANIFEST})")
    return "override"


def freeze_record(res, p_test, status, source_json, role="model", round_note=None, current_p_test=None):
    """Freeze a configuration at its first scoring (status 'new') or re-freeze it after a logged override.
    role / round_note / current_p_test let scripts other than evaluate() (e.g. recalibrate.py) register what they score."""
    if status not in ("new", "override"):
        return
    man = _load_manifest()
    os.makedirs(FROZEN_DIR, exist_ok=True)
    rel = os.path.join("frozen", f"{res['slug']}_p_test.npy")
    np.save(os.path.join(RESULTS, rel), p_test)
    man["models"][res["slug"]] = dict(
        name=res["name"], member=res.get("member", ""), best_params=res.get("best_params") or {},
        config_sha256=config_fingerprint(res["name"], res.get("best_params")),
        thresholds_chosen_on_training_oof=dict(f1=res["at_tuned_f1"]["threshold"], cost=res["at_tuned_cost"]["threshold"]),
        cv_auc=res.get("cv_auc"), oof_auc=res.get("oof_auc"), test_auc=res["test_auc"], test_ap=res["test_ap"],
        first_scored=time.strftime("%Y-%m-%d %H:%M"),
        round=(round_note or "after the freeze (frozen at first scoring by evaluate())") if status == "new"
        else "re-frozen after a logged override", source_json=source_json, frozen_p_test=rel,
        p_test_sha256=p_test_fingerprint(p_test), role=role)
    if current_p_test:
        man["models"][res["slug"]]["current_p_test"] = current_p_test
    _write_manifest(man)


def evaluate(name, model, X_train, y_train, X_test, y_test, member="", best_params=None, cv_auc=None, cv_auc_std=None,
             notes="", fit_seconds=None, p_oof=None, extra=None, slug=None):
    """
    Standard evaluation of a FITTED model. Computes threshold-free metrics (AUC, AP) and metrics at
    threshold 0.5 and at a threshold tuned on training out-of-fold probabilities (F1 and cost criteria).
    p_oof: training out-of-fold probabilities (from cross_val_predict on the tuned pipeline); if None they are computed here.
    Writes results/<slug>.json and figures/<slug>_eval.png. Returns the result dict.
    """
    slug = slug or name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("+", "plus")
    t0 = time.time()
    p_test = _proba(model, X_test); p_train = _proba(model, X_train)
    freeze_status = freeze_check(slug, name, best_params, p_test)      # before any test label is read
    if p_oof is None:
        from sklearn.base import clone
        p_oof = cross_val_predict(clone(model), X_train, y_train, cv=CV, method="predict_proba" if hasattr(model, "predict_proba") else "decision_function")
        if p_oof.ndim == 2: p_oof = p_oof[:, 1]
        elif not hasattr(model, "predict_proba"): p_oof = 1.0 / (1.0 + np.exp(-p_oof))
    t_f1, tab_f1 = tune_threshold(y_train.values, p_oof, "f1")
    t_cost, _ = tune_threshold(y_train.values, p_oof, "cost")
    fpr, tpr, _ = roc_curve(y_test, p_test); prec, rec, _ = precision_recall_curve(y_test, p_test)
    res = dict(name=name, slug=slug, member=member, best_params=best_params or {}, notes=notes,
               cv_auc=cv_auc, cv_auc_std=cv_auc_std, fit_seconds=fit_seconds, eval_seconds=None,
               train_auc=float(roc_auc_score(y_train, p_train)), oof_auc=float(roc_auc_score(y_train, p_oof)),
               test_auc=float(roc_auc_score(y_test, p_test)), test_ap=float(average_precision_score(y_test, p_test)),
               test_brier=float(brier_score_loss(y_test, p_test)), test_logloss=float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6))),
               at_0_5=metrics_at(y_test.values, p_test, 0.5), at_tuned_f1=metrics_at(y_test.values, p_test, t_f1),
               at_tuned_cost=metrics_at(y_test.values, p_test, t_cost),
               threshold_table_test=threshold_table(y_test.values, p_test).round(4).to_dict(orient="list"),
               roc=dict(fpr=np.round(fpr, 4).tolist(), tpr=np.round(tpr, 4).tolist()),
               pr=dict(precision=np.round(prec, 4).tolist(), recall=np.round(rec, 4).tolist()),
               n_train=int(len(y_train)), n_test=int(len(y_test)), extra=extra or {})
    res["eval_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(RESULTS, f"{slug}.json"), "w"), indent=1)
    np.save(os.path.join(RESULTS, f"{slug}_p_test.npy"), p_test); np.save(os.path.join(RESULTS, f"{slug}_p_oof.npy"), p_oof)
    plot_eval(res, y_test.values, p_test, os.path.join(FIGURES, f"{slug}_eval.png"))
    freeze_record(res, p_test, freeze_status, f"{slug}.json")
    return res

def plot_eval(res, y_test, p_test, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    m = res["at_tuned_f1"]; cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[0].imshow(cm, cmap="Blues"); ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1]); ax[0].set_xticklabels(["Pred 0", "Pred 1"]); ax[0].set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2): ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax[0].set_title(f"Confusion matrix (test, threshold {m['threshold']:.2f})")
    ax[1].plot(res["roc"]["fpr"], res["roc"]["tpr"], color="C3", lw=2, label=f"AUC = {res['test_auc']:.3f}"); ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].set_title("ROC (test)"); ax[1].legend(loc="lower right")
    ax[2].plot(res["pr"]["recall"], res["pr"]["precision"], color="C0", lw=2, label=f"AP = {res['test_ap']:.3f}"); ax[2].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"prevalence = {y_test.mean():.3f}")
    ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].set_title("Precision–recall (test)"); ax[2].legend(loc="upper right")
    fig.suptitle(res["name"], fontsize=13); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def save_extra_figure(fig, filename):
    path = os.path.join(FIGURES, filename); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); return path

def load_all_results():
    out = []
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith(".json") and f not in ("split_meta.json", "frozen_manifest.json"):
            out.append(json.load(open(os.path.join(RESULTS, f))))
    return out

if __name__ == "__main__":
    meta = make_split(); print(json.dumps({k: v for k, v in meta.items() if k not in ("features", "num_cols")}, indent=1))
    print(f"\nPAY_RATIO_MEAN undefined (all six bills non-positive) and imputed 0.0: "
          f"{meta['n_all_bills_nonpositive']:,} of {meta['n_after_dedup']:,} clients "
          f"= {100 * meta['share_all_bills_nonpositive']:.2f}%")
