"""
freeze.py — record every configuration already scored on the test set, and check that nothing has drifted since.
Owner: shared (written 2026-10-19 by LI Jingwen).

Why.  The professor's feedback on the proposal (30 Sep) and the lecture of 7 Oct: the test set may only REPORT the
performance of models whose choices were fixed on training data; using it to compare versions of a model is "kind of
cheating".  Every round-1 model in this project was tuned on training CV and scored on the test set by 18 September
2026 (result files dated 16-18 September), before that feedback; the forest and bagging were scored twice with the same
configuration (16 and 18 Sep), which the ROUND_1 label below, kept identical to the frozen entries, does not say
(see results/TEST_SET_LOG.md).  This script turns that history into an enforceable record:

  results/frozen_manifest.json   one entry per scored configuration: name, member, hyper-parameters and their sha256,
                                 the two thresholds chosen on training out-of-fold probabilities, the CV / OOF / test
                                 AUC as reported, when it was first scored, and the sha256 of its test predictions;
  results/frozen/<slug>_p_test.npy   a copy of those predictions.

common.evaluate() reads the manifest: a frozen slug may be re-run only if it reproduces the frozen predictions; a
changed configuration under a frozen slug is refused before any test label is read (override only via
MH6805_ALLOW_RESCORE="<reason>", which is logged); a new slug is frozen at its first scoring.

Usage (from Project/code):
  python3 freeze.py            create the manifest if it does not exist, otherwise verify it (exit 1 on drift); when the
                               verification is clean it also recomputes the summary block and the reporting-only list
                               (refresh_summary) and, if they changed, rewrites the file atomically with a logged
                               history event — no frozen entry is ever touched by this path, and nothing is written on drift
  python3 freeze.py --check    verify only; never writes
  python3 freeze.py --adopt    add results that were scored ELSEWHERE and copied in with their original file times
                               (e.g. RUAN's v2 results of 30 Sep: copy results/*_v2.* with `cp -p`, then --adopt);
                               the file time is recorded as the scoring time, so copy with timestamps preserved

This script hashes files; it never loads the test labels.  The manifest records the configurations scored through
evaluate() or registered by recalibrate.py; the complete dated list of every test-set read, including exploratory and
descriptive reads that are not configurations, is results/TEST_SET_LOG.md.
"""
import datetime
import glob
import json
import os
import shutil
import sys

import numpy as np

from common import (FROZEN_DIR, FROZEN_MANIFEST, FROZEN_TOL, RESULTS, _write_manifest, config_fingerprint,
                    p_test_fingerprint)

ROUND_1 = ("round 1 — tuned on training CV and scored on the test set once, by 18 September 2026 (result files dated "
           "16–18 September), before the professor's 30 September feedback; not re-scored since")
SPECIAL = {
    "xgboost_monotone": "last configuration of round 1 (18 Sep): constraint signs fixed a priori from credit logic, "
                        "hyper-parameters inherited unchanged from xgboost.json; scored once",
    "imb_leaky_smote_demo": "NEGATIVE CONTROL — the deliberate leakage demonstration (SMOTE before the split); its 'test' "
                            "rows are not the real test set and it is not a model result",
}
TEST_SET_LOG = "results/TEST_SET_LOG.md"     # the complete, dated record of every test-set read (written 21 Oct 2026)
REPORTING_ONLY = [                            # current code (21 Oct 2026); history, including removed reads, in TEST_SET_LOG
    "aggregate.py — paired bootstrap of test-set AUC / AP differences, ROC / PR curves, threshold / cost / calibration "
    "curves on the test rows (the reference model and the top-4 are chosen on CV AUC, not on these; until 19 Oct 2026 "
    "they were ranked by test AUC)",
    "cost_sensitivity.py — test cost of every headline model at its own training-OOF threshold for each cost ratio",
    "threshold_vs_model_saving.py — test cost of the logistic model and XGBoost at their training-OOF cut-offs, 1:1-20:1, "
    "with a paired bootstrap",
    "model_monotone.py — after evaluate(): paired bootstrap of the constrained vs unconstrained XGBoost",
    "model_mlp.py — after evaluate(): paired bootstrap of the MLP against other saved test vectors",
    "model_svm.py — after evaluate(): test AUC of the 3-dp rounded linear-SVM scores and of MAX_DELAY alone",
    "model_discriminant.py — a plain logistic reference refitted and scored on test beside LDA (diagnostic, not an entry)",
    "model_imbalance.py — test agreement / recall statistics between strategies; the negative control re-splits all rows",
    "model_unsupervised.py — test default rate per K = 4 cluster (descriptive)",
    "make_slide_figures.py — ROC / PR and cost curves on the test rows for the slides",
    "eda.py — class balance of the test partition (split check) and the target-distribution figure",
    "model_rf.py — until 21 Oct 2026 also test AUC against the number of trees (removed; N_TREES = 500 was fixed in advance)",
]
TRAINING_ONLY = [
    "every hyper-parameter (GridSearchCV on 5 stratified folds of the training set)",
    "both decision thresholds (training out-of-fold probabilities, common.tune_threshold)",
    "the reference / best model and the top-4 in aggregate.py (CV AUC; OOF AUC for the subsample-tuned RBF SVM)",
    "the imbalance strategy (CV AUC)",
    "the drop-column refits (out-of-bag AUC) and TreeSHAP (training rows) — moved off the test set on 2026-10-19",
    "partial dependence / ICE (training rows)",
]


def scored_results():
    """(json_path, result) for every evaluate() output, including the quarantined leakage demo."""
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json")) + glob.glob(os.path.join(RESULTS, "leakage_demo", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "test_auc" in r and "slug" in r:
            out.append((f, r))
    return out


def p_test_path(json_path, slug):
    return os.path.join(os.path.dirname(json_path), f"{slug}_p_test.npy")


def build():
    os.makedirs(FROZEN_DIR, exist_ok=True)
    models = {}
    for f, r in scored_results():
        s = r["slug"]
        p = np.load(p_test_path(f, s))
        rel = os.path.join("frozen", f"{s}_p_test.npy")
        shutil.copyfile(p_test_path(f, s), os.path.join(RESULTS, rel))
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        models[s] = dict(
            name=r["name"], member=r.get("member", ""), best_params=r.get("best_params") or {},
            config_sha256=config_fingerprint(r["name"], r.get("best_params")),
            thresholds_chosen_on_training_oof=dict(f1=r["at_tuned_f1"]["threshold"], cost=r["at_tuned_cost"]["threshold"]),
            cv_auc=r.get("cv_auc"), oof_auc=r.get("oof_auc"), test_auc=r["test_auc"], test_ap=r["test_ap"],
            first_scored=mtime.strftime("%Y-%m-%d %H:%M"),   # result-file time: scoring happened at or before it
            round=ROUND_1 if mtime.date() <= datetime.date(2026, 9, 18)
            else "scored after round 1", note=SPECIAL.get(s, ""),
            source_json=os.path.relpath(f, RESULTS), frozen_p_test=rel, p_test_sha256=p_test_fingerprint(p),
            role="negative_control" if s.startswith("imb_leaky") else "model")
    man = dict(
        created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), created_by="code/freeze.py", seed=2607526,
        rule=("The test set only reports frozen models. A configuration is frozen when it is first scored on the test "
              "set; common.evaluate() refuses to score a different configuration under a frozen slug."),
        summary={}, selection_uses_training_data_only=TRAINING_ONLY,
        test_set_read_for_reporting_only=REPORTING_ONLY,
        models=dict(sorted(models.items())), history=[])
    refresh_summary(man)
    _write_manifest(man)
    return man


def refresh_summary(man):
    """Recompute the summary block from the entries (it went stale once evaluate() and --adopt started adding entries)
    and set the reporting-only list to the current one. Reads and changes nothing under man["models"]."""
    ms = man["models"]; roles = [m.get("role", "model") for m in ms.values()]
    dates = sorted(m["first_scored"] for m in ms.values())
    by_round = {}
    for m in ms.values():
        key = str(m.get("round", ""))[:60]
        by_round[key] = by_round.get(key, 0) + 1
    man["summary"] = dict(n_frozen=len(ms), n_models=roles.count("model"), n_recalibrated_variants=roles.count("recalibrated_variant"),
                          n_negative_controls=roles.count("negative_control"), first_scored_range=[dates[0], dates[-1]],
                          first_scored_is=("the modification time of the entry's result file: the scoring happened at or "
                                           "before it, so it is not a scoring timestamp (see " + TEST_SET_LOG + ")"),
                          rounds=sorted(by_round), n_by_round=dict(sorted(by_round.items())),
                          all_round_1=all(m.get("round") == ROUND_1 for m in ms.values()),
                          complete_test_set_log=TEST_SET_LOG)
    man["test_set_read_for_reporting_only"] = REPORTING_ONLY
    return man


def write_refreshed_summary(man):
    """Refresh the summary block and the reporting-only list, prove that no frozen entry changed, and write the manifest
    atomically with a history event — only if something actually changed. Returns True when the file was rewritten."""
    models_before = json.dumps(man["models"], sort_keys=True)
    old_summary = json.loads(json.dumps(man.get("summary", {})))
    old_reporting = list(man.get("test_set_read_for_reporting_only", []))
    refresh_summary(man)
    assert json.dumps(man["models"], sort_keys=True) == models_before, "refresh_summary must not touch a frozen entry"
    if man["summary"] == old_summary and man["test_set_read_for_reporting_only"] == old_reporting:
        return False
    man.setdefault("history", []).append(dict(
        event="refresh_summary", at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), previous_summary=old_summary,
        note="summary block and reporting-only list recomputed by freeze.py after a clean check; no frozen entry changed"))
    _write_manifest(man)
    return True


def check(man):
    """Verify every frozen entry against the current files; list scored results that are not frozen."""
    drift, current = [], {r["slug"]: (f, r) for f, r in scored_results()}
    recal_path = os.path.join(RESULTS, "recalibration.json")
    recal = json.load(open(recal_path))["variants"] if os.path.exists(recal_path) else {}
    for s, e in man["models"].items():
        if e.get("role") == "recalibrated_variant":            # scored by recalibrate.py, not by evaluate()
            v = recal.get(s)
            if v is None:
                drift.append(f"{s}: frozen recalibrated variant missing from results/recalibration.json"); continue
            if config_fingerprint(v["name"], v["best_params"]) != e["config_sha256"]:
                drift.append(f"{s}: Platt parameters differ from the frozen ones")
            p, fp = np.load(os.path.join(RESULTS, e["current_p_test"])), np.load(os.path.join(RESULTS, e["frozen_p_test"]))
            if p.shape != fp.shape or float(np.max(np.abs(p - fp))) > FROZEN_TOL:
                drift.append(f"{s}: current recalibrated test probabilities differ from the frozen copy")
            continue
        if s not in current:
            drift.append(f"{s}: frozen but its result file is missing"); continue
        f, r = current[s]
        if config_fingerprint(r["name"], r.get("best_params")) != e["config_sha256"]:
            drift.append(f"{s}: name / hyper-parameters in {os.path.relpath(f, RESULTS)} differ from the frozen configuration")
        p, fp = np.load(p_test_path(f, s)), np.load(os.path.join(RESULTS, e["frozen_p_test"]))
        if p.shape != fp.shape or float(np.max(np.abs(p - fp))) > FROZEN_TOL:
            drift.append(f"{s}: current test predictions differ from the frozen copy")
        if abs(r["test_auc"] - e["test_auc"]) > 1e-12:
            drift.append(f"{s}: reported test AUC {r['test_auc']:.6f} differs from the frozen {e['test_auc']:.6f}")
    unfrozen = sorted(set(current) - set(man["models"]))
    return drift, unfrozen


def adopt(man):
    """Freeze results that are on disk but not yet in the manifest, dated by their result-file time."""
    added = []
    for f, r in scored_results():
        s = r["slug"]
        if s in man["models"]:
            continue
        p = np.load(p_test_path(f, s)); rel = os.path.join("frozen", f"{s}_p_test.npy")
        shutil.copyfile(p_test_path(f, s), os.path.join(RESULTS, rel))
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        man["models"][s] = dict(
            name=r["name"], member=r.get("member", ""), best_params=r.get("best_params") or {},
            config_sha256=config_fingerprint(r["name"], r.get("best_params")),
            thresholds_chosen_on_training_oof=dict(f1=r["at_tuned_f1"]["threshold"], cost=r["at_tuned_cost"]["threshold"]),
            cv_auc=r.get("cv_auc"), oof_auc=r.get("oof_auc"), test_auc=r["test_auc"], test_ap=r["test_ap"],
            first_scored=mtime.strftime("%Y-%m-%d %H:%M"),
            round=f"adopted {datetime.date.today()}: scored elsewhere, result file dated {mtime:%Y-%m-%d %H:%M}",
            note="", source_json=os.path.relpath(f, RESULTS), frozen_p_test=rel, p_test_sha256=p_test_fingerprint(p),
            role="negative_control" if s.startswith("imb_leaky") else "model")
        added.append(s)
    if added:
        man.setdefault("history", []).append(dict(event="adopt", slugs=added,
                                                  at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        _write_manifest(man)
    return added


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    if os.path.exists(FROZEN_MANIFEST):
        man = json.load(open(FROZEN_MANIFEST))
        roles = [m.get("role", "model") for m in man["models"].values()]
        print(f"manifest exists ({len(roles)} frozen: {roles.count('model')} models, "
              f"{roles.count('recalibrated_variant')} recalibrated variants, {roles.count('negative_control')} negative "
              f"control; created {man['created']}); verifying ...")
        if "--adopt" in sys.argv:
            added = adopt(man)
            print(f"adopted {len(added)} result(s): {added}" if added else "nothing to adopt")
    elif check_only:
        sys.exit(f"no manifest at {FROZEN_MANIFEST}; run python3 freeze.py once to create it")
    else:
        man = build()
        sm = man["summary"]
        print(f"wrote {FROZEN_MANIFEST}: {sm['n_frozen']} configurations ({sm['n_models']} models + negative control), "
              f"first scored {sm['first_scored_range'][0]} – {sm['first_scored_range'][1]}, all in round 1: {sm['all_round_1']}")
    drift, unfrozen = check(man)
    if not check_only:
        if drift or unfrozen:
            print("summary NOT refreshed: the manifest does not match the files (see below); nothing written")
        elif write_refreshed_summary(man):
            print(f"summary refreshed: {json.dumps(man['summary']['n_by_round'], ensure_ascii=False)} "
                  f"(history event 'refresh_summary' added; no frozen entry changed)")
        else:
            print("summary already up to date; nothing written")
    for d in drift:
        print("DRIFT:", d)
    for s in unfrozen:
        print(f"NOT FROZEN: {s} has a test-set result but no manifest entry")
    overrides = [h for h in man.get("history", []) if h.get("event") == "override_rescore"]
    adopts = [h for h in man.get("history", []) if h.get("event") == "adopt"]
    if overrides or adopts:
        print(f"history: {len(overrides)} logged override(s), {len(adopts)} adoption(s) "
              f"({sum(len(a.get('slugs', [])) for a in adopts)} adopted result(s))")
    print("OK: every frozen configuration matches its record" if not drift and not unfrozen else "")
    sys.exit(1 if drift or unfrozen else 0)
