"""Experiments for the four sub-questions. Every function writes its tables to the results folder."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from . import labelling, models, moments
from .config import Paths
from .evaluation import (
    bootstrap_ci,
    calibration,
    driver_folds,
    metrics,
    paired_permutation,
    per_driver,
    type_accuracy,
)
from .features import FEATURE_GROUPS, MODEL_FEATURES, REFERENCES, TYPE_GROUP, build_windows, load_windows, raw_windows
from .preprocess import load_prepared

SCORES = {"xgb": "score_xgb", "seq": "score_seq", "labelling_functions": "score_lf",
          "time_loss": "score_time_loss", "random": "score_random"}
KEEP = ["window_id", "driver", "level", "circuit", "session_id", "segment", "segment_name", "lap", "is_corner",
        "time_loss", "exit_loss", "total_loss"]


def load_expert_labels(cfg: dict) -> pd.DataFrame | None:
    path = Paths.from_config(cfg).review / "expert_labels.parquet"
    return pd.read_parquet(path) if path.exists() else None


def add_target(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Evaluation target: the expert judgement where it exists, otherwise the weak label.

    Results against weak labels are provisional. They show whether the pipeline works, not how
    good the detection is.
    """
    expert = load_expert_labels(cfg)
    frame = frame.copy()
    if expert is None or expert.empty:
        frame["target"] = frame["weak_label"].astype(float)
        frame["target_type"] = frame["weak_type"]
        return frame, "weak labels (provisional)"
    frame = frame.merge(expert[["window_id", "expert_label", "expert_type"]], on="window_id", how="left")
    frame["target"] = frame["expert_label"]
    frame["target_type"] = frame["expert_type"].fillna("none")
    return frame, "expert-validated labels"


def cross_validate(windows: pd.DataFrame, cfg: dict, data: np.ndarray | None = None,
                   which=("xgb",), seed: int = 0, train_filter=None, label_override=None,
                   scheme: str | None = None, exclude=(), features: list[str] | None = None,
                   explain: bool = False) -> pd.DataFrame:
    """Train on one part of the data, score the other, and return the scores of every window once.

    train_filter    function(train_mask, rng) -> mask, used to train on part of the data
    label_override  function(labels, train_mask) -> labels, used to replace weak labels
    scheme          how the data are split (see evaluation.driver_folds)
    exclude         labelling functions that are switched off
    features        features given to the tree model (default: all)
    explain         also return the contribution of every feature group to the score
    """
    rng = np.random.default_rng(seed)
    scheme = scheme or cfg["evaluation"]["scheme"]
    # Thresholds are per circuit. For a circuit that is held out they can only come from that circuit.
    fit_on_train = cfg["labelling"]["fit_on"] == "train" and scheme != "loco"
    labels_all = labelling.label(windows, cfg, exclude=exclude)
    soft = cfg["labelling"].get("aggregation", "any") == "label_model"
    seq_x = raw_windows(windows, data, cfg["model"]["seq"]["length"]) if "seq" in which else None
    parts = []
    for fold, train, test in driver_folds(windows, cfg, scheme):
        labels = labelling.label(windows, cfg, fit_mask=train, exclude=exclude) if fit_on_train else labels_all
        if label_override is not None:
            labels = label_override(labels.copy(), train)
        fit = train_filter(train, rng) if train_filter is not None else train
        y = labels["weak_label"].to_numpy()
        prob = labels["weak_prob"].to_numpy() if soft else None
        out = windows.loc[test, [c for c in KEEP if c in windows]].copy()
        out["fold"] = fold
        out["weak_label"], out["weak_prob"] = y[test], labels["weak_prob"].to_numpy()[test]
        out["weak_type"] = labels["weak_type"].to_numpy()[test]
        if "xgb" in which:
            detector = models.fit_detector(windows[fit], y[fit], cfg, seed, features, None if prob is None else prob[fit])
            out["score_xgb"] = models.predict(detector, windows[test])
            positive = fit & (y == 1)
            typer = models.TypeModel(cfg, seed).fit(windows[positive], labels["weak_type"][positive])
            out["type_xgb"] = typer.predict(windows[test])
            if explain:
                contrib = moments.group_contributions(models.contributions(detector, windows[test]), FEATURE_GROUPS)
                for group in contrib:
                    out[f"shap_{group}"] = contrib[group].to_numpy()
        if "seq" in which:
            from .seqmodel import SequenceDetector
            target = prob[fit] if soft else y[fit]
            out["score_seq"] = SequenceDetector(cfg, seed).fit(seq_x[fit], target).predict_proba(seq_x[test])
        out["score_lf"] = models.score_labelling_functions(labels)[test]
        out["score_time_loss"] = models.score_time_loss(windows[test])
        out["score_random"] = models.score_random(int(test.sum()), seed)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ SQ1

def sq1(cfg: dict) -> dict:
    """Which types of key moments can be observed: behaviour of the labelling functions."""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    labels = labelling.label(windows, cfg)
    stats = labelling.statistics(labels, cfg)
    # Reliability of every function as estimated without true labels.
    soft_cfg = copy.deepcopy(cfg)
    soft_cfg["labelling"]["aggregation"] = "label_model"
    soft, model = labelling.label(windows, soft_cfg, return_model=True)
    stats = stats.merge(model.table(), on="function")
    stats.to_csv(paths.results / "sq1_labelling_functions.csv", index=False)
    labelling.fit_thresholds(windows, cfg).to_csv(paths.results / "sq1_thresholds.csv", index=False)
    joined = pd.concat([windows[["level", "circuit", "segment"]], labels], axis=1)
    by_level = pd.crosstab(joined["level"], joined["weak_type"])
    by_level.to_csv(paths.results / "sq1_types_by_level.csv")
    pd.crosstab([joined["circuit"], joined["segment"]], joined["weak_type"]).to_csv(
        paths.results / "sq1_types_by_segment.csv")
    fired = (labels[labelling.TYPES] == 1).astype(int)
    co = fired.T.dot(fired)
    co.to_csv(paths.results / "sq1_co_firing.csv")
    labels.assign(window_id=windows["window_id"]).to_parquet(paths.prepared / "weak_labels.parquet", index=False)
    return {"statistics": stats, "by_level": by_level, "positive_rate": float(labels["weak_label"].mean()),
            "positive_rate_label_model": float(soft["weak_label"].mean()), "co_firing": co}


# ------------------------------------------------------------------ SQ2

def sq2(cfg: dict, with_seq: bool = True) -> dict:
    """Which modelling approach detects key moments best for drivers it has not seen."""
    paths = Paths.from_config(cfg).ensure()
    ev = cfg["evaluation"]
    windows = load_windows(cfg)
    _, data, _ = load_prepared(cfg)
    which = ("xgb", "seq") if with_seq else ("xgb",)
    pred = cross_validate(windows, cfg, data, which=which, seed=ev["seeds"][0], explain=True)
    pred.to_parquet(paths.prepared / "predictions.parquet", index=False)
    frame, source = add_target(pred, cfg)
    budgets, n_boot, spread = ev["budgets"], ev["bootstrap"], ev.get("max_per_segment")
    present = {name: col for name, col in SCORES.items() if col in frame}
    rows = []
    for name, column in present.items():
        # The scores of the baselines are not probabilities, so no threshold metric is given for them.
        threshold = 0.5 if name in ("xgb", "seq") else (1.0 if name == "labelling_functions" else np.inf)
        ci = bootstrap_ci(frame, column, "target", budgets, n_boot, threshold=threshold, max_per_segment=spread)
        for metric, r in ci.iterrows():
            if threshold == np.inf and metric in ("precision", "recall", "f1"):
                continue
            rows.append(dict(model=name, metric=metric, **r.to_dict()))
    table = pd.DataFrame(rows)
    table.to_csv(paths.results / "sq2_metrics.csv", index=False)
    per_level = pd.DataFrame([dict(level=lv, model=name, **metrics(g, col, "target", budgets, max_per_segment=spread))
                              for lv, g in frame.groupby("level") for name, col in present.items()])
    per_level.to_csv(paths.results / "sq2_metrics_by_level.csv", index=False)

    # Per driver, and paired tests between the tree model and every alternative.
    drivers = per_driver(frame, present, "target")
    drivers.to_csv(paths.results / "sq2_per_driver.csv", index=False)
    tests = pd.DataFrame([dict(comparison=f"xgb vs {other}", **paired_permutation(drivers["xgb"], drivers[other]))
                          for other in present if other != "xgb"])
    tests.to_csv(paths.results / "sq2_paired_tests.csv", index=False)

    # Calibration of the two models whose scores are probabilities.
    calib = {}
    for name in ("xgb", "seq"):
        if name in present:
            reliability, summary = calibration(frame, present[name], "target")
            reliability.to_csv(paths.results / f"sq2_calibration_{name}.csv", index=False)
            calib[name] = summary

    types = type_accuracy(frame, "type_xgb", "target_type")
    positive = frame["target_type"] != "none"
    pd.crosstab(frame.loc[positive, "target_type"], frame.loc[positive, "type_xgb"]).to_csv(
        paths.results / "sq2_type_confusion.csv")

    # What the tree model looks at.
    labels = labelling.label(windows, cfg)
    detector = models.fit_detector(windows, labels["weak_label"].to_numpy(), cfg)
    importance = pd.Series(detector.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    importance.rename("gain").to_csv(paths.results / "sq2_feature_importance.csv")
    shap_cols = [c for c in pred.columns if c.startswith("shap_")]
    flagged = pred[pred["score_xgb"] >= cfg["moments"]["threshold"]]
    group_share = flagged[shap_cols].abs().mean().rename(lambda c: c[5:]).sort_values(ascending=False)
    group_share.rename("mean_abs_contribution").to_csv(paths.results / "sq2_explanation_groups.csv")
    strongest = flagged[[c for c in shap_cols if c != "shap_time"]].idxmax(axis=1).str[5:]
    expected = flagged["type_xgb"].map(TYPE_GROUP)
    consistency = float((strongest == expected).mean()) if len(flagged) else float("nan")

    # Moments at session level (requirement R1).
    scored = windows.merge(pred[["window_id", "score_xgb", "type_xgb"]], on="window_id")
    scored.loc[scored["is_reference"], "score_xgb"] = 0.0      # a reference lap cannot be a moment
    session = moments.session_moments(scored, cfg, "score_xgb", "type_xgb")
    session.drop(columns=[c for c in ("laps",) if c in session]).to_parquet(paths.prepared / "moments.parquet", index=False)
    pattern_share = session["pattern"].value_counts(normalize=True).to_dict() if len(session) else {}

    summary = {"evaluated_against": source, "scheme": ev["scheme"], "type_accuracy": types,
               "drivers": int(windows["driver"].nunique()), "windows": int(len(windows)), "calibration": calib,
               "explanation_matches_type": consistency, "session_moments": int(len(session)),
               "pattern_share": pattern_share, "max_per_segment": spread, "reference": cfg["features"]["reference"],
               "aggregation": cfg["labelling"]["aggregation"]}
    (paths.results / "sq2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"metrics": table, "summary": summary, "per_level": per_level, "per_driver": drivers, "tests": tests}


# ------------------------------------------------------------------ SQ3

def _requirement(curve: pd.DataFrame, size_col: str, ci_low: float):
    """Smallest size whose mean PR-AUC lies inside the confidence interval obtained with all data."""
    ok = curve[curve["pr_auc_mean"] >= ci_low]
    return None if ok.empty else ok[size_col].iloc[0]


def sq3(cfg: dict) -> dict:
    """What the data requirements are: training size, laps per driver and label quality."""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    _, data, _ = load_prepared(cfg)
    ev = cfg["evaluation"]
    which = tuple(ev["sq3_models"])
    column = SCORES[which[0]]
    lap_ids = windows["lap_uid"].to_numpy()

    def pr_auc(pred):
        frame, _ = add_target(pred, cfg)
        return metrics(frame, column, "target", [])["pr_auc"]

    # 1. Share of the training laps.
    rows = []
    for fraction in ev["train_fractions"]:
        for seed in ev["seeds"]:
            def keep(train, rng, fraction=fraction):
                laps = np.unique(lap_ids[train])
                chosen = rng.choice(laps, max(1, int(round(fraction * len(laps)))), replace=False)
                return train & np.isin(lap_ids, chosen)
            pred = cross_validate(windows, cfg, data, which, seed, train_filter=keep)
            rows.append(dict(fraction=fraction, seed=seed, pr_auc=pr_auc(pred)))
    size = pd.DataFrame(rows)
    size_curve = size.groupby("fraction")["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std").reset_index()

    # 2. Laps per driver per circuit, taken in the order in which they were driven.
    order = windows.groupby(["driver", "circuit"])["lap_uid"].transform(
        lambda x: pd.factorize(x, sort=True)[0]).to_numpy()
    rows = []
    for n_laps in ev["laps_per_driver"]:
        for seed in ev["seeds"]:
            def keep(train, rng, n_laps=n_laps):
                return train if n_laps is None else train & (order < n_laps)
            pred = cross_validate(windows, cfg, data, which, seed, train_filter=keep)
            rows.append(dict(laps="all" if n_laps is None else n_laps, seed=seed, pr_auc=pr_auc(pred)))
    laps = pd.DataFrame(rows)
    laps_curve = laps.groupby("laps", sort=False)["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std").reset_index()

    # 3. Number of training drivers: every fold trains on a random subset of the other drivers.
    drivers = windows["driver"].to_numpy()
    n_train = windows["driver"].nunique() - 1
    rows = []
    for k in range(1, n_train + 1):
        for seed in ev["seeds"]:
            def keep(train, rng, k=k):
                chosen = rng.choice(np.unique(drivers[train]), min(k, len(np.unique(drivers[train]))), replace=False)
                return train & np.isin(drivers, chosen)
            pred = cross_validate(windows, cfg, data, which, seed, train_filter=keep, scheme="lodo")
            rows.append(dict(training_drivers=k, seed=seed, pr_auc=pr_auc(pred)))
    n_drivers = pd.DataFrame(rows)
    drivers_curve = n_drivers.groupby("training_drivers")["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std").reset_index()

    # Confidence interval with all data, from resampling drivers.
    full, source = add_target(cross_validate(windows, cfg, data, which, ev["seeds"][0]), cfg)
    ci = bootstrap_ci(full, column, "target", [], ev["bootstrap"]).loc["pr_auc"]
    requirement = {"evaluated_against": source, "model": which[0],
                   "pr_auc_all_data": float(ci["value"]), "ci_low": float(ci["ci_low"]), "ci_high": float(ci["ci_high"]),
                   "training_fraction": _requirement(size_curve, "fraction", ci["ci_low"]),
                   "laps_per_driver": _requirement(laps_curve, "laps", ci["ci_low"]),
                   "training_drivers": _requirement(drivers_curve, "training_drivers", ci["ci_low"])}

    size.to_csv(paths.results / "sq3_training_size_runs.csv", index=False)
    size_curve.to_csv(paths.results / "sq3_training_size.csv", index=False)
    laps.to_csv(paths.results / "sq3_laps_per_driver_runs.csv", index=False)
    laps_curve.to_csv(paths.results / "sq3_laps_per_driver.csv", index=False)
    drivers_curve.to_csv(paths.results / "sq3_training_drivers.csv", index=False)
    out = {"training_size": size_curve, "laps_per_driver": laps_curve, "training_drivers": drivers_curve,
           "requirement": requirement}

    # 4. Label quality. This part needs expert labels and is skipped until they exist.
    expert = load_expert_labels(cfg)
    if expert is not None and len(expert):
        out["label_quality"] = _label_quality(windows, data, expert, cfg, which, column)
        out["label_quality"].to_csv(paths.results / "sq3_label_quality.csv", index=False)
    (paths.results / "sq3_requirement.json").write_text(json.dumps(requirement, indent=2, default=str), encoding="utf-8")
    return out


def _label_quality(windows, data, expert, cfg, which, column) -> pd.DataFrame:
    """Weak labels against weak labels corrected with expert judgements.

    A growing number of expert judgements replaces the weak labels of training windows. The
    windows are picked either at random or where the weak-label model is least certain.
    """
    ev = cfg["evaluation"]
    ids = windows["window_id"].to_numpy()
    judged = dict(zip(expert["window_id"], expert["expert_label"]))
    judged_type = dict(zip(expert["window_id"], expert["expert_type"]))
    has = np.isin(ids, list(judged))
    base = cross_validate(windows, cfg, data, which, ev["seeds"][0])
    uncertainty = dict(zip(base["window_id"], -np.abs(base[column] - 0.5)))
    rows = []
    for strategy in ("random", "uncertainty"):
        for n in ev["review_steps"]:
            for seed in ev["seeds"]:
                def override(labels, train, n=n, seed=seed, strategy=strategy):
                    pool = np.flatnonzero(train & has)
                    if strategy == "random":
                        pick = np.random.default_rng(seed).permutation(pool)[:n]
                    else:
                        pick = pool[np.argsort([-uncertainty.get(ids[i], -1.0) for i in pool])][:n]
                    values, kinds = labels["weak_label"].to_numpy().copy(), labels["weak_type"].to_numpy().copy()
                    probs = labels["weak_prob"].to_numpy().copy()
                    values[pick] = [int(judged[ids[i]]) for i in pick]
                    probs[pick] = values[pick]
                    kinds[pick] = [judged_type[ids[i]] for i in pick]
                    labels["weak_label"], labels["weak_type"], labels["weak_prob"] = values, kinds, probs
                    return labels
                pred = cross_validate(windows, cfg, data, which, seed, label_override=override)
                frame, _ = add_target(pred, cfg)
                rows.append(dict(strategy=strategy, expert_labels=n, seed=seed,
                                 pr_auc=metrics(frame, column, "target", [])["pr_auc"]))
    runs = pd.DataFrame(rows)
    return runs.groupby(["strategy", "expert_labels"])["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std").reset_index()


# ------------------------------------------------------------------ further studies

def _summary_row(pred: pd.DataFrame, cfg: dict, column: str = "score_xgb") -> dict:
    frame, source = add_target(pred, cfg)
    ev = cfg["evaluation"]
    m = metrics(frame, column, "target", ev["budgets"], max_per_segment=ev.get("max_per_segment"))
    b = cfg["review"]["budget"]
    return {"pr_auc": m["pr_auc"], "f1": m["f1"], f"precision@{b}": m.get(f"precision@{b}"),
            "type_accuracy": type_accuracy(frame, "type_xgb", "target_type")["accuracy"] if "type_xgb" in frame else np.nan,
            "n": m["n"], "positives": m["positives"], "evaluated_against": source}


def reference_study(cfg: dict) -> pd.DataFrame:
    """Does the choice of reference lap change which moments are found, and for whom?

    The pipeline is run once per reference: the own best lap, the ideal lap built from the
    driver's best segments, and the best lap of another experienced driver.
    """
    paths = Paths.from_config(cfg).ensure()
    rows, by_level = [], []
    for mode in REFERENCES:
        windows = build_windows(cfg, reference=mode) if mode != cfg["features"]["reference"] else load_windows(cfg)
        labels = labelling.label(windows, cfg)
        pred = cross_validate(windows, cfg, which=("xgb",), seed=cfg["evaluation"]["seeds"][0])
        rows.append(dict(reference=mode, positive_rate=float(labels["weak_label"].mean()),
                         mean_time_loss=float(windows["time_loss"].mean()), **_summary_row(pred, cfg)))
        frame, _ = add_target(pred, cfg)
        scores = {level: metrics(g, "score_xgb", "target", [])["pr_auc"] for level, g in frame.groupby("level")}
        joined = pd.concat([windows[["level", "time_loss"]], labels[["weak_label", "weak_type"]]], axis=1)
        for level, g in joined.groupby("level"):
            kinds = g.loc[g["weak_label"] == 1, "weak_type"].value_counts(normalize=True)
            by_level.append(dict(reference=mode, level=level, positive_rate=float(g["weak_label"].mean()),
                                 mean_time_loss=float(g["time_loss"].mean()), pr_auc=scores.get(level),
                                 **{f"share_{k}": float(kinds.get(k, 0.0)) for k in labelling.TYPES}))
    table = pd.DataFrame(rows)
    table.to_csv(paths.results / "reference_overall.csv", index=False)
    pd.DataFrame(by_level).to_csv(paths.results / "reference_by_level.csv", index=False)
    return table


def generalisation(cfg: dict) -> pd.DataFrame:
    """How far does the model carry: to a new driver, a new circuit and a new experience level?"""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    rows = []
    for scheme, question in (("lodo", "unseen driver"), ("loco", "unseen circuit"), ("lolo", "unseen experience level")):
        try:
            pred = cross_validate(windows, cfg, which=("xgb",), seed=cfg["evaluation"]["seeds"][0], scheme=scheme)
        except ValueError as exc:                # for example a single circuit in the data
            rows.append(dict(scheme=scheme, question=question, note=str(exc)))
            continue
        frame, _ = add_target(pred, cfg)
        rows.append(dict(scheme=scheme, question=question, fold="all", **_summary_row(pred, cfg),
                         pr_auc_time_loss=metrics(frame, "score_time_loss", "target", [])["pr_auc"]))
        for fold, g in frame.groupby("fold"):
            rows.append(dict(scheme=scheme, question=question, fold=fold,
                             pr_auc=metrics(g, "score_xgb", "target", [])["pr_auc"],
                             pr_auc_time_loss=metrics(g, "score_time_loss", "target", [])["pr_auc"],
                             n=int(g["target"].notna().sum())))
    table = pd.DataFrame(rows)
    table.to_csv(paths.results / "generalisation.csv", index=False)
    return table


def ablation(cfg: dict) -> dict:
    """What each part contributes: feature groups, labelling functions and aggregation choices."""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    seed = cfg["evaluation"]["seeds"][0]
    base = _summary_row(cross_validate(windows, cfg, seed=seed), cfg)

    rows = [dict(variant="all features", **base)]
    for group, columns in FEATURE_GROUPS.items():
        kept = [f for f in MODEL_FEATURES if f not in columns]
        rows.append(dict(variant=f"without {group}", **_summary_row(cross_validate(windows, cfg, seed=seed, features=kept), cfg)))
    deltas = [f for f in MODEL_FEATURES if f.startswith("d_") or f in ("time_loss", "exit_loss", "min_speed_ratio",
                                                                     "line_dev_max", "line_dev_mean", "dtw_speed")]
    rows.append(dict(variant="without comparison with the reference",
                     **_summary_row(cross_validate(windows, cfg, seed=seed, features=[f for f in MODEL_FEATURES if f not in deltas]), cfg)))
    feature_table = pd.DataFrame(rows)
    feature_table["delta_pr_auc"] = feature_table["pr_auc"] - base["pr_auc"]

    rows = [dict(variant="all functions", **base)]
    for t in labelling.TYPES:
        rows.append(dict(variant=f"without {t}", **_summary_row(cross_validate(windows, cfg, seed=seed, exclude=(t,)), cfg)))
    function_table = pd.DataFrame(rows)
    function_table["delta_pr_auc"] = function_table["pr_auc"] - base["pr_auc"]

    rows = []
    for aggregation in ("any", "label_model"):
        for rule in ("priority", "strongest"):
            variant = copy.deepcopy(cfg)
            variant["labelling"].update(aggregation=aggregation, type_rule=rule)
            labels = labelling.label(windows, variant)
            rows.append(dict(aggregation=aggregation, type_rule=rule, positive_rate=float(labels["weak_label"].mean()),
                             **_summary_row(cross_validate(windows, variant, seed=seed), variant)))
    aggregation_table = pd.DataFrame(rows)

    feature_table.to_csv(paths.results / "ablation_features.csv", index=False)
    function_table.to_csv(paths.results / "ablation_labelling_functions.csv", index=False)
    aggregation_table.to_csv(paths.results / "ablation_aggregation.csv", index=False)
    return {"features": feature_table, "labelling_functions": function_table, "aggregation": aggregation_table}
