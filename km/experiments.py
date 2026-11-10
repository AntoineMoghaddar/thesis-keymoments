"""Experiments for the four sub-questions. Every function writes its tables to the results folder."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import labelling, models
from .config import Paths
from .evaluation import bootstrap_ci, driver_folds, metrics, type_accuracy
from .features import MODEL_FEATURES, load_windows, raw_windows
from .preprocess import load_prepared

SCORES = {"xgb": "score_xgb", "seq": "score_seq", "labelling_functions": "score_lf",
          "time_loss": "score_time_loss", "random": "score_random"}


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
                   which=("xgb",), seed: int = 0, train_filter=None, label_override=None) -> pd.DataFrame:
    """Train on some drivers, score the others, and return the scores of every window once.

    train_filter    function(train_mask, rng) -> mask, used to train on part of the data
    label_override  function(labels, train_mask) -> labels, used to replace weak labels
    """
    rng = np.random.default_rng(seed)
    fit_on_train = cfg["labelling"]["fit_on"] == "train"
    labels_all = labelling.label(windows, cfg)
    seq_x = raw_windows(windows, data, cfg["model"]["seq"]["length"]) if "seq" in which else None
    parts = []
    for fold, train, test in driver_folds(windows, cfg):
        labels = labelling.label(windows, cfg, fit_mask=train) if fit_on_train else labels_all
        if label_override is not None:
            labels = label_override(labels.copy(), train)
        fit = train_filter(train, rng) if train_filter is not None else train
        y = labels["weak_label"].to_numpy()
        out = windows.loc[test, ["window_id", "driver", "level", "circuit", "session_id", "segment"]].copy()
        out["fold"] = fold
        out["weak_label"] = y[test]
        out["weak_type"] = labels["weak_type"].to_numpy()[test]
        if "xgb" in which:
            detector = models.fit_detector(windows[fit], y[fit], cfg, seed)
            out["score_xgb"] = detector.predict_proba(windows.loc[test, MODEL_FEATURES])[:, 1]
            positive = fit & (y == 1)
            typer = models.TypeModel(cfg, seed).fit(windows[positive], labels["weak_type"][positive])
            out["type_xgb"] = typer.predict(windows[test])
        if "seq" in which:
            from .seqmodel import SequenceDetector
            out["score_seq"] = SequenceDetector(cfg, seed).fit(seq_x[fit], y[fit]).predict_proba(seq_x[test])
        out["score_lf"] = models.score_labelling_functions(labels)[test]
        out["score_time_loss"] = models.score_time_loss(windows[test])
        out["score_random"] = models.score_random(int(test.sum()), seed)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def sq1(cfg: dict) -> dict:
    """Which types of key moments can be observed: behaviour of the labelling functions."""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    labels = labelling.label(windows, cfg)
    stats = labelling.statistics(labels, cfg)
    stats.to_csv(paths.results / "sq1_labelling_functions.csv", index=False)
    labelling.fit_thresholds(windows, cfg).to_csv(paths.results / "sq1_thresholds.csv", index=False)
    joined = pd.concat([windows[["level", "circuit", "segment"]], labels], axis=1)
    by_level = pd.crosstab(joined["level"], joined["weak_type"])
    by_level.to_csv(paths.results / "sq1_types_by_level.csv")
    pd.crosstab([joined["circuit"], joined["segment"]], joined["weak_type"]).to_csv(
        paths.results / "sq1_types_by_segment.csv")
    labels.assign(window_id=windows["window_id"]).to_parquet(paths.prepared / "weak_labels.parquet", index=False)
    return {"statistics": stats, "by_level": by_level, "positive_rate": float(labels["weak_label"].mean())}


def sq2(cfg: dict, with_seq: bool = True) -> dict:
    """Which modelling approach detects key moments best for drivers it has not seen."""
    paths = Paths.from_config(cfg).ensure()
    windows = load_windows(cfg)
    _, data, _ = load_prepared(cfg)
    which = ("xgb", "seq") if with_seq else ("xgb",)
    pred = cross_validate(windows, cfg, data, which=which, seed=cfg["evaluation"]["seeds"][0])
    pred.to_parquet(paths.prepared / "predictions.parquet", index=False)
    frame, source = add_target(pred, cfg)
    budgets, n_boot = cfg["evaluation"]["budgets"], cfg["evaluation"]["bootstrap"]
    rows = []
    for name, column in SCORES.items():
        if column not in frame:
            continue
        # The scores of the baselines are not probabilities, so no threshold metric is given for them.
        threshold = 0.5 if name in ("xgb", "seq") else (1.0 if name == "labelling_functions" else np.inf)
        ci = bootstrap_ci(frame, column, "target", budgets, n_boot, threshold=threshold)
        for metric, r in ci.iterrows():
            if threshold == np.inf and metric in ("precision", "recall", "f1"):
                continue
            rows.append(dict(model=name, metric=metric, **r.to_dict()))
    table = pd.DataFrame(rows)
    table.to_csv(paths.results / "sq2_metrics.csv", index=False)
    per_level = pd.DataFrame([dict(level=lv, model=name, **metrics(g, col, "target", budgets))
                              for lv, g in frame.groupby("level") for name, col in SCORES.items() if col in g])
    per_level.to_csv(paths.results / "sq2_metrics_by_level.csv", index=False)
    types = type_accuracy(frame, "type_xgb", "target_type")
    confusion = pd.crosstab(frame.loc[frame["target_type"] != "none", "target_type"],
                            frame.loc[frame["target_type"] != "none", "type_xgb"])
    confusion.to_csv(paths.results / "sq2_type_confusion.csv")

    labels = labelling.label(windows, cfg)
    detector = models.fit_detector(windows, labels["weak_label"].to_numpy(), cfg)
    importance = pd.Series(detector.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    importance.rename("gain").to_csv(paths.results / "sq2_feature_importance.csv")
    summary = {"evaluated_against": source, "scheme": cfg["evaluation"]["scheme"], "type_accuracy": types,
               "drivers": int(windows["driver"].nunique()), "windows": int(len(windows))}
    (paths.results / "sq2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"metrics": table, "summary": summary, "per_level": per_level}


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

    # Confidence interval with all data, from resampling drivers.
    full, source = add_target(cross_validate(windows, cfg, data, which, ev["seeds"][0]), cfg)
    ci = bootstrap_ci(full, column, "target", [], ev["bootstrap"]).loc["pr_auc"]
    requirement = {"evaluated_against": source, "model": which[0],
                   "pr_auc_all_data": float(ci["value"]), "ci_low": float(ci["ci_low"]), "ci_high": float(ci["ci_high"]),
                   "training_fraction": _requirement(size_curve, "fraction", ci["ci_low"]),
                   "laps_per_driver": _requirement(laps_curve, "laps", ci["ci_low"])}

    size.to_csv(paths.results / "sq3_training_size_runs.csv", index=False)
    size_curve.to_csv(paths.results / "sq3_training_size.csv", index=False)
    laps.to_csv(paths.results / "sq3_laps_per_driver_runs.csv", index=False)
    laps_curve.to_csv(paths.results / "sq3_laps_per_driver.csv", index=False)
    out = {"training_size": size_curve, "laps_per_driver": laps_curve, "requirement": requirement}

    # 3. Label quality. This part needs expert labels and is skipped until they exist.
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
                    values[pick] = [int(judged[ids[i]]) for i in pick]
                    kinds[pick] = [judged_type[ids[i]] for i in pick]
                    labels["weak_label"], labels["weak_type"] = values, kinds
                    return labels
                pred = cross_validate(windows, cfg, data, which, seed, label_override=override)
                frame, _ = add_target(pred, cfg)
                rows.append(dict(strategy=strategy, expert_labels=n, seed=seed,
                                 pr_auc=metrics(frame, column, "target", [])["pr_auc"]))
    runs = pd.DataFrame(rows)
    return runs.groupby(["strategy", "expert_labels"])["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std").reset_index()
