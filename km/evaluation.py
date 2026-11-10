"""Splits by driver, metrics and confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def driver_folds(windows: pd.DataFrame, cfg: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Train and test masks. No driver is ever on both sides of a split."""
    ev = cfg["evaluation"]
    drivers = windows["driver"].to_numpy()
    if ev["scheme"] == "fixed":
        test = np.isin(drivers, ev["fixed_test_drivers"])
        if not test.any() or test.all():
            raise ValueError("evaluation.fixed_test_drivers must name some, but not all, drivers")
        return [("fixed", ~test, test)]
    if ev["scheme"] != "lodo":
        raise ValueError(f"Unknown evaluation scheme: {ev['scheme']}")
    return [(d, drivers != d, drivers == d) for d in sorted(set(drivers))]


def top_b(frame: pd.DataFrame, score: str, b: int) -> np.ndarray:
    """Mask of the B highest-scoring windows of every session (requirement R1)."""
    rank = frame.groupby("session_id")[score].rank(method="first", ascending=False)
    return (rank <= b).to_numpy()


def metrics(frame: pd.DataFrame, score: str, target: str, budgets: list[int], threshold: float = 0.5) -> dict:
    """PR-AUC, precision, recall and F1 at a threshold, and precision and recall at every budget B.

    The target may be missing for windows that no expert has judged. Those windows still take
    part in the ranking per session, but only judged windows are counted.
    """
    target_values = frame[target].to_numpy(float)
    known = ~np.isnan(target_values)
    y = target_values[known].astype(int)
    s = frame[score].to_numpy(float)
    out = {"n": int(known.sum()), "positives": int(y.sum()),
           "pr_auc": float(average_precision_score(y, s[known])) if 0 < y.sum() < len(y) else np.nan}
    p, r, f, _ = precision_recall_fscore_support(y, s[known] >= threshold, average="binary", zero_division=0)
    out.update(precision=float(p), recall=float(r), f1=float(f))
    for b in budgets:
        chosen = top_b(frame, score, b) & known
        hits = target_values[chosen]
        out[f"precision@{b}"] = float(hits.mean()) if chosen.any() else np.nan
        out[f"recall@{b}"] = float(hits.sum() / y.sum()) if y.sum() else np.nan
    return out


def bootstrap_ci(frame: pd.DataFrame, score: str, target: str, budgets: list[int], n_boot: int,
                 seed: int = 0, threshold: float = 0.5) -> pd.DataFrame:
    """Percentile confidence intervals from resampling drivers, not windows.

    Windows of one driver are not independent, so the driver is the unit that is resampled.
    """
    rng = np.random.default_rng(seed)
    groups = {d: g for d, g in frame.groupby("driver")}
    names = list(groups)
    rows = []
    for _ in range(n_boot):
        draw = rng.choice(names, len(names), replace=True)
        # A driver drawn twice counts as two drivers, so the session ids are made unique.
        sample = pd.concat([groups[d].assign(session_id=groups[d]["session_id"] + f"#{k}")
                            for k, d in enumerate(draw)])
        rows.append(metrics(sample, score, target, budgets, threshold))
    boot = pd.DataFrame(rows).drop(columns=["n", "positives"])
    point = metrics(frame, score, target, budgets, threshold)
    return pd.DataFrame({"value": pd.Series(point).drop(["n", "positives"]),
                         "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)})


def type_accuracy(frame: pd.DataFrame, predicted: str, target: str) -> dict:
    """Agreement on the moment type among the windows that are key moments."""
    sub = frame[frame[target] != "none"]
    if sub.empty:
        return {"n": 0, "accuracy": np.nan}
    return {"n": len(sub), "accuracy": float((sub[predicted] == sub[target]).mean())}
