"""Splits by driver, metrics and confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def driver_folds(windows: pd.DataFrame, cfg: dict, scheme: str | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Train and test masks.

    lodo   leave one driver out (no driver is on both sides)
    fixed  the drivers named in the configuration form the test set
    loco   leave one circuit out (the model is tested on a circuit it has not seen)
    lolo   leave one experience level out (the model is tested on a level it has not seen)
    """
    ev = cfg["evaluation"]
    scheme = scheme or ev["scheme"]
    drivers = windows["driver"].to_numpy()
    if scheme == "fixed":
        test = np.isin(drivers, ev["fixed_test_drivers"])
        if not test.any() or test.all():
            raise ValueError("evaluation.fixed_test_drivers must name some, but not all, drivers")
        return [("fixed", ~test, test)]
    column = {"lodo": "driver", "loco": "circuit", "lolo": "level"}.get(scheme)
    if column is None:
        raise ValueError(f"Unknown evaluation scheme: {scheme}")
    values = windows[column].to_numpy()
    groups = sorted(set(values))
    if len(groups) < 2:
        raise ValueError(f"Scheme {scheme} needs at least two different values of {column}")
    return [(str(g), values != g, values == g) for g in groups]


def top_b(frame: pd.DataFrame, score: str, b: int, max_per_segment: int | None = None) -> np.ndarray:
    """Mask of the B highest-scoring windows of every session (requirement R1).

    With `max_per_segment` the selection is spread over corners: without it, the same corner in
    three different laps could fill the whole budget.
    """
    eligible = np.ones(len(frame), dtype=bool)
    if max_per_segment and "segment" in frame:
        within = frame.groupby(["session_id", "segment"])[score].rank(method="first", ascending=False)
        eligible = (within <= max_per_segment).to_numpy()
    rank = frame[score].where(eligible).groupby(frame["session_id"]).rank(method="first", ascending=False)
    return (rank <= b).fillna(False).to_numpy()


def metrics(frame: pd.DataFrame, score: str, target: str, budgets: list[int], threshold: float = 0.5,
            max_per_segment: int | None = None) -> dict:
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
        chosen = top_b(frame, score, b, max_per_segment) & known
        hits = target_values[chosen]
        out[f"precision@{b}"] = float(hits.mean()) if chosen.any() else np.nan
        out[f"recall@{b}"] = float(hits.sum() / y.sum()) if y.sum() else np.nan
    return out


def bootstrap_ci(frame: pd.DataFrame, score: str, target: str, budgets: list[int], n_boot: int,
                 seed: int = 0, threshold: float = 0.5, max_per_segment: int | None = None) -> pd.DataFrame:
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
        rows.append(metrics(sample, score, target, budgets, threshold, max_per_segment))
    boot = pd.DataFrame(rows).drop(columns=["n", "positives"])
    point = metrics(frame, score, target, budgets, threshold, max_per_segment)
    return pd.DataFrame({"value": pd.Series(point).drop(["n", "positives"]),
                         "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)})


def type_accuracy(frame: pd.DataFrame, predicted: str, target: str) -> dict:
    """Agreement on the moment type among the windows that are key moments."""
    sub = frame[frame[target] != "none"]
    if sub.empty:
        return {"n": 0, "accuracy": np.nan}
    return {"n": len(sub), "accuracy": float((sub[predicted] == sub[target]).mean())}


def per_driver(frame: pd.DataFrame, scores: dict[str, str], target: str) -> pd.DataFrame:
    """PR-AUC per driver and model. With few drivers this table says more than one pooled number."""
    rows = []
    for driver, g in frame.groupby("driver"):
        row = {"driver": driver, "level": g["level"].iloc[0] if "level" in g else "", "n": int(g[target].notna().sum())}
        for name, column in scores.items():
            if column in g:
                row[name] = metrics(g, column, target, [])["pr_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_permutation(a, b, n_random: int = 20000, seed: int = 0) -> dict:
    """Paired sign-flip test on per-driver differences (two-sided).

    Under the null hypothesis the sign of each driver's difference is arbitrary. All sign
    patterns are enumerated for up to 15 drivers; beyond that they are sampled.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return {"n": 0, "mean_difference": np.nan, "p_value": np.nan}
    if n <= 15:
        signs = np.array([[1 if (k >> i) & 1 else -1 for i in range(n)] for k in range(2 ** n)])
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_random, n))
    null = np.abs((signs * diff).mean(axis=1))
    return {"n": n, "mean_difference": float(diff.mean()),
            "p_value": float((null >= abs(diff.mean()) - 1e-12).mean()), "smallest_possible_p": float(2 / 2 ** n) if n <= 15 else np.nan}


def calibration(frame: pd.DataFrame, score: str, target: str, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Reliability table, expected calibration error and Brier score for a probability score."""
    known = frame[target].notna().to_numpy()
    y, p = frame.loc[known, target].to_numpy(float), frame.loc[known, score].to_numpy(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for k in range(bins):
        mask = which == k
        if mask.any():
            rows.append(dict(bin_low=edges[k], bin_high=edges[k + 1], n=int(mask.sum()),
                             mean_score=float(p[mask].mean()), observed=float(y[mask].mean())))
    table = pd.DataFrame(rows)
    ece = float((table["n"] * (table["mean_score"] - table["observed"]).abs()).sum() / max(len(y), 1)) if len(table) else np.nan
    return table, {"ece": ece, "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan, "n": int(len(y))}
