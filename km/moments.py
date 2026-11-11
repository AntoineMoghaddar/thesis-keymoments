"""From flagged windows to the moments of a session.

A coach does not discuss one corner of one lap. A coach discusses what a driver does in a
corner: a pattern that returns lap after lap, or an incident that happened once. This module
groups the flagged windows of a session by corner and moment type, ranks the groups by the
time that can be gained and selects at most B of them, each in a different corner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advice import advice_text

PATTERN = {
    "en": {"structural": "Seen in {k} of {n} laps (recurring). Average values: ",
           "incidental": "Seen in {k} of {n} laps (incidental). Values: "},
    "nl": {"structural": "Gezien in {k} van {n} rondes (terugkerend). Gemiddelde waarden: ",
           "incidental": "Gezien in {k} van {n} rondes (incidenteel). Waarden: "},
}


def session_moments(frame: pd.DataFrame, cfg: dict, score: str = "score", kind: str = "type") -> pd.DataFrame:
    """One row per session, corner and moment type that was flagged at least once.

    `frame` holds the windows with their features, a score and a predicted type.
    expected_gain  sum over the flagged laps of score x time lost, in seconds per session
    """
    mc = cfg["moments"]
    n_laps = frame.groupby("session_id")["lap"].nunique()
    flagged = frame[(frame[score] >= mc["threshold"]) & (frame[kind] != "none")]
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
               and c not in ("lap", "lap_row", "ref_row", "start_idx", "end_idx", "is_corner")]
    rows = []
    for (session, segment, moment), g in flagged.groupby(["session_id", "segment", kind]):
        total = int(n_laps[session])
        loss = g["total_loss"] if "total_loss" in g else g["time_loss"]
        row = g[numeric].mean().to_dict()
        row.update(session_id=session, segment=segment, segment_name=g["segment_name"].iloc[0], type=moment,
                   driver=g["driver"].iloc[0], circuit=g["circuit"].iloc[0], n_flagged=len(g), n_laps=total,
                   share=len(g) / total, mean_loss=float(loss.mean()),
                   expected_gain=float((g[score] * loss.clip(lower=0)).sum()),
                   pattern="structural" if len(g) / total >= mc["structural_share"] else "incidental",
                   example_window=g.loc[g[score].idxmax(), "window_id"], laps=sorted(g["lap"].tolist()))
        rows.append(row)
    return pd.DataFrame(rows)


def select(moments: pd.DataFrame, budget: int) -> pd.DataFrame:
    """At most `budget` moments per session, each in a different corner, largest expected gain first."""
    if moments.empty:
        return moments
    best = moments.sort_values("expected_gain", ascending=False).drop_duplicates(["session_id", "segment"])
    best = best.assign(rank=best.groupby("session_id")["expected_gain"].rank(method="first", ascending=False))
    return best[best["rank"] <= budget].sort_values(["session_id", "rank"]).reset_index(drop=True)


def baseline_moments(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Time-loss baseline at session level: the corners with the largest summed time loss."""
    loss = "total_loss" if "total_loss" in frame else "time_loss"
    corners = frame[frame["is_corner"] == 1]
    agg = corners.groupby(["session_id", "segment"])[loss].apply(lambda x: float(x.clip(lower=0).sum())).rename("expected_gain").reset_index()
    agg["rank"] = agg.groupby("session_id")["expected_gain"].rank(method="first", ascending=False)
    return agg[agg["rank"] <= budget].assign(type="none").sort_values(["session_id", "rank"]).reset_index(drop=True)


def describe(moment: pd.Series | dict, lang: str = "en") -> str:
    """Advice for a moment: how often it occurred, then the advice filled with its average values."""
    head = PATTERN[lang][moment["pattern"]].format(k=int(moment["n_flagged"]), n=int(moment["n_laps"]))
    values = dict(moment)
    values["time_loss"] = moment["mean_loss"]
    return head + advice_text(values, moment["type"], lang)


def explanation(contrib_row: pd.Series, groups: dict[str, list[str]], top: int = 3) -> list[tuple[str, float]]:
    """The features that pushed the score of one window up most."""
    order = contrib_row.sort_values(ascending=False).head(top)
    return [(name, float(value)) for name, value in order.items() if value > 0]


def group_contributions(contrib: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame({g: contrib[[f for f in cols if f in contrib]].sum(axis=1) for g, cols in groups.items()})


def explanation_consistency(contrib: pd.DataFrame, types: np.ndarray, groups: dict, type_group: dict) -> float:
    """Share of flagged windows whose strongest feature group matches the group of the predicted type."""
    grouped = group_contributions(contrib, {g: c for g, c in groups.items() if g != "time"})
    top = grouped.idxmax(axis=1).to_numpy()
    expected = np.array([type_group.get(t, "") for t in types])
    mask = expected != ""
    return float((top[mask] == expected[mask]).mean()) if mask.any() else float("nan")
