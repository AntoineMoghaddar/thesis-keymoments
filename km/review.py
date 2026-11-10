"""Expert validation (SQ4): the review package, the response store and the agreement analysis."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import labelling
from .advice import advice_text
from .config import Paths
from .evaluation import top_b
from .features import load_windows
from .preprocess import CH, load_prepared

TRACE_COLUMNS = ["speed", "gas", "brake", "steer"]


# ---------------------------------------------------------------- access codes

def hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode()).hexdigest()


def make_codes(review_dir: Path, n_experts: int) -> pd.DataFrame:
    """Create one access code per expert and one for the researcher.

    The review tool only gets the hashes (access.csv). The readable codes are written to
    access_codes_PRIVATE.csv, which stays with the researcher and is never deployed.
    """
    rows = [dict(expert_id=f"E{i:02d}", role="expert") for i in range(1, n_experts + 1)]
    rows.append(dict(expert_id="RESEARCHER", role="admin"))
    for row in rows:
        row["code"] = "-".join(secrets.token_hex(2).upper() for _ in range(2))
    private = pd.DataFrame(rows)
    review_dir.mkdir(parents=True, exist_ok=True)
    private.to_csv(review_dir / "access_codes_PRIVATE.csv", index=False)
    private.assign(code_hash=private["code"].map(hash_code)).drop(columns="code").to_csv(
        review_dir / "access.csv", index=False)
    return private


def check_code(review_dir: Path, code: str) -> dict | None:
    path = Path(review_dir) / "access.csv"
    if not path.exists() or not code.strip():
        return None
    access = pd.read_csv(path)
    hit = access[access["code_hash"] == hash_code(code)]
    return None if hit.empty else hit.iloc[0].to_dict()


# ---------------------------------------------------------------- review package

def _traces(windows: pd.DataFrame, data: np.ndarray, lengths: dict, cfg: dict) -> pd.DataFrame:
    """Signals of the lap and its reference around each window, for plotting."""
    points, context = cfg["review"]["trace_points"], cfg["review"]["context"]
    n = data.shape[1]
    parts = []
    for r in windows.itertuples():
        pad = int((r.end_idx - r.start_idx) * context)
        idx = np.linspace(max(r.start_idx - pad, 0), min(r.end_idx + pad, n) - 1, points).astype(int)
        lap, ref = data[r.lap_row][idx], data[r.ref_row][idx]
        ds = lengths[r.circuit] / n
        frame = pd.DataFrame({"window_id": r.window_id, "dist_m": (idx - r.start_idx) * ds,
                              "in_segment": (idx >= r.start_idx) & (idx < r.end_idx)})
        for c in TRACE_COLUMNS:
            frame[c], frame[f"ref_{c}"] = lap[:, CH[c]], ref[:, CH[c]]
        frame["delta_t"] = (lap[:, CH["lap_t"]] - lap[0, CH["lap_t"]]) - (ref[:, CH["lap_t"]] - ref[0, CH["lap_t"]])
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def build_package(cfg: dict, seed: int = 0) -> dict:
    """Select the windows for the experts and write everything the review tool needs.

    Windows come from three groups: selected by the model, selected by the time-loss baseline
    only, and selected by neither. The experts do not see the group. The key that links a
    window to its group is stored in a separate file for the researcher.
    """
    paths = Paths.from_config(cfg).ensure()
    rv = cfg["review"]
    rng = np.random.default_rng(seed)
    windows = load_windows(cfg)
    _, data, circuits = load_prepared(cfg)
    pred = pd.read_parquet(paths.prepared / "predictions.parquet")
    frame = windows.merge(pred[["window_id", "score_xgb", "type_xgb", "weak_label", "weak_type"]], on="window_id")
    frame["in_model"] = top_b(frame, "score_xgb", rv["budget"])
    frame["in_baseline"] = top_b(frame, "score_time_loss" if "score_time_loss" in frame else "time_loss", rv["budget"])
    frame = frame[~frame["is_reference"]]
    frame["group"] = np.where(frame["in_model"], "model", np.where(frame["in_baseline"], "baseline", "neither"))
    pools = {"model": frame[frame["group"] == "model"], "baseline": frame[frame["group"] == "baseline"],
             "neither": frame[(frame["group"] == "neither") & (frame["is_corner"] == 1)]}

    def draw(n_total: int, taken: set) -> list[str]:
        chosen = []
        for group, share in rv["mix"].items():
            pool = pools[group]
            pool = pool[~pool["window_id"].isin(taken | set(chosen))]
            k = min(int(round(share * n_total)), len(pool), n_total - len(chosen))
            chosen += list(rng.choice(pool["window_id"].to_numpy(), k, replace=False))
        # When a group runs out, the remaining places go to windows selected by neither.
        rest = pools["neither"][~pools["neither"]["window_id"].isin(taken | set(chosen))]
        k = min(n_total - len(chosen), len(rest))
        if k > 0:
            chosen += list(rng.choice(rest["window_id"].to_numpy(), k, replace=False))
        return chosen

    shared = draw(rv["shared"], set())
    assignments, used = [], set(shared)
    experts = [f"E{i:02d}" for i in range(1, rv["n_experts"] + 1)]
    for expert in experts:
        own = draw(rv["per_expert"] - len(shared), used)
        used |= set(own)
        order = rng.permutation(shared + own)
        assignments += [dict(expert_id=expert, position=k, window_id=w, shared=w in set(shared))
                        for k, w in enumerate(order)]
    assignments = pd.DataFrame(assignments)

    items = frame[frame["window_id"].isin(used)].copy()
    items["shown_type"] = items["type_xgb"]
    for lang in ("en", "nl"):
        items[f"advice_{lang}"] = [advice_text(r, r["shown_type"], lang) for _, r in items.iterrows()]
    lengths = {c: info["length_m"] for c, info in circuits.items()}
    blind = ["window_id", "circuit", "segment", "segment_name", "shown_type", "advice_en", "advice_nl",
             "time_loss", "seg_time", "ref_seg_time"]
    items[blind].to_parquet(paths.review / "items.parquet", index=False)
    items[["window_id", "driver", "level", "session_id", "lap", "group", "in_model", "in_baseline", "score_xgb",
           "weak_label", "weak_type"]].to_parquet(paths.review / "key_PRIVATE.parquet", index=False)
    assignments.to_parquet(paths.review / "assignments.parquet", index=False)
    _traces(items, data, lengths, cfg).to_parquet(paths.review / "traces.parquet", index=False)
    if not (paths.review / "access.csv").exists():
        make_codes(paths.review, rv["n_experts"])
    return {"windows": len(items), "shared": len(shared), "per_expert": assignments.groupby("expert_id").size().to_dict(),
            "groups": items["group"].value_counts().to_dict()}


# ---------------------------------------------------------------- responses

DDL = """
CREATE TABLE IF NOT EXISTS responses (
    expert_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    relevant INTEGER NOT NULL,          -- would discuss this segment in a debrief
    type_correct INTEGER,               -- assigned moment type is correct
    correct_type TEXT,                  -- type chosen by the expert when it is not
    advice_ok INTEGER,                  -- assigned advice category is suitable
    comment TEXT,
    seconds REAL,                       -- time spent on the window
    saved_at TEXT NOT NULL,
    PRIMARY KEY (expert_id, window_id)
);
CREATE TABLE IF NOT EXISTS expert_profile (
    expert_id TEXT PRIMARY KEY,
    answers TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
"""


def connect(review_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(review_dir) / "responses.db", timeout=30)
    con.executescript(DDL)
    return con


def save_response(review_dir: Path, expert_id: str, window_id: str, relevant: bool, type_correct, correct_type,
                  advice_ok, comment: str, seconds: float) -> None:
    def as_int(v):
        return None if v is None else int(bool(v))
    with connect(review_dir) as con:
        con.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
                    (expert_id, window_id, int(bool(relevant)), as_int(type_correct), correct_type or None,
                     as_int(advice_ok), comment.strip(), float(seconds), time.strftime("%Y-%m-%dT%H:%M:%S")))


def save_profile(review_dir: Path, expert_id: str, answers_json: str) -> None:
    with connect(review_dir) as con:
        con.execute("INSERT OR REPLACE INTO expert_profile VALUES (?,?,?)",
                    (expert_id, answers_json, time.strftime("%Y-%m-%dT%H:%M:%S")))


def load_responses(review_dir: Path) -> pd.DataFrame:
    with connect(review_dir) as con:
        return pd.read_sql_query("SELECT * FROM responses", con)


# ---------------------------------------------------------------- analysis

def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for a table (items x categories) of rating counts with equal raters per item."""
    table = np.asarray(table, dtype=float)
    n_raters = table.sum(axis=1)[0]
    if len(table) == 0 or n_raters < 2:
        return float("nan")
    p_cat = table.sum(axis=0) / table.sum()
    p_item = ((table**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    expected = float((p_cat**2).sum())
    return float("nan") if expected == 1 else float((p_item.mean() - expected) / (1 - expected))


def analyse(cfg: dict) -> dict:
    """Agreement among experts and between experts and the model, and the reference labels."""
    paths = Paths.from_config(cfg).ensure()
    resp = load_responses(paths.review)
    if resp.empty:
        raise RuntimeError("No expert responses found.")
    key = pd.read_parquet(paths.review / "key_PRIVATE.parquet")
    items = pd.read_parquet(paths.review / "items.parquet")
    assignments = pd.read_parquet(paths.review / "assignments.parquet")
    resp = resp.merge(key, on="window_id").merge(items[["window_id", "shown_type"]], on="window_id")
    out = {}

    # Agreement among the experts on the shared windows that all of them completed.
    shared_ids = assignments.loc[assignments["shared"], "window_id"].unique()
    shared = resp[resp["window_id"].isin(shared_ids)]
    wide = shared.pivot(index="window_id", columns="expert_id", values="relevant").dropna()
    if wide.shape[1] >= 2 and len(wide):
        counts = np.column_stack([(wide == 0).sum(axis=1), (wide == 1).sum(axis=1)])
        out["fleiss_kappa_relevance"] = fleiss_kappa(counts)
        out["shared_windows_complete"], out["experts"] = int(len(wide)), int(wide.shape[1])

    # Agreement of every expert with the selection of the model and of the baseline.
    rows = []
    for expert, g in resp.groupby("expert_id"):
        row = dict(expert_id=expert, n=len(g), relevant_rate=float(g["relevant"].mean()))
        for name in ("in_model", "in_baseline"):
            both = len(set(g["relevant"])) > 1 and len(set(g[name])) > 1
            row[f"cohen_kappa_{name[3:]}"] = float(cohen_kappa_score(g["relevant"], g[name].astype(int))) if both else np.nan
        rows.append(row)
    per_expert = pd.DataFrame(rows)

    # Reference label per window: the majority of the experts who judged it. A tie counts as not relevant.
    def reference(g):
        relevant = int(g["relevant"].mean() > 0.5)
        agree = g["type_correct"].dropna()
        if not relevant:
            kind = "none"
        elif len(agree) and agree.mean() >= 0.5:
            kind = g["shown_type"].iloc[0]
        else:
            other = g["correct_type"].dropna()
            kind = other.mode().iloc[0] if len(other) else g["shown_type"].iloc[0]
        return pd.Series({"expert_label": relevant, "expert_type": kind, "n_experts": len(g),
                          "advice_ok": g["advice_ok"].mean()})
    labels = resp.groupby("window_id").apply(reference, include_groups=False).reset_index()
    labels.to_parquet(paths.review / "expert_labels.parquet", index=False)

    joined = labels.merge(key, on="window_id").merge(items[["window_id", "shown_type"]], on="window_id")
    by_group = joined.groupby("group").agg(n=("expert_label", "size"), relevant=("expert_label", "mean"),
                                           advice_ok=("advice_ok", "mean")).reset_index()
    positives = joined[joined["expert_label"] == 1]
    out["type_agreement"] = float((positives["shown_type"] == positives["expert_type"]).mean()) if len(positives) else np.nan
    out["precision_model_selection"] = float(joined.loc[joined["in_model"], "expert_label"].mean())
    out["precision_baseline_selection"] = float(joined.loc[joined["in_baseline"], "expert_label"].mean())

    # Precision of every labelling function against the reference labels.
    windows = load_windows(cfg)
    votes = labelling.label(windows, cfg).assign(window_id=windows["window_id"])
    judged = votes.merge(labels, on="window_id")
    rows = []
    for t in labelling.TYPES:
        fired = judged[judged[t] == 1]
        rows.append(dict(function=t, n_judged=len(fired),
                         precision_relevant=float(fired["expert_label"].mean()) if len(fired) else np.nan,
                         precision_type=float((fired["expert_type"] == t).mean()) if len(fired) else np.nan))
    lf = pd.DataFrame(rows)

    per_expert.to_csv(paths.results / "sq4_per_expert.csv", index=False)
    by_group.to_csv(paths.results / "sq4_by_group.csv", index=False)
    lf.to_csv(paths.results / "sq4_labelling_function_precision.csv", index=False)
    pd.Series(out).to_csv(paths.results / "sq4_summary.csv", header=["value"])
    out.update(per_expert=per_expert, by_group=by_group, labelling_functions=lf)
    return out
