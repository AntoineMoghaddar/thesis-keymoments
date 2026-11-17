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

from . import labelling, moments, track
from .advice import advice_text
from .config import Paths
from .evaluation import top_b
from .features import load_windows
from .preprocess import CH, load_prepared

TRACE_COLUMNS = ["speed", "gas", "brake", "steer"]
CAR_HALF_WIDTH_M = track.CAR_HALF_WIDTH_M      # the recorded position is the centre of the car


def track_outline(laps: pd.DataFrame, data: np.ndarray, edges: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Outline per circuit for the top view, built from the driven lines.

    When the circuit has surveyed edges (km/track.py) these are the outline. Otherwise the outline
    is the band between the 1st and 99th percentile of the lateral position of all laps, widened
    by half a car width: the part of the road that was used, not the track itself. The column
    `source` says which of the two it is.

    The simulator's ground plane is mirrored when x is drawn to the right and z upwards. The
    second return value holds, per circuit, the factor for z (1 or -1) with which steering to
    the right is drawn as a turn to the right, together with the middle line and its normal.
    """
    parts, flip = [], {}
    circuit_of = laps["circuit"].to_numpy()
    for circuit in sorted(set(circuit_of)):
        rows = np.where(circuit_of == circuit)[0]
        x, z = data[rows][:, :, CH["x"]], data[rows][:, :, CH["z"]]
        cx, cz = np.median(x, axis=0), np.median(z, axis=0)
        mine = edges[edges["circuit"] == circuit].sort_values("idx") if edges is not None else []
        surveyed = len(mine) == len(cx)
        if surveyed:                               # the middle of the track instead of the middle of the laps
            cx, cz = (mine["a_x"].to_numpy() + mine["b_x"].to_numpy()) / 2, (mine["a_z"].to_numpy() + mine["b_z"].to_numpy()) / 2
        tx, tz = np.gradient(cx), np.gradient(cz)
        turn = np.gradient(np.unwrap(np.arctan2(tz, tx)))
        steer = np.median(data[rows][:, :, CH["steer"]], axis=0)
        ok = np.isfinite(turn) & np.isfinite(steer)
        sign = -1.0 if ok.sum() > 2 and np.corrcoef(steer[ok], turn[ok])[0, 1] > 0 else 1.0
        norm = np.hypot(tx, tz)
        norm[norm == 0] = 1.0
        nx, nz = -tz / norm, tx / norm
        lateral = (x - cx) * nx + (z - cz) * nz
        if surveyed:
            side_a = (mine["a_x"].to_numpy() - cx) * nx + (mine["a_z"].to_numpy() - cz) * nz
            side_b = (mine["b_x"].to_numpy() - cx) * nx + (mine["b_z"].to_numpy() - cz) * nz
            lo, hi = np.minimum(side_a, side_b), np.maximum(side_a, side_b)
        else:
            lo = np.percentile(lateral, 1, axis=0) - CAR_HALF_WIDTH_M
            hi = np.percentile(lateral, 99, axis=0) + CAR_HALF_WIDTH_M
        # Seen from above after mirroring, the normal points to the left when sign is 1 and to the right otherwise.
        band = np.sort(np.stack([sign * lo, sign * hi]), axis=0)
        flip[circuit] = dict(sign=sign, cx=cx, cz=cz, nx=nx, nz=nz)
        parts.append(pd.DataFrame({
            "circuit": circuit, "idx": np.arange(len(cx)),
            "left_x": cx + nx * hi, "left_y": sign * (cz + nz * hi),
            "right_x": cx + nx * lo, "right_y": sign * (cz + nz * lo),
            "band_lo": band[0], "band_hi": band[1], "source": "survey" if surveyed else "laps"}))
    return pd.concat(parts, ignore_index=True), flip


def _position(lap: np.ndarray, idx: np.ndarray, g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-view coordinates of a lap and its distance to the left of the middle of the used road (m)."""
    x, z = lap[idx, CH["x"]], lap[idx, CH["z"]]
    lateral = g["sign"] * ((x - g["cx"][idx]) * g["nx"][idx] + (z - g["cz"][idx]) * g["nz"][idx])
    return x, g["sign"] * z, lateral


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

def _traces(windows: pd.DataFrame, data: np.ndarray, lengths: dict, cfg: dict, flip: dict) -> pd.DataFrame:
    """Signals of the lap and its reference around each window, for plotting."""
    points, context = cfg["review"]["trace_points"], cfg["review"]["context"]
    n = data.shape[1]
    parts = []
    for r in windows.itertuples():
        pad = int((r.end_idx - r.start_idx) * context)
        idx = np.linspace(max(r.start_idx - pad, 0), min(r.end_idx + pad, n) - 1, points).astype(int)
        lap, ref = data[r.lap_row][idx], data[r.ref_row][idx]
        ds = lengths[r.circuit] / n
        frame = pd.DataFrame({"window_id": r.window_id, "dist_m": (idx - r.start_idx) * ds, "idx": idx,
                              "in_segment": (idx >= r.start_idx) & (idx < r.end_idx)})
        frame["px"], frame["py"], frame["lateral"] = _position(data[r.lap_row], idx, flip[r.circuit])
        frame["ref_px"], frame["ref_py"], frame["ref_lateral"] = _position(data[r.ref_row], idx, flip[r.circuit])
        for c in TRACE_COLUMNS:
            frame[c], frame[f"ref_{c}"] = lap[:, CH[c]], ref[:, CH[c]]
        frame["delta_t"] = (lap[:, CH["lap_t"]] - lap[0, CH["lap_t"]]) - (ref[:, CH["lap_t"]] - ref[0, CH["lap_t"]])
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def build_package(cfg: dict, seed: int = 0, force: bool = False) -> dict:
    """Select the windows for the experts and write everything the review tool needs.

    Windows come from three groups: selected by the model, selected by the time-loss baseline
    only, and selected by neither. The experts do not see the group. The key that links a
    window to its group is stored in a separate file for the researcher.
    """
    paths = Paths.from_config(cfg).ensure()
    if has_answers(paths.review) and not force:
        raise RuntimeError(f"{paths.review} already holds expert answers. A new package would no longer match them. "
                           "Move the folder away first, or pass force to overwrite the package.")
    rv = cfg["review"]
    rng = np.random.default_rng(seed)
    windows = load_windows(cfg)
    _, data, circuits = load_prepared(cfg)
    pred = pd.read_parquet(paths.prepared / "predictions.parquet")
    frame = windows.merge(pred[["window_id", "score_xgb", "type_xgb", "weak_label", "weak_type"]], on="window_id")
    spread = cfg["evaluation"].get("max_per_segment")
    frame["in_model"] = top_b(frame, "score_xgb", rv["budget"], spread)
    frame["in_baseline"] = top_b(frame, "time_loss", rv["budget"], spread)
    with_reference = frame            # the session task shows every lap of a session, the reference lap included
    frame = frame[~frame["is_reference"]]
    frame["group"] = np.where(frame["in_model"], "model", np.where(frame["in_baseline"], "baseline", "neither"))
    corners = frame[frame["is_corner"] == 1]
    pools = {"model": frame[frame["group"] == "model"], "baseline": frame[frame["group"] == "baseline"],
             "neither": corners[corners["group"] == "neither"], "random": corners}
    drawn_from: dict[str, str] = {}

    def draw(n_total: int, taken: set) -> list[str]:
        chosen = []

        def take(group: str, k: int) -> None:
            pool = pools[group]
            pool = pool[~pool["window_id"].isin(taken | set(chosen))]
            k = min(k, len(pool), n_total - len(chosen))
            if k > 0:
                picked = list(rng.choice(pool["window_id"].to_numpy(), k, replace=False))
                drawn_from.update({w: group for w in picked})
                chosen.extend(picked)

        for group, share in rv["mix"].items():
            take(group, int(round(share * n_total)))
        take("random", n_total - len(chosen))      # when a group runs out, the places go to the random draw
        return chosen

    shared = draw(rv["shared"], set())
    assignments, used = [], set(shared)
    experts = [f"E{i:02d}" for i in range(1, rv["n_experts"] + 1)]
    n_repeat = min(int(rv.get("repeats", 0)), (rv["per_expert"] - len(shared)) // 2)
    repeats = {}                                  # id under which a window is shown a second time -> the window
    for expert in experts:
        own = draw(rv["per_expert"] - len(shared) - n_repeat, used)
        used |= set(own)
        order = [str(w) for w in rng.permutation(shared + own)]
        # A few windows from the first half are shown again in the last third, under another id. The two
        # answers of the same expert show how consistent an expert is, which is the ceiling for any model.
        again = [str(w) for w in rng.choice(order[:len(order) // 2], n_repeat, replace=False)] if n_repeat else []
        for w in again:
            shown_as = f"{w}~{expert}"
            repeats[shown_as] = w
            order.insert(int(rng.integers(2 * len(order) // 3, len(order) + 1)), shown_as)
        assignments += [dict(expert_id=expert, position=k, window_id=w, shared=w in set(shared),
                             repeat_of=repeats.get(w)) for k, w in enumerate(order)]
    assignments = pd.DataFrame(assignments)

    items = frame[frame["window_id"].isin(used)].copy()
    items["shown_type"] = items["type_xgb"]
    for lang in ("en", "nl"):
        items[f"advice_{lang}"] = [advice_text(r, r["shown_type"], lang) for _, r in items.iterrows()]
    lengths = {c: info["length_m"] for c, info in circuits.items()}
    blind = ["window_id", "circuit", "segment", "segment_name", "shown_type", "advice_en", "advice_nl",
             "time_loss", "seg_time", "ref_seg_time"]

    def with_repeats(table: pd.DataFrame) -> pd.DataFrame:
        """The rows of the repeated windows once more, under the id with which they are shown again."""
        copies = [table[table["window_id"] == w].assign(window_id=shown_as) for shown_as, w in repeats.items()]
        return pd.concat([table] + copies, ignore_index=True) if copies else table
    with_repeats(items[blind]).to_parquet(paths.review / "items.parquet", index=False)
    # Inclusion probability of every reviewed window: it could be drawn from its own group and from the
    # random draw. The inverse is the weight with which it stands for all windows in the evaluation.
    rate = {g: sum(v == g for v in drawn_from.values()) / max(len(pools[g]), 1) for g in pools}
    is_corner = items["is_corner"].to_numpy() == 1
    own = np.array([rate[g] if g != "neither" or c else 0.0 for g, c in zip(items["group"], is_corner)])
    inclusion = 1 - (1 - own) * (1 - np.where(is_corner, rate["random"], 0.0))
    items["inclusion_probability"] = np.clip(inclusion, 1e-6, 1.0)
    items["weight"] = 1.0 / items["inclusion_probability"]
    items["drawn_from"] = items["window_id"].map(drawn_from)
    items[["window_id", "driver", "level", "session_id", "lap", "group", "drawn_from", "in_model", "in_baseline",
           "score_xgb", "weak_label", "weak_type", "inclusion_probability", "weight"]].to_parquet(
        paths.review / "key_PRIVATE.parquet", index=False)
    assignments.to_parquet(paths.review / "assignments.parquet", index=False)
    laps, _, _ = load_prepared(cfg)
    outline, flip = track_outline(laps, data, track.load_edges(cfg))
    outline.astype({c: "float32" for c in outline.columns if c not in ("circuit", "idx", "source")}).to_parquet(
        paths.review / "track.parquet", index=False)
    with_repeats(_traces(items, data, lengths, cfg, flip)).to_parquet(paths.review / "traces.parquet", index=False)
    if not (paths.review / "access.csv").exists():
        make_codes(paths.review, rv["n_experts"])
    session_info = _session_task(frame, with_reference, data, lengths, experts, cfg, rng, paths, flip)
    return {"sessions": session_info, "drawn_from": pd.Series(drawn_from).value_counts().to_dict(), "windows": len(items), "shared": len(shared), "per_expert": assignments.groupby("expert_id").size().to_dict(),
            "groups": items["group"].value_counts().to_dict()}


def _session_task(frame, shown, data, lengths, experts, cfg, rng, paths, flip) -> dict:
    """Second task for the experts: choose the moments of a whole session themselves.

    The expert sees every corner of every lap of a session and picks at most B corners to
    discuss. Nothing from the model is shown. This gives what the window review cannot give:
    how many of the moments an expert would raise are also found by the model.
    """
    rv = cfg["review"]
    n = data.shape[1]
    sessions = rng.permutation(sorted(frame["session_id"].unique()))
    need = rv["sessions_shared"] + len(experts) * (rv["sessions_per_expert"] - rv["sessions_shared"])
    sessions = list(sessions[:min(need, len(sessions))])
    shared, rest = sessions[:rv["sessions_shared"]], sessions[rv["sessions_shared"]:]
    keys = {sid: f"S{k + 1:02d}" for k, sid in enumerate(rng.permutation(sessions))}
    own = rv["sessions_per_expert"] - len(shared)
    assignments = []
    for e, expert in enumerate(experts):
        mine = list(shared) + [rest[(e * own + k) % len(rest)] for k in range(own)] if rest else list(shared)
        for position, sid in enumerate(rng.permutation(list(dict.fromkeys(mine)))):
            assignments.append(dict(expert_id=expert, position=position, session_key=keys[sid], shared=sid in shared))

    corners = shown[shown["session_id"].isin(sessions) & (shown["is_corner"] == 1)].copy()
    corners["session_key"] = corners["session_id"].map(keys)
    corners["lap_no"] = corners.groupby("session_id")["lap"].transform(lambda x: pd.factorize(x, sort=True)[0] + 1)
    cols = ["session_key", "circuit", "segment", "segment_name", "lap_no", "seg_time", "ref_seg_time", "time_loss",
            "exit_loss", "total_loss", "tyres_out_max", "off_track", "start_idx"]
    corners[cols].to_parquet(paths.review / "session_items.parquet", index=False)

    points = rv["session_trace_points"]
    parts = []
    for r in corners.itertuples():
        idx = np.linspace(r.start_idx, r.end_idx - 1, points).astype(int)
        base = dict(session_key=r.session_key, segment=r.segment, dist_m=(idx - r.start_idx) * lengths[r.circuit] / n,
                    idx=idx)

        def one(row: int, lap_no: int) -> pd.DataFrame:
            signals = {c: data[row][idx, CH[c]] for c in TRACE_COLUMNS}
            px, py, lateral = _position(data[row], idx, flip[r.circuit])
            return pd.DataFrame(dict(base, lap_no=lap_no, **signals, px=px, py=py, lateral=lateral))

        parts.append(one(r.lap_row, r.lap_no))
        if r.lap_no == 1:                        # the reference trace is stored once per corner, as lap 0
            parts.append(one(r.ref_row, 0))
    pd.concat(parts, ignore_index=True).astype({c: "float32" for c in TRACE_COLUMNS + ["px", "py", "lateral"]}).to_parquet(
        paths.review / "session_traces.parquet", index=False)
    pd.DataFrame(assignments).to_parquet(paths.review / "session_assignments.parquet", index=False)

    # What the model and the baseline would select in these sessions. For the researcher only.
    scored = frame[frame["session_id"].isin(sessions)]
    chosen = moments.select(moments.session_moments(scored, cfg, "score_xgb", "type_xgb"), rv["budget"])
    model = chosen[["session_id", "segment", "type", "rank", "expected_gain", "pattern"]].assign(source="model") \
        if len(chosen) else pd.DataFrame(columns=["session_id", "segment", "type", "rank", "expected_gain", "pattern", "source"])
    baseline = moments.baseline_moments(scored, rv["budget"]).assign(source="baseline", pattern="")
    private = pd.concat([model, baseline[model.columns]], ignore_index=True)
    private["session_key"] = private["session_id"].map(keys)
    private.to_parquet(paths.review / "session_key_PRIVATE.parquet", index=False)
    return {"sessions": len(sessions), "shared": len(shared)}


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
CREATE TABLE IF NOT EXISTS session_picks (
    expert_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    rank INTEGER NOT NULL,              -- 1 = most important; 0 = the expert would discuss nothing
    segment TEXT,
    type TEXT,
    comment TEXT,
    seconds REAL,
    saved_at TEXT NOT NULL,
    PRIMARY KEY (expert_id, session_key, rank)
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


def complete(responses: pd.DataFrame) -> pd.Series:
    """True for a window that needs no further answer.

    A window that the expert would not discuss is complete after the first answer: the questions on
    the assigned type and advice are only asked for windows the expert would discuss.
    """
    return (responses["relevant"] == 0) | responses["type_correct"].notna()


def save_relevance(review_dir: Path, expert_id: str, window_id: str, relevant: bool, seconds: float,
                   comment: str = "") -> None:
    """Store the first answer on a window as soon as it is given. An existing answer is never replaced.

    The assigned type and advice are shown after this answer. Storing it at once means that an expert who
    leaves the page and comes back cannot give the first answer again with that knowledge.
    """
    with connect(review_dir) as con:
        con.execute("INSERT OR IGNORE INTO responses (expert_id, window_id, relevant, comment, seconds, saved_at) "
                    "VALUES (?,?,?,?,?,?)", (expert_id, window_id, int(bool(relevant)), comment.strip(), float(seconds),
                                             time.strftime("%Y-%m-%dT%H:%M:%S")))


def has_answers(review_dir: Path) -> bool:
    """True when at least one expert answer is stored for the package in this folder."""
    if not (Path(review_dir) / "responses.db").exists():
        return False
    with connect(review_dir) as con:
        return any(con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() for table in ("responses", "session_picks"))


def save_session_picks(review_dir: Path, expert_id: str, session_key: str, picks: list[dict], seconds: float) -> None:
    """Store the corners an expert selected for a session. An empty list is stored as 'nothing to discuss'."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = [(expert_id, session_key, k + 1, p["segment"], p.get("type"), (p.get("comment") or "").strip(), float(seconds), now)
            for k, p in enumerate(picks)] or [(expert_id, session_key, 0, None, None, "", float(seconds), now)]
    with connect(review_dir) as con:
        con.execute("DELETE FROM session_picks WHERE expert_id = ? AND session_key = ?", (expert_id, session_key))
        con.executemany("INSERT INTO session_picks VALUES (?,?,?,?,?,?,?,?)", rows)


def load_session_picks(review_dir: Path) -> pd.DataFrame:
    with connect(review_dir) as con:
        return pd.read_sql_query("SELECT * FROM session_picks", con)


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


def krippendorff_alpha(wide: pd.DataFrame) -> float:
    """Krippendorff's alpha for nominal ratings (items x raters). Missing ratings are allowed."""
    values = sorted({v for v in wide.to_numpy().ravel() if not pd.isna(v)})
    index = {v: k for k, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for row in wide.to_numpy():
        rated = [index[v] for v in row if not pd.isna(v)]
        if len(rated) < 2:
            continue
        for a in range(len(rated)):
            for b in range(len(rated)):
                if a != b:
                    coincidence[rated[a], rated[b]] += 1.0 / (len(rated) - 1)
    total = coincidence.sum()
    if total <= 1:
        return float("nan")
    marginal = coincidence.sum(axis=0)
    observed = total - np.trace(coincidence)
    expected = (marginal.sum() ** 2 - (marginal ** 2).sum()) / (total - 1)
    return float("nan") if expected == 0 else float(1 - observed / expected)


def gwet_ac1(wide: pd.DataFrame) -> float:
    """Gwet's AC1 for two categories. Less sensitive than kappa to a skewed share of 'yes'."""
    complete = wide.dropna()
    if complete.shape[1] < 2 or complete.empty:
        return float("nan")
    r = complete.shape[1]
    yes = complete.sum(axis=1).to_numpy(float)
    agreement = ((yes * (yes - 1) + (r - yes) * (r - yes - 1)) / (r * (r - 1))).mean()
    share = yes.sum() / (r * len(complete))
    chance = 2 * share * (1 - share)
    return float("nan") if chance == 1 else float((agreement - chance) / (1 - chance))


def _mean_pairwise_kappa(wide: pd.DataFrame) -> float:
    values = []
    cols = list(wide.columns)
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            pair = wide[[cols[a], cols[b]]].dropna()
            if len(pair) > 1 and pair[cols[a]].nunique() > 1 and pair[cols[b]].nunique() > 1:
                values.append(cohen_kappa_score(pair[cols[a]].astype(int), pair[cols[b]].astype(int)))
    return float(np.mean(values)) if values else float("nan")


def _overlap(a: set, b: set) -> dict:
    return {"recall": len(a & b) / len(a) if a else np.nan, "precision": len(a & b) / len(b) if b else np.nan,
            "jaccard": len(a & b) / len(a | b) if a | b else np.nan}


def analyse_sessions(cfg: dict) -> dict | None:
    """Compare the corners that experts selected per session with those of the model and the baseline.

    recall     share of the corners an expert selected that the model also selected
    precision  share of the corners the model selected that the expert also selected
    The overlap between two experts on the shared sessions shows how much agreement is attainable.
    """
    paths = Paths.from_config(cfg)
    picks = load_session_picks(paths.review)
    key_path = paths.review / "session_key_PRIVATE.parquet"
    if picks.empty or not key_path.exists():
        return None
    key = pd.read_parquet(key_path)
    chosen = {(e, s): set(g.loc[g["rank"] > 0, "segment"]) for (e, s), g in picks.groupby(["expert_id", "session_key"])}
    rows = []
    for (expert, session), mine in chosen.items():
        for source in ("model", "baseline"):
            theirs = key[(key["session_key"] == session) & (key["source"] == source)]
            row = dict(expert_id=expert, session_key=session, source=source, n_expert=len(mine), **_overlap(mine, set(theirs["segment"])))
            if source == "model":
                both = picks[(picks["expert_id"] == expert) & (picks["session_key"] == session) & (picks["rank"] > 0)].merge(
                    theirs[["segment", "type"]], on="segment", suffixes=("_expert", "_model"))
                row["type_match"] = float((both["type_expert"] == both["type_model"]).mean()) if len(both) else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    pairs = []
    for session in {s for _, s in chosen}:
        raters = sorted(e for e, s in chosen if s == session)
        for a in range(len(raters)):
            for b in range(a + 1, len(raters)):
                pairs.append(_overlap(chosen[(raters[a], session)], chosen[(raters[b], session)])["jaccard"])
    summary = table.groupby("source")[["recall", "precision", "jaccard"]].mean()
    out = {"table": table, "summary": summary,
           "expert_expert_jaccard": float(np.nanmean(pairs)) if pairs else float("nan"),
           "type_match": float(table.loc[table["source"] == "model", "type_match"].mean())}
    table.to_csv(paths.results / "sq4_session_task.csv", index=False)
    summary.assign(expert_expert_jaccard=out["expert_expert_jaccard"]).to_csv(paths.results / "sq4_session_summary.csv")
    return out


def analyse(cfg: dict) -> dict:
    """Agreement among experts and between experts and the model, and the reference labels."""
    paths = Paths.from_config(cfg).ensure()
    resp = load_responses(paths.review)
    if resp.empty:
        raise RuntimeError("No expert responses found.")
    key = pd.read_parquet(paths.review / "key_PRIVATE.parquet")
    items = pd.read_parquet(paths.review / "items.parquet")
    assignments = pd.read_parquet(paths.review / "assignments.parquet")
    out = {}

    # Windows that an expert saw twice: the second answer only serves to measure the consistency of the expert.
    again = assignments.dropna(subset=["repeat_of"]) if "repeat_of" in assignments else assignments.iloc[:0]
    second = resp.merge(again[["expert_id", "window_id", "repeat_of"]], on=["expert_id", "window_id"])
    resp = resp[~resp["window_id"].isin(again["window_id"])]
    pairs = second.merge(resp[["expert_id", "window_id", "relevant"]], left_on=["expert_id", "repeat_of"],
                         right_on=["expert_id", "window_id"], suffixes=("_second", "_first"))
    if len(pairs):
        same = pairs["relevant_first"] == pairs["relevant_second"]
        out["repeated_windows_answered"] = int(len(pairs))
        out["intra_expert_agreement"] = float(same.mean())
        varied = pairs["relevant_first"].nunique() > 1 and pairs["relevant_second"].nunique() > 1
        out["intra_expert_kappa"] = float(cohen_kappa_score(pairs["relevant_first"], pairs["relevant_second"])) if varied else float("nan")
        pairs.assign(same=same).groupby("expert_id").agg(n=("same", "size"), agreement=("same", "mean")).reset_index().to_csv(
            paths.results / "sq4_intra_expert.csv", index=False)

    resp = resp.merge(key, on="window_id").merge(items[["window_id", "shown_type"]], on="window_id")

    # Agreement among the experts on the shared windows that all of them completed.
    shared_ids = assignments.loc[assignments["shared"], "window_id"].unique()
    shared = resp[resp["window_id"].isin(shared_ids)]
    wide = shared.pivot(index="window_id", columns="expert_id", values="relevant").dropna()
    if wide.shape[1] >= 2 and len(wide):
        counts = np.column_stack([(wide == 0).sum(axis=1), (wide == 1).sum(axis=1)])
        out["fleiss_kappa_relevance"] = fleiss_kappa(counts)
        # Interval from resampling the shared windows.
        rng = np.random.default_rng(0)
        boot = [fleiss_kappa(counts[rng.integers(0, len(counts), len(counts))]) for _ in range(1000)]
        out["fleiss_kappa_ci_low"], out["fleiss_kappa_ci_high"] = (float(v) for v in np.nanpercentile(boot, [2.5, 97.5]))
        out["krippendorff_alpha_relevance"] = krippendorff_alpha(wide)
        out["gwet_ac1_relevance"] = gwet_ac1(wide)
        out["mean_pairwise_kappa_experts"] = _mean_pairwise_kappa(wide)      # the attainable level for the model
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
                          "advice_ok": g["advice_ok"].mean(), "tie": float(g["relevant"].mean() == 0.5)})
    labels = resp.groupby("window_id").apply(reference, include_groups=False).reset_index()
    if "weight" in key:
        labels = labels.merge(key[["window_id", "weight", "drawn_from"]], on="window_id", how="left")
    labels.to_parquet(paths.review / "expert_labels.parquet", index=False)
    several = labels[labels["n_experts"] > 1]
    out["windows_with_several_experts"], out["ties_counted_as_not_relevant"] = int(len(several)), int(several["tie"].sum())

    joined = labels.drop(columns=[c for c in ("weight", "drawn_from") if c in labels]).merge(key, on="window_id").merge(
        items[["window_id", "shown_type"]], on="window_id")
    by_group = joined.groupby("group").agg(n=("expert_label", "size"), relevant=("expert_label", "mean"),
                                           advice_ok=("advice_ok", "mean")).reset_index()
    positives = joined[joined["expert_label"] == 1]
    out["type_agreement"] = float((positives["shown_type"] == positives["expert_type"]).mean()) if len(positives) else np.nan
    if "weight" in joined:
        # Share of all corner windows that the panel would discuss, estimated from the weighted sample.
        out["relevant_share_all_windows"] = float(np.average(joined["expert_label"], weights=joined["weight"]))
        random_part = joined[joined["drawn_from"] == "random"]
        out["relevant_share_random_draw"] = float(random_part["expert_label"].mean()) if len(random_part) else np.nan
    # The windows of a selection were not all drawn with the same probability (a window selected by both the
    # model and the baseline is drawn as a model window), so the precision is weighted by the inverse probability.
    def precision(mask: pd.Series) -> float:
        part = joined[mask]
        if part.empty:
            return float("nan")
        return float(np.average(part["expert_label"], weights=part["weight"])) if "weight" in part else float(part["expert_label"].mean())
    out["precision_model_selection"] = precision(joined["in_model"])
    out["precision_baseline_selection"] = precision(joined["in_baseline"])
    out["n_model_selection"], out["n_baseline_selection"] = int(joined["in_model"].sum()), int(joined["in_baseline"].sum())

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
    out["mean_kappa_model_vs_expert"] = float(per_expert["cohen_kappa_model"].mean())
    out["mean_kappa_baseline_vs_expert"] = float(per_expert["cohen_kappa_baseline"].mean())
    pd.Series(out).to_csv(paths.results / "sq4_summary.csv", header=["value"])
    sessions = analyse_sessions(cfg)
    if sessions is not None:
        out.update(session_task=sessions["summary"], session_expert_expert_jaccard=sessions["expert_expert_jaccard"],
                   session_type_match=sessions["type_match"])
    out.update(per_expert=per_expert, by_group=by_group, labelling_functions=lf)
    return out
