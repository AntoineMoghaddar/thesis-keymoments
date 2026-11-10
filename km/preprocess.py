"""Data preparation: lap selection and resampling on track position."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Paths
from .schema import list_sessions

# Channels kept after resampling. The order is the order of the last axis of the array.
CHANNELS = ["lap_t", "speed", "gas", "brake", "steer", "slip_rear", "slip_front", "tc_on", "abs_on",
            "tyres_out", "yaw_rate", "acc_lat", "acc_lon", "x", "z"]
CH = {name: i for i, name in enumerate(CHANNELS)}


def _lap_position(pos: np.ndarray) -> np.ndarray:
    """Unwrap the position at both ends of a lap.

    The lap counter and the track position do not switch on exactly the same sample, so the
    first samples of a lap can still read 0.99 and the last ones can already read 0.00.
    """
    pos = pos.astype(float).copy()
    half = len(pos) // 2
    head, tail = pos[:half], pos[half:]
    head[head > 0.8] -= 1.0
    tail[tail < 0.2] += 1.0
    return pos


def unique_laps(frame: pd.DataFrame) -> pd.DataFrame:
    """Make lap numbers unique when the lap counter of the simulator was reset during a recording.

    This happens when the session is restarted in the game. Laps after a reset are numbered
    from 1000, 2000 and so on, and the first lap after a reset is treated like an out lap.
    """
    resets = (frame["lap"].diff() < 0).cumsum()
    if resets.max() == 0:
        return frame
    return frame.assign(lap=frame["lap"] + 1000 * resets)


def split_laps(frame: pd.DataFrame, cfg: dict) -> list[dict]:
    """Describe every lap of a session and decide whether it can be used."""
    frame = unique_laps(frame)
    laps = []
    numbers = sorted(frame["lap"].unique())
    groups = {n: g for n, g in frame.groupby("lap", sort=True)}
    for n in numbers:
        g = groups[n]
        pos = _lap_position(g["norm_pos"].to_numpy())
        t = g["t"].to_numpy()
        nxt = groups.get(n + 1)
        reasons = []
        if n == numbers[0] or (n - 1) not in groups:
            reasons.append("first lap of the recording")
        if nxt is None:
            reasons.append("not completed")
        if pos.min() > 0.02 or pos.max() < 0.98:
            reasons.append("incomplete")
        if g["in_pit"].max() > 0 or g["in_pit_lane"].max() > 0:
            reasons.append("pit")
        if len(t) > 1 and np.diff(t).max() > cfg["preprocess"]["max_gap_s"]:
            reasons.append("gap in samples")
        if nxt is not None:
            clock_time = float(nxt["t"].iloc[0] - t[0])
            game_ms = int(nxt["last_time_ms"].iloc[min(5, len(nxt) - 1)])
            # The lap time of the simulator is exact. The clock of the logger is the fallback.
            lap_time = game_ms / 1000.0 if game_ms > 0 and abs(game_ms / 1000.0 - clock_time) < 1.0 else clock_time
        else:
            clock_time = lap_time = float(t[-1] - t[0])
        laps.append(dict(lap=int(n), lap_time=lap_time, clock_time=clock_time, n_samples=len(g),
                         tyres_out_max=int(g["tyres_out"].max()),
                         game_valid=bool(g["valid_lap"].iloc[-1] > 0),
                         usable=not reasons, reason="; ".join(reasons)))
    usable = [lap for lap in laps if lap["usable"]]
    for lap in usable[: cfg["preprocess"]["warmup_laps"]]:
        lap["usable"], lap["reason"] = False, "warm-up"
    return laps


def resample_lap(g: pd.DataFrame, lap_time: float, n_points: int) -> tuple[np.ndarray, float]:
    """Interpolate one lap onto a fixed grid of track positions.

    Returns the array (n_points, channels) and the distance driven in metres.
    """
    pos = _lap_position(g["norm_pos"].to_numpy())
    keep = np.concatenate([[True], pos[1:] > np.maximum.accumulate(pos)[:-1]])
    pos = pos[keep]
    g = g.loc[keep]
    t = g["t"].to_numpy()
    grid = np.arange(n_points) / n_points
    speed = g["speed_kmh"].to_numpy(float)
    source = {
        "lap_t": t - t[0],
        "speed": speed,
        "gas": g["gas"], "brake": g["brake"], "steer": g["steer"],
        "slip_rear": np.maximum(g["wslip_rl"].abs(), g["wslip_rr"].abs()),
        "slip_front": np.maximum(g["wslip_fl"].abs(), g["wslip_fr"].abs()),
        "tc_on": g["tc_on"], "abs_on": g["abs_on"], "tyres_out": g["tyres_out"],
        "yaw_rate": g["yaw_rate"], "acc_lat": g["acc_x"], "acc_lon": g["acc_z"],
        "x": g["x"], "z": g["z"],
    }
    out = np.column_stack([np.interp(grid, pos, np.asarray(source[c], dtype=float)) for c in CHANNELS])
    # Scale the clock so that the lap ends exactly at the lap time of the simulator.
    span = t[-1] - t[0]
    if span > 0:
        out[:, CH["lap_t"]] *= min(1.0, lap_time / span) if lap_time < span else 1.0
    distance = float(np.sum(speed[:-1] / 3.6 * np.diff(t)))
    return out, distance


def prepare(cfg: dict) -> pd.DataFrame:
    """Read all raw sessions, keep the usable laps and store them resampled."""
    paths = Paths.from_config(cfg).ensure()
    n_points = cfg["preprocess"]["grid_points"]
    levels = {}
    for p in paths.participants.glob("*.json"):
        info = json.loads(p.read_text(encoding="utf-8"))
        levels[info.get("driver", p.stem)] = info.get("experience_level", "unknown")

    rows, arrays, log = [], [], []
    for meta in list_sessions(paths.raw):
        frame = unique_laps(pd.read_parquet(meta["path"]))
        for lap in split_laps(frame, cfg):
            base = dict(session_id=meta["session_id"], driver=meta["driver"], circuit=meta["circuit"],
                        session_number=meta.get("session_number", 0), **lap)
            log.append(base)
            if not lap["usable"]:
                continue
            arr, distance = resample_lap(frame[frame["lap"] == lap["lap"]], lap["lap_time"], n_points)
            rows.append(dict(base, lap_uid=f"{meta['session_id']}_L{lap['lap']:02d}", distance_m=distance,
                             level=levels.get(meta["driver"], "unknown")))
            arrays.append(arr)
    if not rows:
        raise RuntimeError(f"No usable laps found below {paths.raw}")

    laps = pd.DataFrame(rows).drop(columns=["usable", "reason"])
    # Reference lap: the fastest lap of the same driver on the same circuit, preferably a clean one.
    laps["clean"] = laps["game_valid"] & (laps["tyres_out_max"] < 3)
    laps["is_reference"] = False
    for _, g in laps.groupby(["driver", "circuit"]):
        pool = g[g["clean"]] if g["clean"].any() else g
        laps.loc[pool["lap_time"].idxmin(), "is_reference"] = True
    circuits = {c: {"length_m": float(g["distance_m"].median()), "laps": int(len(g))}
                for c, g in laps.groupby("circuit")}

    laps.to_parquet(paths.prepared / "laps.parquet", index=False)
    np.save(paths.prepared / "resampled.npy", np.stack(arrays).astype(np.float32))
    pd.DataFrame(log).to_csv(paths.prepared / "lap_log.csv", index=False)
    (paths.prepared / "circuits.json").write_text(json.dumps(circuits, indent=2), encoding="utf-8")
    return laps


def load_prepared(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Laps, the resampled array (laps, grid points, channels) and the circuit information."""
    prepared = Paths.from_config(cfg).prepared
    laps = pd.read_parquet(prepared / "laps.parquet")
    data = np.load(prepared / "resampled.npy")
    circuits = json.loads((prepared / "circuits.json").read_text(encoding="utf-8"))
    return laps, data, circuits


def reference_index(laps: pd.DataFrame) -> np.ndarray:
    """For every lap the row number of its reference lap."""
    ref = {(r.driver, r.circuit): i for i, r in enumerate(laps.itertuples()) if r.is_reference}
    return np.array([ref[(r.driver, r.circuit)] for r in laps.itertuples()])
