"""Track edges from survey laps.

The simulator does not publish the edges of the track. They are measured once per circuit by
the researcher: one recording driven along one edge of the track and one along the other,
stored under the survey pseudonym. The laps of a recording are averaged, and the edge is placed
half a car width outside the driven line, because the recorded position is the centre of the car.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .preprocess import _lap_position, unique_laps
from .schema import list_sessions

CAR_HALF_WIDTH_M = 1.0


def _lines(frame: pd.DataFrame, n_points: int) -> np.ndarray:
    """Position (x, z) of every complete lap of a recording on the grid of track positions."""
    frame = unique_laps(frame)
    grid = np.arange(n_points) / n_points
    out = []
    for _, g in frame.groupby("lap"):
        pos = _lap_position(g["norm_pos"].to_numpy())
        if len(g) < 100 or pos.min() > 0.01 or pos.max() < 0.99:
            continue
        keep = np.concatenate([[True], pos[1:] > np.maximum.accumulate(pos)[:-1]])
        out.append(np.column_stack([np.interp(grid, pos[keep], g["x"].to_numpy(float)[keep]),
                                    np.interp(grid, pos[keep], g["z"].to_numpy(float)[keep])]))
    return np.array(out)


def edges_from_lines(one: np.ndarray, other: np.ndarray) -> pd.DataFrame:
    """Two edges from the laps driven along each side: arrays (laps, grid points, 2)."""
    a, b = np.median(one, axis=0), np.median(other, axis=0)
    across = a - b
    width = np.hypot(across[:, 0], across[:, 1])
    unit = across / np.where(width > 0, width, 1.0)[:, None]
    a, b = a + unit * CAR_HALF_WIDTH_M, b - unit * CAR_HALF_WIDTH_M
    return pd.DataFrame({"idx": np.arange(len(a)), "a_x": a[:, 0], "a_z": a[:, 1], "b_x": b[:, 0], "b_z": b[:, 1],
                         "width_m": width + 2 * CAR_HALF_WIDTH_M})


def reference(cfg: dict, circuit: str) -> dict:
    """Width and length of the circuit according to an independent source, when one is configured.

    The source describes the real circuit and has its own coordinates, so only summary values are
    compared with the survey: they tell whether the measured width is plausible.
    """
    name = cfg.get("track", {}).get("reference", {}).get(circuit)
    path = Path(__file__).resolve().parent.parent / "configs" / "reference_tracks" / f"{name}.csv"
    if not name or not path.exists():
        return {}
    ref = pd.read_csv(path, comment="#", header=None, names=["x", "y", "right", "left"])
    width = ref["right"] + ref["left"]
    x, y = np.append(ref["x"], ref["x"].iloc[0]), np.append(ref["y"], ref["y"].iloc[0])
    return dict(ref_width_median=round(float(width.median()), 1), ref_width_min=round(float(width.min()), 1),
                ref_width_max=round(float(width.max()), 1), ref_length_m=round(float(np.hypot(np.diff(x), np.diff(y)).sum())))


def survey(cfg: dict) -> pd.DataFrame:
    """Build the edges of every circuit that has two survey recordings and store them."""
    paths = Paths.from_config(cfg).ensure()
    who = cfg["study"]["survey_pseudonym"]
    sessions = [m for m in list_sessions(paths.raw) if m["driver"] == who]
    parts, summary = [], []
    for circuit in sorted({m["circuit"] for m in sessions}):
        mine = sorted((m for m in sessions if m["circuit"] == circuit), key=lambda m: m["session_id"])
        if len(mine) < 2:
            summary.append(dict(circuit=circuit, status="needs two recordings, one per edge", **reference(cfg, circuit)))
            continue
        lines = [_lines(pd.read_parquet(m["path"]), cfg["preprocess"]["grid_points"]) for m in mine[-2:]]
        if min(len(x) for x in lines) == 0:
            summary.append(dict(circuit=circuit, status="a recording holds no complete lap"))
            continue
        edges = edges_from_lines(*lines).assign(circuit=circuit)
        parts.append(edges)
        w = edges["width_m"]
        summary.append(dict(circuit=circuit, status="ok", laps_first=len(lines[0]), laps_second=len(lines[1]),
                            width_median=round(float(w.median()), 1), width_min=round(float(w.min()), 1),
                            width_max=round(float(w.max()), 1), **reference(cfg, circuit)))
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(paths.prepared / "track_edges.parquet", index=False)
    return pd.DataFrame(summary)


def load_edges(cfg: dict) -> pd.DataFrame | None:
    path = Paths.from_config(cfg).prepared / "track_edges.parquet"
    return pd.read_parquet(path) if path.exists() else None
