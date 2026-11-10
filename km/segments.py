"""Track segmentation into corners and straights.

Corner boundaries are proposed from the driving data and written to a YAML file per circuit.
That file is the source of truth afterwards: check it against a track map, correct it by hand
where needed and give the corners their usual names. It is not overwritten unless asked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .preprocess import CH


def _runs(mask: np.ndarray) -> list[list[int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return [[a, b] for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def detect(profile: np.ndarray, cfg: dict) -> list[dict]:
    """Propose corner segments from a median profile of the reference laps.

    A corner starts where braking or steering starts and ends where the throttle is fully
    open again after the steering has returned to the centre.
    """
    s = cfg["segments"]
    n = len(profile)
    brake, steer, gas = profile[:, CH["brake"]], np.abs(profile[:, CH["steer"]]), profile[:, CH["gas"]]
    runs = _runs((brake > s["brake_threshold"]) | (steer > s["steer_threshold"]))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < s["merge_gap"] * n:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [r for r in merged if r[1] - r[0] >= s["min_length"] * n]
    margin = int(s["margin"] * n)
    out = []
    for k, (a, b) in enumerate(merged):
        limit = merged[k + 1][0] - margin - 1 if k + 1 < len(merged) else n
        end = b
        while end < limit and gas[min(end, n - 1)] < cfg["features"]["gas_full"]:
            end += 1
        start = max(a - margin, out[-1]["end_idx"] if out else 0)
        end = min(end + margin, limit, n)
        out.append(dict(id=f"T{k + 1}", name=f"Turn {k + 1}", kind="corner", start_idx=start, end_idx=end))
    return out


def with_straights(corners: list[dict], n: int, min_len: int) -> list[dict]:
    """Fill the space between corners with straight segments."""
    out, cursor, k = [], 0, 0
    for c in corners + [None]:
        stop = c["start_idx"] if c else n
        if stop - cursor >= min_len:
            k += 1
            out.append(dict(id=f"S{k}", name=f"Straight {k}", kind="straight", start_idx=cursor, end_idx=stop))
        if c:
            out.append(c)
            cursor = c["end_idx"]
    return out


def propose(laps: pd.DataFrame, data: np.ndarray, circuit: str, cfg: dict) -> list[dict]:
    refs = np.where((laps["circuit"] == circuit) & laps["is_reference"])[0]
    profile = np.median(data[refs], axis=0)
    profile[:, CH["steer"]] = np.median(np.abs(data[refs][:, :, CH["steer"]]), axis=0)
    n = data.shape[1]
    corners = detect(profile, cfg)
    return with_straights(corners, n, int(cfg["segments"]["min_length"] * n))


def save(path: Path, circuit: str, segments: list[dict], n: int) -> None:
    doc = {
        "circuit": circuit,
        "note": "Proposed from data. Check against a track map and correct by hand if needed.",
        "segments": [dict(id=s["id"], name=s["name"], kind=s["kind"],
                          start=float(round(s["start_idx"] / n, 5)), end=float(round(s["end_idx"] / n, 5))) for s in segments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def load(path: Path, n: int) -> list[dict]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for s in doc["segments"]:
        out.append(dict(id=s["id"], name=s.get("name", s["id"]), kind=s.get("kind", "corner"),
                        start_idx=int(round(s["start"] * n)), end_idx=int(round(s["end"] * n))))
    return out


def get_segments(laps: pd.DataFrame, data: np.ndarray, circuit: str, cfg: dict, seg_dir: Path,
                 force: bool = False) -> list[dict]:
    """Load the segment file of a circuit, or create it from the data when it does not exist."""
    path = Path(seg_dir) / f"{circuit}.yaml"
    n = data.shape[1]
    if force or not path.exists():
        save(path, circuit, propose(laps, data, circuit, cfg), n)
    return load(path, n)
