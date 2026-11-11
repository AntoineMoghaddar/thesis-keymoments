"""Windows and their features.

A window is one segment of one lap. Each window is described twice: by engineered features
that compare it with the reference lap of the same driver, and by the resampled signals
themselves (used by the sequence model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Paths
from .preprocess import CH, load_prepared, reference_index
from .segments import get_segments

# Groups are used for the ablation and for reading the explanations of the model.
FEATURE_GROUPS = {
    "time": ["time_loss", "exit_loss", "seg_time", "dtw_speed"],
    "braking": ["d_brake_onset_m", "peak_brake", "d_peak_brake", "brake_rise_s", "brake_release_s",
                "d_brake_release_s", "d_brake_len_m", "abs_frac", "entry_speed", "d_entry_speed"],
    "speed": ["min_speed", "d_min_speed", "min_speed_ratio"],
    "throttle": ["d_throttle_pickup_m", "d_full_throttle_m", "throttle_lifts", "d_throttle_lifts",
                 "exit_speed", "d_exit_speed"],
    "traction": ["slip_exit_max", "d_slip_exit_max", "tc_frac", "yaw_rate_max"],
    "steering_line": ["steer_integral", "d_steer_integral", "steer_reversals", "d_steer_reversals",
                      "line_dev_max", "line_dev_mean"],
    "incident": ["tyres_out_max"],
    "consistency": ["time_cv_ratio"],
}
TYPE_GROUP = {"K1": "braking", "K2": "braking", "K3": "speed", "K4": "throttle", "K5": "traction",
              "K6": "steering_line", "K7": "incident", "K8": "consistency"}

# Features given to the tree model. Identifiers and absolute positions are left out on purpose.
MODEL_FEATURES = [
    "time_loss", "exit_loss", "seg_time", "entry_speed", "min_speed", "exit_speed",
    "d_entry_speed", "d_min_speed", "d_exit_speed", "min_speed_ratio",
    "d_brake_onset_m", "peak_brake", "d_peak_brake", "brake_rise_s", "brake_release_s",
    "d_brake_release_s", "d_brake_len_m",
    "d_throttle_pickup_m", "d_full_throttle_m", "throttle_lifts", "d_throttle_lifts",
    "steer_integral", "d_steer_integral", "steer_reversals", "d_steer_reversals",
    "slip_exit_max", "d_slip_exit_max", "tc_frac", "abs_frac",
    "line_dev_max", "line_dev_mean", "tyres_out_max", "yaw_rate_max", "dtw_speed",
    "time_cv_ratio", "is_corner",
]

SEQ_CHANNELS = ["speed", "gas", "brake", "steer", "slip_rear", "ref_speed", "ref_gas", "ref_brake",
                "ref_steer", "time_delta"]


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 6) -> float:
    """Dynamic time warping distance with a Sakoe-Chiba band, normalised by path length."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost[i, j] = abs(a[i - 1] - b[j - 1]) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def _count_lifts(gas: np.ndarray, drop: float = 0.1) -> int:
    """Number of times the throttle is released by more than `drop` after it was applied."""
    lifts, peak, lifted = 0, 0.0, False
    for g in gas:
        if g > peak:
            peak, lifted = g, False
        elif not lifted and peak - g > drop:
            lifts, lifted = lifts + 1, True
            peak = g
        elif lifted and g < peak:
            peak = g
    return lifts


def _reversals(steer: np.ndarray, amplitude: float = 0.02) -> int:
    """Number of direction changes of the steering wheel larger than `amplitude`."""
    count, direction, anchor = 0, 0, steer[0]
    for s in steer[1:]:
        if direction >= 0 and s < anchor - amplitude:
            count += direction > 0
            direction, anchor = -1, s
        elif direction <= 0 and s > anchor + amplitude:
            count += direction < 0
            direction, anchor = 1, s
        elif (direction > 0 and s > anchor) or (direction < 0 and s < anchor):
            anchor = s
    return int(count)


def _phase(lap: np.ndarray, a: int, b: int, lo: int, hi: int, ds: float, fc: dict) -> dict:
    """Measurements of one lap in one segment. Positions are in metres from the segment start."""
    n = len(lap)
    t = lap[:, CH["lap_t"]]
    speed, gas, brake, steer = (lap[:, CH[c]] for c in ("speed", "gas", "brake", "steer"))
    seg = slice(a, b)
    i_min = a + int(np.argmin(speed[seg]))
    out = dict(entry_speed=speed[a], exit_speed=speed[min(b, n) - 1], min_speed=speed[i_min],
               min_speed_m=(i_min - a) * ds)

    onset = _first(brake[lo:i_min + 1] > fc["brake_on"])
    if onset is not None:
        onset += lo
        zone = brake[onset:b]
        peak = float(zone.max())
        released = _first(zone < fc["brake_on"])
        end = onset + (released if released is not None else len(zone) - 1)
        high = np.flatnonzero(brake[onset:end + 1] >= 0.9 * peak) + onset
        out.update(brake_onset_m=(onset - a) * ds, peak_brake=peak,
                   brake_rise_s=t[high[0]] - t[onset], brake_release_s=t[end] - t[high[-1]],
                   brake_len_m=(end - onset) * ds)
    pickup = _first(gas[i_min:hi] > fc["gas_on"])
    if pickup is not None:
        pickup += i_min
        out["throttle_pickup_m"] = (pickup - a) * ds
        full = _first(gas[pickup:hi] >= fc["gas_full"])
        if full is not None:
            out["full_throttle_m"] = (pickup + full - a) * ds
        out["throttle_lifts"] = _count_lifts(gas[pickup:hi])
    out["steer_integral"] = float(np.sum(np.abs(steer[seg])) * ds)
    out["steer_reversals"] = _reversals(steer[seg])
    out["slip_exit_max"] = float(lap[i_min:b, CH["slip_rear"]].max())
    out["tc_frac"] = float(lap[i_min:b, CH["tc_on"]].mean())
    out["abs_frac"] = float(lap[a:i_min + 1, CH["abs_on"]].mean())
    out["tyres_out_max"] = float(lap[seg, CH["tyres_out"]].max())
    out["yaw_rate_max"] = float(np.abs(lap[seg, CH["yaw_rate"]]).max())
    return out


def _seg_time(lap: np.ndarray, lap_time: float, a: int, b: int) -> float:
    t = lap[:, CH["lap_t"]]
    return float((lap_time if b >= len(t) else t[b]) - t[a])


DELTAS = {  # feature name: (measurement, sign) with sign +1 for lap minus reference
    "d_entry_speed": ("entry_speed", 1), "d_min_speed": ("min_speed", 1), "d_exit_speed": ("exit_speed", 1),
    "d_brake_onset_m": ("brake_onset_m", -1),      # positive: braking started earlier than the reference
    "d_peak_brake": ("peak_brake", 1), "d_brake_release_s": ("brake_release_s", 1),
    "d_brake_len_m": ("brake_len_m", 1),
    "d_throttle_pickup_m": ("throttle_pickup_m", 1),   # positive: throttle applied later than the reference
    "d_full_throttle_m": ("full_throttle_m", 1), "d_throttle_lifts": ("throttle_lifts", 1),
    "d_steer_integral": ("steer_integral", 1), "d_steer_reversals": ("steer_reversals", 1),
    "d_slip_exit_max": ("slip_exit_max", 1),
}


def window_features(lap: np.ndarray, ref: np.ndarray, lap_time: float, ref_time: float, seg: dict,
                    bounds: tuple[int, int], ds: float, cfg: dict) -> dict:
    fc = cfg["features"]
    a, b = seg["start_idx"], seg["end_idx"]
    lo, hi = bounds
    cur, base = _phase(lap, a, b, lo, hi, ds, fc), _phase(ref, a, b, lo, hi, ds, fc)
    out = {k: v for k, v in cur.items()}
    out.update({f"ref_{k}": base.get(k, np.nan) for k in
                ("min_speed", "brake_onset_m", "peak_brake", "throttle_pickup_m", "steer_integral")})
    for name, (key, sign) in DELTAS.items():
        out[name] = sign * (cur.get(key, np.nan) - base.get(key, np.nan))
    out["seg_time"] = _seg_time(lap, lap_time, a, b)
    out["ref_seg_time"] = _seg_time(ref, ref_time, a, b)
    out["time_loss"] = out["seg_time"] - out["ref_seg_time"]
    out["min_speed_ratio"] = cur["min_speed"] / max(base["min_speed"], 1e-6)
    dev = np.hypot(lap[a:b, CH["x"]] - ref[a:b, CH["x"]], lap[a:b, CH["z"]] - ref[a:b, CH["z"]])
    out["line_dev_max"], out["line_dev_mean"] = float(dev.max()), float(dev.mean())
    pts = np.linspace(a, b - 1, fc["dtw_points"]).astype(int)
    scale = max(float(ref[pts, CH["speed"]].mean()), 1e-6)
    out["dtw_speed"] = dtw_distance(lap[pts, CH["speed"]] / scale, ref[pts, CH["speed"]] / scale)
    return out


REFERENCES = ("own_best", "ideal", "expert")


def _reference_rows(laps: pd.DataFrame, data: np.ndarray, rows: np.ndarray, segs: list[dict], mode: str) -> np.ndarray:
    """Row number of the reference lap for every lap and segment of one circuit.

    own_best  the fastest clean lap of the same driver
    ideal     per segment the lap in which the same driver was fastest in that segment
    expert    the fastest clean lap of another driver with a high experience level
    """
    own = reference_index(laps)[rows]
    out = np.repeat(own[:, None], len(segs), axis=1)
    if mode == "own_best":
        return out
    drivers = laps["driver"].to_numpy()[rows]
    if mode == "expert":
        sub = laps.iloc[rows]
        pool = sub[(sub["level"] == "high") & sub["clean"]]
        for k, driver in enumerate(drivers):
            others = pool[pool["driver"] != driver]
            if len(others):                       # without another experienced driver the own best lap stays
                out[k, :] = laps.index.get_loc(others["lap_time"].idxmin())
        return out
    if mode == "ideal":
        for s_i, seg in enumerate(segs):
            a, b = seg["start_idx"], seg["end_idx"]
            times = np.array([_seg_time(data[i], laps.iloc[i]["lap_time"], a, b) for i in rows])
            off = np.array([data[i][a:b, CH["tyres_out"]].max() >= 3 for i in rows])
            for driver in np.unique(drivers):
                mine = np.flatnonzero(drivers == driver)
                ok = mine[~off[mine]] if (~off[mine]).any() else mine
                out[mine, s_i] = rows[ok[np.argmin(times[ok])]]
        return out
    raise ValueError(f"Unknown reference: {mode}")


def windows_path(cfg: dict, reference: str | None = None):
    mode = reference or cfg["features"]["reference"]
    name = "windows.parquet" if mode == cfg["features"]["reference"] else f"windows_{mode}.parquet"
    return Paths.from_config(cfg).prepared / name


def build_windows(cfg: dict, force_segments: bool = False, reference: str | None = None) -> pd.DataFrame:
    """Compute the feature table for all laps and segments and store it."""
    paths = Paths.from_config(cfg).ensure()
    mode = reference or cfg["features"]["reference"]
    laps, data, circuits = load_prepared(cfg)
    laps = laps.reset_index(drop=True)
    n = data.shape[1]
    look = int(cfg["features"]["lookaround"] * n)
    rows = []
    for circuit, info in circuits.items():
        segs = get_segments(laps, data, circuit, cfg, paths.segments, force=force_segments)
        if not cfg["windows"]["include_straights"]:
            segs = [s for s in segs if s["kind"] == "corner"]
        ds = info["length_m"] / n
        corners = [s for s in segs if s["kind"] == "corner"]
        lap_rows = np.where(laps["circuit"] == circuit)[0]
        refs = _reference_rows(laps, data, lap_rows, segs, mode)
        for k, i in enumerate(lap_rows):
            lap_row = laps.iloc[i]
            for s_i, seg in enumerate(segs):
                j = int(refs[k, s_i])
                ref_time = laps.iloc[j]["lap_time"]
                a, b = seg["start_idx"], seg["end_idx"]
                prev_end = max([c["end_idx"] for c in corners if c["end_idx"] <= a], default=0)
                next_start = min([c["start_idx"] for c in corners if c["start_idx"] >= b], default=n)
                bounds = (max(prev_end, a - look), min(next_start, b + look))
                feats = window_features(data[i], data[j], lap_row["lap_time"], ref_time, seg, bounds, ds, cfg)
                # Time lost on the straight after a corner is the price of a poor exit.
                exit_loss = 0.0
                if seg["kind"] == "corner" and next_start > b:
                    exit_loss = (_seg_time(data[i], lap_row["lap_time"], b, next_start)
                                 - _seg_time(data[j], ref_time, b, next_start))
                rows.append(dict(window_id=f"{lap_row['lap_uid']}_{seg['id']}", lap_uid=lap_row["lap_uid"],
                                 lap_row=int(i), ref_row=j, session_id=lap_row["session_id"],
                                 driver=lap_row["driver"], level=lap_row["level"], circuit=circuit,
                                 lap=int(lap_row["lap"]), segment=seg["id"], segment_name=seg["name"],
                                 is_corner=int(seg["kind"] == "corner"), start_idx=a, end_idx=b,
                                 is_reference=bool(i == j), reference=mode, exit_loss=exit_loss, **feats))
    w = pd.DataFrame(rows)
    w["total_loss"] = w["time_loss"] + w["exit_loss"]
    # Consistency of a driver in a segment within a session, relative to his or her other segments.
    cv = w.groupby(["session_id", "segment"])["seg_time"].transform(lambda x: x.std(ddof=0) / x.mean())
    w["time_cv"] = cv
    w["time_cv_ratio"] = cv / w.groupby("session_id")["time_cv"].transform("median").clip(lower=1e-9)
    w["session_median_time"] = w.groupby(["session_id", "segment"])["seg_time"].transform("median")
    w.to_parquet(windows_path(cfg, mode), index=False)
    return w


def load_windows(cfg: dict, reference: str | None = None) -> pd.DataFrame:
    path = windows_path(cfg, reference)
    return pd.read_parquet(path) if path.exists() else build_windows(cfg, reference=reference)


def raw_windows(windows: pd.DataFrame, data: np.ndarray, length: int) -> np.ndarray:
    """Signals of every window resampled to a fixed length: array (windows, channels, length)."""
    out = np.zeros((len(windows), len(SEQ_CHANNELS), length), dtype=np.float32)
    for k, r in enumerate(windows[["lap_row", "ref_row", "start_idx", "end_idx"]].itertuples(index=False)):
        idx = np.linspace(r.start_idx, r.end_idx - 1, length).astype(int)
        lap, ref = data[r.lap_row][idx], data[r.ref_row][idx]
        delta = (lap[:, CH["lap_t"]] - lap[0, CH["lap_t"]]) - (ref[:, CH["lap_t"]] - ref[0, CH["lap_t"]])
        out[k] = np.stack([lap[:, CH["speed"]] / 300.0, lap[:, CH["gas"]], lap[:, CH["brake"]],
                           lap[:, CH["steer"]], np.clip(lap[:, CH["slip_rear"]], 0, 3),
                           ref[:, CH["speed"]] / 300.0, ref[:, CH["gas"]], ref[:, CH["brake"]],
                           ref[:, CH["steer"]], delta])
    return out
