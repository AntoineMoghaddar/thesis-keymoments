"""Synthetic telemetry in the format of the logger.

The generator exists for one reason: the pipeline has to be developed and tested before any
driver has been recorded. The tracks are invented and carry the names synth_a and synth_b so
that they cannot be mistaken for real circuits. Nothing produced here is research data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RAW_COLUMNS, session_id, write_session

VMAX = 72.0          # m/s
BRAKE_DECEL = 13.0   # m/s^2
ACCEL = 4.5          # m/s^2
HZ = 60

SKILL = {
    "high": dict(pace=1.00, noise=0.004, shift_noise=2.0, p_error=0.06),
    "moderate": dict(pace=0.975, noise=0.010, shift_noise=4.0, p_error=0.14),
    "none": dict(pace=0.94, noise=0.020, shift_noise=7.0, p_error=0.26),
}
ERROR_TYPES = ["K1", "K2", "K3", "K4", "K5", "K6", "K7"]
ERROR_WEIGHTS = [0.2, 0.15, 0.2, 0.17, 0.1, 0.12, 0.06]


def make_track(name: str) -> dict:
    """An invented layout: corner positions, apex speeds and directions."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32 if name not in ("synth_a", "synth_b")
                                else {"synth_a": 11, "synth_b": 23}[name])
    n = {"synth_a": 7, "synth_b": 10}.get(name, 8)
    length = {"synth_a": 4400.0, "synth_b": 4000.0}.get(name, 4200.0)
    apex = np.linspace(450, length - 450, n) + rng.uniform(-60, 60, n)
    return {
        "name": name,
        "length": length,
        "apex_m": apex,
        "apex_ms": rng.uniform(22, 48, n),
        "direction": rng.choice([-1.0, 1.0], n),
    }


def _smooth(a: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(np.pad(a, width, mode="edge"), kernel, mode="same")[width:-width]


def base_path(track: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre line of the invented track and its heading, one point per metre."""
    s = np.arange(int(track["length"]), dtype=float)
    curv = np.zeros_like(s)
    for a, v, d in zip(track["apex_m"], track["apex_ms"], track["direction"]):
        radius = v**2 / 14.0
        curv += d / radius * np.exp(-(((s - a) / 35.0) ** 2))
    heading = np.cumsum(curv)
    return np.cumsum(np.cos(heading)), np.cumsum(np.sin(heading)), heading


def simulate_lap(track: dict, rng: np.random.Generator, skill: dict, errors: dict[int, str],
                 weak_corner: int | None = None) -> dict[str, np.ndarray]:
    """One lap on a 1 m grid. `errors` maps a corner index to the error type injected there."""
    n = int(track["length"])
    s = np.arange(n, dtype=float)
    v2 = np.full(n, VMAX**2)
    steer = np.zeros(n)
    slip = np.abs(rng.normal(0.05, 0.01, n))
    tc = np.zeros(n)
    tyres_out = np.zeros(n)
    lateral = np.zeros(n)
    peak_scale = np.ones(n)

    for i, (a, v_apex, d) in enumerate(zip(track["apex_m"], track["apex_ms"], track["direction"])):
        err = errors.get(i)
        noise = skill["noise"] * (4.0 if i == weak_corner else 1.0)
        va = v_apex * skill["pace"] * rng.normal(1.0, noise)
        shift = abs(rng.normal(0.0, skill["shift_noise"]))
        delay = abs(rng.normal(0.0, skill["shift_noise"]))
        ab, aa, steer_gain = BRAKE_DECEL, ACCEL, 1.0
        rel = s - a
        if err == "K1":
            shift += rng.uniform(28, 45)
        elif err == "K2":
            ab *= rng.uniform(0.55, 0.68)
        elif err == "K3":
            va *= rng.uniform(0.86, 0.91)
        elif err == "K4":
            delay += rng.uniform(30, 45)
        elif err == "K5":
            aa *= 0.8
            zone = (rel > 5) & (rel < 80)
            slip[zone] += rng.uniform(0.6, 1.0) * np.exp(-(((rel[zone] - 35) / 20.0) ** 2))
            tc[(rel > 15) & (rel < 55)] = 1.0
            steer += 0.08 * np.sin(rel / 6.0) * ((rel > 5) & (rel < 70))
        elif err == "K6":
            steer_gain = rng.uniform(1.45, 1.7)
            va *= 0.965
            lateral += rng.uniform(2.5, 4.0) * np.exp(-((rel / 40.0) ** 2))
        elif err == "K7":
            va *= rng.uniform(0.4, 0.55)
            delay += 30
            tyres_out[np.abs(rel) < 35] = 4
            lateral += 7.0 * np.exp(-((rel / 30.0) ** 2))
            steer += 0.2 * np.sin(rel / 5.0) * (np.abs(rel) < 40)

        before = va**2 + 2 * ab * (-shift - rel)
        after = va**2 + 2 * aa * (rel - delay)
        constraint = np.where(rel < -shift, before, np.where(rel > delay, after, va**2))
        v2 = np.minimum(v2, constraint)
        steer += d * 0.35 * steer_gain * np.exp(-((rel / 35.0) ** 2))

    v = np.sqrt(v2)
    acc = v * np.gradient(v)                     # m/s^2 along the lap
    brake = _smooth(np.clip(-acc / 14.0, 0, 1), 7) * peak_scale
    gas = np.where((acc > 0.5) | (v > VMAX - 0.3), 1.0, 0.08)
    gas = np.clip(_smooth(gas, 15), 0, 1)
    gas[brake > 0.05] = 0.0
    return dict(v=v, gas=gas, brake=np.clip(brake, 0, 1), steer=np.clip(steer, -1, 1), slip=slip,
                tc=tc, tyres_out=tyres_out, lateral=lateral, acc=acc)


def _sample(track: dict, lap: dict, path, s_from: float = 0.0, s_to: float | None = None) -> pd.DataFrame:
    """Turn a lap on the 1 m grid into samples at the logger frequency."""
    n = int(track["length"])
    s_to = n - 1 if s_to is None else s_to
    s = np.arange(n, dtype=float)
    t_cum = np.concatenate([[0.0], np.cumsum(1.0 / lap["v"])[:-1]])
    t0, t1 = np.interp([s_from, s_to], s, t_cum)
    lap_total = t_cum[-1] + 1.0 / lap["v"][-1]
    if s_to >= n - 1:
        t1 = lap_total
    times = np.arange(t0, t1, 1.0 / HZ)
    pos = np.interp(times, np.append(t_cum, lap_total), np.append(s, n))
    pos = np.minimum(pos, n - 1e-3)

    def ch(key):
        return np.interp(pos, s, lap[key])

    px, pz, heading = path
    lat = ch("lateral")
    hd = np.interp(pos, s, heading)
    out = pd.DataFrame({
        "t": times - t0,
        "norm_pos": pos / n,
        "speed_kmh": ch("v") * 3.6,
        "gas": ch("gas"), "brake": ch("brake"), "steer": ch("steer"),
        "acc_z": ch("acc") / 9.81,
        "acc_x": ch("steer") * 4.0,
        "wslip_rl": ch("slip"), "wslip_rr": ch("slip"),
        "tc_on": (ch("tc") > 0.5).astype(float),
        "tyres_out": np.round(ch("tyres_out")),
        "x": np.interp(pos, s, px) - lat * np.sin(hd),
        "z": np.interp(pos, s, pz) + lat * np.cos(hd),
        "cur_time_ms": (times - (0.0 if s_from > 0 else t0)) * 1000.0,
    })
    out.attrs["lap_time"] = lap_total
    return out


def simulate_session(track: dict, driver: str, level: str, number: int, n_laps: int,
                     rng: np.random.Generator, weak_corner: int | None) -> tuple[pd.DataFrame, list[dict]]:
    """Out lap from the pit, `n_laps` flying laps and the start of an in lap."""
    skill = SKILL[level]
    path = base_path(track)
    n_corners = len(track["apex_m"])
    parts, truth = [], []
    clock, last_ms = 0.0, 0

    def add(frame, lap_no, valid_from_start=True):
        nonlocal clock
        frame = frame.copy()
        frame["t"] = frame["t"] + clock
        frame["lap"] = lap_no
        frame["last_time_ms"] = last_ms
        off = (frame["tyres_out"] >= 3).to_numpy()
        frame["valid_lap"] = (~np.maximum.accumulate(off)).astype(int) if valid_from_start else 0
        clock = float(frame["t"].iloc[-1]) + 1.0 / HZ
        parts.append(frame)

    out_lap = _sample(track, simulate_lap(track, rng, skill, {}), path, s_from=track["length"] * 0.55)
    out_lap["in_pit"] = (out_lap["t"] < 3).astype(int)
    out_lap["in_pit_lane"] = (out_lap["t"] < 8).astype(int)
    add(out_lap, 0, valid_from_start=False)

    for lap_no in range(1, n_laps + 1):
        errors = {}
        for c in range(n_corners):
            if rng.random() < skill["p_error"]:
                errors[c] = str(rng.choice(ERROR_TYPES, p=ERROR_WEIGHTS))
        frame = _sample(track, simulate_lap(track, rng, skill, errors, weak_corner), path)
        add(frame, lap_no)
        last_ms = int(frame.attrs["lap_time"] * 1000)
        truth += [dict(lap=lap_no, corner=c, apex_pos=float(track["apex_m"][c] / track["length"]), type=k)
                  for c, k in errors.items()]

    in_lap = _sample(track, simulate_lap(track, rng, skill, {}), path, s_to=track["length"] * 0.2)
    in_lap["in_pit_lane"] = (in_lap["norm_pos"] > 0.15).astype(int)
    add(in_lap, n_laps + 1)

    df = pd.concat(parts, ignore_index=True)
    for col in RAW_COLUMNS:
        if col not in df:
            df[col] = 0
    df["packet_id"] = np.arange(len(df)) * 6
    df["gear"] = np.clip((df["speed_kmh"] // 45).astype(int) + 1, 1, 6)
    df["rpm"] = (4000 + (df["speed_kmh"] % 45) * 70).astype(int)
    df["fuel"] = 60.0 - df["t"] * 0.03
    df["wslip_fl"] = df["wslip_fr"] = 0.04
    return df[RAW_COLUMNS], truth


def generate(data_dir: str | Path, drivers_per_level: int = 2, circuits=("synth_a", "synth_b"),
             sessions: int = 3, laps: int = 15, seed: int = 0) -> pd.DataFrame:
    """Write a complete synthetic study and return the injected errors as ground truth."""
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    (data_dir / "participants").mkdir(parents=True, exist_ok=True)
    truth_rows, k = [], 0
    for level in ("none", "moderate", "high"):
        for _ in range(drivers_per_level):
            k += 1
            driver = f"D{k:02d}"
            (data_dir / "participants" / f"{driver}.json").write_text(
                json.dumps({"driver": driver, "experience_level": level, "synthetic": True,
                            "publication_consent": True}, indent=2),
                encoding="utf-8")
            for circuit in circuits:
                track = make_track(circuit)
                weak = int(rng.integers(len(track["apex_m"])))
                for number in range(1, sessions + 1):
                    frame, truth = simulate_session(track, driver, level, number, laps, rng, weak)
                    sid = session_id(driver, circuit, number, "synthetic")
                    meta = dict(session_id=sid, driver=driver, circuit=circuit, session_number=number,
                                car_model="synthetic", track="synthetic", synthetic=True, hz=HZ,
                                n_samples=len(frame))
                    write_session(data_dir / "raw", meta, frame)
                    truth_rows += [dict(session_id=sid, driver=driver, circuit=circuit, **row) for row in truth]
    truth = pd.DataFrame(truth_rows)
    truth.to_parquet(data_dir / "synthetic_truth.parquet", index=False)
    return truth
