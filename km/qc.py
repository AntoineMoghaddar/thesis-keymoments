"""Quality control of recorded sessions, to be run after every recording visit."""
from __future__ import annotations

import pandas as pd

from .config import Paths
from .preprocess import split_laps, unique_laps
from .schema import list_sessions

SETTINGS = ["car_model", "tc_level", "abs_level", "engine_map", "tyre_compound", "aid_stability", "aid_auto_clutch",
            "aid_auto_blip", "ideal_line_on", "aid_fuel_rate", "aid_tyre_rate", "rain_tyres", "track_grip_status",
            "rain_intensity"]


def session_quality(cfg: dict) -> pd.DataFrame:
    """One row per session with what was recorded and a list of things to look at."""
    paths = Paths.from_config(cfg).ensure()
    study = cfg["study"]
    target = study["laps_per_session"] - cfg["preprocess"]["warmup_laps"]
    rows = []
    for meta in list_sessions(paths.raw):
        frame = unique_laps(pd.read_parquet(meta["path"]))
        laps = split_laps(frame, cfg)
        usable = [lap for lap in laps if lap["usable"]]
        gaps = frame["t"].diff().dropna()
        rate = len(frame) / max(frame["t"].iloc[-1] - frame["t"].iloc[0], 1e-9) if len(frame) > 1 else 0.0
        times = [lap["lap_time"] for lap in usable]
        clock_error = max((abs(lap["lap_time"] - lap["clock_time"]) for lap in usable), default=0.0)
        row = dict(session_id=meta["session_id"], driver=meta["driver"], circuit=meta["circuit"],
                   session_number=meta.get("session_number"), samples=len(frame), rate_hz=round(rate, 1),
                   max_gap_s=round(float(gaps.max()) if len(gaps) else 0.0, 3), laps_recorded=len(laps),
                   laps_usable=len(usable), best_lap=min(times, default=None),
                   median_lap=float(pd.Series(times).median()) if times else None,
                   laps_off_track=sum(lap["tyres_out_max"] >= 3 for lap in usable),
                   max_lap_time_error_s=round(clock_error, 3), **{k: meta.get(k) for k in SETTINGS})
        notes = []
        if not meta.get("finished", True):
            notes.append("recording was not closed properly")
        if rate < 0.9 * meta.get("hz", cfg["logger"]["hz"]):
            notes.append("low sample rate")
        if len(usable) < target:
            notes.append(f"{len(usable)} usable laps, {target} expected")
        if clock_error > 0.1:
            notes.append("lap time of the logger differs from the simulator")
        if meta.get("demo") or meta.get("synthetic"):
            notes.append("not a real recording")
        elif (meta.get("car_model") or "").lower() != study["car_model"].lower():
            notes.append("car is not the car of the study")
        if not (meta.get("demo") or meta.get("synthetic")) and meta["circuit"].lower() not in (meta.get("track") or "").lower():
            notes.append("track in the simulator does not match the circuit")
        row["notes"] = notes
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # The set-up has to be the same in every session. Flag sessions that deviate from the most common value.
    for column in SETTINGS:
        values = table[column].dropna()
        if values.nunique() > 1:
            common = values.mode().iloc[0]
            for i in table.index[table[column].notna() & (table[column] != common)]:
                table.at[i, "notes"].append(f"{column} = {table.at[i, column]} (most sessions: {common})")
    per_driver = table.groupby(["driver", "circuit"]).size()
    for (driver, circuit), n in per_driver.items():
        if n != study["sessions_per_circuit"]:
            i = table.index[(table["driver"] == driver) & (table["circuit"] == circuit)][0]
            table.at[i, "notes"].append(f"{n} sessions on this circuit, {study['sessions_per_circuit']} expected")
    table["notes"] = table["notes"].map("; ".join)
    table.to_csv(paths.results / "qc_sessions.csv", index=False)
    return table
