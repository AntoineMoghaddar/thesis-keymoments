"""Export of the dataset for publication.

The public copy carries new, random identifiers. The table that links them to the pseudonyms
used during the study is written to a separate key file, which belongs on the storage of the
university only. Without that file the public data cannot be traced back to a recording visit.

Only drivers whose participant file records consent for publication are included, and experts
named in export.exclude_experts are left out.

Left out of the public copy: dates and times, the questionnaire (unless items are named in the
configuration), all free text, and anything recorded in demo mode.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .labelling import SCHEMA
from .schema import RAW_FIELDS, list_sessions

META_KEEP = ["circuit", "session_number", "car_model", "track", "hz", "n_samples", "laps_completed", "duration_s",
             "mean_rate_hz", "acc_version", "tc_level", "abs_level", "engine_map", "tyre_compound", "aid_stability",
             "aid_auto_clutch", "aid_auto_blip", "ideal_line_on", "air_temp", "road_temp", "track_grip_status",
             "rain_intensity", "synthetic"]
COLUMN_NOTES = {
    "t": "seconds since the start of the recording", "packet_id": "physics step counter of the simulator",
    "lap": "completed laps at the time of the sample", "norm_pos": "position on the track spline, 0 to 1",
    "gas": "throttle input, 0 to 1", "brake": "brake input, 0 to 1", "steer": "steering input, -1 to 1",
    "acc_x": "lateral acceleration (g)", "acc_z": "longitudinal acceleration (g)",
    "tc_on": "traction control intervening", "abs_on": "ABS intervening",
    "tyres_out": "number of tyres outside the track limits", "valid_lap": "lap still valid according to the simulator",
    "x": "world position (m)", "y": "world position, up (m)", "z": "world position (m)",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_dataset(cfg: dict, out_dir: str | Path, key_path: str | Path, seed: int | None = None) -> dict:
    paths = Paths.from_config(cfg)
    out, key_path = Path(out_dir), Path(key_path)
    if out.resolve() == key_path.resolve().parent or out.resolve() in key_path.resolve().parents:
        raise ValueError("The key file must not be stored inside the public dataset folder.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"The output folder is not empty: {out}")
    rng = np.random.default_rng(seed)
    sessions = [m for m in list_sessions(paths.raw) if not m.get("demo")]

    # Publication needs the separate consent that is recorded per participant by the logger.
    def consented(driver: str) -> bool:
        path = paths.participants / f"{driver}.json"
        return path.exists() and bool(json.loads(path.read_text(encoding="utf-8")).get("publication_consent"))
    left_out = sorted({m["driver"] for m in sessions if not consented(m["driver"])})
    sessions = [m for m in sessions if m["driver"] not in left_out]
    if not sessions:
        raise RuntimeError("No sessions to export: no driver has consent for publication on record.")

    # New identifiers in random order, so that the order of recruitment cannot be read from them.
    drivers = sorted({m["driver"] for m in sessions})
    driver_map = {d: f"P{k + 1:02d}" for k, d in enumerate(rng.permutation(drivers))}
    session_map, key_rows = {}, []
    for m in sessions:
        public = f"{driver_map[m['driver']]}_{m['circuit']}_S{m.get('session_number', 0)}"
        while public in session_map.values():
            public += "b"
        session_map[m["session_id"]] = public
        key_rows.append(dict(kind="session", public_id=public, internal_id=m["session_id"],
                             public_driver=driver_map[m["driver"]], pseudonym=m["driver"]))

    (out / "telemetry").mkdir(parents=True)
    for m in sessions:
        public = session_map[m["session_id"]]
        folder = out / "telemetry" / driver_map[m["driver"]] / m["circuit"]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m["path"], folder / f"{public}.parquet")
        meta = {"session_id": public, "driver": driver_map[m["driver"]], **{k: m[k] for k in META_KEEP if k in m}}
        (folder / f"{public}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Participants: experience level, plus the questionnaire items named in the configuration.
    rows = []
    for d in drivers:
        record_path = paths.participants / f"{d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        row = {"driver": driver_map[d], "experience_level": record.get("experience_level", "unknown")}
        for item in cfg["export"]["questionnaire_items"]:
            value = record.get("questionnaire", {}).get(item)
            row[item] = "; ".join(value) if isinstance(value, list) else value
        rows.append(row)
    pd.DataFrame(rows).sort_values("driver").to_csv(out / "participants.csv", index=False)

    def remap(frame: pd.DataFrame, by_window: dict | None = None) -> pd.DataFrame:
        frame = frame.copy()
        if "session_id" in frame:
            old = frame["session_id"]
            new = old.map(session_map)
            for col in ("window_id", "lap_uid", "example_window"):
                if col in frame:
                    frame[col] = [n + v[len(o):] for n, o, v in zip(new, old, frame[col])]
            frame["session_id"] = new
        elif by_window is not None and "window_id" in frame:
            frame["window_id"] = frame["window_id"].map(by_window)
        for col in ("driver", "fold"):          # with leave-one-driver-out the fold carries the pseudonym
            if col in frame:
                frame[col] = frame[col].map(lambda v: driver_map.get(v, v))
        return frame.drop(columns=[c for c in ("lap_row", "ref_row") if c in frame])

    derived = out / "derived"
    derived.mkdir()
    window_map = {}
    for name in ("laps", "windows", "predictions", "moments"):
        src = paths.prepared / f"{name}.parquet"
        if src.exists():
            frame = pd.read_parquet(src)
            frame = frame[frame["session_id"].isin(session_map)]
            public = remap(frame)
            if name == "windows":
                window_map = dict(zip(frame["window_id"], public["window_id"]))
            public.to_parquet(derived / f"{name}.parquet", index=False)
    for name, src in (("weak_labels", paths.prepared / "weak_labels.parquet"),
                      ("expert_labels", paths.review / "expert_labels.parquet")):
        if src.exists() and window_map:
            frame = remap(pd.read_parquet(src), window_map).dropna(subset=["window_id"])
            frame.to_parquet(derived / f"{name}.parquet", index=False)

    # Expert responses: new expert identifiers, no comments, no timestamps.
    db = paths.review / "responses.db"
    if db.exists() and window_map:
        from .review import load_responses
        resp = load_responses(paths.review)
        if len(resp):
            resp = resp[~resp["expert_id"].isin(cfg["export"].get("exclude_experts", []))]
            experts = sorted(resp["expert_id"].unique())
            expert_map = {e: f"X{k + 1:02d}" for k, e in enumerate(rng.permutation(experts))}
            key_rows += [dict(kind="expert", public_id=v, internal_id=k) for k, v in expert_map.items()]
            resp = resp.assign(expert_id=resp["expert_id"].map(expert_map), window_id=resp["window_id"].map(window_map))
            resp.drop(columns=["comment", "saved_at"]).dropna(subset=["window_id"]).to_parquet(
                derived / "expert_responses.parquet", index=False)

    seg_dir = out / "segments"
    seg_dir.mkdir()
    for circuit in sorted({m["circuit"] for m in sessions}):
        src = paths.segments / f"{circuit}.yaml"
        if src.exists():
            shutil.copyfile(src, seg_dir / src.name)

    _datasheet(out, cfg, len(drivers), len(sessions))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("\n".join(f"{_sha(p)}  {p.relative_to(out).as_posix()}" for p in files) + "\n",
                                    encoding="utf-8")

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_rows += [dict(kind="driver", public_id=v, internal_id=k, pseudonym=k) for k, v in driver_map.items()]
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    return {"drivers": len(drivers), "left_out_without_consent": left_out, "sessions": len(sessions), "files": len(files) + 1, "out": str(out), "key": str(key_path)}


def _datasheet(out: Path, cfg: dict, n_drivers: int, n_sessions: int) -> None:
    columns = "\n".join(f"| `{name}` | {COLUMN_NOTES.get(name, '')} |" for name, _ in RAW_FIELDS)
    types = "\n".join(f"| {k} | {v['name']} | {v['phase']} | {v['advice']} |" for k, v in SCHEMA.items())
    text = f"""# Sim racing telemetry with coaching-relevant moments

Telemetry of {n_drivers} drivers in {n_sessions} sessions in Assetto Corsa Competizione, recorded for an MSc thesis at the
University of Twente on the automatic detection of coaching-relevant moments.

Licence: {cfg['export']['licence']}

## Contents

| Path | Content |
|---|---|
| `telemetry/<driver>/<circuit>/<session>.parquet` | Samples of one session at about 60 Hz, with a JSON file with car, track and settings |
| `participants.csv` | Experience level per driver (none, moderate, high) |
| `derived/laps.parquet` | The laps used in the analysis, with lap time and the reference lap per driver and circuit |
| `derived/windows.parquet` | One row per lap and track segment with the engineered features |
| `derived/weak_labels.parquet` | Votes of the labelling functions and the resulting weak labels |
| `derived/expert_labels.parquet` | Reference labels from the expert review (majority of the panel), where available |
| `derived/expert_responses.parquet` | Individual expert judgements, without comments |
| `segments/<circuit>.yaml` | Track segments as shares of the lap |
| `SHA256SUMS` | Checksums of all files |

## Collection

Each driver drove {cfg['study']['sessions_per_circuit']} sessions of {cfg['study']['laps_per_session']} laps per circuit with a fixed car and
fixed fuel and tyre settings. No feedback on driving was given during the sessions. The first
{cfg['preprocess']['warmup_laps']} flying laps of each session are treated as warm-up and are not part of the derived tables. They are part of the telemetry files.

## Identifiers and privacy

Driver, session and expert identifiers were newly assigned in random order for this publication. The
table that links them to the study records is not part of the dataset. Dates, times, questionnaire
answers and free text have been removed. Participants gave consent for publication in this form.

## Telemetry columns

| Column | Meaning |
|---|---|
{columns}

Columns without a description carry the value of the simulator field of the same name (wheel slip,
slip ratio and brake pressure per wheel, fuel, gear, engine speed, lap timers and pit state).

## Moment types

| ID | Moment type | Phase | Advice category |
|---|---|---|---|
{types}

## Known limitations

- Weak labels come from rules, not from people. Use `expert_labels` for evaluation.
- The reference lap is the fastest lap of the same driver, so a label describes a deviation from that
  driver's own best and not from an optimal lap.
- The track length is estimated from the distance driven; the simulator does not publish it.
"""
    (out / "README.md").write_text(text, encoding="utf-8")
