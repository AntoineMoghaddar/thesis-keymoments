"""Layout of a raw session file. The logger and the synthetic generator both write this."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_FIELDS: list[tuple[str, pa.DataType]] = [
    ("t", pa.float64()),              # seconds since the start of the recording (monotonic clock)
    ("packet_id", pa.int32()),        # physics step counter of the simulator
    ("lap", pa.int16()),              # completed laps at the time of the sample
    ("norm_pos", pa.float32()),       # position on the track spline, 0 to 1
    ("speed_kmh", pa.float32()),
    ("gas", pa.float32()),            # 0 to 1
    ("brake", pa.float32()),          # 0 to 1
    ("steer", pa.float32()),          # -1 to 1
    ("gear", pa.int8()),
    ("rpm", pa.int32()),
    ("acc_x", pa.float32()),          # lateral acceleration in g
    ("acc_y", pa.float32()),
    ("acc_z", pa.float32()),          # longitudinal acceleration in g
    ("wslip_fl", pa.float32()), ("wslip_fr", pa.float32()),
    ("wslip_rl", pa.float32()), ("wslip_rr", pa.float32()),
    ("sratio_fl", pa.float32()), ("sratio_fr", pa.float32()),
    ("sratio_rl", pa.float32()), ("sratio_rr", pa.float32()),
    ("bpress_fl", pa.float32()), ("bpress_fr", pa.float32()),
    ("bpress_rl", pa.float32()), ("bpress_rr", pa.float32()),
    ("yaw_rate", pa.float32()),
    ("tc_on", pa.float32()),          # traction control intervening
    ("abs_on", pa.float32()),         # ABS intervening
    ("tyres_out", pa.int8()),         # number of tyres outside the track limits
    ("in_pit", pa.int8()),
    ("in_pit_lane", pa.int8()),
    ("valid_lap", pa.int8()),
    ("cur_time_ms", pa.int32()),
    ("last_time_ms", pa.int32()),
    ("fuel", pa.float32()),
    ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()),   # world position, y is up
]
RAW_SCHEMA = pa.schema(RAW_FIELDS)
RAW_COLUMNS = [name for name, _ in RAW_FIELDS]


def session_id(driver: str, circuit: str, number: int, stamp: str) -> str:
    return f"{driver}_{circuit}_S{number}_{stamp}"


def session_paths(raw_dir: Path, driver: str, circuit: str, sid: str) -> tuple[Path, Path]:
    folder = Path(raw_dir) / driver / circuit
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{sid}.parquet", folder / f"{sid}.json"


def write_session(raw_dir: Path, meta: dict, frame: pd.DataFrame) -> Path:
    """Write a whole session at once (used by the synthetic generator and the tests)."""
    data_path, meta_path = session_paths(raw_dir, meta["driver"], meta["circuit"], meta["session_id"])
    table = pa.Table.from_pandas(frame[RAW_COLUMNS].fillna(0), schema=RAW_SCHEMA, preserve_index=False, safe=False)
    pq.write_table(table, data_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data_path


def list_sessions(raw_dir: Path) -> list[dict]:
    """Metadata of every recorded session, with the path of its data file."""
    out = []
    for meta_path in sorted(Path(raw_dir).glob("*/*/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data_path = meta_path.with_suffix(".parquet")
        if data_path.exists():
            meta["path"] = str(data_path)
            out.append(meta)
    return out
