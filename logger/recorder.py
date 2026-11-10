"""Recording of one session. No user interface in here, so that it can be tested without one."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from km import __version__
from km.schema import RAW_COLUMNS, RAW_SCHEMA, session_id, session_paths

from .acc_shm import STATUS_LIVE


class Recorder:
    """Polls a source and writes the samples of one session to a Parquet file.

    Samples are written to disk every few seconds. When the program or the computer stops
    unexpectedly, at most the last seconds of a session are lost.
    """

    def __init__(self, source, data_dir: str | Path, driver: str, circuit: str, number: int,
                 hz: int = 60, flush_seconds: float = 10.0, expected_car: str | None = None, extra_meta: dict | None = None):
        self.source, self.hz, self.flush_seconds = source, hz, flush_seconds
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.meta = dict(session_id=session_id(driver, circuit, number, stamp), driver=driver, circuit=circuit,
                         session_number=number, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), hz=hz,
                         logger_version=__version__, source=getattr(source, "name", "unknown"),
                         expected_car=expected_car, **(extra_meta or {}))
        self.data_path, self.meta_path = session_paths(Path(data_dir) / "raw", driver, circuit, self.meta["session_id"])
        self._writer = pq.ParquetWriter(self.data_path, RAW_SCHEMA)
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_packet = None
        self._last_flush = time.perf_counter()
        self._t0 = None
        self._first_lap = None
        self.error: str | None = None
        self.stats = dict(samples=0, laps_completed=0, last_lap_ms=0, status=0, speed_kmh=0.0, in_pit=0,
                          valid_lap=1, elapsed=0.0, rate=0.0)
        self._write_meta(finished=False)

    # -- one polling step ---------------------------------------------------
    def step(self) -> bool:
        """Read the source once. Returns True when a new sample was stored."""
        status, sample = self.source.read()
        self.stats["status"] = status
        if status != STATUS_LIVE or sample["packet_id"] == self._last_packet:
            return False        # paused, in a menu, or no new physics step since the last poll
        self._last_packet = sample["packet_id"]
        if self._t0 is None:
            self._t0, self._first_lap = sample["t"], sample["lap"]
        sample["t"] -= self._t0
        with self._lock:
            self._buffer.append(sample)
            s = self.stats
            s["samples"] += 1
            s["laps_completed"] = sample["lap"] - self._first_lap
            s["last_lap_ms"], s["speed_kmh"] = sample["last_time_ms"], sample["speed_kmh"]
            s["in_pit"], s["valid_lap"] = sample["in_pit"] or sample["in_pit_lane"], sample["valid_lap"]
            s["elapsed"] = sample["t"]
            s["rate"] = s["samples"] / max(sample["t"], 1e-6)
        if time.perf_counter() - self._last_flush >= self.flush_seconds:
            self.flush()
        return True

    def flush(self) -> None:
        with self._lock:
            rows, self._buffer = self._buffer, []
        self._last_flush = time.perf_counter()
        if rows:
            columns = {name: [row[name] for row in rows] for name in RAW_COLUMNS}
            self._writer.write_table(pa.table(columns).cast(RAW_SCHEMA, safe=False))

    # -- background thread --------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / self.hz
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                self.step()
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        except Exception as exc:  # shown in the window; the data written so far are kept
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.flush()
        self._writer.close()
        self._write_meta(finished=True)
        return self.meta

    def _write_meta(self, finished: bool) -> None:
        meta = dict(self.meta, finished=finished)
        if finished:
            meta.update(ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"), n_samples=self.stats["samples"],
                        laps_completed=self.stats["laps_completed"], duration_s=round(self.stats["elapsed"], 1),
                        mean_rate_hz=round(self.stats["rate"], 1), error=self.error)
            self.meta = meta
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
