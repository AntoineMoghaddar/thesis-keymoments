"""A stand-in for the simulator, for trying out the logger without ACC.

It plays back a synthetic session. Recordings made with it are marked as demo data.
"""
from __future__ import annotations

import time

import numpy as np

from km import synth

from .acc_shm import STATUS_LIVE


class MockSource:
    name = "demo"

    def __init__(self, speed: float = 1.0, laps: int = 4, realtime: bool = True):
        track = synth.make_track("synth_a")
        frame, _ = synth.simulate_session(track, "DEMO", "moderate", 1, laps, np.random.default_rng(0), None)
        self._rows = frame.to_dict("records")
        self._times = frame["t"].to_numpy()
        self._speed, self._realtime = speed, realtime
        self._start = time.perf_counter()
        self._cursor = 0

    def status(self) -> int:
        return STATUS_LIVE

    def info(self) -> dict:
        return {"car_model": "demo_car", "track": "synth_a"}

    def settings(self) -> dict:
        return {}

    def read(self) -> tuple[int, dict]:
        if self._realtime:
            now = (time.perf_counter() - self._start) * self._speed
            self._cursor = int(np.searchsorted(self._times, now % self._times[-1]))
        else:
            self._cursor = min(self._cursor + 1, len(self._rows) - 1)
        row = dict(self._rows[min(self._cursor, len(self._rows) - 1)])
        row["t"] = time.perf_counter() if self._realtime else float(row["t"])
        return STATUS_LIVE, row

    def close(self) -> None:
        pass
