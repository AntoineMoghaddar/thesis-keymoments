import ctypes
import json
import time
from unittest.mock import MagicMock

import pandas as pd

from km.schema import RAW_COLUMNS
from logger import acc_shm
from logger import questionnaire as qn
from logger.mock import MockSource
from logger.recorder import Recorder
from logger.transfer import transfer


def test_struct_layout():
    # Sizes follow from the ACC Shared Memory Documentation v1.8.12.
    assert ctypes.sizeof(acc_shm.Physics) == 800
    assert ctypes.sizeof(acc_shm.Graphics) == 1580
    assert acc_shm.Physics.packetId.offset == 0 and acc_shm.Physics.speedKmh.offset == 28
    assert acc_shm.Graphics.completedLaps.offset == 132 and acc_shm.Graphics.normalizedCarPosition.offset == 248
    assert acc_shm.Static.carModel.offset == 68 and acc_shm.Static.track.offset == 134


def test_sample_has_all_columns():
    physics, graphics = acc_shm.Physics(), acc_shm.Graphics()
    physics.speedKmh, graphics.completedLaps, graphics.activeCars = 123.0, 4, 1
    graphics.carCoordinates[0][0] = 5.0
    sample = acc_shm.sample_from(physics, graphics, 1.5)
    assert set(sample) == set(RAW_COLUMNS)
    assert sample["speed_kmh"] == 123.0 and sample["lap"] == 4 and sample["x"] == 5.0


def test_text_decoding():
    static = acc_shm.Static()
    for k, ch in enumerate("monza"):
        static.track[k] = ord(ch)
    assert acc_shm.text(static.track) == "monza"


class _Scripted:
    name = "test"

    def __init__(self, steps):
        self.steps = iter(steps)

    def read(self):
        status, packet, lap = next(self.steps)
        row = {c: 0 for c in RAW_COLUMNS}
        row.update(t=time.perf_counter(), packet_id=packet, lap=lap, valid_lap=1)
        return status, row


def test_recorder_skips_duplicates_and_pauses(tmp_path):
    live, pause = acc_shm.STATUS_LIVE, acc_shm.STATUS_PAUSE
    steps = [(live, 1, 3), (live, 1, 3), (pause, 2, 3), (live, 3, 3), (live, 4, 4)]
    rec = Recorder(_Scripted(steps), tmp_path, "D01", "monza", 2, flush_seconds=0)
    stored = [rec.step() for _ in steps]
    meta = rec.stop()
    assert stored == [True, False, False, True, True]
    frame = pd.read_parquet(rec.data_path)
    assert frame["packet_id"].tolist() == [1, 3, 4] and frame["t"].iloc[0] == 0
    assert meta["laps_completed"] == 1 and meta["finished"] and meta["session_number"] == 2
    assert json.loads(rec.meta_path.read_text())["n_samples"] == 3
    assert rec.data_path.parent == tmp_path / "raw" / "D01" / "monza"


def test_recorder_with_demo_source_feeds_the_pipeline(tmp_path):
    from km.config import load_config
    from km.preprocess import split_laps
    rec = Recorder(MockSource(realtime=False, laps=4), tmp_path, "D02", "synth_a", 1, flush_seconds=0.01)
    while rec.source._cursor < len(rec.source._rows) - 1:
        rec.step()
    rec.stop()
    laps = split_laps(pd.read_parquet(rec.data_path), load_config())
    assert sum(lap["usable"] for lap in laps) == 2                   # four flying laps minus two warm-up laps


def test_transfer_verifies_and_deletes_only_when_asked(tmp_path):
    src, dst = tmp_path / "local", tmp_path / "ut"
    (src / "raw").mkdir(parents=True)
    dst.mkdir()
    (src / "raw" / "a.parquet").write_bytes(b"123")
    assert transfer(src, dst)["copied"] == 1 and (src / "raw" / "a.parquet").exists()
    result = transfer(src, dst, delete_source=True)
    assert result["already_there"] == 1 and result["deleted"] == 1
    assert (dst / "raw" / "a.parquet").read_bytes() == b"123" and not (src / "raw" / "a.parquet").exists()


def test_questionnaire_is_complete_in_both_languages(tmp_path):
    ids = [q["id"] for q in qn.ITEMS]
    assert len(ids) == len(set(ids)) and len(ids) >= 20
    for q in qn.ITEMS:
        assert q["en"] and q["nl"]
        if q["kind"] in ("choice", "multi"):
            assert all(en and nl for en, nl in q["options"])
            assert qn.to_english(q, q["options"][0][1]) == q["options"][0][0]
    banned = ("age", "gender", "name", "leeftijd", "geslacht", "naam")
    assert not [q["id"] for q in qn.ITEMS if any(word == q["id"] for word in banned)]
    assert set(qn.missing({})) == set(qn.REQUIRED)
    qn.save_participant(tmp_path, "D05", experience_level="high")
    record = qn.save_participant(tmp_path, "D05", consent_signed=True)
    assert record["experience_level"] == "high" and record["consent_signed"] and record["driver"] == "D05"


def test_logger_window_logic(tmp_path, monkeypatch):
    """The window code, run against a stand-in for tkinter. It does not test the layout on screen."""
    import tk_stub
    box = tk_stub.install()
    import importlib

    import logger.app as app
    importlib.reload(app)
    monkeypatch.setattr(app, "SETTINGS_FILE", tmp_path / "settings.json")
    root = MagicMock()
    root.winfo_children.return_value = []
    window = app.LoggerApp(root)
    window.data_dir.set(str(tmp_path))

    window.driver.set("Jan")                                         # a name is refused
    window.start()
    assert window.recorder is None and box.showerror.called

    window.driver.set("d07")
    window.level_box.get = lambda: "hoog"
    window.consent.set(True)
    window.set_level()                                               # also stores the consent tick
    form = app.QuestionnaireWindow(window, "D07")
    for q in qn.ITEMS:
        var = form.vars[q["id"]]
        if q["kind"] == "choice":
            var.set(qn.option_text(q, "nl")[1])
        elif q["kind"] == "multi":
            next(iter(var.values())).set(True)
        elif q["kind"] == "scale":
            var.set(4)
    form.save()
    record = json.loads((tmp_path / "participants" / "D07.json").read_text(encoding="utf-8"))
    assert record["experience_level"] == "high" and record["questionnaire"]["sim_years"] == "Less than 1 year"

    window.demo.set(True)
    window.start()
    time.sleep(0.5)
    window.tick()
    assert window.recorder is not None and window.status_vars["samples"].get() != "-"
    window.stop()
    assert window.recorder is None and list((tmp_path / "raw" / "D07" / "synth_a").glob("*.parquet"))
    for lang in ("en", "nl"):
        window.lang.set(lang)
        window.build()
