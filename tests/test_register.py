import datetime as dt
import json

import pytest

from km import register
from km.config import Paths, load_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"data_dir": str(tmp_path / "data"), "results_dir": str(tmp_path / "results"),
                                  "register_dir": str(tmp_path / "register")})


def test_register_stays_apart_from_the_research_data(tmp_path):
    bad = load_config(overrides={"data_dir": str(tmp_path / "data"), "register_dir": str(tmp_path / "data" / "register")})
    with pytest.raises(ValueError):
        register.register_dir(bad)


def test_pseudonyms_checks_and_sync(cfg):
    today = dt.date.today()
    a = register.add_participant(cfg, "driver", name="Test Person", email="t@example.org", level="high", is_adult=True,
                                 no_health_exclusion=True, is_independent=True, letter_sent=today.isoformat(),
                                 consent_signed=today.isoformat(), publication_consent=True, wants_summary=True)
    b = register.add_participant(cfg, "driver", name="Second Person")
    e = register.add_participant(cfg, "expert", name="An Expert", is_adult=True, is_independent=True)
    assert (a, b, e) == ("D01", "D02", "E01")
    (Paths.from_config(cfg).raw / "D03").mkdir(parents=True)              # a code that already has recordings
    assert register.next_pseudonym(cfg, "driver") == "D04"

    register.add_appointment(cfg, a, (today + dt.timedelta(days=2)).isoformat(), "14:00", "Sessions Monza")
    points = register.open_points(cfg).set_index("pseudonym")["open"]
    assert "at least 5 required" in points[a]                             # letter sent too late for this appointment
    assert "experience level not set" in points[b] and "consent form not signed" in points[e]
    assert "health" not in points[e]                                      # that criterion is for drivers only
    counts = register.recruitment(cfg).set_index("level")["drivers"]
    assert counts["high"] == 1 and counts["none"] == 0

    assert register.sync_to_data(cfg) == ["D01", "D02"]
    record = json.loads((Paths.from_config(cfg).participants / "D01.json").read_text())
    assert record == {"driver": "D01", "consent_signed": True, "publication_consent": True, "withdrawn": False,
                      "experience_level": "high"}                         # no name, no email
    assert register.experts_without_publication_consent(cfg) == ["E01"]

    register.update_participant(cfg, a, withdrawn=today.isoformat())
    register.sync_to_data(cfg)
    record = json.loads((Paths.from_config(cfg).participants / "D01.json").read_text())
    assert record["withdrawn"] and not record["publication_consent"]
    progress = register.progress(cfg).set_index("pseudonym")
    assert progress.loc["D02", "done"] == 0 and progress.loc["D02", "total"] == 6

    assert register.remove_identities(cfg) == 3
    people = register.participants(cfg)
    assert people["name"].isna().all() and people["email"].isna().all()
    assert list(people["pseudonym"]) == ["D01", "D02", "E01"] and people.loc[0, "consent_signed"] == today.isoformat()


def test_register_app_runs(cfg, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("KM_REGISTER_DIR", cfg["register_dir"])
    monkeypatch.setenv("KM_DATA_DIR", cfg["data_dir"])
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent / "register_app" / "app.py"
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception
    app.sidebar.radio[1].set_value("person").run()
    app.text_input[0].set_value("Demo Driver")
    app.button[0].click().run()
    assert not app.exception and register.participants(cfg)["name"].tolist() == ["Demo Driver"]
    for page in ("overview", "planning", "closing"):
        app.sidebar.radio[1].set_value(page).run()
        assert not app.exception, page
