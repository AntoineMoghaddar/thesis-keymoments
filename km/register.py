"""Register of participants: who takes part, under which pseudonym, and how far they are.

This is the only place where a name and a pseudonym occur together. It is therefore kept in
its own folder, apart from the research data, on the storage of the university. Nothing in the
analysis reads it. After the thesis has been completed the names and contact details are
removed, which is what the information letter promises.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import ROOT, Paths

ROLES = ("driver", "expert")
LEVELS = ("none", "moderate", "high")
CHANNELS = ("POD1UM AI network", "E-Sports Lab", "Personal network", "Other")
MIN_DAYS_LETTER = 5        # the information letter is sent at least five days before the first session
IDENTITY = ("name", "email", "phone", "notes")

DDL = """
CREATE TABLE IF NOT EXISTS participants (
    pseudonym TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    name TEXT, email TEXT, phone TEXT,
    channel TEXT,
    level TEXT,                       -- drivers: none, moderate, high
    background TEXT,                  -- why this level, or why this person counts as an expert
    is_adult INTEGER DEFAULT 0,
    no_health_exclusion INTEGER DEFAULT 0,   -- drivers: no photosensitive epilepsy, no severe motion sickness
    is_independent INTEGER DEFAULT 0,        -- not a student of the researcher, not a relative
    letter_sent TEXT,
    consent_signed TEXT,
    publication_consent INTEGER DEFAULT 0,
    quote_consent INTEGER DEFAULT 0,         -- experts: comments may be quoted anonymously
    wants_summary INTEGER DEFAULT 0,
    future_contact INTEGER DEFAULT 0,
    withdrawn TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pseudonym TEXT NOT NULL,
    day TEXT NOT NULL,
    time TEXT,
    activity TEXT,
    note TEXT
);
"""
BOOLS = ("is_adult", "no_health_exclusion", "is_independent", "publication_consent", "quote_consent",
         "wants_summary", "future_contact")
FIELDS = ("name", "email", "phone", "channel", "level", "background", *BOOLS, "letter_sent", "consent_signed",
          "withdrawn", "notes")


def register_dir(cfg: dict) -> Path:
    """Folder of the register. It may not lie inside the data folder or inside the repository."""
    folder = Path(cfg.get("register_dir") or "register").resolve()
    data = Paths.from_config(cfg).data.resolve()
    if folder == data or data in folder.parents:
        raise ValueError("register_dir must not lie inside data_dir: names have to stay apart from the research data.")
    return folder


def connect(cfg: dict) -> sqlite3.Connection:
    folder = register_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(folder / "register.db", timeout=30)
    con.executescript(DDL)
    return con


def in_repository(cfg: dict) -> bool:
    folder = register_dir(cfg)
    return folder == ROOT or ROOT in folder.parents


def next_pseudonym(cfg: dict, role: str) -> str:
    """Next free code: D01, D02, ... for drivers and E01, E02, ... for experts."""
    prefix = "D" if role == "driver" else "E"
    with connect(cfg) as con:
        used = {r[0] for r in con.execute("SELECT pseudonym FROM participants")}
    raw = Paths.from_config(cfg).raw
    if role == "driver" and raw.exists():
        used |= {p.name for p in raw.iterdir() if p.is_dir()}      # codes that already have recordings
    k = 1
    while f"{prefix}{k:02d}" in used:
        k += 1
    return f"{prefix}{k:02d}"


def add_participant(cfg: dict, role: str, **fields) -> str:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    pseudonym = next_pseudonym(cfg, role)
    with connect(cfg) as con:
        con.execute("INSERT INTO participants (pseudonym, role, created_at) VALUES (?,?,?)",
                    (pseudonym, role, dt.datetime.now().isoformat(timespec="seconds")))
    update_participant(cfg, pseudonym, **fields)
    return pseudonym


def update_participant(cfg: dict, pseudonym: str, **fields) -> None:
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    if fields.get("level") not in (None, "", *LEVELS):
        raise ValueError(f"Unknown level: {fields['level']}")
    values = {k: (int(bool(v)) if k in BOOLS else (str(v) if v not in (None, "") else None)) for k, v in fields.items()}
    if values:
        with connect(cfg) as con:
            con.execute(f"UPDATE participants SET {', '.join(f'{k} = ?' for k in values)} WHERE pseudonym = ?",
                        (*values.values(), pseudonym))


def participants(cfg: dict) -> pd.DataFrame:
    with connect(cfg) as con:
        frame = pd.read_sql_query("SELECT * FROM participants ORDER BY role, pseudonym", con)
    # Empty text fields are returned as None, so that "not filled in" can be tested in one way.
    text = [c for c in frame.columns if c not in BOOLS]
    frame[text] = frame[text].astype(object).where(frame[text].notna(), None)
    return frame


def add_appointment(cfg: dict, pseudonym: str, day: str, time: str = "", activity: str = "", note: str = "") -> None:
    with connect(cfg) as con:
        con.execute("INSERT INTO appointments (pseudonym, day, time, activity, note) VALUES (?,?,?,?,?)",
                    (pseudonym, str(day), time, activity, note))


def delete_appointment(cfg: dict, appointment_id: int) -> None:
    with connect(cfg) as con:
        con.execute("DELETE FROM appointments WHERE id = ?", (int(appointment_id),))


def appointments(cfg: dict) -> pd.DataFrame:
    with connect(cfg) as con:
        return pd.read_sql_query("SELECT * FROM appointments ORDER BY day, time", con)


# ------------------------------------------------------------------ progress and checks

def progress(cfg: dict) -> pd.DataFrame:
    """What has been recorded per participant, read from the research data by pseudonym."""
    paths = Paths.from_config(cfg)
    circuits = cfg["study"]["circuits"]
    people = participants(cfg)
    review_done, review_total, sessions_done, sessions_total = {}, {}, {}, {}
    if (paths.review / "assignments.parquet").exists():
        from . import review
        review_total = pd.read_parquet(paths.review / "assignments.parquet").groupby("expert_id").size().to_dict()
        review_done = review.load_responses(paths.review).groupby("expert_id").size().to_dict()
        if (paths.review / "session_assignments.parquet").exists():
            sessions_total = pd.read_parquet(paths.review / "session_assignments.parquet").groupby("expert_id").size().to_dict()
            sessions_done = review.load_session_picks(paths.review).groupby("expert_id")["session_key"].nunique().to_dict()
    rows = []
    for p in people.itertuples():
        row = {"pseudonym": p.pseudonym}
        if p.role == "driver":
            for circuit in circuits:
                folder = paths.raw / p.pseudonym / circuit
                row[f"sessions_{circuit}"] = len(list(folder.glob("*.parquet"))) if folder.exists() else 0
            record = paths.participants / f"{p.pseudonym}.json"
            answers = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
            row["questionnaire"] = bool(answers.get("questionnaire"))
            done = sum(min(row[f"sessions_{c}"], cfg["study"]["sessions_per_circuit"]) for c in circuits)
            row["done"], row["total"] = done, len(circuits) * cfg["study"]["sessions_per_circuit"]
        else:
            row["done"] = review_done.get(p.pseudonym, 0) + sessions_done.get(p.pseudonym, 0)
            row["total"] = review_total.get(p.pseudonym, 0) + sessions_total.get(p.pseudonym, 0)
        rows.append(row)
    return pd.DataFrame(rows, columns=["pseudonym", *[f"sessions_{c}" for c in circuits], "questionnaire", "done", "total"])


def open_points(cfg: dict) -> pd.DataFrame:
    """Things that have to be settled per participant before or during the study."""
    people = participants(cfg)
    plan = appointments(cfg)
    first = plan.groupby("pseudonym")["day"].min().to_dict() if len(plan) else {}
    rows = []
    for p in people.itertuples():
        if p.withdrawn:
            continue
        issues = []
        if not p.is_adult:
            issues.append("age of 18 or older not confirmed")
        if p.role == "driver" and not p.no_health_exclusion:
            issues.append("health exclusion criteria not checked")
        if not p.is_independent:
            issues.append("independence from the researcher not confirmed")
        if p.role == "driver" and p.level not in LEVELS:
            issues.append("experience level not set")
        if not p.letter_sent:
            issues.append("information letter not sent")
        if not p.consent_signed:
            issues.append("consent form not signed")
        start = first.get(p.pseudonym)
        if start and p.letter_sent:
            gap = (dt.date.fromisoformat(start) - dt.date.fromisoformat(p.letter_sent)).days
            if gap < MIN_DAYS_LETTER:
                issues.append(f"first appointment is {gap} days after the letter, at least {MIN_DAYS_LETTER} required")
        if start and not p.consent_signed and dt.date.fromisoformat(start) < dt.date.today():
            issues.append("an appointment has passed without a signed consent form")
        if issues:
            rows.append({"pseudonym": p.pseudonym, "role": p.role, "open": "; ".join(issues)})
    return pd.DataFrame(rows, columns=["pseudonym", "role", "open"])


def recruitment(cfg: dict) -> pd.DataFrame:
    """Number of active drivers per experience level against the target of the study."""
    people = participants(cfg)
    active = people[(people["role"] == "driver") & people["withdrawn"].isna()]
    counts = active["level"].value_counts()
    return pd.DataFrame([{"level": lv, "drivers": int(counts.get(lv, 0)), "target": 3, "minimum": 2} for lv in LEVELS])


# ------------------------------------------------------------------ link with the research data

def sync_to_data(cfg: dict) -> list[str]:
    """Write level and consent of the drivers to the participant files that the logger and the export use.

    Only the pseudonym goes across. Names and contact details never leave the register.
    """
    paths = Paths.from_config(cfg).ensure()
    written = []
    for p in participants(cfg).itertuples():
        if p.role != "driver":
            continue
        path = paths.participants / f"{p.pseudonym}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"driver": p.pseudonym}
        record.update(driver=p.pseudonym, consent_signed=bool(p.consent_signed) and not p.withdrawn,
                      publication_consent=bool(p.publication_consent) and not p.withdrawn, withdrawn=bool(p.withdrawn))
        if p.level in LEVELS:
            record["experience_level"] = p.level
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(p.pseudonym)
    return written


def experts_without_publication_consent(cfg: dict) -> list[str]:
    people = participants(cfg)
    experts = people[people["role"] == "expert"]
    return sorted(experts.loc[(experts["publication_consent"] == 0) | experts["withdrawn"].notna(), "pseudonym"])


def data_locations(cfg: dict, pseudonym: str) -> list[str]:
    """Where the data of one participant are stored, for handling a withdrawal by hand."""
    paths = Paths.from_config(cfg)
    found = [str(p) for p in (paths.raw / pseudonym, paths.participants / f"{pseudonym}.json") if p.exists()]
    if pseudonym.startswith("E") and (paths.review / "responses.db").exists():
        found.append(f"{paths.review / 'responses.db'} (rows with expert_id = {pseudonym})")
    return found


def remove_identities(cfg: dict) -> int:
    """Delete names, contact details and notes of everyone. Pseudonyms and consent records stay.

    To be done once the thesis has been completed. It cannot be undone.
    """
    with connect(cfg) as con:
        n = con.execute("SELECT COUNT(*) FROM participants WHERE name IS NOT NULL OR email IS NOT NULL").fetchone()[0]
        con.execute(f"UPDATE participants SET {', '.join(f'{k} = NULL' for k in IDENTITY)}")
        con.execute("UPDATE appointments SET note = NULL")
    con = connect(cfg)
    con.isolation_level = None          # outside a transaction, so that the removed text is also erased from the file
    con.execute("VACUUM")
    con.close()
    return int(n)
