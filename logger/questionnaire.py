"""Experience questionnaire for drivers, in Dutch and English.

The answers describe sim racing and driving experience only. Age, gender and name are not
asked. The experience level used in the study is assigned by the researcher on the basis of
these answers and is stored next to them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# kind: choice (one answer), multi (several answers), scale (1 to 5), text (optional free text)
# Options are stored by their English text, so that answers given in either language compare.
QUESTIONS = [
    {"section": {"en": "Sim racing", "nl": "Simracen"}},
    {"id": "sim_years", "kind": "choice",
     "en": "For how long have you been sim racing?", "nl": "Hoe lang simrace je al?",
     "options": [("Never done it", "Nooit gedaan"), ("Less than 1 year", "Korter dan 1 jaar"), ("1 to 3 years", "1 tot 3 jaar"),
                 ("3 to 5 years", "3 tot 5 jaar"), ("More than 5 years", "Langer dan 5 jaar")]},
    {"id": "sim_hours_week", "kind": "choice",
     "en": "How many hours per week did you sim race on average over the past six months?",
     "nl": "Hoeveel uur per week heb je de afgelopen zes maanden gemiddeld gesimracet?",
     "options": [("None", "Geen"), ("Less than 2", "Minder dan 2"), ("2 to 5", "2 tot 5"), ("5 to 10", "5 tot 10"),
                 ("More than 10", "Meer dan 10")]},
    {"id": "sims_used", "kind": "multi",
     "en": "Which simulators or racing games have you driven for more than a few hours?",
     "nl": "Welke simulators of racegames heb je meer dan een paar uur gereden?",
     "options": [("Assetto Corsa Competizione", "Assetto Corsa Competizione"), ("Assetto Corsa", "Assetto Corsa"),
                 ("iRacing", "iRacing"), ("rFactor 2 / Le Mans Ultimate", "rFactor 2 / Le Mans Ultimate"),
                 ("Automobilista 2", "Automobilista 2"), ("F1 series", "F1-serie"), ("Gran Turismo", "Gran Turismo"),
                 ("Forza", "Forza"), ("Other", "Anders"), ("None", "Geen")]},
    {"id": "acc_hours", "kind": "choice",
     "en": "Roughly how many hours have you driven in Assetto Corsa Competizione in total?",
     "nl": "Hoeveel uur heb je in totaal ongeveer in Assetto Corsa Competizione gereden?",
     "options": [("None", "Geen"), ("Less than 10", "Minder dan 10"), ("10 to 50", "10 tot 50"), ("50 to 200", "50 tot 200"),
                 ("200 to 1000", "200 tot 1000"), ("More than 1000", "Meer dan 1000")]},
    {"id": "gt3_familiarity", "kind": "scale",
     "en": "How familiar are you with driving GT3 cars in a simulator? (1 = not at all, 5 = very familiar)",
     "nl": "Hoe vertrouwd ben je met het rijden van GT3-auto's in een simulator? (1 = helemaal niet, 5 = zeer vertrouwd)"},
    {"id": "laps_monza", "kind": "choice",
     "en": "How many laps have you driven at Monza in any simulator?",
     "nl": "Hoeveel rondes heb je in een simulator op Monza gereden?",
     "options": [("None", "Geen"), ("Fewer than 20", "Minder dan 20"), ("20 to 100", "20 tot 100"), ("More than 100", "Meer dan 100")]},
    {"id": "laps_hungaroring", "kind": "choice",
     "en": "How many laps have you driven at the Hungaroring in any simulator?",
     "nl": "Hoeveel rondes heb je in een simulator op de Hungaroring gereden?",
     "options": [("None", "Geen"), ("Fewer than 20", "Minder dan 20"), ("20 to 100", "20 tot 100"), ("More than 100", "Meer dan 100")]},
    {"id": "best_laps", "kind": "text",
     "en": "If you know them: your best lap times in a GT3 car at Monza and the Hungaroring (optional)",
     "nl": "Als je ze weet: je beste rondetijden in een GT3-auto op Monza en de Hungaroring (optioneel)"},

    {"section": {"en": "Equipment at home", "nl": "Apparatuur thuis"}},
    {"id": "home_setup", "kind": "choice",
     "en": "What do you normally drive with?", "nl": "Waarmee rijd je normaal?",
     "options": [("I do not sim race at home", "Ik simrace thuis niet"), ("Controller or keyboard", "Controller of toetsenbord"),
                 ("Wheel and pedals on a desk", "Stuur en pedalen aan een bureau"),
                 ("Wheel and pedals on a cockpit or rig", "Stuur en pedalen op een cockpit of rig")]},
    {"id": "wheel_type", "kind": "choice",
     "en": "What kind of wheel base do you use?", "nl": "Wat voor stuurbasis gebruik je?",
     "options": [("Not applicable", "Niet van toepassing"), ("Gear or belt driven", "Tandwiel- of riemaandrijving"),
                 ("Direct drive", "Direct drive"), ("I do not know", "Weet ik niet")]},
    {"id": "pedal_type", "kind": "choice",
     "en": "What kind of brake pedal do you use?", "nl": "Wat voor rempedaal gebruik je?",
     "options": [("Not applicable", "Niet van toepassing"), ("Position based (potentiometer or hall sensor)", "Op positie (potentiometer of hallsensor)"),
                 ("Load cell", "Load cell"), ("Hydraulic", "Hydraulisch"), ("I do not know", "Weet ik niet")]},

    {"section": {"en": "Competition", "nl": "Competitie"}},
    {"id": "competition_level", "kind": "choice",
     "en": "What is the highest level at which you have raced online?",
     "nl": "Wat is het hoogste niveau waarop je online hebt geracet?",
     "options": [("I have not raced online", "Ik heb niet online geracet"), ("Public lobbies", "Openbare lobby's"),
                 ("Organised leagues", "Georganiseerde competities"),
                 ("Ranked platforms (for example LFM or iRacing official series)", "Platforms met ranking (bijvoorbeeld LFM of officiële iRacing-series)"),
                 ("Esports at national or international level", "Esports op nationaal of internationaal niveau")]},
    {"id": "online_races", "kind": "choice",
     "en": "Roughly how many online races have you driven?", "nl": "Hoeveel online races heb je ongeveer gereden?",
     "options": [("None", "Geen"), ("Fewer than 10", "Minder dan 10"), ("10 to 50", "10 tot 50"), ("50 to 200", "50 tot 200"),
                 ("More than 200", "Meer dan 200")]},
    {"id": "rating", "kind": "text",
     "en": "If you have one: your rating and the platform (for example LFM Elo or iRating) (optional)",
     "nl": "Als je die hebt: je rating en het platform (bijvoorbeeld LFM Elo of iRating) (optioneel)"},

    {"section": {"en": "Driving outside the simulator", "nl": "Rijden buiten de simulator"}},
    {"id": "licence_years", "kind": "choice",
     "en": "For how long have you had a driving licence for a car?", "nl": "Hoe lang heb je een rijbewijs voor de auto?",
     "options": [("No licence", "Geen rijbewijs"), ("Less than 2 years", "Korter dan 2 jaar"), ("2 to 5 years", "2 tot 5 jaar"),
                 ("More than 5 years", "Langer dan 5 jaar")]},
    {"id": "karting", "kind": "choice",
     "en": "What is your experience with karting?", "nl": "Wat is je ervaring met karten?",
     "options": [("None", "Geen"), ("Rental karts now and then", "Af en toe huurkarts"), ("Rental karts regularly", "Regelmatig huurkarts"),
                 ("Competition karting", "Wedstrijdkarten")]},
    {"id": "track_days", "kind": "choice",
     "en": "How often have you driven a car on a real circuit?", "nl": "Hoe vaak heb je met een auto op een echt circuit gereden?",
     "options": [("Never", "Nooit"), ("1 to 5 times", "1 tot 5 keer"), ("More than 5 times", "Vaker dan 5 keer")]},
    {"id": "real_racing", "kind": "text",
     "en": "If you have raced in real motorsport: which class and for how long? (optional)",
     "nl": "Als je in de echte autosport hebt geracet: in welke klasse en hoe lang? (optioneel)"},

    {"section": {"en": "Coaching and data", "nl": "Coaching en data"}},
    {"id": "coaching", "kind": "choice",
     "en": "Have you ever received coaching on your driving?", "nl": "Heb je ooit coaching op je rijden gehad?",
     "options": [("Never", "Nooit"), ("Tips from friends or videos", "Tips van vrienden of video's"),
                 ("Structured coaching in sim racing", "Gestructureerde coaching in simracen"),
                 ("Coaching in real motorsport or karting", "Coaching in de echte autosport of karting")]},
    {"id": "telemetry_use", "kind": "choice",
     "en": "How do you use telemetry to improve your driving?", "nl": "Hoe gebruik je telemetrie om je rijden te verbeteren?",
     "options": [("I do not know what it is", "Ik weet niet wat het is"), ("I know it but do not use it", "Ik ken het, maar gebruik het niet"),
                 ("Now and then", "Af en toe"), ("Regularly", "Regelmatig")]},
    {"id": "telemetry_tools", "kind": "text",
     "en": "Which telemetry tools have you used? (optional)", "nl": "Welke telemetrietools heb je gebruikt? (optioneel)"},
    {"id": "setup_knowledge", "kind": "scale",
     "en": "How well can you adjust a car setup yourself? (1 = not at all, 5 = very well)",
     "nl": "Hoe goed kun je zelf een afstelling van een auto aanpassen? (1 = helemaal niet, 5 = zeer goed)"},

    {"section": {"en": "Driving aids and self-assessment", "nl": "Rijhulpen en zelfinschatting"}},
    {"id": "aids", "kind": "multi",
     "en": "Which driving aids do you normally use?", "nl": "Welke rijhulpen gebruik je normaal?",
     "options": [("Racing line", "Ideale lijn"), ("Automatic gearbox", "Automatische versnellingsbak"),
                 ("Stability control", "Stabiliteitscontrole"), ("Automatic clutch", "Automatische koppeling"),
                 ("None of these", "Geen van deze")]},
    {"id": "self_skill", "kind": "scale",
     "en": "How would you rate your own level as a sim racer? (1 = beginner, 5 = expert)",
     "nl": "Hoe schat je je eigen niveau als simracer in? (1 = beginner, 5 = expert)"},
    {"id": "self_consistency", "kind": "scale",
     "en": "How consistent are your lap times over a stint? (1 = very variable, 5 = very consistent)",
     "nl": "Hoe constant zijn je rondetijden over een stint? (1 = sterk wisselend, 5 = zeer constant)"},
    {"id": "self_weakness", "kind": "multi",
     "en": "Where do you think you lose most time?", "nl": "Waar denk je de meeste tijd te verliezen?",
     "options": [("Braking", "Remmen"), ("Corner entry", "Insturen"), ("Mid corner", "Midden van de bocht"),
                 ("Corner exit", "Uitkomen van de bocht"), ("Consistency", "Constantheid"), ("I do not know", "Weet ik niet")]},
    {"id": "gaming_hours_week", "kind": "choice",
     "en": "How many hours per week do you play video games other than racing games?",
     "nl": "Hoeveel uur per week speel je andere videogames dan racegames?",
     "options": [("None", "Geen"), ("Less than 2", "Minder dan 2"), ("2 to 10", "2 tot 10"), ("More than 10", "Meer dan 10")]},
]

ITEMS = [q for q in QUESTIONS if "id" in q]
REQUIRED = [q["id"] for q in ITEMS if q["kind"] != "text"]


def option_text(question: dict, lang: str) -> list[str]:
    return [en if lang == "en" else nl for en, nl in question["options"]]


def to_english(question: dict, shown: str) -> str:
    """Translate a shown option back to its English text, which is what gets stored."""
    for en, nl in question.get("options", []):
        if shown in (en, nl):
            return en
    return shown


def missing(answers: dict) -> list[str]:
    return [qid for qid in REQUIRED if answers.get(qid) in (None, "", [])]


def participant_path(data_dir: str | Path, driver: str) -> Path:
    folder = Path(data_dir) / "participants"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{driver}.json"


def load_participant(data_dir: str | Path, driver: str) -> dict:
    path = participant_path(data_dir, driver)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"driver": driver}


def save_participant(data_dir: str | Path, driver: str, **fields) -> dict:
    """Update the file of a participant. Only the pseudonym identifies the file."""
    record = load_participant(data_dir, driver)
    record.update(fields, driver=driver, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    participant_path(data_dir, driver).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
