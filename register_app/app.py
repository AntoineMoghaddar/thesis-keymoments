"""Register of participants, for the researcher only. Start with: streamlit run register_app/app.py

Names and contact details are stored in the folder `register_dir` of the configuration, apart
from the research data. Set KM_CONFIG to the configuration file that points to the university storage.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from km import register  # noqa: E402
from km.config import load_config  # noqa: E402

T = {
    "title": {"en": "Participant register", "nl": "Deelnemersregister"},
    "private": {"en": "This register holds names and contact details. It is stored in {folder}, apart from the research data.",
                "nl": "Dit register bevat namen en contactgegevens. Het staat in {folder}, los van de onderzoeksdata."},
    "in_repo": {"en": "The register folder lies inside the repository. Set register_dir to a folder on the university storage before entering real participants.",
                "nl": "De registermap ligt in de repository. Zet register_dir op een map op de universiteitsopslag voordat je echte deelnemers invoert."},
    "overview": {"en": "Overview", "nl": "Overzicht"}, "person": {"en": "Add or edit", "nl": "Toevoegen of bewerken"},
    "planning": {"en": "Planning", "nl": "Planning"}, "closing": {"en": "Data and closing", "nl": "Data en afsluiten"},
    "recruitment": {"en": "Drivers per experience level", "nl": "Coureurs per ervaringsniveau"},
    "open": {"en": "Open points", "nl": "Openstaande punten"}, "none_open": {"en": "Nothing is open.", "nl": "Er staat niets open."},
    "people": {"en": "Participants and progress", "nl": "Deelnemers en voortgang"},
    "empty": {"en": "No participants yet.", "nl": "Nog geen deelnemers."},
    "who": {"en": "Participant", "nl": "Deelnemer"}, "new_driver": {"en": "New driver", "nl": "Nieuwe coureur"},
    "new_expert": {"en": "New expert", "nl": "Nieuwe expert"},
    "name": {"en": "Name", "nl": "Naam"}, "email": {"en": "Email", "nl": "E-mail"}, "phone": {"en": "Phone", "nl": "Telefoon"},
    "channel": {"en": "Recruited through", "nl": "Geworven via"}, "level": {"en": "Experience level", "nl": "Ervaringsniveau"},
    "background": {"en": "Basis for the level, or for inclusion as an expert", "nl": "Onderbouwing van het niveau, of van opname als expert"},
    "criteria": {"en": "Inclusion criteria", "nl": "Inclusiecriteria"},
    "is_adult": {"en": "18 years or older", "nl": "18 jaar of ouder"},
    "no_health_exclusion": {"en": "No photosensitive epilepsy and no severe motion sickness (drivers)",
                            "nl": "Geen fotosensitieve epilepsie en geen ernstige bewegingsziekte (coureurs)"},
    "is_independent": {"en": "Not a student in a course of the researcher and not a relative",
                       "nl": "Geen student in een vak van de onderzoeker en geen familielid"},
    "consent": {"en": "Information and consent", "nl": "Informatie en toestemming"},
    "letter_sent": {"en": "Information letter sent on", "nl": "Informatiebrief verstuurd op"},
    "consent_signed": {"en": "Consent form signed on", "nl": "Toestemmingsformulier getekend op"},
    "publication_consent": {"en": "Consent for publication of the data", "nl": "Toestemming voor publicatie van de data"},
    "quote_consent": {"en": "Comments may be quoted anonymously (experts)", "nl": "Opmerkingen mogen anoniem geciteerd worden (experts)"},
    "wants_summary": {"en": "Wants a summary of the results", "nl": "Wil een samenvatting van de resultaten"},
    "future_contact": {"en": "May be contacted for future research", "nl": "Mag benaderd worden voor vervolgonderzoek"},
    "withdrawn": {"en": "Withdrawn on", "nl": "Teruggetrokken op"}, "notes": {"en": "Notes", "nl": "Notities"},
    "not_yet": {"en": "not yet", "nl": "nog niet"}, "save": {"en": "Save", "nl": "Opslaan"},
    "saved": {"en": "Saved as {code}.", "nl": "Opgeslagen als {code}."},
    "need_name": {"en": "Enter a name.", "nl": "Vul een naam in."},
    "day": {"en": "Date", "nl": "Datum"}, "time": {"en": "Time", "nl": "Tijd"}, "activity": {"en": "Activity", "nl": "Activiteit"},
    "add": {"en": "Add appointment", "nl": "Afspraak toevoegen"}, "upcoming": {"en": "Appointments", "nl": "Afspraken"},
    "remove": {"en": "Remove appointment", "nl": "Afspraak verwijderen"},
    "sync": {"en": "Write level and consent to the research data", "nl": "Niveau en toestemming naar de onderzoeksdata schrijven"},
    "sync_help": {"en": "Only pseudonyms go across. The logger and the dataset export read these files.",
                  "nl": "Alleen pseudoniemen gaan mee. De logger en de datasetexport lezen deze bestanden."},
    "synced": {"en": "Written for: {codes}", "nl": "Geschreven voor: {codes}"},
    "experts_out": {"en": "Experts without consent for publication (put these in export.exclude_experts): {codes}",
                    "nl": "Experts zonder toestemming voor publicatie (zet deze in export.exclude_experts): {codes}"},
    "withdrawal": {"en": "Where are the data of a participant?", "nl": "Waar staan de data van een deelnemer?"},
    "summary_list": {"en": "Addresses for the summary of the results", "nl": "Adressen voor de samenvatting van de resultaten"},
    "wipe": {"en": "Remove names and contact details", "nl": "Namen en contactgegevens verwijderen"},
    "wipe_help": {"en": "To be done once the thesis has been completed, as promised in the information letter. Pseudonyms and consent records stay. This cannot be undone. Type REMOVE to confirm.",
                  "nl": "Uit te voeren zodra de thesis is afgerond, zoals toegezegd in de informatiebrief. Pseudoniemen en toestemmingen blijven staan. Dit kan niet ongedaan worden gemaakt. Typ REMOVE om te bevestigen."},
    "wiped": {"en": "Identities of {n} participants removed.", "nl": "Identiteit van {n} deelnemers verwijderd."},
    "password": {"en": "Password", "nl": "Wachtwoord"},
}
ACTIVITIES = ["Introduction and questionnaire", "Sessions Monza", "Sessions Hungaroring", "Expert review", "Other"]


def t(key: str, lang: str, **values) -> str:
    text = T[key][lang]
    return text.format(**values) if values else text


def as_date(value):
    return dt.date.fromisoformat(value) if value else None


def page_overview(cfg: dict, lang: str) -> None:
    st.subheader(t("recruitment", lang))
    st.dataframe(register.recruitment(cfg), hide_index=True)
    st.subheader(t("open", lang))
    points = register.open_points(cfg)
    if points.empty:
        st.success(t("none_open", lang))
    else:
        st.dataframe(points, hide_index=True, width="stretch")
    st.subheader(t("people", lang))
    people = register.participants(cfg)
    if people.empty:
        st.info(t("empty", lang))
        return
    table = people[["pseudonym", "role", "name", "level", "channel", "letter_sent", "consent_signed",
                    "publication_consent", "withdrawn"]].merge(register.progress(cfg), on="pseudonym", how="left")
    st.dataframe(table, hide_index=True, width="stretch")


def page_person(cfg: dict, lang: str) -> None:
    people = register.participants(cfg)
    options = ["new_driver", "new_expert"] + people["pseudonym"].tolist()
    names = dict(zip(people["pseudonym"], people["name"].fillna("")))
    choice = st.selectbox(t("who", lang), options,
                          format_func=lambda v: t(v, lang) if v.startswith("new_") else f"{v}  {names[v]}")
    new = choice.startswith("new_")
    role = choice[4:] if new else people.set_index("pseudonym").loc[choice, "role"]
    old = {} if new else people.set_index("pseudonym").loc[choice].to_dict()

    def text(key):
        value = old.get(key)
        return "" if value is None or pd.isna(value) else str(value)

    with st.form(f"person_{choice}"):
        c1, c2, c3 = st.columns(3)
        fields = {"name": c1.text_input(t("name", lang), text("name")), "email": c2.text_input(t("email", lang), text("email")),
                  "phone": c3.text_input(t("phone", lang), text("phone"))}
        c1, c2 = st.columns(2)
        channel = text("channel") or register.CHANNELS[0]
        fields["channel"] = c1.selectbox(t("channel", lang), register.CHANNELS, index=register.CHANNELS.index(channel)
                                         if channel in register.CHANNELS else 0)
        if role == "driver":
            levels = ["", *register.LEVELS]
            fields["level"] = c2.selectbox(t("level", lang), levels, index=levels.index(text("level")) if text("level") in levels else 0)
        fields["background"] = st.text_area(t("background", lang), text("background"), height=80)
        st.markdown(f"**{t('criteria', lang)}**")
        for key in ("is_adult", "no_health_exclusion", "is_independent"):
            if key == "no_health_exclusion" and role != "driver":
                continue
            fields[key] = st.checkbox(t(key, lang), bool(old.get(key)))
        st.markdown(f"**{t('consent', lang)}**")
        c1, c2, c3 = st.columns(3)
        fields["letter_sent"] = c1.date_input(t("letter_sent", lang), as_date(text("letter_sent")), format="DD-MM-YYYY")
        fields["consent_signed"] = c2.date_input(t("consent_signed", lang), as_date(text("consent_signed")), format="DD-MM-YYYY")
        fields["withdrawn"] = c3.date_input(t("withdrawn", lang), as_date(text("withdrawn")), format="DD-MM-YYYY")
        for key in ("publication_consent", "quote_consent", "wants_summary", "future_contact"):
            if key == "quote_consent" and role != "expert":
                continue
            fields[key] = st.checkbox(t(key, lang), bool(old.get(key)))
        fields["notes"] = st.text_area(t("notes", lang), text("notes"), height=80)
        if st.form_submit_button(t("save", lang), type="primary"):
            if not fields["name"].strip():
                st.error(t("need_name", lang))
            else:
                code = register.add_participant(cfg, role, **fields) if new else choice
                if not new:
                    register.update_participant(cfg, code, **fields)
                st.success(t("saved", lang, code=code))


def page_planning(cfg: dict, lang: str) -> None:
    people = register.participants(cfg)
    if people.empty:
        st.info(t("empty", lang))
        return
    names = dict(zip(people["pseudonym"], people["name"].fillna("")))
    with st.form("appointment"):
        c1, c2, c3, c4 = st.columns(4)
        who = c1.selectbox(t("who", lang), people["pseudonym"].tolist(), format_func=lambda v: f"{v}  {names[v]}")
        day = c2.date_input(t("day", lang), dt.date.today(), format="DD-MM-YYYY")
        time = c3.text_input(t("time", lang), "14:00")
        activity = c4.selectbox(t("activity", lang), ACTIVITIES)
        if st.form_submit_button(t("add", lang)):
            register.add_appointment(cfg, who, day.isoformat(), time, activity)
    plan = register.appointments(cfg)
    st.subheader(t("upcoming", lang))
    if plan.empty:
        st.info(t("empty", lang))
        return
    plan["name"] = plan["pseudonym"].map(names)
    st.dataframe(plan[["id", "day", "time", "pseudonym", "name", "activity"]], hide_index=True, width="stretch")
    with st.form("remove"):
        target = st.selectbox(t("remove", lang), plan["id"].tolist(),
                              format_func=lambda i: " ".join(map(str, plan.set_index("id").loc[i, ["day", "time", "pseudonym", "activity"]])))
        if st.form_submit_button(t("remove", lang)):
            register.delete_appointment(cfg, target)
            st.rerun()


def page_closing(cfg: dict, lang: str) -> None:
    st.caption(t("sync_help", lang))
    if st.button(t("sync", lang)):
        st.success(t("synced", lang, codes=", ".join(register.sync_to_data(cfg)) or "-"))
    out = register.experts_without_publication_consent(cfg)
    if out:
        st.warning(t("experts_out", lang, codes=", ".join(out)))
    people = register.participants(cfg)
    if len(people):
        st.subheader(t("withdrawal", lang))
        who = st.selectbox(t("who", lang), people["pseudonym"].tolist())
        st.code("\n".join(register.data_locations(cfg, who)) or "-")
        st.subheader(t("summary_list", lang))
        wants = people[(people["wants_summary"] == 1) & people["email"].notna()]
        st.code("; ".join(wants["email"]) or "-")
    st.subheader(t("wipe", lang))
    st.caption(t("wipe_help", lang))
    with st.form("wipe"):
        word = st.text_input("REMOVE")
        if st.form_submit_button(t("wipe", lang)) and word == "REMOVE":
            st.success(t("wiped", lang, n=register.remove_identities(cfg)))


def main() -> None:
    st.set_page_config(page_title="Register", layout="wide")
    cfg = load_config(os.environ.get("KM_CONFIG"))
    if os.environ.get("KM_REGISTER_DIR"):
        cfg["register_dir"] = os.environ["KM_REGISTER_DIR"]
    if os.environ.get("KM_DATA_DIR"):
        cfg["data_dir"] = os.environ["KM_DATA_DIR"]
    with st.sidebar:
        lang = st.radio("Language / Taal", ["nl", "en"], horizontal=True, format_func=str.upper, key="lang")
        pages = {"overview": page_overview, "person": page_person, "planning": page_planning, "closing": page_closing}
        page = st.radio(" ", list(pages), format_func=lambda k: t(k, lang), label_visibility="collapsed", key="page")
    password = os.environ.get("KM_REGISTER_PASSWORD")
    if password and not st.session_state.get("unlocked"):
        with st.form("unlock"):
            given = st.text_input(t("password", lang), type="password")
            if st.form_submit_button("OK") and given == password:
                st.session_state["unlocked"] = True
                st.rerun()
        return
    st.title(t("title", lang))
    st.caption(t("private", lang, folder=register.register_dir(cfg)))
    if register.in_repository(cfg):
        st.warning(t("in_repo", lang))
    pages[page](cfg, lang)


main()
