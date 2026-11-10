"""Texts of the review tool in Dutch and English."""

T = {
    "app_title": {"en": "Review of driving segments", "nl": "Beoordeling van rijsegmenten"},
    "language": {"en": "Language", "nl": "Taal"},
    "login_intro": {"en": "Enter the personal access code that you received from the researcher.",
                    "nl": "Vul de persoonlijke toegangscode in die je van de onderzoeker hebt gekregen."},
    "code": {"en": "Access code", "nl": "Toegangscode"},
    "login": {"en": "Log in", "nl": "Inloggen"},
    "bad_code": {"en": "This code is not known.", "nl": "Deze code is niet bekend."},
    "logout": {"en": "Log out", "nl": "Uitloggen"},
    "logged_in": {"en": "Logged in as", "nl": "Ingelogd als"},
    "page_intro": {"en": "Instructions", "nl": "Instructies"},
    "page_review": {"en": "Review", "nl": "Beoordelen"},
    "page_dash": {"en": "Dashboard", "nl": "Dashboard"},
    "no_package": {"en": "No review package was found. Create it first with: km review-package",
                   "nl": "Er is geen beoordelingspakket gevonden. Maak het eerst aan met: km review-package"},
    "intro": {
        "en": """
**What you are asked to do**

You will see a series of segments. A segment is one corner, or one straight, of one recorded lap in Assetto Corsa Competizione (GT3). For each segment you see the speed, throttle, brake and steering of that lap, together with the fastest lap of the same driver on the same circuit as a reference.

For each segment you answer three questions:

1. Would you discuss this segment with the driver in a debrief?
2. A moment type has been assigned to the segment. Is it correct? If not, which type fits?
3. An advice category has been assigned to the segment. Is it suitable?

You can add a short comment. There are no right or wrong answers: your own judgement as a coach or driver is what is asked for. Some segments contain nothing remarkable, and that is intended.

**Good to know**

- The session takes about one hour. Your answers are saved after every segment, so you can stop and continue later with the same code.
- You do not see who the driver is, and the drivers do not see your answers.
- You can go back to a previous segment and change your answer.
- You can stop at any time without giving a reason.
""",
        "nl": """
**Wat er van je gevraagd wordt**

Je krijgt een reeks segmenten te zien. Een segment is een bocht, of een recht stuk, van een opgenomen ronde in Assetto Corsa Competizione (GT3). Per segment zie je de snelheid, het gas, de rem en het stuur van die ronde, samen met de snelste ronde van dezelfde coureur op hetzelfde circuit als referentie.

Bij elk segment beantwoord je drie vragen:

1. Zou je dit segment met de coureur bespreken in een debrief?
2. Aan het segment is een type moment toegekend. Klopt dat? Zo niet, welk type past wel?
3. Aan het segment is een adviescategorie toegekend. Is die passend?

Je kunt een korte opmerking toevoegen. Er zijn geen goede of foute antwoorden: het gaat om je eigen oordeel als coach of coureur. Sommige segmenten bevatten niets opvallends, en dat is de bedoeling.

**Goed om te weten**

- De sessie duurt ongeveer een uur. Je antwoorden worden na elk segment opgeslagen, dus je kunt stoppen en later met dezelfde code verdergaan.
- Je ziet niet wie de coureur is, en de coureurs zien jouw antwoorden niet.
- Je kunt terug naar een vorig segment om je antwoord te wijzigen.
- Je kunt op elk moment stoppen zonder opgaaf van reden.
"""},
    "schema_title": {"en": "Moment types and advice categories", "nl": "Typen momenten en adviescategorieën"},
    "col_type": {"en": "Moment type", "nl": "Type moment"},
    "col_phase": {"en": "Phase of the corner", "nl": "Fase van de bocht"},
    "col_advice": {"en": "Advice category", "nl": "Adviescategorie"},
    "phases": {"en": {"entry": "Entry", "middle": "Middle", "exit": "Exit", "whole corner": "Whole corner", "any": "Any"},
               "nl": {"entry": "Ingang", "middle": "Midden", "exit": "Uitgang", "whole corner": "Hele bocht", "any": "Overal"}},
    "profile_title": {"en": "About your background", "nl": "Over je achtergrond"},
    "profile_intro": {"en": "These four questions are used to describe the group of experts in the thesis. They are stored under your code, not under your name.",
                      "nl": "Deze vier vragen worden gebruikt om de groep experts in de scriptie te beschrijven. Ze worden onder je code opgeslagen, niet onder je naam."},
    "p_role": {"en": "Which roles apply to you?", "nl": "Welke rollen zijn op jou van toepassing?"},
    "p_roles": {"en": ["Driver coach (real motorsport)", "Sim racing coach", "Race engineer or data engineer",
                       "Racing driver", "Experienced sim racer"],
                "nl": ["Rijderscoach (echte autosport)", "Simracecoach", "Race-engineer of data-engineer",
                       "Coureur", "Ervaren simracer"]},
    "p_years": {"en": "Years of experience in motorsport or sim racing", "nl": "Jaren ervaring in autosport of simracen"},
    "p_coach_years": {"en": "Years of experience in coaching drivers", "nl": "Jaren ervaring in het coachen van coureurs"},
    "p_telemetry": {"en": "How often do you work with telemetry?", "nl": "Hoe vaak werk je met telemetrie?"},
    "p_telemetry_opts": {"en": ["Never", "Now and then", "Regularly", "Daily"], "nl": ["Nooit", "Af en toe", "Regelmatig", "Dagelijks"]},
    "saved": {"en": "Saved.", "nl": "Opgeslagen."},
    "save": {"en": "Save", "nl": "Opslaan"},
    "progress": {"en": "Segment {i} of {n}", "nl": "Segment {i} van {n}"},
    "all_done": {"en": "You have reviewed all segments. Thank you. You can still change answers with the buttons below.",
                 "nl": "Je hebt alle segmenten beoordeeld. Dank je wel. Met de knoppen hieronder kun je antwoorden nog wijzigen."},
    "this_lap": {"en": "This lap", "nl": "Deze ronde"},
    "reference": {"en": "Reference lap", "nl": "Referentieronde"},
    "speed": {"en": "Speed (km/h)", "nl": "Snelheid (km/u)"},
    "gas": {"en": "Throttle (%)", "nl": "Gas (%)"},
    "brake": {"en": "Brake (%)", "nl": "Rem (%)"},
    "steer": {"en": "Steering", "nl": "Stuur"},
    "delta": {"en": "Time against reference (s)", "nl": "Tijd t.o.v. referentie (s)"},
    "distance": {"en": "Distance from the start of the segment (m)", "nl": "Afstand vanaf het begin van het segment (m)"},
    "segment_area": {"en": "The shaded area is the segment that you assess. The parts before and after it are shown for context.",
                     "nl": "Het gearceerde deel is het segment dat je beoordeelt. De delen ervoor en erna zijn ter context."},
    "seg_time": {"en": "Segment time", "nl": "Segmenttijd"},
    "ref_time": {"en": "Reference", "nl": "Referentie"},
    "difference": {"en": "Difference", "nl": "Verschil"},
    "assigned_type": {"en": "Assigned moment type", "nl": "Toegekend type moment"},
    "assigned_advice": {"en": "Assigned advice category", "nl": "Toegekende adviescategorie"},
    "q1": {"en": "1. Would you discuss this segment with the driver in a debrief?",
           "nl": "1. Zou je dit segment met de coureur bespreken in een debrief?"},
    "q2": {"en": "2. Is the assigned moment type correct?", "nl": "2. Klopt het toegekende type moment?"},
    "q2b": {"en": "If not: which type fits?", "nl": "Zo niet: welk type past wel?"},
    "q3": {"en": "3. Is the assigned advice category suitable?", "nl": "3. Is de toegekende adviescategorie passend?"},
    "comment": {"en": "Comment (optional)", "nl": "Opmerking (optioneel)"},
    "yes": {"en": "Yes", "nl": "Ja"},
    "no": {"en": "No", "nl": "Nee"},
    "none_type": {"en": "None of these", "nl": "Geen van deze"},
    "choose": {"en": "Not applicable", "nl": "Niet van toepassing"},
    "save_next": {"en": "Save and continue", "nl": "Opslaan en verder"},
    "previous": {"en": "Previous segment", "nl": "Vorig segment"},
    "next": {"en": "Next segment", "nl": "Volgend segment"},
    "first_open": {"en": "Go to first open segment", "nl": "Naar eerste open segment"},
    "need_answers": {"en": "Please answer questions 1, 2 and 3.", "nl": "Beantwoord vraag 1, 2 en 3."},
    "need_type": {"en": "You answered that the type is not correct. Please choose the type that fits.",
                  "nl": "Je gaf aan dat het type niet klopt. Kies het type dat wel past."},
    "dash_done": {"en": "Reviewed", "nl": "Beoordeeld"},
    "dash_relevant": {"en": "Would discuss", "nl": "Zou bespreken"},
    "dash_type_ok": {"en": "Type correct", "nl": "Type correct"},
    "dash_advice_ok": {"en": "Advice suitable", "nl": "Advies passend"},
    "dash_time": {"en": "Median time per segment", "nl": "Mediane tijd per segment"},
    "dash_by_type": {"en": "Your answers per assigned moment type", "nl": "Je antwoorden per toegekend type moment"},
    "dash_empty": {"en": "No answers yet.", "nl": "Nog geen antwoorden."},
    "dash_all": {"en": "Progress of all experts", "nl": "Voortgang van alle experts"},
    "dash_download": {"en": "Download all responses (CSV)", "nl": "Alle antwoorden downloaden (CSV)"},
    "dash_mine": {"en": "Your answers", "nl": "Je antwoorden"},
    "count": {"en": "Number of segments", "nl": "Aantal segmenten"},
}


def t(key: str, lang: str, **values) -> str:
    text = T[key][lang]
    return text.format(**values) if values else text
