"""Advice text for a detected moment, filled with the values measured in the window (R3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labelling import SCHEMA

TYPE_NAMES = {
    "en": {k: v["name"] for k, v in SCHEMA.items()},
    "nl": {"K1": "Rempunt", "K2": "Remopbouw en lossen van de rem", "K3": "Minimale bochtsnelheid",
           "K4": "Gas geven", "K5": "Tractie en stabiliteit", "K6": "Sturen en lijn",
           "K7": "Fout of incident", "K8": "Inconsistentie"},
}
ADVICE_NAMES = {
    "en": {k: v["advice"] for k, v in SCHEMA.items()},
    "nl": {"K1": "Rempunt verleggen", "K2": "Harder aanremmen, rem geleidelijk lossen",
           "K3": "Meer snelheid meenemen naar de apex", "K4": "Eerder gas geven zonder te liften",
           "K5": "Gas geleidelijker opbouwen", "K6": "Minder sturen, lijn aanpassen",
           "K7": "Oorzaak van het incident nagaan", "K8": "Deze bocht gericht oefenen"},
}

TEMPLATES = {
    "en": {
        "K1_early": "{seg}: braking started {d:.0f} m earlier than on the reference lap, with a loss of {loss:.2f} s. Brake later.",
        "K1_late": "{seg}: braking started {d:.0f} m later than on the reference lap, with a loss of {loss:.2f} s. Brake earlier so that the corner entry stays under control.",
        "K2": "{seg}: peak brake input was {peak:.0f}% against {ref_peak:.0f}% on the reference lap, with a loss of {loss:.2f} s. Brake harder at the start and release the brake more gradually.",
        "K3": "{seg}: minimum speed was {v:.0f} km/h against {ref_v:.0f} km/h on the reference lap, with a loss of {loss:.2f} s. Carry more speed to the apex.",
        "K4": "{seg}: throttle was applied {d:.0f} m later than on the reference lap, with {lifts:.0f} lift(s) after application and a loss of {loss:.2f} s. Apply the throttle earlier and keep it applied.",
        "K5": "{seg}: rear wheel slip on exit peaked at {slip:.2f} and traction control was active for {tc:.0f}% of the exit, with a loss of {loss:.2f} s. Apply the throttle more progressively.",
        "K6": "{seg}: the line deviated up to {dev:.1f} m from the reference lap and steering input was {steer:+.0f}% compared with it, with a loss of {loss:.2f} s. Reduce steering input and adjust the line.",
        "K7": "{seg}: {loss:.2f} s was lost against the reference lap{off}. Review what caused this incident.",
        "K8": "{seg}: segment times in this session varied {ratio:.1f} times as much as in the other segments. Focus practice on this corner.",
    },
    "nl": {
        "K1_early": "{seg}: er werd {d:.0f} m eerder geremd dan in de referentieronde, met een verlies van {loss:.2f} s. Rem later.",
        "K1_late": "{seg}: er werd {d:.0f} m later geremd dan in de referentieronde, met een verlies van {loss:.2f} s. Rem eerder, zodat het insturen onder controle blijft.",
        "K2": "{seg}: de maximale reminput was {peak:.0f}% tegen {ref_peak:.0f}% in de referentieronde, met een verlies van {loss:.2f} s. Rem harder aan en los de rem geleidelijker.",
        "K3": "{seg}: de minimumsnelheid was {v:.0f} km/u tegen {ref_v:.0f} km/u in de referentieronde, met een verlies van {loss:.2f} s. Neem meer snelheid mee naar de apex.",
        "K4": "{seg}: er werd {d:.0f} m later gas gegeven dan in de referentieronde, met {lifts:.0f} keer liften daarna en een verlies van {loss:.2f} s. Geef eerder gas en houd het gas vast.",
        "K5": "{seg}: de slip van de achterwielen bij het uitkomen liep op tot {slip:.2f} en de tractiecontrole greep in gedurende {tc:.0f}% van de uitgang, met een verlies van {loss:.2f} s. Bouw het gas geleidelijker op.",
        "K6": "{seg}: de lijn week tot {dev:.1f} m af van de referentieronde en de stuurinput was {steer:+.0f}% ten opzichte daarvan, met een verlies van {loss:.2f} s. Stuur minder en pas de lijn aan.",
        "K7": "{seg}: er ging {loss:.2f} s verloren ten opzichte van de referentieronde{off}. Ga na wat dit incident veroorzaakte.",
        "K8": "{seg}: de segmenttijden in deze sessie varieerden {ratio:.1f} keer zo sterk als in de andere segmenten. Oefen deze bocht gericht.",
    },
}
OFF_TRACK = {"en": ", and the car left the track", "nl": ", en de auto kwam buiten de baan"}


def _num(row, key, default=0.0) -> float:
    value = row.get(key, default)
    return default if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


def advice_text(row: pd.Series | dict, moment_type: str, lang: str = "en") -> str:
    """One or two sentences that state what was measured and what to change."""
    if moment_type not in SCHEMA:
        return ""
    t = TEMPLATES[lang]
    loss = _num(row, "time_loss")
    seg = row.get("segment_name", row.get("segment", ""))
    if moment_type == "K1":
        d = _num(row, "d_brake_onset_m")
        return t["K1_early" if d >= 0 else "K1_late"].format(seg=seg, d=abs(d), loss=loss)
    if moment_type == "K2":
        return t["K2"].format(seg=seg, peak=100 * _num(row, "peak_brake"),
                              ref_peak=100 * _num(row, "ref_peak_brake"), loss=loss)
    if moment_type == "K3":
        return t["K3"].format(seg=seg, v=_num(row, "min_speed"), ref_v=_num(row, "ref_min_speed"), loss=loss)
    if moment_type == "K4":
        return t["K4"].format(seg=seg, d=_num(row, "d_throttle_pickup_m"), lifts=_num(row, "throttle_lifts"), loss=loss)
    if moment_type == "K5":
        return t["K5"].format(seg=seg, slip=_num(row, "slip_exit_max"), tc=100 * _num(row, "tc_frac"), loss=loss)
    if moment_type == "K6":
        ref = _num(row, "ref_steer_integral")
        steer = 100 * _num(row, "d_steer_integral") / ref if ref else 0.0
        return t["K6"].format(seg=seg, dev=_num(row, "line_dev_max"), steer=steer, loss=loss)
    if moment_type == "K7":
        off = OFF_TRACK[lang] if _num(row, "tyres_out_max") >= 3 else ""
        return t["K7"].format(seg=seg, loss=loss, off=off)
    return t["K8"].format(seg=seg, ratio=_num(row, "time_cv_ratio"))
