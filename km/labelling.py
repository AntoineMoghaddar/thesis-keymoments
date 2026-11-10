"""Labelling functions for the key-moment schema K1 to K8.

Every function looks at one window and answers 1 (this type of moment is present), 0 (it is
not) or -1 (abstain, the measurement it needs is missing). Thresholds are relative to the
reference lap and are set per segment at a percentile of the observed distribution, with an
absolute floor so that sensor noise on an easy corner does not produce labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ABSTAIN = -1
TYPES = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"]

SCHEMA = {
    "K1": dict(phase="entry", name="Braking point", advice="Move braking point"),
    "K2": dict(phase="entry", name="Brake application and release",
               advice="Brake harder at onset, release progressively"),
    "K3": dict(phase="middle", name="Minimum corner speed", advice="Carry more speed to the apex"),
    "K4": dict(phase="exit", name="Throttle application", advice="Apply throttle earlier and without lifting"),
    "K5": dict(phase="exit", name="Traction and stability", advice="Apply throttle more progressively"),
    "K6": dict(phase="whole corner", name="Steering and line", advice="Reduce steering input, adjust line"),
    "K7": dict(phase="any", name="Error or incident", advice="Review cause of incident"),
    "K8": dict(phase="any", name="Inconsistency", advice="Focus practice on this corner"),
}

# Quantities for which a per-segment percentile threshold is fitted: name -> (column, transform)
QUANTITIES = {
    "K1_onset": ("d_brake_onset_m", np.abs),
    "K2_peak": ("d_peak_brake", lambda x: -x),          # peak pressure below the reference
    "K2_release": ("d_brake_release_s", lambda x: -x),  # brake released more abruptly than the reference
    "K3_speed": ("d_min_speed", lambda x: -x),
    "K4_pickup": ("d_throttle_pickup_m", lambda x: x),
    "K5_slip": ("slip_exit_max", lambda x: x),
    "K5_tc": ("tc_frac", lambda x: x),
    "K6_steer": ("d_steer_integral", lambda x: x),
    "K6_line": ("line_dev_max", lambda x: x),
    "K7_loss": ("time_loss", lambda x: x),
}
PERCENTILE_KEY = {"K1_onset": "K1", "K2_peak": "K2", "K2_release": "K2", "K3_speed": "K3", "K4_pickup": "K4",
                  "K5_slip": "K5", "K5_tc": "K5", "K6_steer": "K6", "K6_line": "K6", "K7_loss": "K7"}
FLOORS = {"K1_onset": ("K1", "floor_m"), "K2_peak": ("K2", "floor_peak"), "K3_speed": ("K3", "floor_kmh"),
          "K4_pickup": ("K4", "floor_m"), "K7_loss": ("K7", "floor_s")}


def _value(windows: pd.DataFrame, quantity: str) -> pd.Series:
    column, transform = QUANTITIES[quantity]
    return transform(windows[column].astype(float))


def fit_thresholds(windows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Threshold per circuit, segment and quantity. Reference laps are left out of the fit."""
    lab = cfg["labelling"]
    base = windows[~windows["is_reference"]]
    rows = []
    for (circuit, segment), g in base.groupby(["circuit", "segment"]):
        row = dict(circuit=circuit, segment=segment)
        for q in QUANTITIES:
            values = _value(g, q).dropna()
            thr = float(np.percentile(values, lab[PERCENTILE_KEY[q]]["percentile"])) if len(values) else np.inf
            if q in FLOORS:
                key, name = FLOORS[q]
                thr = max(thr, float(lab[key][name]))
            row[q] = max(thr, 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def apply(windows: pd.DataFrame, thresholds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Votes of the eight labelling functions: one column per type with values 1, 0 or -1."""
    lab = cfg["labelling"]
    w = windows.merge(thresholds, on=["circuit", "segment"], how="left", suffixes=("", "_thr"))
    corner = w["is_corner"].to_numpy() == 1
    slower = (w["time_loss"] > lab["min_time_loss_s"]).to_numpy()

    def exceeds(q):
        return (_value(w, q) > w[q]).to_numpy(), _value(w, q).isna().to_numpy()

    votes = {}

    def vote(name, fired, missing, needs_corner=True, needs_loss=True):
        fired = fired & (slower if needs_loss else True)
        out = np.where(fired, 1, np.where(missing, ABSTAIN, 0))
        if needs_corner:
            out = np.where(corner, out, ABSTAIN)
        votes[name] = out

    f, m = exceeds("K1_onset")
    vote("K1", f, m)
    (f1, m1), (f2, m2) = exceeds("K2_peak"), exceeds("K2_release")
    vote("K2", f1 | f2, m1 & m2)
    f, m = exceeds("K3_speed")
    vote("K3", f, m)
    f, m = exceeds("K4_pickup")
    lifts = (w["d_throttle_lifts"] >= 1).to_numpy()
    vote("K4", f | lifts, m & w["d_throttle_lifts"].isna().to_numpy())
    (f1, m1), (f2, m2) = exceeds("K5_slip"), exceeds("K5_tc")
    vote("K5", f1 | f2, m1 & m2)
    (f1, m1), (f2, m2) = exceeds("K6_steer"), exceeds("K6_line")
    vote("K6", f1 | f2, m1 & m2)
    f, m = exceeds("K7_loss")
    off = (w["tyres_out_max"] >= lab["K7"]["tyres_out"]).to_numpy()
    vote("K7", f | off, m, needs_corner=False, needs_loss=False)
    inconsistent = (w["time_cv_ratio"] > lab["K8"]["ratio"]) & (w["seg_time"] > w["session_median_time"])
    vote("K8", inconsistent.to_numpy(), w["time_cv_ratio"].isna().to_numpy(), needs_loss=False)

    out = pd.DataFrame(votes, index=windows.index)
    # A reference lap is compared with itself and therefore carries no information.
    out.loc[windows["is_reference"].to_numpy(), [t for t in TYPES if t != "K7"]] = ABSTAIN
    return out


def aggregate(votes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weak label per window: key moment yes or no, and the primary type when several functions fire."""
    priority = cfg["labelling"]["priority"]
    fired = votes[priority] == 1
    n_fired = fired.sum(axis=1)
    primary = np.where(n_fired > 0, fired.idxmax(axis=1), "none")
    return pd.DataFrame({"weak_label": (n_fired > 0).astype(int), "weak_type": primary,
                         "n_fired": n_fired.astype(int)}, index=votes.index)


def label(windows: pd.DataFrame, cfg: dict, fit_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Votes and aggregated labels. `fit_mask` restricts the windows the thresholds are fitted on."""
    thresholds = fit_thresholds(windows if fit_mask is None else windows[fit_mask], cfg)
    votes = apply(windows, thresholds, cfg)
    return pd.concat([votes, aggregate(votes, cfg)], axis=1)


def statistics(votes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Coverage, firing rate, overlap and conflict per labelling function.

    coverage  share of windows on which the function does not abstain
    fires     share of windows on which it answers 1
    overlap   share of its positive votes that coincide with a positive vote of another function
    conflict  share of its positive votes for which another type became the primary type
    """
    agg = aggregate(votes[TYPES], cfg)
    fired = votes[TYPES] == 1
    rows = []
    for t in TYPES:
        own = fired[t]
        others = fired.drop(columns=t).any(axis=1)
        n = int(own.sum())
        rows.append(dict(function=t, coverage=float((votes[t] != ABSTAIN).mean()), fires=float(own.mean()),
                         n_fired=n, overlap=float((own & others).sum() / n) if n else np.nan,
                         conflict=float((own & (agg["weak_type"] != t)).sum() / n) if n else np.nan,
                         n_primary=int((agg["weak_type"] == t).sum())))
    return pd.DataFrame(rows)
