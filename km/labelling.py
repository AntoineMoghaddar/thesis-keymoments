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
    """Votes of the eight labelling functions and how strongly each one fires.

    One column per type with values 1, 0 or -1, and one column `<type>_strength` with the
    largest ratio between a measured quantity and its threshold (0 when the function does not fire).
    """
    lab = cfg["labelling"]
    w = windows.merge(thresholds, on=["circuit", "segment"], how="left", suffixes=("", "_thr"))
    corner = w["is_corner"].to_numpy() == 1
    exit_loss = w["exit_loss"].to_numpy() if "exit_loss" in w else 0.0
    slower = (w["time_loss"] > lab["min_time_loss_s"]).to_numpy()
    # A poor exit shows on the straight that follows, so the exit types also count that time.
    slower_exit = (w["time_loss"].to_numpy() + exit_loss) > lab["min_time_loss_s"]

    def exceeds(q):
        value = _value(w, q)
        ratio = (value / w[q]).clip(lower=0, upper=10).fillna(0).to_numpy()
        return (value > w[q]).to_numpy(), value.isna().to_numpy(), ratio

    votes, strengths = {}, {}

    def vote(name, parts, extra=None, needs_corner=True, gate=None):
        fired = np.any([p[0] for p in parts], axis=0)
        missing = np.all([p[1] for p in parts], axis=0)
        strength = np.max([np.where(p[0], p[2], 0.0) for p in parts], axis=0)
        if extra is not None:
            fired, strength = fired | extra[0], np.maximum(strength, np.where(extra[0], extra[1], 0.0))
        if gate is not None:
            fired = fired & gate
        out = np.where(fired, 1, np.where(missing, ABSTAIN, 0))
        if needs_corner:
            out = np.where(corner, out, ABSTAIN)
        votes[name] = out
        strengths[f"{name}_strength"] = np.where(out == 1, strength, 0.0)

    vote("K1", [exceeds("K1_onset")], gate=slower)
    vote("K2", [exceeds("K2_peak"), exceeds("K2_release")], gate=slower)
    vote("K3", [exceeds("K3_speed")], gate=slower)
    lifts = (w["d_throttle_lifts"] >= 1).to_numpy()
    vote("K4", [exceeds("K4_pickup")], extra=(lifts, 1.0 + w["d_throttle_lifts"].fillna(0).to_numpy()), gate=slower_exit)
    vote("K5", [exceeds("K5_slip"), exceeds("K5_tc")], gate=slower_exit)
    vote("K6", [exceeds("K6_steer"), exceeds("K6_line")], gate=slower)
    off = (w["tyres_out_max"] >= lab["K7"]["tyres_out"]).to_numpy()
    vote("K7", [exceeds("K7_loss")], extra=(off, 2.0), needs_corner=False)
    ratio = (w["time_cv_ratio"] / lab["K8"]["ratio"]).fillna(0).to_numpy()
    inconsistent = ((ratio > 1) & (w["seg_time"] > w["session_median_time"]).to_numpy())
    none = np.zeros(len(w), dtype=bool)
    vote("K8", [(none, w["time_cv_ratio"].isna().to_numpy(), ratio)], extra=(inconsistent, ratio))

    out = pd.DataFrame({**votes, **strengths}, index=windows.index)
    # A reference lap is compared with itself and therefore carries no information.
    ref = windows["is_reference"].to_numpy()
    for t in TYPES:
        if t != "K7":
            out.loc[ref, t] = ABSTAIN
            out.loc[ref, f"{t}_strength"] = 0.0
    return out


class LabelModel:
    """Learns how reliable each labelling function is, without any true labels.

    A window is a key moment (y = 1) or not (y = 0). Each function fires with one probability
    when y = 1 and with another when y = 0, independently of the other functions given y. The
    probabilities are estimated with expectation maximisation (Dawid and Skene, 1979). A function
    that fires on many windows on which no other function fires ends up with little weight.
    """

    def __init__(self, n_iter: int = 200, tol: float = 1e-6):
        self.n_iter, self.tol = n_iter, tol
        self.prior = self.fire_pos = self.fire_neg = None

    def fit(self, votes: np.ndarray) -> "LabelModel":
        votes = np.asarray(votes)
        seen = votes != ABSTAIN
        fired = (votes == 1) & seen
        k = votes.shape[1]
        self.prior = float(np.clip(fired.any(axis=1).mean(), 0.02, 0.6))
        rate = fired.sum(axis=0) / np.maximum(seen.sum(axis=0), 1)
        self.fire_pos = np.clip(rate / self.prior, 0.05, 0.9)
        self.fire_neg = np.full(k, 0.01)
        for _ in range(self.n_iter):
            post = self.predict_proba(votes)
            w_pos, w_neg = post[:, None] * seen, (1 - post)[:, None] * seen
            new_pos = np.clip((w_pos * fired).sum(axis=0) / np.maximum(w_pos.sum(axis=0), 1e-9), 1e-3, 0.999)
            # A function may fire on at most one in five windows that are not key moments;
            # without this bound the two classes can swap during estimation.
            new_neg = np.clip((w_neg * fired).sum(axis=0) / np.maximum(w_neg.sum(axis=0), 1e-9), 1e-4, 0.2)
            new_prior = float(np.clip(post.mean(), 0.01, 0.9))
            shift = max(np.abs(new_pos - self.fire_pos).max(), np.abs(new_neg - self.fire_neg).max())
            self.fire_pos, self.fire_neg, self.prior = new_pos, new_neg, new_prior
            if shift < self.tol:
                break
        return self

    def predict_proba(self, votes: np.ndarray) -> np.ndarray:
        votes = np.asarray(votes)
        seen = votes != ABSTAIN
        fired = votes == 1
        log_pos = np.where(fired, np.log(self.fire_pos), np.log1p(-self.fire_pos))
        log_neg = np.where(fired, np.log(self.fire_neg), np.log1p(-self.fire_neg))
        diff = ((log_pos - log_neg) * seen).sum(axis=1) + np.log(self.prior) - np.log1p(-self.prior)
        return 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))

    def table(self) -> pd.DataFrame:
        precision = self.fire_pos * self.prior / (self.fire_pos * self.prior + self.fire_neg * (1 - self.prior))
        return pd.DataFrame({"function": TYPES, "fires_given_moment": self.fire_pos,
                             "fires_given_no_moment": self.fire_neg, "estimated_precision": precision})


def aggregate(votes: pd.DataFrame, cfg: dict, prob: np.ndarray | None = None) -> pd.DataFrame:
    """Weak label per window: key moment yes or no, its probability and the primary type."""
    lab = cfg["labelling"]
    priority = lab["priority"]
    fired = votes[priority] == 1
    n_fired = fired.sum(axis=1)
    strength_cols = [f"{t}_strength" for t in priority]
    if lab.get("type_rule", "priority") == "strongest" and set(strength_cols) <= set(votes.columns):
        strongest = votes[strength_cols].to_numpy().argmax(axis=1)
        primary = np.where(n_fired > 0, np.array(priority, dtype=object)[strongest], "none")
    else:
        primary = np.where(n_fired > 0, fired.idxmax(axis=1), "none")
    weak_prob = (n_fired > 0).astype(float).to_numpy() if prob is None else np.where(n_fired > 0, prob, 0.0)
    return pd.DataFrame({"weak_label": (weak_prob >= 0.5).astype(int), "weak_prob": weak_prob,
                         "weak_type": np.where(weak_prob >= 0.5, primary, "none"),
                         "n_fired": n_fired.astype(int)}, index=votes.index)


def label(windows: pd.DataFrame, cfg: dict, fit_mask: np.ndarray | None = None, exclude=(),
          return_model: bool = False):
    """Votes and aggregated labels.

    fit_mask  restricts the windows on which thresholds and label model are fitted
    exclude   labelling functions that are switched off (used for the ablation)
    """
    thresholds = fit_thresholds(windows if fit_mask is None else windows[fit_mask], cfg)
    votes = apply(windows, thresholds, cfg)
    for t in exclude:
        votes[t], votes[f"{t}_strength"] = ABSTAIN, 0.0
    prob = model = None
    if cfg["labelling"].get("aggregation", "any") == "label_model":
        matrix = votes[TYPES].to_numpy()
        model = LabelModel().fit(matrix if fit_mask is None else matrix[fit_mask])
        prob = model.predict_proba(matrix)
    out = pd.concat([votes, aggregate(votes, cfg, prob)], axis=1)
    return (out, model) if return_model else out


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
