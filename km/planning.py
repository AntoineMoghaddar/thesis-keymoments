"""What the study can show for a given number of drivers and experts.

Three simulations, each with its assumptions stated in the table. They replace an argument from
saturation by the width of the intervals and the power of the test that the study will report.
They do not use recorded data.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import Paths
from .review import fleiss_kappa


def agreement(cfg: dict, n_sim: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Spread of Fleiss' kappa on the shared windows, and how often a window ends in a tie.

    Assumption: a window is relevant or not, 40% are, and every expert judges it correctly with the
    same probability (`expert_accuracy`), independently of the other experts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for accuracy, experts, windows in itertools.product((0.75, 0.85), (5, 6, 7), (20, 30, 40)):
        truth = rng.random((n_sim, windows)) < 0.4
        correct = rng.random((n_sim, windows, experts)) < accuracy
        yes = np.where(correct, truth[:, :, None], ~truth[:, :, None]).sum(axis=2)
        kappa = np.array([fleiss_kappa(np.column_stack([experts - y, y])) for y in yes])
        low, high = np.nanpercentile(kappa, [2.5, 97.5])
        rows.append(dict(expert_accuracy=accuracy, experts=experts, shared_windows=windows,
                         kappa_expected=float(np.nanmean(kappa)), kappa_low=float(low), kappa_high=float(high),
                         width=float(high - low), share_ties=float((2 * yes == experts).mean())))
    return pd.DataFrame(rows)


def precision(cfg: dict) -> pd.DataFrame:
    """Half-width of the 95% interval on the precision of the model selection (Wilson interval).

    Assumption: reviewed windows are independent. Windows of one driver resemble each other, so
    the real interval will be wider than this.
    """
    rv = cfg["review"]
    rows = []
    for experts, per_expert in itertools.product((5, 6, 7), sorted({40, rv["per_expert"]})):
        shared = min(rv["shared"], per_expert // 2)
        unique = shared + experts * (per_expert - shared - rv.get("repeats", 0))
        n = int(round(rv["mix"]["model"] * unique))
        for p in (0.6, 0.8):
            z = 1.96
            centre = (p + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append(dict(experts=experts, windows_per_expert=per_expert, windows_reviewed=unique, model_windows=n,
                             true_precision=p, interval_low=centre - half, interval_high=centre + half, half_width=half))
    return pd.DataFrame(rows)


def power(cfg: dict, n_sim: int = 2000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Power of the paired sign-flip test over drivers that compares the model with a baseline.

    Assumption: the per-driver differences are normally distributed. `effect` is the mean difference
    divided by its standard deviation over drivers. With n drivers the smallest attainable two-sided
    p-value is 2 / 2^n, whatever the size of the effect.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for drivers in (6, 9, 12):
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=drivers)))
        for effect in (0.8, 1.2, 1.6):
            diff = rng.normal(effect, 1.0, (n_sim, drivers))
            observed = np.abs(diff.mean(axis=1))
            null = np.abs(diff @ signs.T) / drivers                      # every sign pattern, per simulated study
            p = (null >= observed[:, None] - 1e-12).mean(axis=1)
            rows.append(dict(drivers=drivers, effect=effect, power=float((p <= alpha).mean()),
                             smallest_p=2.0 / 2 ** drivers))
    return pd.DataFrame(rows)


def plan(cfg: dict) -> dict:
    paths = Paths.from_config(cfg).ensure()
    out = {"agreement between experts": agreement(cfg), "precision of the model selection": precision(cfg),
           "power of the test over drivers": power(cfg)}
    for name, table in out.items():
        table.to_csv(paths.results / f"planning_{name.split()[0]}.csv", index=False)
    return out
