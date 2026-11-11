"""Figures for the thesis, drawn from the result tables. Written as PDF and PNG."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths

COLOURS = {"xgb": "#1f4e79", "seq": "#2a7f7f", "time_loss": "#b8860b", "labelling_functions": "#7a5195", "random": "#9aa0a6"}
NAMES = {"xgb": "Gradient-boosted trees", "seq": "Sequence model", "time_loss": "Time-loss baseline",
         "labelling_functions": "Labelling functions", "random": "Random"}


def _save(fig, folder: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png", dpi=200)
    return name


def make_figures(cfg: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    from .experiments import SCORES, add_target

    paths = Paths.from_config(cfg)
    res = paths.results
    folder = res / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.figsize": (5.2, 3.4)})
    made = []

    def table(name):
        path = res / name
        return pd.read_csv(path) if path.exists() else None

    stats = table("sq1_labelling_functions.csv")
    if stats is not None:
        fig, ax = plt.subplots()
        x = np.arange(len(stats))
        ax.bar(x - 0.2, stats["fires"], 0.4, label="Share of windows on which it fires", color=COLOURS["xgb"])
        ax.bar(x + 0.2, stats["n_primary"] / stats["n_primary"].sum() * stats["fires"].sum(), 0.4,
               label="Share as primary type (rescaled)", color=COLOURS["time_loss"])
        ax.set_xticks(x, stats["function"])
        ax.set_ylabel("Share of windows")
        ax.legend(frameon=False)
        made.append(_save(fig, folder, "sq1_labelling_functions"))
        plt.close(fig)

    pred_path = paths.prepared / "predictions.parquet"
    if pred_path.exists():
        frame, source = add_target(pd.read_parquet(pred_path), cfg)
        known = frame[frame["target"].notna()]
        fig, ax = plt.subplots()
        for name, column in SCORES.items():
            if column in known and name != "labelling_functions" and known["target"].nunique() > 1:
                p, r, _ = precision_recall_curve(known["target"].astype(int), known[column])
                ax.plot(r, p, label=NAMES[name], color=COLOURS[name], lw=1.6)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Evaluated against {source}", fontsize=8)
        ax.legend(frameon=False)
        made.append(_save(fig, folder, "sq2_precision_recall"))
        plt.close(fig)

    met = table("sq2_metrics.csv")
    if met is not None:
        at_b = met[met["metric"].str.startswith("precision@")].copy()
        at_b["B"] = at_b["metric"].str.split("@").str[1].astype(int)
        fig, ax = plt.subplots()
        for name, g in at_b.groupby("model"):
            g = g.sort_values("B")
            ax.plot(g["B"], g["value"], marker="o", ms=3, label=NAMES.get(name, name), color=COLOURS.get(name))
            ax.fill_between(g["B"], g["ci_low"], g["ci_high"], color=COLOURS.get(name), alpha=0.12, lw=0)
        ax.set_xlabel("Moments per session (B)")
        ax.set_ylabel("Precision@B")
        ax.set_xticks(sorted(at_b["B"].unique()))
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7)
        made.append(_save(fig, folder, "sq2_precision_at_b"))
        plt.close(fig)

    drivers = table("sq2_per_driver.csv")
    if drivers is not None:
        fig, ax = plt.subplots()
        models = [m for m in ("xgb", "seq", "time_loss") if m in drivers]
        for k, m in enumerate(models):
            ax.scatter(np.full(len(drivers), k) + np.linspace(-0.12, 0.12, len(drivers)), drivers[m],
                       color=COLOURS[m], s=18, zorder=3)
            ax.hlines(drivers[m].mean(), k - 0.25, k + 0.25, color=COLOURS[m], lw=2)
        ax.set_xticks(range(len(models)), [NAMES[m] for m in models])
        ax.set_ylabel("PR-AUC per held-out driver")
        made.append(_save(fig, folder, "sq2_per_driver"))
        plt.close(fig)

    calib = table("sq2_calibration_xgb.csv")
    if calib is not None and len(calib):
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=1, ls="--")
        ax.plot(calib["mean_score"], calib["observed"], marker="o", ms=4, color=COLOURS["xgb"])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed share of key moments")
        made.append(_save(fig, folder, "sq2_calibration"))
        plt.close(fig)

    for name, column, label in (("sq3_training_size", "fraction", "Share of training laps"),
                                ("sq3_laps_per_driver", "laps", "Laps per driver and circuit"),
                                ("sq3_training_drivers", "training_drivers", "Number of training drivers")):
        curve = table(f"{name}.csv")
        if curve is not None:
            fig, ax = plt.subplots()
            x = np.arange(len(curve))
            ax.errorbar(x, curve["pr_auc_mean"], yerr=curve["pr_auc_std"].fillna(0), marker="o", ms=4, capsize=3,
                        color=COLOURS["xgb"])
            ax.set_xticks(x, curve[column].astype(str))
            ax.set_xlabel(label)
            ax.set_ylabel("PR-AUC (mean and SD over seeds)")
            made.append(_save(fig, folder, name))
            plt.close(fig)

    quality = table("sq3_label_quality.csv")
    if quality is not None:
        fig, ax = plt.subplots()
        for (strategy, g), colour in zip(quality.groupby("strategy"), (COLOURS["time_loss"], COLOURS["xgb"])):
            ax.errorbar(g["expert_labels"], g["pr_auc_mean"], yerr=g["pr_auc_std"].fillna(0), marker="o", ms=4,
                        capsize=3, label=f"Selection: {strategy}", color=colour)
        ax.set_xlabel("Expert judgements added to the weak labels")
        ax.set_ylabel("PR-AUC")
        ax.legend(frameon=False)
        made.append(_save(fig, folder, "sq3_label_quality"))
        plt.close(fig)

    for name, column in (("ablation_features", "variant"), ("ablation_labelling_functions", "variant")):
        abl = table(f"{name}.csv")
        if abl is not None:
            abl = abl.iloc[1:]
            fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(abl) + 1))
            ax.barh(abl[column], abl["delta_pr_auc"], color=[COLOURS["xgb"] if v < 0 else COLOURS["random"] for v in abl["delta_pr_auc"]])
            ax.axvline(0, color="#4d4d4d", lw=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Change in PR-AUC")
            made.append(_save(fig, folder, name))
            plt.close(fig)

    gen = table("generalisation.csv")
    if gen is not None and "fold" in gen:
        total = gen[gen["fold"] == "all"]
        fig, ax = plt.subplots()
        x = np.arange(len(total))
        ax.bar(x - 0.2, total["pr_auc"], 0.4, color=COLOURS["xgb"], label=NAMES["xgb"])
        ax.bar(x + 0.2, total["pr_auc_time_loss"], 0.4, color=COLOURS["time_loss"], label=NAMES["time_loss"])
        ax.set_xticks(x, total["question"])
        ax.set_ylabel("PR-AUC")
        ax.legend(frameon=False)
        made.append(_save(fig, folder, "generalisation"))
        plt.close(fig)

    ref = table("reference_by_level.csv")
    if ref is not None:
        fig, ax = plt.subplots()
        levels = [lv for lv in ("none", "moderate", "high") if lv in set(ref["level"])]
        modes = list(dict.fromkeys(ref["reference"]))
        for k, mode in enumerate(modes):
            g = ref[ref["reference"] == mode].set_index("level").reindex(levels)
            ax.bar(np.arange(len(levels)) + (k - 1) * 0.26, g["positive_rate"], 0.26, label=mode.replace("_", " "),
                   color=list(COLOURS.values())[k])
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel("Experience level")
        ax.set_ylabel("Share of windows labelled as key moment")
        ax.legend(frameon=False, title="Reference lap")
        made.append(_save(fig, folder, "reference_by_level"))
        plt.close(fig)
    return made
