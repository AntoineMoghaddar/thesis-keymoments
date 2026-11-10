"""Command line entry point: km <command>."""
from __future__ import annotations

import argparse

import pandas as pd

from .config import Paths, load_config


def _advise(cfg: dict, session: str, budget: int, lang: str) -> None:
    """Deployment view: the B moments of one session with their advice text."""
    from . import labelling, models
    from .advice import TYPE_NAMES, advice_text
    from .evaluation import top_b
    from .features import MODEL_FEATURES, load_windows
    windows = load_windows(cfg)
    if session not in set(windows["session_id"]):
        raise SystemExit(f"Unknown session. Available: {sorted(windows['session_id'].unique())[:10]} ...")
    driver = windows.loc[windows["session_id"] == session, "driver"].iloc[0]
    train = (windows["driver"] != driver).to_numpy()          # the driver of the session stays unseen (R4)
    labels = labelling.label(windows, cfg, fit_mask=train if cfg["labelling"]["fit_on"] == "train" else None)
    y = labels["weak_label"].to_numpy()
    detector = models.fit_detector(windows[train], y[train], cfg)
    typer = models.TypeModel(cfg).fit(windows[train & (y == 1)], labels["weak_type"][train & (y == 1)])
    sess = windows[(windows["session_id"] == session) & ~windows["is_reference"]].copy()
    sess["score"] = detector.predict_proba(sess[MODEL_FEATURES])[:, 1]
    sess["type"] = typer.predict(sess)
    chosen = sess[top_b(sess, "score", budget)].sort_values("score", ascending=False)
    for r in chosen.to_dict("records"):
        print(f"[{r['type']} {TYPE_NAMES[lang][r['type']]}] lap {r['lap']}, score {r['score']:.2f}")
        print("  " + advice_text(r, r["type"], lang))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="km", description="Key-moment detection pipeline")
    parser.add_argument("--config", help="YAML file with values that replace the defaults")
    parser.add_argument("--data-dir", help="folder with raw/, participants/, prepared/ and review/")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synth", help="write synthetic sessions for testing (not research data)")
    p.add_argument("--drivers-per-level", type=int, default=2)
    p = sub.add_parser("prepare", help="select laps and resample them on track position")
    p = sub.add_parser("segments", help="propose corner segments per circuit and write them to YAML")
    p.add_argument("--force", action="store_true", help="overwrite existing segment files")
    sub.add_parser("features", help="compute the window features")
    sub.add_parser("sq1", help="labelling functions: coverage, overlap, conflict")
    p = sub.add_parser("sq2", help="model comparison on unseen drivers")
    p.add_argument("--no-seq", action="store_true", help="skip the sequence model")
    sub.add_parser("sq3", help="data requirements")
    sub.add_parser("review-package", help="select windows for the experts and create access codes")
    sub.add_parser("sq4", help="analyse the expert responses")
    p = sub.add_parser("advise", help="print the selected moments of one session with advice")
    p.add_argument("session")
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--lang", choices=["en", "nl"], default="en")
    p = sub.add_parser("all", help="prepare, features, sq1 and sq2 in one go")
    p.add_argument("--no-seq", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"data_dir": args.data_dir} if args.data_dir else None
    cfg = load_config(args.config, overrides)
    paths = Paths.from_config(cfg).ensure()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    from . import experiments, features, preprocess, review, segments, synth
    if args.command == "synth":
        truth = synth.generate(paths.data, drivers_per_level=args.drivers_per_level)
        print(f"Synthetic sessions written to {paths.raw} ({len(truth)} injected errors).")
    if args.command in ("prepare", "all"):
        laps = preprocess.prepare(cfg)
        print(laps.groupby(["circuit", "driver"]).agg(laps=("lap", "size"), best=("lap_time", "min")))
        print(f"Dropped laps and the reason are listed in {paths.prepared / 'lap_log.csv'}")
    if args.command == "segments":
        laps, data, circuits = preprocess.load_prepared(cfg)
        for circuit in circuits:
            segs = segments.get_segments(laps, data, circuit, cfg, paths.segments, force=args.force)
            print(f"{circuit}: {sum(s['kind'] == 'corner' for s in segs)} corners in {paths.segments / (circuit + '.yaml')}")
    if args.command in ("features", "all"):
        w = features.build_windows(cfg)
        print(f"{len(w)} windows written to {paths.prepared / 'windows.parquet'}")
    if args.command in ("sq1", "all"):
        out = experiments.sq1(cfg)
        print(out["statistics"].round(3).to_string(index=False))
        print(out["by_level"])
    if args.command in ("sq2", "all"):
        out = experiments.sq2(cfg, with_seq=not args.no_seq)
        print(out["summary"])
        print(out["metrics"].pivot(index="metric", columns="model", values="value").round(3))
    if args.command == "sq3":
        out = experiments.sq3(cfg)
        for name in ("training_size", "laps_per_driver", "label_quality"):
            if name in out:
                print(out[name].round(3).to_string(index=False))
        print(out["requirement"])
    if args.command == "review-package":
        print(review.build_package(cfg))
        print(f"Access codes: {paths.review / 'access_codes_PRIVATE.csv'} (keep this file to yourself)")
    if args.command == "sq4":
        out = review.analyse(cfg)
        for name, value in out.items():
            print(f"\n{name}:\n{value.round(3).to_string(index=False) if hasattr(value, 'to_string') else value}")
    if args.command == "advise":
        _advise(cfg, args.session, args.budget, args.lang)


if __name__ == "__main__":
    main()
