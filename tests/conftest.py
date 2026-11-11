import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def study(tmp_path_factory):
    """A small synthetic study taken through preparation and feature computation."""
    from km import features, preprocess, synth
    from km.config import load_config

    root = tmp_path_factory.mktemp("study")
    truth = synth.generate(root, drivers_per_level=1, sessions=2, laps=9, seed=3)
    cfg = load_config(overrides={
        "data_dir": str(root), "results_dir": str(root / "results"), "segments_dir": str(root / "segments"),
        "evaluation": {"bootstrap": 30, "seeds": [0], "train_fractions": [0.5, 1.0], "laps_per_driver": [4, None],
                       "review_steps": [0, 10]},
        "model": {"xgb": {"n_estimators": 60}, "seq": {"epochs": 2}},
        "review": {"n_experts": 3, "per_expert": 20, "shared": 8, "sessions_per_expert": 3, "sessions_shared": 2},
    })
    laps = preprocess.prepare(cfg)
    windows = features.build_windows(cfg)
    return dict(root=root, cfg=cfg, truth=truth, laps=laps, windows=windows)
