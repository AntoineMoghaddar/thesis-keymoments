import numpy as np
import pandas as pd
import pytest

from km import experiments, features, labelling, preprocess, review, segments, synth
from km.advice import advice_text
from km.config import Paths
from km.evaluation import driver_folds, metrics, top_b


def _with_truth(windows, truth, n):
    """Attach the injected error type to the window that contains its apex."""
    key = {}
    for r in truth.itertuples():
        key.setdefault((r.session_id, r.lap), []).append((r.apex_pos, r.type))
    out = []
    for r in windows.itertuples():
        hit = [k for p, k in key.get((r.session_id, r.lap), []) if r.start_idx / n <= p < r.end_idx / n]
        out.append(hit[0] if hit else "none")
    return np.array(out)


def test_lap_selection(study):
    log = pd.read_csv(Paths.from_config(study["cfg"]).prepared / "lap_log.csv")
    per_session = study["laps"].groupby("session_id").size()
    assert (per_session == 9 - 2).all()                       # nine flying laps, two of them warm-up
    assert (log.loc[log["lap"] == 0, "reason"].str.contains("first lap")).all()
    assert log["reason"].fillna("").str.contains("pit").any()
    assert study["laps"].groupby(["driver", "circuit"])["is_reference"].sum().eq(1).all()


def test_restart_of_the_session_does_not_merge_laps(study):
    from km.schema import list_sessions
    frame = pd.read_parquet(list_sessions(Paths.from_config(study["cfg"]).raw)[0]["path"])
    second = frame.copy()
    second["t"] += frame["t"].iloc[-1] + 5
    laps = preprocess.split_laps(pd.concat([frame, second], ignore_index=True), study["cfg"])
    numbers = [lap["lap"] for lap in laps]
    assert len(numbers) == len(set(numbers)) == 2 * frame["lap"].nunique()
    assert "first lap" in next(lap for lap in laps if lap["lap"] == 1000)["reason"]


def test_resampling_keeps_lap_time(study):
    laps, data, circuits = preprocess.load_prepared(study["cfg"])
    assert data.shape[:2] == (len(laps), study["cfg"]["preprocess"]["grid_points"])
    lap_t = data[:, :, preprocess.CH["lap_t"]]
    assert (np.diff(lap_t, axis=1) > 0).all()                 # time increases along the lap
    assert np.allclose(lap_t[:, -1], laps["lap_time"], atol=0.2)
    for name, info in circuits.items():
        assert abs(info["length_m"] - synth.make_track(name)["length"]) < 15


def test_segments_cover_every_corner(study):
    cfg = study["cfg"]
    laps, data, circuits = preprocess.load_prepared(cfg)
    n = data.shape[1]
    for circuit in circuits:
        segs = segments.get_segments(laps, data, circuit, cfg, Paths.from_config(cfg).segments)
        corners = [s for s in segs if s["kind"] == "corner"]
        track = synth.make_track(circuit)
        assert len(corners) == len(track["apex_m"])
        for apex, seg in zip(track["apex_m"] / track["length"], corners):
            assert seg["start_idx"] / n < apex < seg["end_idx"] / n
        ends = [s["end_idx"] for s in segs]
        starts = [s["start_idx"] for s in segs]
        assert starts[0] == 0 and ends[-1] == n and starts[1:] == ends[:-1]   # no gaps, no overlap


def test_labelling_functions_find_injected_errors(study):
    cfg, w = study["cfg"], study["windows"]
    labels = labelling.label(w, cfg)
    assert set(np.unique(labels[labelling.TYPES].to_numpy())) <= {-1, 0, 1}
    assert (labels.loc[w["is_corner"] == 0, "K1"] == labelling.ABSTAIN).all()
    truth = _with_truth(w, study["truth"], cfg["preprocess"]["grid_points"])
    injected = truth != "none"
    assert labels.loc[injected, "weak_label"].mean() > 0.75
    stats = labelling.statistics(labels, cfg)
    assert stats["coverage"].between(0, 1).all() and (stats["n_primary"] <= stats["n_fired"]).all()
    assert labels.loc[labels["weak_label"] == 0, "weak_type"].eq("none").all()


def test_priority_decides_primary_type(study):
    cfg = study["cfg"]
    votes = pd.DataFrame([[0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], columns=labelling.TYPES)
    agg = labelling.aggregate(votes, cfg)
    assert agg["weak_type"].tolist() == ["K2", "none"] and agg["weak_label"].tolist() == [1, 0]


def test_folds_never_share_a_driver(study):
    cfg, w = study["cfg"], study["windows"]
    for _, train, test in driver_folds(w, cfg):
        assert not set(w.loc[train, "driver"]) & set(w.loc[test, "driver"])
    fixed = dict(cfg, evaluation=dict(cfg["evaluation"], scheme="fixed", fixed_test_drivers=["D01"]))
    (_, train, test), = driver_folds(w, fixed)
    assert set(w.loc[test, "driver"]) == {"D01"}


def test_metrics_and_budget():
    frame = pd.DataFrame({"session_id": ["a"] * 4 + ["b"] * 4, "driver": ["x"] * 8,
                          "score": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.3, 0.2],
                          "target": [1, 0, 0, 0, 1, 1, 0, np.nan]})
    assert top_b(frame, "score", 1).sum() == 2
    m = metrics(frame, "score", "target", [1, 2])
    assert m["precision@1"] == 1.0 and m["precision@2"] == 0.75 and m["n"] == 7


def test_model_beats_baselines_on_unseen_drivers(study):
    cfg, w = study["cfg"], study["windows"]
    pred = experiments.cross_validate(w, cfg, which=("xgb",))
    assert len(pred) == len(w) and pred["window_id"].is_unique
    truth = pd.Series(_with_truth(w, study["truth"], cfg["preprocess"]["grid_points"]), index=w["window_id"])
    pred["truth"] = (truth.reindex(pred["window_id"]).to_numpy() != "none").astype(float)
    scores = {c: metrics(pred, c, "truth", [])["pr_auc"] for c in ("score_xgb", "score_time_loss", "score_random")}
    assert scores["score_xgb"] > scores["score_random"] + 0.2
    assert scores["score_xgb"] > 0.5


def test_sequence_model_runs(study):
    pytest.importorskip("torch")
    cfg, w = study["cfg"], study["windows"]
    _, data, _ = preprocess.load_prepared(cfg)
    x = features.raw_windows(w, data, cfg["model"]["seq"]["length"])
    assert x.shape == (len(w), len(features.SEQ_CHANNELS), 64) and np.isfinite(x).all()
    from km.seqmodel import SequenceDetector
    y = labelling.label(w, cfg)["weak_label"].to_numpy()
    p = SequenceDetector(cfg).fit(x[:600], y[:600]).predict_proba(x[600:700])
    assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all()


def test_advice_text_in_both_languages(study):
    row = study["windows"][study["windows"]["is_corner"] == 1].iloc[5].to_dict()
    for lang in ("en", "nl"):
        for kind in labelling.TYPES:
            text = advice_text(row, kind, lang)
            assert text.startswith(row["segment_name"]) and "nan" not in text.lower()


def test_fleiss_kappa_reference_value():
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert review.fleiss_kappa(np.array(table)) == pytest.approx(0.210, abs=0.001)


def test_experiments_and_expert_review(study):
    """SQ1 to SQ4 in order, with simulated experts who know the injected errors."""
    cfg, w = study["cfg"], study["windows"]
    paths = Paths.from_config(cfg)
    assert experiments.sq1(cfg)["positive_rate"] > 0
    out = experiments.sq2(cfg, with_seq=False)
    assert out["summary"]["evaluated_against"].startswith("weak")
    assert {"xgb", "time_loss", "random", "labelling_functions"} <= set(out["metrics"]["model"])

    info = review.build_package(cfg)
    assert all(n == 20 for n in info["per_expert"].values())
    items = pd.read_parquet(paths.review / "items.parquet")
    assert not {"group", "driver", "level", "session_id", "in_model", "score_xgb"} & set(items.columns)   # blind
    codes = pd.read_csv(paths.review / "access_codes_PRIVATE.csv")
    assert review.check_code(paths.review, codes["code"][0].lower())["expert_id"] == "E01"
    assert review.check_code(paths.review, "0000-0000") is None
    assert "code" not in pd.read_csv(paths.review / "access.csv").columns

    truth = pd.Series(_with_truth(w, study["truth"], cfg["preprocess"]["grid_points"]), index=w["window_id"])
    assignments = pd.read_parquet(paths.review / "assignments.parquet")
    rng = np.random.default_rng(0)
    for r in assignments.itertuples():
        kind = truth[r.window_id]
        shown = items.set_index("window_id").loc[r.window_id, "shown_type"]
        relevant = (kind != "none") != (rng.random() < 0.1)          # ten percent disagreement
        review.save_response(paths.review, r.expert_id, r.window_id, relevant, kind == shown,
                             None if kind == shown else kind, True, "", 30.0)
    result = review.analyse(cfg)
    assert result["experts"] == 3 and result["shared_windows_complete"] == 8
    assert -1 <= result["fleiss_kappa_relevance"] <= 1
    assert len(result["labelling_functions"]) == 8
    labels = pd.read_parquet(paths.review / "expert_labels.parquet")
    assert set(labels["expert_label"]) <= {0, 1} and len(labels) == assignments["window_id"].nunique()

    out = experiments.sq2(cfg, with_seq=False)                       # now against the expert labels
    assert out["summary"]["evaluated_against"].startswith("expert")
    out = experiments.sq3(cfg)
    assert list(out["training_size"]["fraction"]) == [0.5, 1.0]
    assert "label_quality" in out and set(out["label_quality"]["strategy"]) == {"random", "uncertainty"}
