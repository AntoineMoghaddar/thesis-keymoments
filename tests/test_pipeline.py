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
    assert all(n == 20 for n in info["per_expert"].values()) and info["sessions"]["shared"] == 2
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
    assert list(out["training_drivers"]["training_drivers"]) == [1, 2]
    assert "label_quality" in out and set(out["label_quality"]["strategy"]) == {"random", "uncertainty"}

    # Session task: every expert selects corners, the analysis compares them with the model.
    sessions = pd.read_parquet(paths.review / "session_assignments.parquet")
    session_items = pd.read_parquet(paths.review / "session_items.parquet")
    assert not {"driver", "session_id", "level"} & set(session_items.columns)
    private = pd.read_parquet(paths.review / "session_key_PRIVATE.parquet")
    for r in sessions.itertuples():
        model = private[(private["session_key"] == r.session_key) & (private["source"] == "model")]
        picks = [dict(segment=m.segment, type=m.type) for m in model.head(2).itertuples()]
        review.save_session_picks(paths.review, r.expert_id, r.session_key, picks, 60.0)
    review.save_session_picks(paths.review, "E01", sessions["session_key"].iloc[0], [], 5.0)   # nothing to discuss
    task = review.analyse_sessions(cfg)
    assert task["summary"].loc["model", "recall"] > 0.8 and 0 <= task["expert_expert_jaccard"] <= 1
    result = review.analyse(cfg)
    assert {"krippendorff_alpha_relevance", "gwet_ac1_relevance", "mean_pairwise_kappa_experts"} <= set(result)

    from km import figures, qc
    made = figures.make_figures(cfg)
    assert {"sq2_precision_recall", "sq3_training_size", "sq1_labelling_functions"} <= set(made)
    table = qc.session_quality(cfg)
    assert len(table) == 12 and (table["laps_usable"] == 7).all() and table["rate_hz"].between(59, 61).all()


def test_dataset_export_carries_no_internal_identifiers(study, tmp_path):
    from km import export
    cfg = study["cfg"]
    experiments.sq1(cfg)
    out, key = tmp_path / "public", tmp_path / "ut" / "key.csv"
    with pytest.raises(ValueError):
        export.export_dataset(cfg, out, out / "key.csv")
    info = export.export_dataset(cfg, out, key, seed=1)
    assert info["sessions"] == 12 and key.exists() and not list(out.rglob("key.csv"))
    internal = set(study["windows"]["driver"]) | set(study["windows"]["session_id"])
    for path in out.rglob("*"):
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            for col in frame.select_dtypes(include=["object", "string"]).columns:
                values = set(frame[col].dropna().astype(str))
                assert not values & internal, (path.name, col)
                assert not any(v.startswith(tuple(study["windows"]["session_id"].unique())) for v in values), (path.name, col)
        elif path.is_file() and path.suffix in (".json", ".csv", ".md"):
            text = path.read_text(encoding="utf-8")
            assert "synthetic_L" not in text and "started_at" not in text and "D01_" not in text
    keys = pd.read_csv(key)
    assert set(keys.loc[keys["kind"] == "driver", "internal_id"]) == set(study["windows"]["driver"])
    public = pd.read_parquet(out / "derived" / "windows.parquet")
    assert len(public) == len(study["windows"]) and public["window_id"].is_unique

    # A driver without consent for publication is left out completely.
    record = Paths.from_config(cfg).participants / "D02.json"
    original = record.read_text(encoding="utf-8")
    record.write_text(original.replace('"publication_consent": true', '"publication_consent": false'), encoding="utf-8")
    try:
        partial = export.export_dataset(cfg, tmp_path / "public2", tmp_path / "ut" / "key2.csv", seed=1)
    finally:
        record.write_text(original, encoding="utf-8")
    assert partial["left_out_without_consent"] == ["D02"] and partial["drivers"] == 2
    assert pd.read_parquet(tmp_path / "public2" / "derived" / "windows.parquet")["driver"].nunique() == 2
    assert "D02" not in set(pd.read_csv(tmp_path / "ut" / "key2.csv")["internal_id"])
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == info["files"] - 1


def test_reference_modes(study):
    cfg = study["cfg"]
    base = study["windows"]
    for mode in ("ideal", "expert"):
        w = features.build_windows(cfg, reference=mode)
        assert list(w["window_id"]) == list(base["window_id"]) and (w["reference"] == mode).all()
    ideal = features.load_windows(cfg, "ideal")
    corners = ideal[ideal["is_corner"] == 1]
    assert corners["time_loss"].mean() > base.loc[base["is_corner"] == 1, "time_loss"].mean()   # a stricter reference
    expert = features.load_windows(cfg, "expert")
    other = expert[expert["level"] != "high"]
    laps = study["laps"].reset_index(drop=True)
    assert (laps.loc[other["ref_row"], "level"] == "high").all()
    assert (laps.loc[other["ref_row"], "driver"].to_numpy() != other["driver"].to_numpy()).all()


def test_label_model_downweights_an_unreliable_function():
    rng = np.random.default_rng(0)
    truth = rng.random(4000) < 0.2
    good = [(truth & (rng.random(4000) < 0.6)) | (~truth & (rng.random(4000) < 0.01)) for _ in range(3)]
    noisy = rng.random(4000) < 0.15                               # fires regardless of the truth
    votes = np.column_stack(good + [noisy]).astype(int)
    model = labelling.LabelModel().fit(votes)
    precision = model.fire_pos * model.prior / (model.fire_pos * model.prior + model.fire_neg * (1 - model.prior))
    assert precision[:3].min() > 0.8 and precision[3] < 0.5
    only_noisy = model.predict_proba(np.array([[0, 0, 0, 1]]))[0]
    one_good = model.predict_proba(np.array([[1, 0, 0, 0]]))[0]
    assert only_noisy < 0.5 < one_good


def test_selection_is_spread_over_corners_and_schemes_split_correctly(study):
    cfg, w = study["cfg"], study["windows"]
    frame = pd.DataFrame({"session_id": ["a"] * 4, "segment": ["T1", "T1", "T2", "T3"], "score": [0.9, 0.8, 0.5, 0.4]})
    assert top_b(frame, "score", 2).tolist() == [True, True, False, False]
    assert top_b(frame, "score", 2, max_per_segment=1).tolist() == [True, False, True, False]
    for scheme, column in (("loco", "circuit"), ("lolo", "level")):
        folds = driver_folds(w, cfg, scheme)
        assert len(folds) == w[column].nunique()
        for _, train, test in folds:
            assert not set(w.loc[train, column]) & set(w.loc[test, column])


def test_session_moments_and_statistics(study):
    from km import moments
    from km.evaluation import calibration, paired_permutation
    cfg, w = study["cfg"], study["windows"]
    scored = w.assign(score=(w["time_loss"] > 0.3).astype(float), type="K1")
    found = moments.session_moments(scored, cfg)
    assert (found["n_flagged"] <= found["n_laps"]).all() and set(found["pattern"]) <= {"structural", "incidental"}
    chosen = moments.select(found, 3)
    assert chosen.groupby("session_id").size().max() <= 3
    assert not chosen.duplicated(["session_id", "segment"]).any()
    assert "laps" in moments.describe(chosen.iloc[0], "en") and "rondes" in moments.describe(chosen.iloc[0], "nl")
    test = paired_permutation([0.9, 0.8, 0.85, 0.7], [0.5, 0.6, 0.55, 0.4])
    assert test["p_value"] == pytest.approx(2 / 16) and test["mean_difference"] > 0
    frame = pd.DataFrame({"p": [0.1] * 50 + [0.9] * 50, "y": [0.0] * 45 + [1.0] * 5 + [1.0] * 45 + [0.0] * 5})
    _, summary = calibration(frame, "p", "y")
    assert summary["ece"] == pytest.approx(0.0, abs=1e-9)
    alpha = review.krippendorff_alpha(pd.DataFrame([[1, 1], [0, 0], [1, 1], [0, 0]]))
    assert alpha == pytest.approx(1.0)


def test_generalisation_and_ablation_run(study):
    cfg = study["cfg"]
    gen = experiments.generalisation(cfg)
    assert set(gen["scheme"]) == {"lodo", "loco", "lolo"} and gen.loc[gen["fold"] == "all", "pr_auc"].notna().all()
    ref = experiments.reference_study(cfg)
    assert list(ref["reference"]) == ["own_best", "ideal", "expert"]
    abl = experiments.ablation(cfg)
    assert len(abl["features"]) == 10 and len(abl["labelling_functions"]) == 9 and len(abl["aggregation"]) == 4
