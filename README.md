# thesis-keymoments

Software for the MSc thesis "Selecting the Moments That Matter: Data-Driven Identification of
Coaching-Relevant Events for AI-Supported Motorsport Training" (University of Twente).

The repository contains everything between the simulator and the results tables:

| Part | Folder | What it does |
|---|---|---|
| Telemetry logger | `logger/` | Records driving sessions from Assetto Corsa Competizione, holds the experience questionnaire and copies the data to university storage |
| Pipeline | `km/` | Lap selection, resampling, track segmentation, features, labelling functions K1 to K8, models, evaluation, advice text |
| Review tool | `review_app/` | Web page on which experts judge segments blind, with a personal access code |
| Configuration | `configs/` | All thresholds and experiment settings in one YAML file |
| Tests | `tests/` | Run on synthetic telemetry, no simulator needed |

The logger and the review tool are available in Dutch and English.

## Rules for data

- No research data in this repository. `data/`, `results/`, Parquet files and databases are ignored by git.
- Participants are stored under a pseudonym (D01, D02, ...). The logger refuses anything that looks like a name.
- The logger does not read the player name that the simulator publishes.
- The lab computer is temporary storage only. After a visit, the data are copied to the university storage with the button in the logger, checked by checksum, and removed locally.
- `access_codes_PRIVATE.csv` and `key_PRIVATE.parquet` in the review folder stay with the researcher and are never placed on a server.

## Installation

Python 3.11 or newer.

```
git clone https://github.com/AntoineMoghaddar/thesis-keymoments.git
cd thesis-keymoments
python -m venv .venv
.venv\Scripts\activate            (Windows)
pip install -e .[seq,dev]
pytest
```

`seq` installs PyTorch for the sequence model. Everything else works without it.

## Before the first driver

The logger could only be tested against synthetic data. The reading of the simulator memory follows
the official ACC Shared Memory Documentation v1.8.12, but it has not yet run against the game.
Do this once on the lab computer:

1. Start ACC, select the car and circuit of the study, drive out of the pit.
2. Start the logger (`python -m logger.app`), fill in a test pseudonym such as `T01`, record three laps and stop.
3. Run `km --data-dir <local data folder> prepare` and check in `prepared/lap_log.csv` that
   - the lap times equal the lap times shown in the game (`lap_time` against `clock_time`),
   - the sample rate in the session JSON is close to 60 Hz,
   - the out lap is marked as dropped and the flying laps as usable.
4. Check that `car_model` and `track` in the session JSON match `study.car_model` and `study.circuits`
   in `configs/default.yaml`. Correct the configuration when the game uses another identifier.
5. Delete the test recording.

To try the logger without the game, tick "Demo mode". A demo recording is marked as such and is stored
under the circuit name `synth_a`.

A single program for the lab computer can be built with `logger\build_exe.bat`.

## A recording visit

1. Fill in the pseudonym. Tick the consent box after the form has been signed.
2. Let the participant fill in the questionnaire (button in the logger). Select the experience level afterwards.
3. Choose the circuit. The session number is proposed from the recordings that already exist.
4. Start the recording when the car is in the pit, stop it after the last lap has been completed
   (drive a short distance into the next lap, otherwise the last lap counts as not completed).
5. After the visit: "Copy data to university storage".

Samples are written to disk every ten seconds, so a crash costs at most the last seconds.

## Analysis

```
km --config configs/local.yaml prepare          lap selection and resampling
km --config configs/local.yaml segments         proposes corner segments, then check configs/segments/*.yaml by hand
km --config configs/local.yaml features         window features
km --config configs/local.yaml sq1              labelling functions: coverage, overlap, conflict
km --config configs/local.yaml sq2              models and baselines on unseen drivers
km --config configs/local.yaml sq3              data requirements
km --config configs/local.yaml review-package   selects windows for the experts, creates access codes
streamlit run review_app/app.py                 the review tool
km --config configs/local.yaml sq4              agreement, reference labels
km --config configs/local.yaml sq2              again, now evaluated against the expert labels
km --config configs/local.yaml advise <session> --lang nl
```

`km synth` writes a synthetic study to the data folder for trying out the pipeline. The synthetic tracks
are invented and are called `synth_a` and `synth_b`.

### How the method maps to the code

| Thesis | Code |
|---|---|
| Removal of in laps, out laps, pit laps, laps with gaps and warm-up laps | `km/preprocess.py: split_laps` |
| Resampling on a distance grid, reference lap per driver and circuit | `km/preprocess.py: resample_lap, prepare` |
| Corner segments and straights | `km/segments.py`, `configs/segments/` |
| Engineered features and raw representation | `km/features.py` |
| Labelling functions, primary type, coverage, overlap, conflict | `km/labelling.py` |
| Gradient-boosted trees, type model, baselines | `km/models.py` |
| Sequence model | `km/seqmodel.py` |
| Split by driver, PR-AUC, precision@B, bootstrap over drivers | `km/evaluation.py` |
| SQ1 to SQ3 | `km/experiments.py` |
| SQ4: blind review, Fleiss' kappa, Cohen's kappa, reference labels | `km/review.py`, `review_app/` |
| Advice templates (R3) | `km/advice.py` |

### Choices that are made in the code and belong in the thesis

These are not prescribed by the method chapter and were needed to make the pipeline concrete. All of them
are settings in `configs/default.yaml`.

- **Test design.** Leave-one-driver-out by default (`evaluation.scheme: lodo`). A fixed test set is available.
- **Time-loss condition.** K1 to K6 only fire when the segment was slower than on the reference lap
  (`labelling.min_time_loss_s`). A deviation that costs no time is not treated as a coaching moment.
- **Thresholds.** A percentile per segment combined with an absolute floor. The values in the file are
  starting points and have to be set on the collected data.
- **Primary type.** Order K1 to K8 as in the thesis (entry before exit). In the synthetic data an off-track
  excursion is then often labelled K1, because the driver also braked at a different point. Placing K7
  first in `labelling.priority` avoids this.
- **K8.** Fires for windows of a segment whose times vary at least 1.5 times as much within a session as the
  median segment of that driver, and only for the laps that were slower than the session median.
- **Thresholds and unseen drivers.** With `labelling.fit_on: train` the thresholds are fitted on the
  training drivers of each fold only.
- **Evaluation target.** Expert labels where they exist, weak labels otherwise. Results against weak labels
  are marked as provisional in `sq2_summary.json`, because the labelling functions score perfectly against
  their own output.
- **Track length.** ACC does not publish it. It is estimated per circuit as the median distance driven per lap.

## Review tool

Local use:

```
km --config configs/local.yaml review-package
streamlit run review_app/app.py
```

Experts log in with the code from `access_codes_PRIVATE.csv`. The code of the researcher opens an overview
with the progress of all experts and a download of the responses. Answers are saved after every segment in
`review/responses.db`.

For use on a server, the tool needs four files only: `items.parquet`, `assignments.parquet`,
`traces.parquet` and `access.csv` (hashed codes). Set `KM_REVIEW_DIR` to the folder that holds them. These
files contain no pseudonyms, no experience levels and no information on why a segment was selected.

## What has been verified

- 20 tests pass: lap selection, resampling, segmentation, labelling functions, driver splits, metrics,
  the experiments SQ1 to SQ4 with simulated experts, the recorder, the transfer and the questionnaire.
- The review tool has been run in a browser on synthetic data.
- Not verified: reading from the running game, and the layout of the logger window on screen. The logic
  of the window is tested with a stand-in for the display library.
