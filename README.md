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
km --config configs/local.yaml qc               quality control of the recordings (run after every visit)
km --config configs/local.yaml prepare          lap selection and resampling
km --config configs/local.yaml segments         proposes corner segments, then check configs/segments/*.yaml by hand
km --config configs/local.yaml features         window features
km --config configs/local.yaml sq1              labelling functions: coverage, overlap, conflict
km --config configs/local.yaml sq2              models and baselines on unseen drivers
km --config configs/local.yaml sq3              data requirements (laps, drivers, label quality)
km --config configs/local.yaml reference        three kinds of reference lap compared
km --config configs/local.yaml generalisation   unseen driver, unseen circuit, unseen experience level
km --config configs/local.yaml ablation         feature groups, labelling functions, aggregation
km --config configs/local.yaml review-package   selects windows for the experts, creates access codes
streamlit run review_app/app.py                 the review tool
km --config configs/local.yaml sq4              agreement, reference labels
km --config configs/local.yaml sq2              again, now evaluated against the expert labels
km --config configs/local.yaml figures          figures for the thesis as PDF and PNG
km --config configs/local.yaml advise <session> --lang nl
km --config configs/local.yaml export-dataset --out <folder> --key <path on university storage>
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
| SQ1 to SQ3, reference study, generalisation, ablation | `km/experiments.py` |
| Label model (reliability per labelling function without true labels) | `km/labelling.py: LabelModel` |
| Moments at session level: recurring or incidental, spread over corners | `km/moments.py` |
| Explanations per selected moment (SHAP values from XGBoost) | `km/models.py: contributions` |
| Paired tests per driver, calibration | `km/evaluation.py` |
| Quality control of recordings | `km/qc.py` |
| Figures | `km/figures.py` |
| Anonymised dataset for publication | `km/export.py` |
| SQ4: blind review, Fleiss' kappa, Cohen's kappa, reference labels | `km/review.py`, `review_app/` |
| Advice templates (R3) | `km/advice.py` |

### Choices that are made in the code and belong in the thesis

These are not prescribed by the method chapter and were needed to make the pipeline concrete. All of them
are settings in `configs/default.yaml`.

- **Test design.** Leave-one-driver-out by default (`evaluation.scheme: lodo`). A fixed test set is available.
- **Spread over corners.** Precision@B counts at most one window per corner and session
  (`evaluation.max_per_segment: 1`), so that B moments are B different corners.
- **Time lost after the corner.** The time lost on the straight after a corner is attributed to that corner
  (`exit_loss`) and counts for the exit types K4 and K5.
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

### Studies beyond the four sub-questions

- **Reference lap** (`km reference`). The pipeline is run with three references: the driver's own best lap, an
  ideal lap built from the driver's best segments, and the best lap of another experienced driver. The output
  shows per experience level how many and which moments each reference produces.
- **Generalisation** (`km generalisation`). Next to an unseen driver, the model is tested on a circuit it has
  not seen and on an experience level it has not seen. With two circuits the first of these has two folds, so
  it shows a direction and not more than that.
- **Ablation** (`km ablation`). Each feature group and each labelling function is left out in turn. The variant
  without any comparison with the reference shows how much of the detection rests on the reference lap.
  Leaving out a labelling function changes the weak labels themselves, so this table is only meaningful
  when it is evaluated against expert labels.
- **Label model**. `labelling.aggregation: label_model` estimates the reliability of every labelling function
  from their agreement (Dawid and Skene, 1979) and gives every window a probability. The models are then
  trained on these probabilities. The default remains `any`, as described in the method chapter.
- **Session moments**. Flagged windows are grouped per session, corner and type. A group that returns in at
  least 40% of the laps counts as recurring, otherwise as incidental. At most B groups are selected, each in
  a different corner, ranked by the time that can be gained.
- **Statistics**. Per-driver PR-AUC, an exact paired sign-flip test between models, bootstrap intervals over
  drivers and calibration. With n drivers the smallest attainable p-value of the paired test is 2 / 2^n
  (0.031 with six drivers, 0.004 with nine); the output states this value.

## Publishing the dataset

`km export-dataset --out <folder> --key <path>` writes a copy of the telemetry, the derived tables and the expert
judgements with newly assigned random identifiers for drivers, sessions and experts. Dates, times, the
questionnaire and all free text are left out. The key file links the public identifiers to the study
pseudonyms. It has to be stored on the university storage, and the command refuses to write it inside the
public folder. Questionnaire items can be added to the public copy through `export.questionnaire_items`;
check for each item whether it could single out a person in a group of six to nine drivers.

## Review tool

The review has two parts. In **Sessions** the expert sees a whole session (time lost per corner and lap, and
all laps of a corner on top of each other) and chooses at most B corners to discuss. Nothing from the model
is shown. This gives the share of expert-selected moments that the model also finds, and the overlap between
two experts as the attainable level. In **Review** the expert judges single segments blind.

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

- 27 tests pass: lap selection, resampling, segmentation, labelling functions, the label model, the three
  references, splits by driver, circuit and level, metrics and tests, the experiments SQ1 to SQ4 with simulated
  experts, the session task, the dataset export (including a check that no study identifier is left in the
  public copy), the recorder, the transfer and the questionnaire.
- The review tool, including the session page, has been run in a browser on synthetic data.
- Not verified: reading from the running game, and the layout of the logger window on screen. The logic
  of the window is tested with a stand-in for the display library.
