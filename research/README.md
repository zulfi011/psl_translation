# PSL Research

Research codebase for signer-independent **Pakistani Sign Language** recognition.
Every stage is versioned and pluggable, so new extractors, preprocessing
pipelines and models can be tried side by side without touching each other.

## Pipeline

```
data/raw/<sign>/<signer> (<take>).mp4                       original videos (never modified)
        │  scripts/build_manifest.py
        ▼
data/metadata/clips.csv                                     one row per clip: sign, signer, take
        │  scripts/extract_landmarks.py --extractor <name>
        ▼
data/interim/<extractor>/<sign>/<clip>.npy                  full-length landmarks, one folder per extractor
        │  scripts/make_splits.py --strategy loso --name <split>
        ▼
data/splits/<split>.json                                    signer-independent folds
        │  scripts/train.py configs/experiments/<exp>.yaml
        ▼   (transforms + augmentation applied on the fly, defined in the config)
experiments/runs/<exp>/fold<k>/                             config, history, weights, metrics
        │  scripts/summarize.py
        ▼
reports/leaderboard.md                                      mean ± std across folds, every experiment
```

## Quick start

```bash
pip install -e "research[dev]"          # from the repo root
```

```bash
cd research
python scripts/build_manifest.py
python scripts/extract_landmarks.py --extractor mediapipe_holistic_v1
python scripts/make_splits.py --strategy loso --val next_signer --name loso_nextval
python scripts/train.py configs/experiments/exp001_tcn_v1_baseline.yaml
python scripts/train.py configs/experiments/exp002_tcn_v2.yaml
python scripts/summarize.py
```

## Layout

```
research/
├── configs/
│   ├── data/          preprocessing pipelines   (extractor + split + ordered transforms)
│   ├── model/         architectures + hyper-parameters
│   ├── train/         optimiser, epochs, seed …
│   └── experiments/   one file per experiment = data + model + train (+ overrides)
├── data/              raw / interim / processed / splits / metadata   (see data/README.md)
├── src/psl_research/
│   ├── registry.py    plug-in registries: EXTRACTORS, TRANSFORMS, MODELS
│   ├── data/
│   │   ├── extractors/    video → landmarks     (mediapipe_holistic_v1, …)
│   │   ├── transforms/    normalise / temporal / augment
│   │   ├── manifest.py    raw folder → clips.csv
│   │   ├── splits.py      LOSO, holdout_v1
│   │   └── dataset.py
│   ├── models/        model zoo                 (tcn_v1, tcn, …)
│   ├── training/      trainer (one fold → run folder)
│   └── evaluation/    accuracy, macro-F1, top-5, ECE, confusion matrix
├── scripts/           thin command-line entry points (numbered by pipeline order in the docs above)
├── experiments/runs/  outputs — configs & metrics are committed, weights are not
├── notebooks/         exploration: NN_initials_topic.ipynb
├── reports/           leaderboard, figures for the paper
├── paper/             manuscript (private until submission)
├── docs/              dataset card, model card
└── tests/             transform correctness + end-to-end smoke test (synthetic data)
```

## Adding something new

| You want to try… | Do this |
|---|---|
| a new model | add `src/psl_research/models/<name>.py` with `@MODELS.register("<name>")`, import it in `models/__init__.py`, add `configs/model/<name>.yaml` |
| a new preprocessing step / augmentation | add a class with `@TRANSFORMS.register("<name>")` in `data/transforms/`, use it in a `configs/data/*.yaml` |
| a new preprocessing pipeline | new `configs/data/<pipeline>.yaml` — the transforms list is the whole pipeline |
| a new landmark extractor (other MediaPipe settings, a different pose model, face features…) | add a class with `@EXTRACTORS.register("<name>")` in `data/extractors/`; its output lands in its own `data/interim/<name>/` |
| a new evaluation protocol | add a function to `data/splits.py` → new `data/splits/<name>.json` |
| a new experiment | new `configs/experiments/expNNN_<short_name>.yaml`; never edit one that already has results — copy it |

Rules that keep results trustworthy:

- Experiment names are numbered and never reused (`exp004_…`). Results of a
  finished experiment are never overwritten.
- Every run folder contains the fully resolved `config.yaml`, so any number in
  the paper can be traced back to the exact settings that produced it.
- Report test accuracy as **mean ± std over LOSO folds**. Never tune on the test signer.

## Experiments planned

| ID | Purpose |
|---|---|
| exp001_tcn_v1_baseline | FYP v1 model + v1 preprocessing, under LOSO — the honest baseline |
| exp002_tcn_v2 | redesigned TCN + v2 preprocessing |
| exp003_tcn_v1_v2data | v1 model + v2 preprocessing — isolates the effect of preprocessing |
