# Spokoyno ML playground

The ML pipeline is deliberately separate from `spokoyno.user.js`. Production remains on the frozen v5.7 event detector; exported models are uncalibrated shadow scorers.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r research/requirements.txt
```

FFmpeg and FFprobe must also be available on `PATH`.

## Local data layout

Git tracks the label policy in `corpus/labels.json`, but not the media-derived corpus:

- `corpus/audio/` — stereo float32 WAV audio at 16 kHz;
- `corpus/index.json` — canonical media identity, extraction status, hashes, and deduplication metadata;
- `corpus/manifests/` — optional saved thread/API inputs;
- `research/artifacts/` — extracted feature matrices and frozen scoring snapshots.

The historical corpus already exists locally in this working copy. Back up the whole `corpus/` directory, including the index. A fresh clone does not contain the audio, and expired imageboard threads may be impossible to recover.

## 1. Extend the corpus

Append a future thread directly:

```bash
.venv/bin/python research/build_corpus.py \
  --thread https://2ch.life/b/res/THREAD.html
```

The builder also accepts saved analysis manifests or raw 2ch API JSON files as positional arguments. It extracts only the first audio track, tries all configured mirrors, retains existing index entries, and hard-links duplicates first by source MD5 and then by decoded WAV SHA-256. `--replace-index` intentionally drops absent catalog entries; it never deletes audio files.

## 2. Extract features

```bash
.venv/bin/python research/extract_features.py
```

The extractor processes one representative of each exact-audio group and reuses its vector for equivalent media paths. Existing vectors are reused by audio hash; use `--refresh` after changing the feature schema.

Feature version 1 contains 150 values covering:

- whole-file loudness, dynamic range, derivatives, clipping, and loud/quiet occupancy;
- a dedicated first-second loud-start summary;
- the top three non-overlapping candidate events;
- 100/300/600/1000 ms event levels and 0.5/1/3/6 s baselines;
- baseline MAD/IQR/derivative and preceding abrupt-event count;
- attack, persistence, duration, and loudness area;
- near-clipping, spectral flux, Hellinger spectral distance, centroid/flatness change, and six band-energy changes.

The JSON and companion CSV are written to ignored `research/artifacts/`.

## 3. Train grouped experiments

```bash
.venv/bin/python research/train_models.py
```

Training collapses identical labeled audio into content groups and excludes unlabeled and visual-only clips. Evaluation holds out one positive-bearing source thread at a time; each fold learns its conservative threshold from training negatives without inspecting test scores.

Five constrained candidates are compared: two L2-logistic models, two compact depth-2/3 forests, and one rich depth-3 forest. Automatic selection first constrains grouped out-of-fold false positives, then prefers held-out recall, average precision, and simplicity.

Promoted outputs are tracked in `research/models/`:

- `MODEL_CARD.md` — readable grouped results and every held-out error;
- `model-results.json` — fold metrics and error records;
- `shadow-model.json` — conservative selected model;
- `challenger-model.json` — higher-recall experimental model.

The model files contain pure numerical inference definitions and identify their untracked training matrix by SHA-256. Forest artifacts require float32 inputs; JavaScript inference must apply `Math.fround()` after imputation to match scikit-learn split decisions.

## 4. Freeze predictions before labeling

```bash
.venv/bin/python research/score_models.py
```

Run the scorer after importing a new thread but before changing `corpus/labels.json` or retraining. Preserve the ignored scoring snapshot separately when it is intended as prospective evidence. Do not estimate population false-positive rates from detector-selected review candidates; label complete threads or random samples too.

## Current result

The labeled dataset contains 10 positive and 849 negative exact-audio groups. Under grouped thread holdout, the conservative logistic candidate detects 3/10 positives with 0/849 false positives; the rich shallow forest detects 7/10 with 2/849 false positives. These small-sample results are model-development diagnostics, not production accuracy or calibrated probabilities.

At zero-training-false-positive thresholds, the final all-data fits mark 5/10 and 9/10 positives respectively. Those are training results, not validation.

## Checks

```bash
.venv/bin/python -m unittest discover -s research -p 'test_*.py' -v
.venv/bin/ruff check research
.venv/bin/ruff format --check research
```
