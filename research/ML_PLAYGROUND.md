# Spokoyno ML playground

The ML pipeline is deliberately separate from `spokoyno.user.js`. The v5.8.1 userscript preserves the existing red-decision rules with continuous evidence scores, a yellow tier, and parallel audio analysis; exported ML models are uncalibrated shadow scorers.

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

Import a thread the user has reviewed, listing every known audio screamer:

```bash
.venv/bin/python research/build_corpus.py \
  --thread https://2ch.life/b/res/THREAD.html --reviewed \
  --screamer FIRST.mp4 --screamer SECOND.webm
```

Every other video attachment in this fetched snapshot becomes a negative, without individual negative labels. For an all-safe thread, use `--reviewed` without any `--screamer`. A misspelled screamer filename is rejected before downloads or label writes. Existing explicit positive labels and visual-only exclusions are preserved. Later unseen attachments do not inherit the review. Failed downloads remain indexed as failures and are excluded from audio training; a negative content label does not imply successful analysis.

Importing without `--reviewed` retains the previous unlabeled-import workflow. Timing annotations are optional and separate:

```bash
.venv/bin/python research/event_annotations.py \
  --set FIRST.mp4 --interval 7.76:10.42 --uncertainty 0.05
.venv/bin/python research/event_annotations.py \
  --set SECOND.webm --interval 15.67:end --uncertainty 0.05
```

Repeat `--interval` for separate events. `end` resolves to the actual WAV duration. The uncertainty is an estimated plus/minus measurement error, not a guaranteed bound. Positive recordings without timings stay positive at clip level; their event windows are not silently trained as negatives.

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

Training collapses identical labeled audio into content groups and excludes visual-only clips and any future unlabeled clips. Evaluation holds out one positive-bearing source thread at a time; each fold learns its conservative threshold from training negatives without inspecting test scores.

Five constrained candidates are compared: two L2-logistic models, two compact depth-2/3 forests, and one rich depth-3 forest. Automatic selection first constrains grouped out-of-fold false positives, then prefers held-out recall, average precision, and simplicity.

Promoted outputs are tracked in `research/models/`:

- `MODEL_CARD.md` — readable grouped results and every held-out error;
- `model-results.json` — fold metrics and error records;
- `shadow-model.json` — conservative selected model;
- `challenger-model.json` — higher-recall experimental model.

The model files contain pure numerical inference definitions and identify their untracked training matrix by SHA-256. Forest artifacts require float32 inputs; JavaScript inference must apply `Math.fround()` after imputation to match scikit-learn split decisions.

## 4. Freeze predictions before retraining

```bash
.venv/bin/python research/score_models.py
```

In the reviewed-thread workflow, labels are recorded during import. Score with the previously frozen models before fitting anything on the new thread, and retain model hashes and training membership. These are future-thread results against frozen models, not blinded labels. If predictions can be captured before the human review, preserve those too. The original scorer's default file is overwritten; use an explicit fresh `--output` when preserving that scorer's snapshots. The event scorer below automatically creates timestamped snapshots.

Do not estimate population false-positive rates from detector-selected review candidates; whole reviewed threads remain the primary negative collection method.

The event-level workflow and measured comparison are in [Event results](EVENT_RESULTS.md). Both old and new scorers output a continuous score plus `low` / `maybe` / `alert`; the 0.6/0.8 cutoffs are display/decision policy, not evidence of probability calibration.

## Current result

The labeled dataset contains 10 positive and 1,181 negative exact-audio groups. Under grouped thread holdout, the conservative logistic candidate detects 4/10 positives with 0/1,181 false positives; the rich shallow forest detects 7/10 with 2/1,181 false positives. These small-sample results are model-development diagnostics, not production accuracy or calibrated probabilities.

At zero-training-false-positive thresholds, the final all-data fits mark 5/10 and 10/10 positives respectively. Those are training results, not validation.

## Checks

For the separate frozen-YAMNet comparison, known-screamer fingerprint matcher, and user-confirmed event timings, see [Audio experiments](AUDIO_EXPERIMENTS.md). These additions preserve the existing thread-wide negative-label workflow and do not replace production warnings or the original shadow models.

```bash
.venv/bin/python -m unittest discover -s research -p 'test_*.py' -v
.venv/bin/ruff check research
.venv/bin/ruff format --check research
node --test research/test_userscript.cjs research/test_analysis_concurrency.cjs
```
