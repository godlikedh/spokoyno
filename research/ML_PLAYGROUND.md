# Spokoyno ML playground

The ML pipeline is intentionally separate from `spokoyno.user.js`. Production remains on the frozen v5.7 event detector. Exported models are shadow scorers whose numbers are not calibrated probabilities.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r research/requirements.txt
```

FFmpeg and FFprobe must also be available on `PATH`.

## 1. Build or extend the audio corpus

The corpus builder accepts saved analysis manifests, raw 2ch thread-API JSON, or a live thread URL. It extracts only the first audio track as stereo float32 WAV at 16 kHz. Mirror fallback is built in.

```bash
.venv/bin/python research/build_corpus.py \
  research/thread-336185346.json \
  research/supplemental-positives.json \
  research/thread-336272252.json \
  research/thread-336291305.json
```

To append a future thread directly:

```bash
.venv/bin/python research/build_corpus.py \
  --thread https://2ch.life/b/res/THREAD.html
```

Existing index entries are retained unless `--replace-index` is supplied. Source-media MD5 matches are hard-linked before downloading. After extraction, normalized WAV SHA-256 matches are hard-linked as well. `corpus/index.json` records media identity, audio hashes, statuses, and duplicate ownership; `corpus/audio/` remains ignored by Git.

## 2. Extract model features

```bash
.venv/bin/python research/extract_features.py
```

The extractor processes one representative of every exact-audio group, then attaches the resulting vector to each logical media path. Existing vectors are reused by audio hash. Use `--refresh` only after changing the feature schema.

Feature version 1 contains:

- whole-file loudness, dynamic range, derivative, clipping, and loud/quiet occupancy;
- a dedicated first-second loud-start summary;
- the top three non-overlapping candidate events;
- 100/300/600/1000 ms event levels and 0.5/1/3/6 s baselines;
- baseline MAD/IQR/derivative and preceding abrupt-event count;
- attack, persistence, multiple duration definitions, and loudness area;
- near-clipping, spectral flux, nearby flux, Hellinger spectral distance, centroid/flatness change, and six band-energy changes.

The generated JSON is the canonical training input. The CSV is convenient for manual exploration.

## 3. Train grouped experiments

```bash
.venv/bin/python research/train_models.py
```

Training collapses identical labeled audio to one content group. Visual-only and unlabeled clips are excluded. Evaluation holds out one positive-bearing source thread at a time; a conservative threshold is learned from each training fold without looking at that fold's test scores.

Five intentionally constrained candidates are compared: two L2-logistic models, two compact random forests of depth 2/3, and one rich depth-3 forest. Automatic export first requires zero grouped out-of-fold false positives, then prefers recall, average precision, and simplicity. `--max-oof-fp` can change that research constraint, and `--select` can export a named experiment explicitly.

Outputs:

- `MODEL_CARD.md`: readable grouped results and every held-out error;
- `model-results.json`: folds, positive predictions, and false-positive/negative records;
- `shadow-model.json`: conservative selected model;
- `challenger-model.json`: higher-recall experimental model.

Both model formats are compact JSON with a pure numerical inference definition. Forest artifacts declare float32 input precision; JavaScript inference must apply `Math.fround()` after imputation before traversing their splits to match scikit-learn exactly. Training remains offline; no server, upload, TensorFlow, or WASM runtime is required.

## 4. Freeze predictions before adding labels

```bash
.venv/bin/python research/score_models.py
```

Run this after importing and extracting a new thread but before editing `corpus/labels.json` or retraining. Commit or archive `scores-v1.json`; it preserves the label state, model score, and threshold decision made at that time. This is the prospective evidence needed to distinguish genuine generalization from post-label tuning.

Do not estimate population false-positive rate from detector-selected review candidates. Continue labeling complete threads or random samples as well.

## Current result

The dataset contains 10 positive and 849 negative exact-audio groups. The conservative logistic candidate obtains 3/10 positive warnings with 0/849 false positives under grouped thread holdout. The rich shallow forest obtains 7/10 with 2/849 false positives. The forest's high ranking AUC does not make its output calibrated or production-safe, and both short-burst positives are missed when their entire source thread is excluded.

At their zero-training-FP thresholds, the final all-data conservative fit marks 5/10 positives and the challenger marks 9/10. These fits are provided only for shadow scoring. Their training performance is not validation and must not be compared to production's post-label regression as though the numbers measured the same thing.

Run the fast pipeline tests with:

```bash
.venv/bin/python -m unittest discover -s research -p 'test_*.py' -v
```
