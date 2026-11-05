# Training

This package supports two routine tasks: importing a thread with screamer labels and retraining; and adding exact screamer timings before trying models that use event supervision. `python -m training --help` is the entry point.

## Setup

Install FFmpeg/FFprobe and Node 22+. Use Python 3.14 for training and variant generation, and Python 3.12/3.13 for TensorFlow:

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r training/requirements-dev.txt
python3.12 -m venv .venv-audio
.venv-audio/bin/pip install -r training/requirements-audio.txt
npm ci
```

Restore your `corpus/` backup or start collecting reviewed threads. A new import initializes its local label/timing files automatically. `python -m training init` can also create an empty store. It preserves existing annotations.

## Import and retrain

```sh
.venv/bin/python -m training add https://2ch.life/b/res/THREAD.html \
  --screamer FIRST.mp4 --screamer SECOND.webm --train
```

Only attachments in the fetched, reviewed snapshot are labeled. Named screamers become positive; its other attachments become negative. Later posts remain unlabeled. Existing positive, visual-only, and ambiguous reviews are preserved. For an all-safe thread, omit `--screamer`. Omit `--train` to import without training.

The importer saves the thread snapshot, extracts audio, and shares storage for exact duplicates. Unavailable media remain failed and are excluded from training; they never become successful safe analyses. Old failed downloads are not retried on every import. To retry them explicitly, use `python -m training.data.build_corpus --thread URL --retry-failed`.

Routine training compares logistic, forest, boosted-tree, embedding, hybrid, and event/MIL models. It rebuilds inputs in order, reuses valid extraction caches, and evaluates with source-thread and known-family exclusions. Positive clips without timings can train clip/MIL models; supervised event models require timed positives. A useful grouped evaluation requires sufficient independently labeled positive and negative sources across threads.

## Record exact timings

```sh
.venv/bin/python -m training labels
.venv/bin/python -m training timings FIRST.mp4 \
  --interval 7.76:10.42 --uncertainty 0.05
.venv/bin/python -m training timings SECOND.webm --interval 15.67:end
.venv/bin/python -m training train
```

Repeat `--interval` for separate events in the same clip. Timings are seconds from the start of retained audio. Only user-confirmed audio positives can receive timings. Untimed positives are not silently relabeled as negative event windows. `labels` shows confirmed screamer filenames, timings, and audio availability.

## Experiments

| Selection                       | What is compared                                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--experiment models` (default) | Clip/event model families using physical features, embeddings, and hybrid inputs                                                  |
| `--experiment features`         | Acoustic feature families, feature ablations, semantic event models, and synthetic MIL weights; includes the shipped model family |
| `--experiment augmentation`     | Codec/delay/gain variants, generated examples, robust MIL, hard-negative mining, trees, and additional sound-context features     |
| `--experiment all`              | All three groups                                                                                                                  |

```sh
.venv/bin/python -m training train --experiment features
.venv/bin/python -m training train --experiment augmentation
.venv/bin/python -m training train --experiment all --dry-run
```

Feature and synthetic studies require exact timings. `--dry-run` prints all commands without running training. `--output artifacts/runs/NAME` chooses a fresh run directory; otherwise a timestamped directory is created. Existing runs are never overwritten. `--jobs 1` reduces parallel training; `--audio-python PATH` selects the TensorFlow environment.

New runs build their dependencies without referring to dated experiments. Results are written to `artifacts/runs/NAME/summary.json`, and portable/final snapshots to its `final/` directory. Context-tree fits retain their local evaluation; the historical full-matrix portable export failed, so the workflow does not silently promote them. Training never automatically replaces the browser model.

Generated derivatives stay with their source families during evaluation. Synthetic variants do not increase the number of independent positives. Held-out scores and final-fit training diagnostics remain separate.

## Code and local data

| Location        | Responsibility                                                            |
| --------------- | ------------------------------------------------------------------------- |
| `data/`         | Import, labels, timings, event examples, and duplicate/family matching    |
| `features/`     | Physical/acoustic/context features and frozen YAMNet encoding             |
| `models/`       | Estimators, pooling objectives, grouped training, and numerical inference |
| `augmentation/` | Codec variants, generated examples, encoding, and matrix assembly         |
| `evaluation/`   | Feature ablations, robustness checks, and aggregate comparisons           |
| `export/`       | Final model and encoder serialization with parity checks                  |
| `../tools/`     | Browser builds, replay fixtures, and integration checks                   |
| `../tests/`     | Synthetic tests and optional checks against local audio                   |

`corpus/labels.json` stores human screamer labels under `confirmed_positives`, reviewed negative snapshots, and exceptions. `corpus/events.json` stores exact intervals under `events`. Both belong with `corpus/audio/`, `corpus/index.json`, and `corpus/manifests/`; all are ignored by Git. No real-thread identifiers or personal annotations are required for the unit tests.

Use `--corpus PATH` and `--cache PATH` **before** the subcommand to select other local directories. The corresponding environment variables are `SPOKOYNO_CORPUS` and `SPOKOYNO_CACHE`. The default generated cache is `artifacts/cache/`. Historical snapshots are retained locally in `research/models/`; their frozen source hashes refer to the earlier source layout. That directory and earlier local runs are excluded from Git.

Reproducing the historical numbers requires the corresponding saved dataset and code. Collecting new threads produces a new experiment; expired URLs alone are not a reproducible dataset.

## Completed results

[results.csv](results.csv) contains the compact comparison across model families, feature studies, and original/codec views. Each row identifies its dataset counts, decision policy, evaluation type, true/false positives, yellow-only outcomes, and average precision where available. The initial study used a training-selected threshold; later studies use the fixed 0.6/0.8 display policy. Training diagnostics are marked explicitly.

The shipped `synthetic-mil-20pct` model detected 30/30 positives with 34 red and 64 yellow-only false warnings among 3,466 negatives in the development evaluation. Subsequent robust/context candidates did not replace it. Repeated research on this small corpus limits generalization claims; the score is not a calibrated probability.

## Browser release

The shipped model and encoder are self-contained in `ml/assets/`. Rebuilding the existing runtime requires only the committed assets and `npm ci`:

```sh
python -m tools.build_browser_ml
```

To explicitly package a newly trained acoustic model, first export its encoder with `python -m training.export.export_robust_encoder --output-dir artifacts/cache/portable-encoder` in the audio environment, then run:

```sh
.venv/bin/python -m tools.build_browser_ml \
  --snapshot artifacts/runs/NAME/final/acoustic
```

`tools/prepare_browser_ml.py` takes explicit `--run` and `--acoustic` inputs for corpus parity fixtures, and `--data-dir` for browser-replay fixtures. The `check_*` tools cover features, model inference, runtime loading, and Tampermonkey installation. Their `--help` or argument checks describe the inputs.

After parity/integration verification, commit the assets first. Run `python -m tools.pin_browser_ml FULL_ASSET_COMMIT_SHA`, format and verify the generated userscript, then commit it. Preserve that asset commit so pinned download URLs remain valid. Keep the TensorFlow.js and YAMNet licenses with the release.
