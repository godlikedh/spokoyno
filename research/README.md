# Training and final results

The production model is `synthetic-mil-20pct` from [acoustic-mil-20261025](models/acoustic-mil-20261025/manifest.json). Training and evaluation run locally; browser inference uses committed assets in `ml/assets/`.

## Final results

These are development results on 30 positive and 3,466 negative audio groups, with source-thread and known-family exclusions. Red means score ≥ 0.8. Codec columns evaluate transformations of the same sources, not additional independent examples.

| Candidate                          | Original positives red | Original negatives red | AAC positives red | Opus positives red | MP3 positives red |
| ---------------------------------- | ---------------------: | ---------------------: | ----------------: | -----------------: | ----------------: |
| **Shipped synthetic MIL**          |                 **30** |                 **34** |            **25** |             **28** |            **28** |
| Robust consistent hybrid           |                     30 |                     66 |                23 |                 29 |                30 |
| Codec-stable hybrid                |                     28 |                     50 |                25 |                 28 |                28 |
| Robust boosted trees, seven leaves |                     29 |                    154 |                29 |                 29 |                29 |

The shipped candidate also produced 64 yellow-only negative warnings. Its score is uncalibrated evidence. Repeated model selection on this corpus, incomplete family matching, and only 30 positive groups limit generalization claims. Gains, codecs, and timing changes can still change decisions. Proposed event times are machine candidates, not human annotations.

| Directory under `models/`                                                          | Retained result                                                                                 |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Root `shadow-model.json`, `challenger-model.json`, `model-results.json`            | Initial tabular baselines on the earlier 10-positive corpus                                     |
| `retrained-20260930/`, `retrained-20261025/`                                       | Ten final model/manifest pairs per run and aggregate results                                    |
| `event-research-20261025/`                                                         | Event models and two rounds of aggregate results                                                |
| `rules-mil-20261025/`                                                              | Exploratory fusion policy and evaluation; not shipped                                           |
| `acoustic-mil-20261025/`                                                           | Acoustic models, fitted transform, and results including the shipped candidate                  |
| `robust-audio-20261026/`, `codec-stable-audio-20261026/`, `robust-trees-20261026/` | Portable models and codec evaluation                                                            |
| `hard-negative-audio-20261026/`, `context-audio-20261026/`                         | Hard-negative and sound-context models and evaluation                                           |
| `context-trees-20261026/`                                                          | Rejected evaluation only; no loadable model. PCA rounding crossed tree thresholds during export |
| `robust-comparison-20261026/`, `context-comparison-20261026/`                      | Aggregate comparisons across originals and codec variants                                       |

Later robust/context experiments did not replace the shipped model. Complete aggregate metrics remain in each `evaluation.json`, `results-summary.json`, or `comparison.json`. Final-fit training scores are separate from held-out results.

Manifests validate retained model files and fitted transforms. Training membership identifies previously seen audio when scoring. `original_manifest_sha256`, where present, identifies the historical bundle before evidence files were removed; the existing browser release still records that historical hash. Model coefficients and production assets were not changed by repository cleanup.

## Setup and data

Run commands from the repository root. Install FFmpeg/FFprobe and Node 22+. The main training environment uses Python 3.14 for the bounded process-pool generator; TensorFlow uses a separate Python 3.12/3.13 environment.

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r research/requirements-dev.txt
python3.12 -m venv .venv-audio
.venv-audio/bin/pip install -r research/requirements-audio.txt
npm ci
```

Restore `corpus/audio/`, `corpus/index.json`, and `corpus/manifests/` from local backup. Preserve corpus filenames: the index refers to them. Exact duplicates can share hard links; generated codec variants and browser PCM copies belong in `research/artifacts/`.

Import a reviewed thread and identify its audio screamers:

```sh
.venv/bin/python research/build_corpus.py \
  --thread https://2ch.life/b/res/THREAD.html --reviewed \
  --screamer FIRST.mp4 --screamer SECOND.webm
.venv/bin/python research/event_annotations.py \
  --set FIRST.mp4 --interval 7.76:10.42 --uncertainty 0.05
```

Other attachments in that reviewed snapshot become negatives; later posts remain unlabeled. An all-safe thread uses `--reviewed` without `--screamer`. Explicit positives, visual-only exclusions, and ambiguous reviews are preserved. Timings are optional; untimed positive clips must not become negative event windows. Rebuild feature/event datasets after changing labels. Evaluate new threads with frozen models before retraining.

Build shared inputs in order:

```sh
.venv/bin/python research/extract_features.py
.venv-audio/bin/python research/extract_audio_context.py
.venv-audio/bin/python research/event_dataset.py
.venv/bin/python research/fingerprint_audio.py build
.venv/bin/python research/fingerprint_audio.py evaluate
```

Defaults produce `features-v1.json`, `audio-context-v1/`, `event-data-v1/`, and `fingerprints/evaluation.json` under `research/artifacts/`. Frozen experiments record input hashes. Reproducing historical numbers requires the matching corpus/label snapshot; rebuilding on newer labels is a new experiment. Use fresh output directories where commands refuse overwrites. Joblib loading requires `--trust-models` and the recorded scikit-learn version.

## Training

Use `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2` when running several training jobs. Keep intermediate fits, predictions, and audits in `research/artifacts/`; publish only final snapshots to `research/models/`.

For the ten baseline configurations:

```sh
.venv/bin/python research/retrain_models.py \
  --output-dir research/artifacts/retrain-new \
  --families research/artifacts/fingerprints/evaluation.json --jobs 2
.venv/bin/python research/score_retrained.py publish \
  --run research/artifacts/retrain-new \
  --output research/models/retrained-new --trust-models
```

The original tabular, pretrained-context, and event baselines use `train_models.py`, `train_audio_context.py`, and `train_events.py`. Pass `--output-dir research/artifacts/tabular-new` to the original tabular trainer to preserve the committed baseline. `improve_events.py` and `improve_event_semantics.py` create the event studies used by subsequent acoustic experiments; their `--help` lists source-run arguments.

The shipped candidate's pipeline uses shared inputs and an event-study source run (default `research/artifacts/improve-events-20261025`). Create that run first or pass `--source` to `improve_acoustics.py`:

```sh
.venv/bin/python research/acoustic_features.py \
  --output-dir research/artifacts/acoustic-new --workers 4
.venv-audio/bin/python research/synthetic_events.py \
  --output-dir research/artifacts/synthetic-new
.venv/bin/python research/improve_acoustics.py \
  --features research/artifacts/acoustic-new \
  --synthetic research/artifacts/synthetic-new \
  --output-dir research/artifacts/acoustic-study-new --jobs 2 --trust-models
.venv/bin/python research/improve_augmentation.py \
  --features research/artifacts/acoustic-new \
  --synthetic research/artifacts/synthetic-new \
  --source research/artifacts/acoustic-study-new \
  --output-dir research/artifacts/augmentation-new
.venv/bin/python research/ablate_acoustics.py \
  --features research/artifacts/acoustic-new \
  --synthetic research/artifacts/synthetic-new \
  --source research/artifacts/augmentation-new \
  --output-dir research/artifacts/ablation-new
.venv/bin/python research/acoustic_artifacts.py publish \
  --run research/artifacts/acoustic-study-new \
  --followup research/artifacts/augmentation-new \
  --ablation research/artifacts/ablation-new \
  --features research/artifacts/acoustic-new \
  --output-dir research/models/acoustic-new
```

Use `audit_acoustics.py` for browser and perturbation checks. Export still checks every model against saved final-fit scores and verifies the serialized round trip. Detailed audit files stay local.

The robustness/context pipeline remains available:

| Stage                                                         | Commands                                                                           |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Generate source-linked codec/delay/gain variants and mixtures | `generate_robust_data.py`, then `encode_robust_data.py` in the audio environment   |
| Assemble training matrices                                    | `assemble_robust_data.py`                                                          |
| Train robustness candidates                                   | `train_robust_models.py`, `train_codec_stable.py`, `train_robust_trees.py`         |
| Add semantic context                                          | `export_context_head.py` in the audio environment, then `assemble_context_data.py` |
| Train hard-negative/context candidates                        | `train_hard_negative_mil.py`, `train_context_models.py`, `train_context_trees.py`  |
| Audit and compare                                             | `audit_robust_models.py`, `audit_tree_preprocessing.py`, `summarize_robust.py`     |
| Publish portable models                                       | `export_robust_models.py --data-dir MATRIX --run RUN --output-dir SNAPSHOT`        |

These commands expose their paths with `--help`. `train_robust_models.py` also uses the event study's split plan. Generated matrices, reencodes, and browser fixtures are disposable; regenerate them before rerunning checks that consume them. Keep final models, transforms, encoder dependencies, and the original corpus.

## Frozen scoring

```sh
.venv-audio/bin/python research/acoustic_artifacts.py score \
  --snapshot research/models/acoustic-mil-20261025 \
  --media /path/to/video.mp4 --output research/artifacts/acoustic-score.json
.venv-audio/bin/python research/robust_detector.py \
  --snapshot research/models/robust-audio-20261026 \
  --media /path/to/video.mp4 --output research/artifacts/robust-score.json
.venv/bin/python research/score_retrained.py score \
  --run research/models/retrained-20261025 \
  --output research/artifacts/retrained-scores.json --trust-models
```

## Browser release

Edit `ml/*.cjs`. `build_browser_ml.py` requires the acoustic snapshot, `research/artifacts/yamnet/LICENSE`, and the portable encoder at `research/artifacts/robust-portable-encoder-20261026-v3`. Regenerate the encoder with `export_robust_encoder.py --output-dir PATH` in the audio environment. The npm lockfile pins TensorFlow.js 4.22.0. Installation and normal Node tests use committed browser assets directly.

```sh
.venv/bin/python research/build_browser_ml.py
.venv/bin/python research/prepare_browser_ml.py \
  --output-dir research/artifacts/ml-corpus-check
node research/check_browser_ml_features.cjs \
  research/artifacts/ml-corpus-check research/artifacts/ml-corpus-check.json 4
```

Corpus parity preparation uses matching feature/event matrices and frozen augmentation fold fits. Browser-original checks additionally require replay fixtures, produced with `browser_replay.cjs` and `extract_browser_events.py`; regenerate these from retained source media. `check_ml_userscript_browser.cjs` and `check_ml_tampermonkey.cjs` cover the bundled runtime, cache/fallback behavior, and installation.

After verification, commit assets first. Run `research/pin_browser_ml.py FULL_ASSET_COMMIT_SHA`, format and verify the generated userscript, then commit it. Preserve the asset commit in history so pinned URLs remain valid. Check remote pinned resources before release.
