# Detector stability and saved-model inventory — 2026-09-21

Status: **candidate only; do not merge as a production improvement yet**.
The requested condition of fewer false positives than comparable saved ML is
not satisfied once the forest's already-saved threshold is included.

## What "frozen forest" means

It means inference with the existing `forest-rich-depth3` artifact, without
retraining its trees or adjusting its saved decision threshold on this review.
It is not a new model. The forest has 400 trees of maximum depth three and uses
150 physical features (loudness, clipping, persistence, spectral changes, etc.).

There are seven saved ML variants, none deployed in the userscript:

| Family             | Saved variant     | Input / decision                                                                                |
| ------------------ | ----------------- | ----------------------------------------------------------------------------------------------- |
| Original tabular   | Logistic shadow   | Regularized logistic regression over 48 selected physical features                              |
| Original tabular   | Forest challenger | 400 shallow decision trees over 150 physical features                                           |
| Whole-clip context | Physical          | Logistic regression over physical features; equivalent predictions to the original logistic fit |
| Whole-clip context | Embedding         | Frozen pretrained YAMNet audio embeddings, PCA-8, then logistic regression                      |
| Whole-clip context | Hybrid            | Physical features plus YAMNet/PCA context, then logistic regression                             |
| Event-level        | Physical          | Logistic regression on each proposed event; maximum score across proposals                      |
| Event-level        | Hybrid            | Event physical features plus before/event/after YAMNet context; maximum event score             |

The original study also compared five tabular configurations (two logistic and
three forest configurations). Only the selected logistic and rich depth-three
forest were exported as final whole-dataset scorers. Event fold models are
validation fits, not additional competing deployed models. `model.json` files
in experiment directories are aliases, not extra model families.

The separate audio-fingerprint experiment is a known-audio reuse matcher, not
a general screamer classifier. It remains unsuitable for production because
short annotated references produced many collisions; see `EVENT_RESULTS.md`.

## Full retained-audio comparison

No model was retrained. Physical features, YAMNet context, and event matrices
were refreshed using current labels. All 3,392 non-visual exact-audio groups
were scored. Accuracy below uses 11 positive and 2,810 negative groups; 571
unlabeled groups are excluded. There are also 72 no-audio attachments in the
index; these are not successful audio classifications. Exact duplicates count
once, and visual-only screamers are excluded from this audio task.

The shared display policy is red at score >= 0.8, yellow at 0.6 <= score < 0.8.
Scores are uncalibrated and are not interchangeable probabilities.

| Detector / saved model                | Red positives / 11 | Red false positives / 2,810 | Yellow-only negatives |
| ------------------------------------- | -----------------: | --------------------------: | --------------------: |
| Production 5.8.1, retained-WAV replay |                 11 |                           2 |                    28 |
| Candidate neighborhood fix            |                 11 |                           5 |                    35 |
| Logistic shadow                       |                 11 |                          19 |                    32 |
| Forest challenger                     |                  8 |                           0 |                     3 |
| Whole-clip physical                   |                 11 |                          19 |                    32 |
| Whole-clip embedding                  |                  2 |                          89 |                   530 |
| Whole-clip hybrid                     |                 11 |                          18 |                    25 |
| Event physical                        |                 11 |                          47 |                    44 |
| Event hybrid                          |                 11 |                          47 |                    43 |

### Historical saved thresholds also matter

Comparing only raw score 0.8 would conceal a stronger forest operating point.
Its threshold was already present in the saved artifact before this thread:

| Model                          | Previously saved cutoff (strict >) | Positive alerts / 11 | False alerts / 2,810 |
| ------------------------------ | ---------------------------------: | -------------------: | -------------------: |
| Logistic / whole-clip physical |                 0.9800518511071886 |                    6 |                    0 |
| Forest challenger              |                 0.6453151545544709 |                   11 |                    1 |
| Whole-clip embedding           |                 0.9999999988389673 |                    0 |                    0 |
| Whole-clip hybrid              |                 0.9817314047070838 |                    7 |                    0 |

The forest's sole false alert at its saved cutoff is
`/b/src/336679639/17896836443193031385.webm` (score 0.7951716543).
Its three positives below 0.8 are yellow, not absent scores.

Most positives were used in ML training. The subset not present as identical
training audio contains only **one positive and 1,629 negatives**. At its
historical cutoff, the forest detects that positive and alerts on one negative.
This is not enough new positive data to establish generalization. The older
source-thread-held-out forest experiment detected 7/10 positives with two false
alerts; those results must not be replaced with the optimistic all-corpus fit
results above. See `models/MODEL_CARD.md` for its original evaluation protocol.

## Browser discrepancy and candidate fix

The baseline replay is commit `5f32be88c3dfe0ee0601252c34cb68562b2ec79d`.
The retained audio is FFmpeg-decoded float32 WAV, not original-media browser
decoding. The earlier report of 0.906 for `17898262773380676776.webm` described
that offline pipeline and did **not** establish its actual browser score.

A fresh, isolated, muted Brave/Chromium 150 profile reproduces approximately
0.6664 on the original WebM with 5.8.1. Slight decode/resample differences move
the selected window from 26.80 to 26.75 seconds. Its single-frame spectral flux
changes from about 0.313 to 0.012 despite essentially the same loudness jump.

The candidate uses the already-computed maximum spectral change within
plus/minus 100 ms for the main transition score, leaving loudness, duration,
red thresholds, and rescue eligibility unchanged. It scores the original WebM
about **0.9357, red**, in both browser worker and fallback paths. The analysis
cache version is incremented, without invalidating downloaded media.

However, this simple fix adds three red false positives on retained audio:

- `17881699236063425442.mp4`: 0.7300 -> 0.9747.
- `17882990553793679533.webm`: 0.6546 -> 0.9171.
- `17896818223550599544.webm`: 0.7264 -> 0.8866.

The previously reviewed ambiguous loud meme remains yellow. The browser-decoded
regression sample remains red with -25, -12.5, 0, +12.5, and +25 ms timing shifts.
The corpus was not re-downloaded in its original encoded form: an attempted
fetch of the oldest positive returned 404. Full-corpus numbers must therefore
remain labeled retained-WAV results, not an all-browser original-media test.

## Recommendation / merge gate

Do not silently deploy this candidate: it fixes the reported browser failure,
but increases false positives and loses to the forest at matched corpus recall.
Also do not silently promote the forest. A production port needs numerical
feature/inference parity, real-browser decoding tests, and an explicit decision
about mapping its saved cutoff to the desired 0.8 UI boundary. A remapped score
would still be evidence, not a calibrated screamer probability.

Keep all saved ML artifacts unchanged. Preserve this candidate and evaluation
in a reviewable PR while the deployment choice is resolved. New reviewed threads
remain necessary to compare future behavior without repeatedly tuning to these
eleven screamers.

## Reproduction and artifacts

```sh
.venv/bin/python research/extract_features.py --workers 4
TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv-audio/bin/python research/extract_audio_context.py --threads 4
.venv/bin/python research/event_dataset.py
.venv/bin/python research/evaluate_frozen_models.py --output research/artifacts/fresh-model-comparison.json
node research/evaluate_userscript.cjs \
  --baseline 5f32be88c3dfe0ee0601252c34cb68562b2ec79d \
  --output research/artifacts/fresh-userscript-comparison.json --workers 4
node research/browser_replay.cjs --media /path/to/original.webm \
  --baseline 5f32be88c3dfe0ee0601252c34cb68562b2ec79d \
  --output research/artifacts/fresh-browser-comparison.json
```

The browser harness accepts `--browser /path/to/chromium-binary`, uses a new
temporary profile, never plays audio, and preserves decoded float WAV alongside
its JSON report. Run it only with trusted local media. Output snapshots refuse
to overwrite previous results. Per-model hashes, training membership, labels,
and all per-group predictions are retained in ignored artifacts:

- `research/artifacts/frozen-model-comparison-with-cutoffs-20260921.json`
- `research/artifacts/scoring-nearby-final-20260921.json`
- `research/artifacts/browser-nearby-final-20260921.json`

Checks: 36 Python tests, 18 JavaScript tests, Ruff, Prettier, and syntax checks.
The real-browser PCM regression test is skipped in fresh clones without the
local-only fixture; the actual original-media browser run above was performed
in this workspace. None of these results is an independent accuracy guarantee.
