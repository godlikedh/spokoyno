# Audio context and known-screamer matching

These are local, offline shadow experiments. Neither their models nor fingerprints drive `spokoyno.user.js`. The separate v5.8 UI update adds a numeric score and yellow tier while preserving v5.7 red decisions. Visual-only screamers are excluded. The workflow of marking specific screamers and treating the rest of a reviewed thread snapshot as negative is preserved.

For the completed annotation-aware event comparison, revised fingerprint results, and browser replay, see [Event results](EVENT_RESULTS.md). The original results below are retained as historical baselines.

## Timing annotations

Clip labels remain in `corpus/labels.json`. User-confirmed event timings go in `corpus/events.json`. No detector-generated timings have been inserted as ground truth.

Mark the beginning and end of the actual screamer audio, in seconds from the beginning of the retained recording. About 0.1-second precision is sufficient to start. Include a ramp if it is part of the startling sound; exclude ordinary setup audio. Repeat for multiple events. Negative clips do not need timings.

The first annotation batch covers all ten positive recordings with twelve intervals. The user measured the boundaries visually and estimates about ±0.05 s error, recorded as `estimated_boundary_error_s` on each entry. This is an approximate uncertainty estimate, not a guaranteed bound or sample-accurate ground truth. User-provided timestamps are preserved without rounding or automatic adjustment; explicit end-of-file boundaries use the exact retained WAV duration. The second interval in `17885460222141837902.mp4` repeats the same audio and must not be counted as an independent positive example. Coverage below uses the nominal onset times and does not adjust its 250 ms threshold for this uncertainty.

List the ten positive recordings and their annotation status:

```sh
.venv/bin/python research/event_annotations.py
```

Record a confirmed interval (replace the illustrative times with your own):

```sh
.venv/bin/python research/event_annotations.py \
  --set 17884174673280229863.mp4 --interval 3.0:3.9
```

Repeat `--interval START:END` for multiple events; `END` may be `end`. Use `--uncertainty 0.05` for an estimated ±0.05 s boundary error. Setting a recording replaces that recording's previous intervals and preserves its previous uncertainty unless a new value is supplied. The command validates bounds against the retained WAV and records its content hash. It does not change whether a clip is positive or negative.

Check existing event candidates against the confirmed onsets:

```sh
.venv/bin/python research/event_annotations.py --coverage
```

This reports the nearest candidate and whether it falls within 250 ms of each annotated onset. Annotations do not change candidate selection. Before any intervals are supplied, the coverage report explicitly contains zero annotated events.

## Pretrained context experiment

The extractor uses official frozen YAMNet weights. Source code is pinned to TensorFlow Models commit `d598fb8b23d9cd2fb26b5789b8242de3f494aca7`; the weights are checked against SHA-256 `13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744`. Downloads remain under ignored `research/artifacts/yamnet/`, retaining the upstream license and a hash manifest. Audio is processed locally and is not uploaded.

YAMNet needs a separate environment because the existing `.venv` uses Python 3.14, which has no TensorFlow wheel. An initialized `.venv-audio` already exists in this checkout. For a fresh setup with Python 3.12 available:

```sh
python3.12 -m venv .venv-audio
.venv-audio/bin/pip install -r research/requirements-audio.txt
```

Alternatively, install `uv` in the original environment and let it obtain a local interpreter:

```sh
.venv/bin/pip install uv
UV_CACHE_DIR="$PWD/research/artifacts/uv-cache" \
UV_PYTHON_INSTALL_DIR="$PWD/research/artifacts/python" \
  .venv/bin/uv venv .venv-audio --python 3.12
UV_CACHE_DIR="$PWD/research/artifacts/uv-cache" \
  .venv/bin/uv pip install --python .venv-audio/bin/python -r research/requirements-audio.txt
```

Extract the features, then compare the three models:

```sh
TF_CPP_MIN_LOG_LEVEL=2 .venv-audio/bin/python research/extract_audio_context.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python research/train_audio_context.py
```

The existing feature extractor provides three event candidates and one opening candidate, without using event annotations. Ten independent 975 ms waveform patches cover before/event/after context for the three candidates plus the opening. Out-of-bounds context is zero-padded. The more energetic real channel is selected to avoid cancellation of antiphase stereo; patches exceeding nominal full scale are scaled into YAMNet's input range. Separate physical features retain raw level, clipping, and contrast measurements.

Each patch produces a 1,024-dimensional embedding. Eight PCA components summarize the concatenated embeddings; PCA, imputation, and scaling are fitted using training rows only. Three fixed, regularized logistic models compare physical features, embeddings, and their combination. Candidate and feature design has already seen the current corpus, so even grouped results remain development diagnostics.

Exact audio groups are deduplicated. Each group is counted once in evaluation, and every training group with any membership in the held-out source thread is excluded. Fold thresholds are above the largest fitted-training negative score, which does not guarantee a low population false-alarm rate. Model selection uses these same folds and therefore also needs prospective validation.

The batched encoder checks its output against the unmodified official waveform model. Classifier exports use numerical JSON, including PCA and scaling transforms, with inference parity checks. There is no pickle loading or browser implementation of the new features.

### Initial result — 2026-09-05

| Features                  | Held-out positives | False alarms / 1,181 unique negatives | Average precision |
| ------------------------- | -----------------: | ------------------------------------: | ----------------: |
| Physical features         |               4/10 |                                     0 |             0.594 |
| YAMNet context, PCA-8     |               0/10 |                                     1 |             0.030 |
| Physical + YAMNet context |               3/10 |                                     1 |             0.506 |

This particular pretrained-context representation did not improve the baseline. It is retained as a reproducible experiment, not promoted. These results do not establish that every pretrained model or event representation would fail. Event timings can help identify proposal misses; more naturally collected examples can support future comparisons without waiting for a large dataset up front.

Extraction ran with Python 3.12.14, TensorFlow/tf-keras 2.21.0, and NumPy 2.5.2. Model evaluation ran with scikit-learn 1.9.0 and SciPy 1.18.1 in the original environment.

Artifacts:

- `research/artifacts/audio-context-v1/`: cached embeddings and hash manifests; existing vectors are reused when extending the corpus.
- `research/artifacts/audio-models/physical-model.json`, `embedding-model.json`, `hybrid-model.json`: all three fitted candidates.
- `research/artifacts/audio-models/model.json`: the selected candidate, currently physical features alone.
- `research/artifacts/audio-models/results.json` and `REPORT.md`: all fold predictions, errors, thresholds, and training results.

Score an exported candidate without retraining:

```sh
.venv/bin/python research/train_audio_context.py \
  --score research/artifacts/audio-models/hybrid-model.json
```

The scorer writes a new timestamped snapshot with the model hash, feature provenance, labels at scoring, and whether each recording appeared in training. Extend the corpus and extract both kinds of features before scoring a new thread. Freeze predictions before changing labels or retraining. Preserve the ignored artifacts with the corpus backup when they matter as prospective evidence.

## Known-screamer fingerprints

This is an independent local landmark matcher, inspired by Wang's audio fingerprinting method. It builds time/frequency peak pairs, votes for a consistent time offset, and requires multiple anchors, hash/frequency diversity, and reference coverage. Spectrogram memory is bounded by processing overlapping chunks. It is intended to recognize reuse of known audio; it does not learn how to recognize a new kind of screamer.

Build references and score clips:

```sh
.venv/bin/python research/fingerprint_audio.py build
.venv/bin/python research/fingerprint_audio.py score
```

To score specific recordings, repeat `--file FILENAME` or use their canonical media paths:

```sh
.venv/bin/python research/fingerprint_audio.py score \
  --file 17884174673280229863.mp4
```

If timings are confirmed, references contain the annotated sound intervals, and matches report estimated event locations in the query. Before timings are available, references cover the whole positive recording. A partial match to an unannotated recording is only an identity hint: it might match ordinary setup audio. It is never automatically represented as a confirmed event. Exact audio-hash matches are preserved when scoring reuploads.

Evaluate with source exclusion and controlled variants:

```sh
.venv/bin/python research/fingerprint_audio.py evaluate
```

The corpus evaluation excludes the reference's identical audio hash, including aliases, from its own query. It reports negative identity matches and matches to other positive recordings separately. The controlled reuse test intentionally keeps the source reference and measures recovery of the same recording, a 25 ms shift, -3 dB gain, and an AAC 96 kb/s roundtrip. These transformed copies are not independent positive examples and are not a measure of detection on unfamiliar content.

References, per-audio cached landmarks, timestamped scoring snapshots, and `evaluation.json` are under ignored `research/artifacts/fingerprints/`. `score` loads the existing index; it does not silently rebuild it. Rebuild after confirming timings or adding positive references. All matches remain offline shadow results.

### Initial matching result — 2026-09-05

| Controlled query against its known reference                        | Recovered |
| ------------------------------------------------------------------- | --------: |
| Same recording, using landmarks rather than the exact hash shortcut |     10/10 |
| Prepend 25 ms silence                                               |     10/10 |
| Reduce gain by 3 dB                                                 |     10/10 |
| AAC 96 kb/s encode/decode                                           |     10/10 |

With the identical source excluded, none of the ten positives matches a different positive reference. This is expected to be a reuse tool rather than a new-screamer classifier. One of 1,181 negative recordings produces an identity match. This initial evaluation preceded the timing annotations and used no user-annotated event references, so it does not estimate an event-specific false-alarm rate. Saving annotations alone does not rebuild the fingerprint index or update these historical results.

The negative `17884283770650516151.mp4` matches positive `17885024222391568056.mp4` across 14.78 seconds, with 366 matching anchor times. The existing physical features distinguish them sharply: the negative's selected event is approximately -16.60 dB with a 9.79 dB rise, versus -3.48 dB and a 34.47 dB rise in the positive. This is evidence of substantial shared audio material, not a reason to change the user's negative label. A gain-insensitive identity matcher can recognize ordinary material that was used in a boosted screamer edit. Even with timing annotations, identity must be interpreted alongside acoustic level and local contrast before it can support a warning.

These parameters were not retuned to remove that negative after observing the result. The identity matcher, classifier, and original userscript remain separate. Targeted scoring also verifies that the known short-burst recording receives an exact-audio match, while the reviewed phonk example does not match.

## References

- [Official YAMNet implementation](https://github.com/tensorflow/models/tree/d598fb8b23d9cd2fb26b5789b8242de3f494aca7/research/audioset/yamnet)
- [YAMNet transfer learning tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Wang, An Industrial-Strength Audio Search Algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
