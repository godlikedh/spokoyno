# Codec, delay, and synthetic-audio research — 2026-10-26

**This run did not produce an overall replacement for the previous best ML candidate.** It generated 25,794 recording variants, extracted 348,112 event windows, and evaluated twelve configurations across 67 distinct source-excluded training partitions: 804 saved model fits. Some models improved codec recall, but every apparent gain came with worse recall or more false alerts elsewhere. None is promoted to the userscript.

The concrete engineering gains are an exactly silence-delay-invariant feature pipeline, corrected codec tests, reproducible source-tracked augmentation, and portable Python/JavaScript inference verified against real-browser execution. The [complete comparison](models/robust-comparison-20261026/comparison.json) retains all configurations and their failure cases. The previous `synthetic-mil-20pct` snapshot remains the research baseline; this is not a claim that it is ready for production either.

## Main results

All thresholds remain **0.6 for yellow and 0.8 for red**. Scores are uncalibrated risk evidence. These are held-out predictions on **30 positive and 3,466 negative original audio groups**, with source threads, duplicate memberships, and known transitive audio families excluded from training.

| Model                                     | Original screamers red / 30 | Original normal recordings red / 3,466 | AAC screamers red / 30 | Opus red / 30 | MP3 red / 30 |
| ----------------------------------------- | --------------------------: | -------------------------------------: | ---------------------: | ------------: | -----------: |
| Previous best ML: synthetic MIL           |                          30 |                                     34 |                     25 |            28 |           28 |
| New primary: consistent hybrid            |                          30 |                                     66 |                     23 |            29 |           30 |
| No clipping feature, stronger consistency |                          28 |                                     50 |                     25 |            28 |           28 |
| Small boosted trees, seven leaves         |                          29 |                                    154 |                     29 |            29 |           29 |

The previous model's codec columns were recomputed using the **same new bounded codec recipes** on all thirty positives. Its codec negative set contains only nine selected hard negatives, so this report does not compare its codec false-positive rate with the full new-model evaluation set. Original-file false-alert counts use the full 3,466 negatives for every model.

The original baseline produced 98 yellow-or-red warnings on negatives. The new primary produced 168. The new softmax variant came closer at 104, with all thirty positives yellow-or-red, but only 27 red. The boosted-tree primary produced 263 warnings on negatives. These are unfavorable product tradeoffs for rare screamers.

### Every fixed configuration

| Configuration         | Original red positives | Original red negatives | AAC red positives | Opus red positives | MP3 red positives |
| --------------------- | ---------------------: | ---------------------: | ----------------: | -----------------: | ----------------: |
| consistent-hybrid     |                     30 |                     66 |                23 |                 29 |                30 |
| augmented-hybrid      |                     30 |                     65 |                23 |                 29 |                30 |
| consistent-physical   |                     30 |                     66 |                24 |                 30 |                30 |
| augmented-physical    |                     30 |                     67 |                24 |                 30 |                30 |
| unaugmented-hybrid    |                     25 |                     34 |                12 |                 26 |                25 |
| augmented-softmax     |                     27 |                     33 |                23 |                 28 |                29 |
| codec-stable-hybrid   |                     28 |                     50 |                25 |                 28 |                28 |
| codec-stable-physical |                     27 |                     54 |                24 |                 26 |                26 |
| codec-stable-softmax  |                     23 |                     20 |                23 |                 24 |                23 |
| noclip-hybrid         |                     28 |                     50 |                25 |                 28 |                28 |
| mil-boost-7           |                     29 |                    154 |                29 |                 29 |                29 |
| mil-boost-15          |                     29 |                    158 |                28 |                 28 |                29 |

Augmentation helped the new architecture: its unaugmented control caught only 25 originals and 12 AAC positives. That does not establish an improvement over the previous architecture. The codecs and synthetic mixing were introduced together in that comparison, so it also does not isolate which augmentation contributed how much.

Automatic model selection was evaluated inside the outer splits. The first six-model selector caught 27/30 original positives with 63 false alerts; the no-clipping selector caught 28/30 with 51; the tree selector caught 29/30 with 154. Choosing the nicest fixed row after inspecting all results would overstate the evidence. The published defaults select a model within each experiment, not a production winner.

## What changed

The production heuristic was not used as a blend, fallback, teacher, or promotion target. The new pipeline changes preprocessing, event selection, and learning:

1. **Bound PCM to the playback range before codec transforms and feature extraction.** This prevents the old raw-overrange AAC experiment from conflating compression with extreme signal-level changes. Channels remain separate; stereo cancellation is not introduced by averaging channels.
2. **Remove leading digital silence and insert a fixed six-second analysis prefix.** Absolute timestamps, opening flags, and the old absolute-threshold attack feature are omitted. Display timestamps retain the removed offset.
3. **Propose events at multiple durations.** Use six separated candidates for 50 ms transitions, six for 300 ms transitions, a candidate per twenty-second block, and the loudest opening candidate. This covers short bursts and long videos better than only three global candidates, but also increases opportunities for false alarms.
4. **Use 68 physical measurements**, including loudness changes, burst duration and relative attack, rhythm, spectral change, and absolute frequency-band energy. These reuse feature families from the previous acoustic study. Roughness features are omitted. Hybrid heads additionally use eight training-only PCA components from frozen YAMNet before/event/after embeddings.
5. **Train recording-level multiple-instance models.** A positive recording need not have every event labeled positive. Strong timing labels supervise only confidently labeled windows. A high-order pooling function reduces dilution of a strong event by weak candidates; the lower-order pooling control exposes the recall/false-alert tradeoff.
6. **Penalize score differences between training variants of the same source.** Compare ordinary augmentation with consistency weights of one and ten. Weight original sources equally within each class, regardless of how many copies were generated.
7. **Test a nonlinear alternative.** Small histogram-boosted trees use seven or fifteen leaves, depth three or four, 120 boosting iterations, and two fixed rounds of selecting likely events in weakly labeled positive recordings. Their seed models use the exact same training partition. They improve AAC recall but produce too many false alarms.

The use of weak recording labels, strong event labels, synthetic data, and consistency training is informed by the [DCASE sound-event detection baseline](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cornell_baseline_t4.pdf). Its success on other sound-event tasks does not establish success on this corpus; the measurements above are the deciding evidence here.

## Generated data and labels

| Group                                          | Recordings |
| ---------------------------------------------- | ---------: |
| Original audio groups, including 518 unknowns  |      4,014 |
| Real-source training codec/delay/gain variants |      7,292 |
| Synthetic positive recordings                  |      2,000 |
| Matched normal-background controls             |      2,000 |
| Reserved evaluation codec variants             |     10,488 |
| **Total**                                      | **25,794** |

Training uses AAC at 64/96 kbit/s and Opus at 32/64 kbit/s. Real-source leading delays range from one sample to seven seconds; synthetic delays extend to twenty seconds. Real-source gain variations include ±1.5 dB. Test profiles use AAC 80, Opus 48, and **MP3 128 kbit/s, a codec absent from training**, with additional delays. Evaluation profiles never enter fitting or inner selection.

The ten timed screamer recordings provide 200 mixes each. Segments stay inside human intervals after allowing the stated ±0.05 s boundary uncertainty. Mixes vary foreground duration (0.2–3 s), insertion time, level, contrast, background, and additive versus replacement insertion, with five-millisecond fades. Matched controls retain the normal background's temporal structure, with optional uniform gain. An arbitrary newly inserted loud sound is not declared negative merely because its source recording was normal.

Synthetic positives are generated training labels, not newly reviewed human examples. Some mixes may be less realistic than the original thread content. The failed promotion results are evidence against assuming that more such mixes automatically improve the product.

Every derivative records its recipe and parent hashes. A synthetic recording is eligible for a training fold only if **both its donor and background** are eligible. Synthetic examples carry 20% of the real examples' total objective weight; copies do not increase an original source's weight. The 471 synthetic positives without a fully contained positive proposal remain weakly supervised, with nearby windows unknown rather than falsely negative. Human labels and timings were not changed.

The local artifacts retain recipes, hashes, extracted features, and frozen embeddings. Temporary encoder patches are removed after verified embedding writes. The immutable [generation summary](models/robust-comparison-20261026/generation-plan.json) identifies the full local recipe plan and extractor source.

## What actually caused the remaining AAC failures

The previous stress test fed float audio with amplitudes far outside ±1 directly into a 16 kHz AAC round trip. Some clips suffered large level changes as well as encoder delay. The new tests bound PCM before encoding, retain channels, and measure codec alignment. Aligned and unaligned tests are reported separately.

Correcting that test did **not** eliminate codec sensitivity. The new linear heads still learned a large positive weight for `near_clip_fraction`. AAC can strongly reduce the fraction of samples exceeding a fixed amplitude threshold while preserving a very similar waveform.

For `17929291025992177046.webm`, the first-round primary's recording score fell from approximately **0.842 to 0.423** under AAC, despite waveform correlation of **0.992**. At the strongest proposed event, the change in the clipping feature contributed about **−1.59 to the model logit**. Similar drops were −3.02 and −3.86 in two other examined misses. These are model diagnostics, not extra hand-written rules for those filenames.

The follow-up retrained with that input fixed to zero, making its coefficient exactly zero, and increased consistency pressure. It improved AAC recall to 25/30, but original-file recall fell to 28/30. The trees then reached 29/30 on AAC at a much higher false-alert rate. Neither fix satisfies the overall product requirement.

## Delay and runtime verification

**Pure leading zero padding is exactly invariant in the new pipeline.** On thirty positives and nine hard negatives, five delays—0.013, 0.125, 1.75, 11, and 30 seconds—produced **195 exact comparisons**. Physical features and encoder inputs were identical, with maximum difference zero; displayed event times shifted by the expected offset. A unit test also covers a non-frame-aligned sample offset. These are paired checks on 39 originals, not 195 independent examples.

Lossy encoding changes the waveform. Exact score equality for arbitrary codecs is not established; warning-decision stability remains a requirement the new candidates fail on some examples. Gain changes can also alter perceived severity. The reports retain aligned AAC, 16 kHz AAC, gain ±3 dB, and Opus plus eleven seconds of delay, including failures.

The portable runtime is implemented in [robust_features.cjs](robust_features.cjs), [robust_encoder.cjs](robust_encoder.cjs), and [robust_model.cjs):

- On all thirty positive originals plus ten negatives, 40,451 physical values differed from Python by at most **0.000032**. Event times and encoder patch samples matched exactly.
- The exported frozen YAMNet encoder uses standard [TensorFlow.js](https://www.tensorflow.org/js/tutorials/setup) operations with the WASM backend. Three numerical fixtures had maximum embedding difference **0.000358**. The classifier head is not exported from YAMNet.
- Full browser execution on 27 preserved original-media fixtures matched all six first-round models within **0.0000019**, and all four no-clipping models within **0.00000062**, with **no warning-tier changes**. The 27 fixtures contain fourteen positives, nine selected hard negatives, and four unknowns. The models exclude their source thread.
- Both tree heads matched Python scores **exactly** on those 27 browser-decoded fixtures using Node JavaScript. This particular tree check did not launch another browser. Tree export is also checked directly against sklearn and JavaScript in a unit test.

The first browser pass took roughly 0.5–2.4 seconds per fixture for feature extraction, frozen encoding, and all six heads on this machine. The second ran under concurrent training load and was slower. These are development-machine measurements; they do not establish mobile performance or extension concurrency performance. The standalone runtime has not been integrated into the userscript's worker pool.

The browser checks were muted and used isolated temporary profiles. Audio was not uploaded. A runtime/decode failure is an error, not a zero/safe prediction.

## Evaluation limits and release decision

The [family audit](models/robust-comparison-20261026/family-audit.json) recomputed all 121 outer/inner partition definitions from source membership and the transitive known-family graph. Per-fit audits verify derivative eligibility; tree audits additionally verify the same-fold seed model. Unknown recordings are excluded from accuracy calculations. Thresholds were not retuned against the reserved codec results.

The first six configurations were declared before this run's fits. The no-clipping follow-up was motivated by its held-out failure analysis. The two tree configurations were declared before inspecting their results or the no-clipping follow-up results. These are **adaptive experiments on an existing development corpus**, not fresh prospective tests. Nested selection and source grouping reduce leakage; they cannot prove absence of overfitting or detect every unknown shared sound family. The comparison includes descriptive thread-bootstrap intervals with that limitation explicitly recorded.

The source count remains **30 positive recordings, ten with timings**. More generated views teach robustness to chosen transformations; they do not create new independent screamer styles. The results support retaining this infrastructure and rejecting the new heads as overall replacements. Further model choices should be judged against frozen predictions on subsequently received original threads. No additional tuning of this run's held-out thresholds is presented as progress.

The userscript remains 5.8.2. No new model is silently blended into it. The snapshots below are runnable research artifacts, including unsuccessful configurations, so the experiment can be inspected and reproduced:

- [First six models](models/robust-audio-20261026/manifest.json)
- [Four no-clipping models](models/codec-stable-audio-20261026/manifest.json)
- [Two nonlinear models](models/robust-trees-20261026/manifest.json)
- [Comparison, codec baseline, browser parity, environment, and provenance](models/robust-comparison-20261026/manifest.json)

Each snapshot checks file and extractor hashes and stores portable JSON rather than executable pickle. Serialization is checked against saved final-fit predictions on all 25,794 records. Final-fit scores on training recordings are not validation accuracy.

Repository validation passed: 77 Python tests, 19 JavaScript tests, Ruff lint/format checks, and Prettier. Both the hybrid and physical-only command-line inference paths were exercised with frozen snapshots. Browser and corpus parity checks above are additional to those unit tests.

## Reproduction

Use the retained local corpus and existing frozen event/family snapshots. The generator uses Python 3.14's bounded process-pool mapping; the TensorFlow encoder runs in the existing Python 3.12 audio environment. Exact library/build versions are recorded in [environment.json](models/robust-comparison-20261026/environment.json). Choose new output directories for a new experiment.

```sh
# Run these two concurrently; the consumer processes atomic patch files.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/generate_robust_data.py \
  --output-dir research/artifacts/robust-data-rerun --mixes-per-donor 200 --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-audio/bin/python research/encode_robust_data.py \
  --data-dir research/artifacts/robust-data-rerun

.venv/bin/python research/assemble_robust_data.py \
  --data-dir research/artifacts/robust-data-rerun --output-dir research/artifacts/robust-matrix-rerun
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_robust_models.py \
  --data-dir research/artifacts/robust-matrix-rerun --output-dir research/artifacts/robust-study-rerun --jobs 2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_codec_stable.py \
  --data-dir research/artifacts/robust-matrix-rerun --output-dir research/artifacts/codec-stable-rerun --jobs 2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_robust_trees.py \
  --data-dir research/artifacts/robust-matrix-rerun --output-dir research/artifacts/robust-trees-rerun \
  --seed-run research/artifacts/codec-stable-rerun --jobs 2
```

For each completed run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-audio/bin/python research/audit_robust_models.py \
  --data-dir research/artifacts/robust-matrix-rerun --run research/artifacts/robust-study-rerun \
  --output-dir research/artifacts/robust-audit-rerun
.venv/bin/python research/export_robust_models.py \
  --data-dir research/artifacts/robust-matrix-rerun --run research/artifacts/robust-study-rerun \
  --audit research/artifacts/robust-audit-rerun --output-dir research/artifacts/robust-export-rerun
```

Score an existing media file with a frozen snapshot:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-audio/bin/python research/robust_detector.py \
  --snapshot research/models/robust-audio-20261026 \
  --media /path/to/video.webm --output /tmp/spokoyno-robust-score.json
```

The result contains a continuous score, tier, and proposed event times. Add `--model` to select another head within that snapshot. Times are machine proposals, not verified screamer boundaries.

The browser reproduction tools are `export_robust_encoder.py`, `prepare_robust_browser.py`, and `check_robust_browser.cjs`; their arguments are documented in the executable usage checks. The JavaScript dependency directory must contain `@tensorflow/tfjs@4.22.0` and `@tensorflow/tfjs-backend-wasm@4.22.0`. `check_robust_node.cjs` verifies physical-only heads without those packages. `benchmark_previous_robust.py` reproduces the corrected previous-model codec comparison, and `summarize_robust.py` produces paired codec changes and clustered comparisons.

Actual local datasets/runs use the `20261026` suffix; the final tree run is `robust-tree-study-20261026-v2`. Ten completed tree partitions were retained through a formatting-only process restart before inspecting tree predictions; its resume provenance is included in the snapshot. This did not add candidates or change their parameters. Raw audio, generated matrices, and complete per-fold fit caches stay local.
