# New acoustic features and synthetic screamer training

**A fixed candidate improves the measured development results: 30/30 red detections and 34 false red alerts, versus event-hybrid's 29/30 and 58.** That is 41% fewer false red alerts across 3,466 negative audio groups. Negative warnings at either yellow or red decrease from 100 to 98. The candidate also improves the saved browser-PCM comparison: 14/14 red detections and two false red alerts among nine hard negatives, versus 13/14 and six.

This is a stronger research candidate, not proof of deployment accuracy. The same development corpus informed several rounds of experiments. Automatic nested model/threshold selection regressed, and codec/level stress still exposes misses. **Freeze `synthetic-mil-20pct` at the existing 0.6/0.8 boundaries for future evaluation; production remains 5.8.2.** The runnable model, score/timecode CLI, all comparisons, and reproduction commands are included below.

## What changed

This phase extracts **33 genuinely new acoustic measurements** from the complete 65.04-hour, 4,014-group audio corpus. It retains the original 43 measurements and frozen YAMNet context encoder. Old measurements at original proposals reproduce within 1e-10 across the corpus. The baseline's held-out predictions reproduce within 1e-12.

| New family           | Count | Measurements and purpose                                                                                                                                                                       |
| -------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Burst shape          |    12 | Energy and duration of the active burst, peak energy over 10/50/100/200 ms, relative attack time, quiet-tail drop, and activity fraction. Separate a short loud event from subsequent silence. |
| Rhythmic context     |     4 | Prior onset periodicity, onset surprise, prior onset strength, and nearest prior spectral profile. Test whether an abrupt passage repeats an established pattern.                              |
| Absolute band energy |    12 | Absolute event energy and pre-event contrast in six frequency bands, rather than only normalized spectral shape.                                                                               |
| Roughness proxy      |     5 | Envelope modulation energy in 30–150 Hz, modulation depth, and changes from preceding audio. Computed from raw audio with a 500 Hz envelope, not the coarse level timeline.                    |

The roughness hypothesis comes from research on human alarm vocalizations, which identifies rapid amplitude modulations in roughly this range. Our envelope measure is a simplified proxy, not a reproduction of the paper's complete modulation-power analysis or evidence that normal vocal alarm semantics identify 2ch edits. [Arnal et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4562283/)

A separate proposal experiment adds up to four short onsets, ranked by 50 ms energy against preceding context, without consulting labels. It improves yellow coverage but increases false alerts. The frozen candidate therefore uses the **original proposals**, so its improvement does not come from scanning more locations.

## Synthetic examples and training objective

The generator makes **120 positive edits and 120 matched negative controls** from the ten precisely timed positive recordings. It crops confirmed screamer intervals, excluding the stated timing uncertainty; inserts 0.35, 0.75, or 1.5 seconds into a real negative background; varies position, target foreground level, and contrast; and applies a 5 ms crossfade. Peak headroom can lower the requested foreground level. Negative controls retain the background through the same edit envelope. Waveform features and frozen context embeddings are recomputed from every derivative.

These are not 120 new independent screamer identities. They provide new combinations of existing content, duration, level, and lead-in. Duration and target level share the deterministic variant schedule, so their effects are not independently factorial. The controls are relatively quiet; this limited synthesis distribution may explain some false-alert inflation. No synthetic recording is evaluated as a new real test example or added to human corpus labels.

Every derivative records **both original parent hashes**. It enters training only when both parents belong to that exact outer or inner training partition, after thread, duplicate, and known-family exclusions. This also conservatively excludes a negative control when its paired donor is excluded. Synthetic weight is balanced by class and original donor, so creating more copies cannot increase a donor's total influence. All 134 saved training partitions across the two rounds passed the parent-eligibility audit.

Combining real weak labels with synthetic strong labels is an established sound-event research approach; it does not establish validity for this particular target. DCASE likewise distinguishes synthetic training material from real evaluation data. [DCASE sound-event detection task](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments)

The final candidate combines:

1. A regularized linear model over 76 physical measurements and eight training-fold PCA components of frozen audio embeddings.
2. Linear-softmax multiple-instance learning (MIL): recording score is `sum(event_score²) / sum(event_score)`, allowing the twenty untimed positive recordings to participate without fabricated timestamps.
3. Genuine timed-window supervision and synthetic-window supervision.

Its objective is balanced real-recording cross-entropy, plus 0.25 times human-window loss, plus **0.20 times synthetic-window loss**, plus L2 regularization. The `20pct` MIL suffix means this auxiliary-loss coefficient; it does **not** mean synthetic examples are 20% of the complete objective. In the ordinary logistic augmentation experiments, percentages instead mean the share of total example weight. Imputation, weighted scaling, PCA, fitting, and latent behavior remain specific to the training partition.

## Evaluation and all main results

The unchanged real evaluation set contains **30 positive and 3,466 negative exact-audio groups**, with 518 unknown groups excluded from binary metrics. Ten positives have verified intervals. For the others, these metrics measure detection somewhere in a positive recording, not independently verified onset localization.

The eleven source-thread folds and known-family exclusions are identical to the previous study. Inner selection never sees the outer thread or its known linked families. There are 67 distinct training partitions. The first round declared twelve configurations, including the reused baseline; the adaptive follow-up declared five more after observing the synthetic recall/false-alert tradeoff. This produced **1,072 newly fitted main configurations**, plus 60 explanatory feature-ablation fits; 67 existing baseline fits were reused.

Each row below uses the same real held-out recordings. Yellow counts exclude red. Scores are evidence scores, not calibrated probabilities.

| Model                                                  | Red positives | Yellow positives | False red alerts | Yellow negatives |
| ------------------------------------------------------ | ------------: | ---------------: | ---------------: | ---------------: |
| Event-hybrid reference                                 |            29 |                0 |               58 |               42 |
| Add burst shape                                        |            28 |                1 |               56 |               32 |
| Replace long-window duration features with burst shape |            28 |                1 |               57 |               34 |
| Add rhythmic context                                   |            27 |                2 |               46 |               36 |
| Add absolute bands                                     |            29 |                0 |               59 |               33 |
| Add roughness                                          |            29 |                0 |               57 |               40 |
| All new acoustic features                              |            28 |                1 |               41 |               43 |
| All new features + MIL                                 |            27 |                1 |               18 |               46 |
| Original physical features, no embeddings              |            28 |                1 |               57 |               53 |
| New physical features, no embeddings                   |            28 |                1 |               46 |               31 |
| New physical features + expanded proposals             |            28 |                2 |               73 |               52 |
| New features + 20% synthetic example weight            |            30 |                0 |              103 |               94 |
| New features + 2% synthetic example weight             |            29 |                0 |               48 |               53 |
| New features + 5% synthetic example weight             |            29 |                1 |               56 |               62 |
| New features + 10% synthetic example weight            |            30 |                0 |               77 |               70 |
| New features + MIL + 0.05 synthetic loss               |            28 |                2 |               21 |               56 |
| **New features + MIL + 0.20 synthetic loss**           |        **30** |            **0** |           **34** |           **64** |

Average precision improves from 0.808 to 0.912 for the fixed candidate. This statistic describes the enriched development corpus and is not an estimate of production alert precision.

The short-burst recording `17929502589060439332.mp4` changes from **0.00551 to 0.92559** with its thread excluded from training. The quieter `17929314751662663260.webm` changes from **0.83270 to 0.90111**. These are held-out component scores, not all-data training scores.

False red alerts decrease or stay unchanged in every source fold:

| Thread    | Positive / negative groups | Baseline red positives / false red | Candidate red positives / false red |
| --------- | -------------------------- | ---------------------------------- | ----------------------------------- |
| 336185346 | 2 / 535                    | 2 / 8                              | 2 / 4                               |
| 336243339 | 2 / 0                      | 2 / 0                              | 2 / 0                               |
| 336272252 | 2 / 333                    | 2 / 1                              | 2 / 0                               |
| 336291305 | 4 / 313                    | 4 / 2                              | 4 / 2                               |
| 336466927 | 1 / 339                    | 1 / 5                              | 1 / 1                               |
| 336603044 | 0 / 412                    | 0 / 2                              | 0 / 1                               |
| 336679639 | 0 / 260                    | 0 / 10                             | 0 / 8                               |
| 336723954 | 1 / 682                    | 1 / 12                             | 1 / 7                               |
| 336991612 | 2 / 212                    | 2 / 8                              | 2 / 6                               |
| 337355133 | 2 / 108                    | 2 / 3                              | 2 / 2                               |
| 337690329 | 14 / 272                   | 13 / 7                             | 14 / 3                              |

A descriptive bootstrap over the eleven threads gives a 95% percentile interval of **−9.44 to −4.41 false red alerts per 1,000 negatives** for the fixed candidate's difference from baseline. This does not account for adaptive research, unrecognized related sounds, or future distribution change. Fourteen positive hashes come from one thread; 30/30 is not evidence of perfect recall.

## What failed

The automatic nested selector was unreliable: first-round selection gave **24/30 red detections and 46 false red alerts**; selection across both rounds gave **23/30 and 23**, with only 28/30 receiving any warning. Its inner requirement to preserve every baseline-detected positive did not generalize. Its all-development final choice was synthetic logistic at a shifted threshold, not the fixed MIL candidate reported here. **Neither selector is recommended or deployed.** Selecting a promising fixed candidate after these experiments remains adaptive, even though each individual score excludes its recording's source thread.

New features alone did not fix the short-event miss. Roughness alone barely changed the result, and extra proposals increased false alerts. Ordinary synthetic logistic training over-alerted. The useful result comes from the combination of feature changes, synthetic contexts, and recording-label training.

An explanatory removal study retrained the fixed candidate with the same objective and folds. It was not used to choose a new winner:

| Removed features      | Red positives | False red alerts | Negative warnings, yellow or red |
| --------------------- | ------------: | ---------------: | -------------------------------: |
| None: fixed candidate |            30 |               34 |                               98 |
| Burst family          |            30 |               32 |                              102 |
| Rhythmic context      |            30 |               37 |                              105 |
| Absolute bands        |            30 |               29 |                              100 |
| Roughness             |            30 |               30 |                               98 |
| All 33 new features   |            28 |               37 |                              111 |

The new representation helps as a whole in this experiment. It does not justify claiming each family helps: removing roughness improves red false alerts, and several removals preserve recall. These small post-discovery differences are simplification leads, not independently validated gains.

## Browser and perturbation checks

All 27 original-media browser fixtures were verified by replay, media, and PCM hashes. They contain fourteen positives, nine selected hard negatives, and four unknowns. The scoring models exclude the entire thread. This is Python inference on browser-decoded PCM, **not a JavaScript implementation of the ML model**.

| Detector                      | Red / yellow positives | False red / yellow negatives |
| ----------------------------- | ---------------------- | ---------------------------- |
| Current production rules      | 10 / 3                 | 5 / 4                        |
| Event-hybrid                  | 13 / 0                 | 6 / 1                        |
| Prior exploratory rules + MIL | 13 / 1                 | 5 / 4                        |
| New fixed synthetic MIL       | 14 / 0                 | 2 / 3                        |

The nine selected hard negatives cannot estimate a population false-positive rate. The current rules and prior fusion figures are from the unchanged [previous audit](EVENT_RESEARCH_20261025.md).

Controlled stress re-extracts proposals and embeddings from all thirty positive recordings plus those nine negatives. Each transformed recording uses its original thread-excluded model; these are diagnostics, not new independent samples.

| Perturbation           | Baseline red / any-warning positives | Candidate red / any-warning positives | Baseline / candidate false red |
| ---------------------- | ------------------------------------ | ------------------------------------- | ------------------------------ |
| Unchanged retained WAV | 29 / 29                              | 30 / 30                               | 5 / 1                          |
| Prepend 25 ms          | 28 / 29                              | 29 / 30                               | 4 / 1                          |
| Gain −3 dB             | 20 / 26                              | 22 / 28                               | 4 / 0                          |
| Gain +3 dB             | 29 / 29                              | 30 / 30                               | 7 / 4                          |
| AAC 96 kbps roundtrip  | 20 / 26                              | 23 / 26                               | 3 / 0                          |

The candidate improves these comparisons but remains sensitive to codec changes and reduced level. Gain changes can change perceived severity, so perfect gain invariance is not necessarily the right target; AAC misses are a clear remaining robustness concern. These failures, adaptive selection, and the missing browser runtime prevent a production-quality claim.

## Delivered artifacts and use

The [frozen snapshot](models/acoustic-mil-20261025/manifest.json) contains all seventeen main models, a shared transform, source/training hashes, all held-out positive scores and false warnings, browser predictions, stress results, and compact experiment plans. Models use portable JSON rather than executable pickle. Published predictions reproduce saved final-fit scores across all 4,014 groups with maximum error below 9e-14. Final-fit parity is a serialization check, not validation accuracy.

Score a media file with the fixed candidate:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-audio/bin/python research/acoustic_artifacts.py score \
  --snapshot research/models/acoustic-mil-20261025 \
  --media /absolute/path/to/video.webm \
  --output research/artifacts/new-video-acoustic-score.json
```

The command decodes media with ffmpeg, returns a continuous score, yellow/red tier, and proposed event times. Add `--all-models` to compare all seventeen snapshots. Times are machine proposals, not verified scream boundaries. Record future-thread predictions before retraining or making model choices from their labels. The final snapshot was trained on current labeled recordings, so using it to score them again does not produce held-out evaluation.

The extension's current detector is unchanged. Integrating this model requires a browser encoder/runtime, feature parity tests, and runtime/memory measurements. The delivered candidate can already be used offline; the evidence does not justify silently replacing the production userscript.

## Reproduction

Use the existing retained corpus and immutable event/encoder snapshots. Output directories must be new. Source baselines use trusted local joblib; published models do not require pickle.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/acoustic_features.py \
  --output-dir research/artifacts/acoustic-v2-rerun --workers 4

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-audio/bin/python research/synthetic_events.py \
  --output-dir research/artifacts/synthetic-rerun

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/improve_acoustics.py \
  --features research/artifacts/acoustic-v2-rerun --synthetic research/artifacts/synthetic-rerun \
  --output-dir research/artifacts/acoustic-study-rerun --jobs 2 --trust-models

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/improve_augmentation.py \
  --features research/artifacts/acoustic-v2-rerun --synthetic research/artifacts/synthetic-rerun \
  --source research/artifacts/acoustic-study-rerun --output-dir research/artifacts/augmentation-study-rerun

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/ablate_acoustics.py \
  --features research/artifacts/acoustic-v2-rerun --synthetic research/artifacts/synthetic-rerun \
  --source research/artifacts/augmentation-study-rerun --output-dir research/artifacts/acoustic-ablation-rerun

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-audio/bin/python research/audit_acoustics.py \
  --run research/artifacts/acoustic-study-rerun --followup research/artifacts/augmentation-study-rerun \
  --synthetic research/artifacts/synthetic-rerun --output-dir research/artifacts/acoustic-audit-rerun
```

The actual local runs use `acoustic-v2-20261025`, `synthetic-events-20261025`, `improve-acoustics-20261025`, `improve-augmentation-20261025`, `ablate-acoustics-20261025`, and `acoustic-followup-audit-20261025` under `research/artifacts/`. Raw media, derivatives, and full per-fold fit caches remain local. Human labels, event intervals, previous models, and production scores retain their existing meanings.
