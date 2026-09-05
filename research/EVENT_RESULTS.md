# Event labels, staged reviews, and score tiers

Measured on 2026-09-05. These are development experiments on the existing corpus, not independent estimates of production accuracy. No ML or fingerprint result has been promoted into the userscript.

## Browser behavior

Spokoyno 5.8 shows the eligible, merged heuristic decision score on a 0–1 scale:

| Score                                    | Presentation                       |
| ---------------------------------------- | ---------------------------------- |
| Below 0.6                                | Low risk                           |
| 0.6 to below 0.8                         | Yellow **MAYBE** badge and outline |
| At least 0.8                             | Red screamer alert and outline     |
| Missing/invalid score or failed analysis | Unknown, not safe                  |

The score is not a probability. Previously the badge showed raw path evidence while the alert used a different score after eligibility checks. The new badge, tier, and red decision use the same `score = decisionScore`; raw evidence remains in the tooltip. Values displayed to three decimals are truncated so 0.79999 cannot appear as 0.800 in a yellow badge. Community reports retain their own red treatment without changing the audio score. Upgrading invalidates old analysis results, not cached media.

Replaying the actual old and new JavaScript detectors over all 1,191 uniquely labeled audio recordings produced:

- 10/10 positive recordings red; no red alerts on 1,181 negatives.
- Four negatives yellow; no red decisions or numeric decision scores changed.
- Yellow examples: `17881699236063425442.mp4` (0.730), `17881887457541053716.mp4` (0.721), `17882990553793679533.webm` (0.654), and `17885509617340308491.mp4` (0.615).

This is a regression check on data used to develop the rules, **not** held-out accuracy. The four yellow warnings are an intentional increase in caution, not four newly discovered screamers. The test harness runs the detector without playback; actual browser decoding/layout has not been manually tested here.

```sh
node --test research/test_userscript.cjs
node research/userscript_harness.cjs
```

The corpus replay compares the working userscript with `HEAD:spokoyno.user.js`. Its JSON output is `research/artifacts/userscript-tiers-v1.json`.

## Event-level physical / YAMNet comparison

Ten recordings contain twelve annotated intervals, with estimated ±0.05 s manually measured boundary error. They remain ten recordings, not twelve independent positives. All twelve intervals already overlap an original YAMNet event patch; onset error alone did not establish that the old model never heard the screamer.

The new dataset uses the existing label-independent top three transition proposals plus the opening. This is a localized classifier over automatically proposed events, **not** an exhaustive sliding-window classifier. All whole-recording proposals remain active at test time; human timings never select test proposals.

There are 4,749 event windows: 25 positive, 4,715 negative, and nine uncertain. Each has 43 physical features recomputed at its own location, plus the frozen before/event/after YAMNet vectors already cached. Opening events use their own physical features and mark missing baselines explicitly. The opening baseline extractor was corrected to avoid a negative slice endpoint reading future audio; existing transition features are unchanged.

A positive target requires its 300 ms core to lie inside an interval, away from uncertain boundaries. A negative target from a positive recording requires its surrounding physical context to avoid all annotated events. An unannotated positive contributes no negative event targets. Labels are not used as features; absolute position fractions are excluded. Clip weights prevent extra windows in a long recording from increasing its total initial training weight, followed by class balancing. Weighted scaling and regularized logistic regression use training windows only; the hybrid additionally fits PCA-8 on training embeddings only.

Evaluation excludes whole source threads, exact-audio duplicates, and the known shared-audio family from the **earlier whole-recording** fingerprint evaluation. It does not use the new, unreliable short-reference matches as family labels. Each test recording is counted once. Clip score is the maximum of all its automatic event scores.

| Model                   | Detected at 0.8 | Red false alerts / 1,181 negatives | Red false alerts / 1,000 | Yellow-only negatives |
| ----------------------- | --------------: | ---------------------------------: | -----------------------: | --------------------: |
| Event physical          |           10/10 |                                 20 |                    16.93 |                    16 |
| Event physical + YAMNet |           10/10 |                                 17 |                    14.39 |                    19 |

At a separate threshold above each fold's highest training-negative clip score, physical detects 8/10 with two false alerts; hybrid detects 4/10 with five. This secondary diagnostic is **not** mapped to 0.8 or used for browser warnings. It underscores that the models' score scales are not calibrated. Selecting the hybrid at the requested 0.8 criterion does not establish that it is generally better, and neither model is ready for browser deployment.

```sh
.venv/bin/python research/event_dataset.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python research/train_events.py
.venv/bin/python research/train_events.py \
  --score research/artifacts/event-models-v1/model.json
```

Run the original feature and frozen-context extractors first after adding new audio. Rebuilding the event dataset after later timings reuses cached physical features and embeddings. The scorer writes a new timestamped snapshot containing the model hash, dataset hash, score, tier, selected event time, labels at scoring, and training membership. It does not retrain anything.

Artifacts are in `research/artifacts/event-data-v1/` and `research/artifacts/event-models-v1/`. Both individual model exports, every held-out prediction, and fold-specific models are retained. JSON inference parity is tested. The original tracked shadow models are unchanged.

### Controlled robustness

Held-out fold models were tested on ten positives and fourteen explicitly reviewed hard negatives. Proposals and features were recomputed on each full transformed recording. Stereo AAC roundtrips preserve channel layout rather than interpreting interleaved stereo as mono.

| Transformation | Physical: positive alerts / 10 | Hybrid: positive alerts / 10 | Physical: hard-negative red alerts / 14 | Hybrid: hard-negative red alerts / 14 |
| -------------- | -----------------------------: | ---------------------------: | --------------------------------------: | ------------------------------------: |
| Unchanged      |                             10 |                           10 |                                       7 |                                     7 |
| Prepend 25 ms  |                              9 |                            9 |                                       7 |                                     6 |
| Gain −3 dB     |                              7 |                            8 |                                       3 |                                     4 |
| Gain +3 dB     |                             10 |                           10 |                                       9 |                                     9 |
| AAC 96 kb/s    |                              7 |                            8 |                                       5 |                                     5 |

These variants are not independent examples. Hard negatives are intentionally difficult and are not a population sample. Gain changes may also change perceived severity. Parameters were not tuned on these variants. Timing/encoding instability and false alerts remain reasons to keep both models in shadow mode.

```sh
TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv-audio/bin/python research/stress_events.py
```

## Annotated fingerprints: failed specificity test

The reference database was rebuilt from all twelve annotated intervals, without changing matcher thresholds. To preserve the historical evaluation and reuse its landmark cache:

```sh
.venv/bin/python research/fingerprint_audio.py evaluate \
  --output-dir research/artifacts/fingerprints-annotated-v1 \
  --cache-dir research/artifacts/fingerprints/cache
```

Controlled reuse still recovers 10/10 recordings for unchanged audio, a 25 ms prefix, −3 dB gain, and AAC 96 kb/s. However, with each source excluded, **314/1,181 negatives match an annotated reference**. The 0.72 s reference from `17885487431220276835.mp4` accounts for matches in 291 negatives. Five positives also match other positive references; given the negative collision rate, those matches cannot establish genuine reuse.

These are matcher outputs, not confirmed new screamers. The legacy field `confirmed_event_match` means the **reference interval** was annotated by the user, not that the query has been verified. Short references expose insufficient specificity in the current landmark acceptance rules. Neither reference restriction alone nor a gain-insensitive match is safe enough to drive warnings. The fresh index and results stay in their separate artifact directory; the old whole-recording results are preserved.

## Subsequent collection

The user supplies a reviewed thread and screamer filenames; the builder downloads its attachments and labels the remaining snapshot attachments negative. Timings can arrive later. See [the labeling commands](ML_PLAYGROUND.md#1-extend-the-corpus).

For a new thread, score with the previously frozen models and save the timestamped snapshot **before retraining**. Because labels arrive during import, call these future-thread frozen-model results, not a blinded study. Do not select thresholds on those same predictions and then present them as independent validation. Fresh naturally collected threads remain essential; more precise relabeling of these ten files is not the immediate bottleneck.

## Verification and methodological references

```sh
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
.venv/bin/ruff check research
.venv/bin/ruff format --check research
node --test research/test_userscript.cjs
node --check spokoyno.user.js
```

The use of grouped splits follows [scikit-learn's grouped evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation). Training uses explicit [logistic-regression sample weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html); class-balanced outputs must not be read as the natural screamer prevalence.
