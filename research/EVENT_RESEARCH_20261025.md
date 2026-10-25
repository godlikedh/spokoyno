# Event model research and rules plus MIL comparison

**The strongest lead is a combination of the existing rules and a model trained from recording labels. It remains an exploratory shadow candidate.** Its corpus replay gives **29 red and one yellow among 30 positives**, with **30 false red alerts and 81 yellow negatives**. Event-hybrid gives 29 red, one below yellow, 58 false red alerts, and 42 yellow negatives. The combination reduces red false alerts by 48%, but increases total negative warnings from 100 to 111.

This is not an independently validated improvement. The ML component is scored with its source thread excluded; the handcrafted rules were developed using part of this corpus, and the combination was investigated after the individual models. The quieter screamer changes from red to yellow, while a short burst changes from missed to red. Freeze the combination for future threads; **production stays at 5.8.2**.

The individual ML experiments did not establish a replacement. Thirteen configurations were tested across 67 distinct training partitions, producing **871 saved fits**. Nested model and threshold selection detected 26/30 at red with 46 false red alerts, losing three detections relative to event-hybrid. The only individual alternative retaining 29/30 red detections reduced false alerts by just one and lost a detection on browser-decoded audio.

## Data and evaluation

The unchanged corpus has **30 positive and 3,466 negative exact-audio groups**, plus 518 unknown groups excluded from binary targets. There are ten timed positive groups and twenty untimed positives. Ambiguous cases remain unknown; no model proposal becomes a human annotation.

Event-hybrid uses 43 physical event features and eight principal components of frozen YAMNet context embeddings. Its training windows come from ten timed positives and the negatives. All candidates retain the existing label-independent proposals: up to three transitions and the opening. The experiments assess recording detection, not verified onset localization for the twenty untimed positives.

The eleven outer folds exclude the entire source thread, every duplicate source membership, and transitive known audio families. Each labeled audio hash is counted once, under its earliest thread. Inner validation intersects those exclusions with a second held-out thread. There are 110 inner validation folds; symmetric training sets share cached fits. Imputation, scaling, PCA, latent-window choices, and model fitting use training rows only.

For each outer fold, the inner selector must preserve **every** positive caught by event-hybrid at 0.8 and achieve at least 20% fewer false red alerts. It then minimizes false alerts, prefers more detections, and uses declaration order and proximity to 0.8 to break ties. Otherwise it retains the baseline. Thresholds are restricted to 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, and 0.99. A separate baseline-only threshold control isolates gains from threshold adjustment. Selected thresholds are mapped monotonically to the usual 0.8 display boundary; this is not probability calibration.

Round one declared eight configurations before fitting. Round two added five after observing round-one results, and is explicitly adaptive. Nested validation separates fitting and selection from the outer fold, but cannot remove prior feature-design exposure or this adaptive research history. This distinction follows the purpose of [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Known family links are the same frozen snapshot used in the previous study. An additional screen found no cross-recording event-context pairs involving positives with cosine similarity at least 0.995. This narrow check does not rule out transformed copies or shared sound families. Fourteen of the thirty positive hashes belong to the newest thread; they are not fourteen independent deployment environments.

## Methods tested

| Configuration      | Change from event-hybrid                                                                                                                           |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| event-hybrid       | Exact existing implementation, including weighted preprocessing and logistic C=0.05.                                                               |
| strong-ridge       | Logistic C=0.01 with the same training-only preprocessing.                                                                                         |
| semantic-contrast  | PCA over event embedding, event minus preceding embedding, and event minus following embedding.                                                    |
| hard-negatives     | Three training-only reweighting rounds; difficult negative windows receive up to eight times the initial relative weight, preserving class totals. |
| mil-hybrid         | Three latent updates select one proposal from each untimed positive training clip; other unknown windows stay unknown.                             |
| mil-linear-softmax | Optimize recording-level loss over all labeled training clips, plus one-quarter weighted human-window loss and L2 regularization.                  |
| additive-spline    | Quadratic splines with three uniform knots for physical features, audio PCA, and regularized logistic regression.                                  |
| rbf-event          | RBF SVM, C=1, gamma=1/51; sigmoid of its margin is an uncalibrated score.                                                                          |
| semantic-taxonomy  | Add 22 fixed pretrained sound categories and their before/after differences; logistic C=0.01.                                                      |
| semantic-mil       | Add three latent-window updates to semantic-taxonomy.                                                                                              |
| semantic-boost     | Depth-two histogram boosting over physical, PCA, and taxonomy features.                                                                            |
| clip-gated         | Multiply event-hybrid by the fourth root of a training-only whole-clip rich-logistic score.                                                        |
| event-boost        | Depth-two histogram boosting over physical and PCA features.                                                                                       |

Both boosting variants use 100 iterations, learning rate 0.05, minimum leaf size 20, and L2=10. All constants and software versions are preserved in the runners and pre-fit plans. The frozen encoder is never fine-tuned.

Multiple-instance learning (MIL) treats a positive recording as containing at least one relevant event, without requiring a timestamp. Linear-softmax pooling is `sum(p²) / sum(p)` across proposals. The [sound-event pooling study by Wang, Li, and Metze](https://arxiv.org/html/1810.09050v3) motivated testing it alongside max pooling; that paper's results do not establish that it works for this corpus. Here a numerical finite-difference check verifies the implemented objective's gradient, and all 67 optimizations converged.

The semantic round uses [YAMNet's pretrained AudioSet classifier](https://www.tensorflow.org/hub/tutorials/yamnet), including speech, screaming, moaning, singing, music, noise, and distortion. Categories were declared from the task and taxonomy rather than selected by corpus correlations. The exported head matches the official network to a maximum absolute error of 7.45e-8 on the parity fixture. Its 22 categories produce 66 event/context features. [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) supplies the separate smooth nonlinear alternative.

## Individual model results

All rows use the same outer thread exclusions and **30 positives / 3,466 negatives**. Yellow columns exclude red. These are development comparisons, not future-thread validation. Event-hybrid reproduces the previous run's predictions exactly: maximum absolute score difference **0.0**.

| Model              | Red positives | Yellow positives | False red alerts | Yellow negatives |
| ------------------ | ------------- | ---------------- | ---------------- | ---------------- |
| event-hybrid       | 29            | 0                | 58               | 42               |
| strong-ridge       | 27            | 2                | 63               | 67               |
| semantic-contrast  | 29            | 0                | 57               | 48               |
| hard-negatives     | 26            | 3                | 31               | 37               |
| mil-hybrid         | 27            | 2                | 59               | 50               |
| mil-linear-softmax | 25            | 4                | 23               | 54               |
| additive-spline    | 27            | 2                | 41               | 35               |
| rbf-event          | 6             | 10               | 0                | 6                |
| semantic-taxonomy  | 25            | 4                | 53               | 65               |
| semantic-mil       | 23            | 6                | 56               | 72               |
| semantic-boost     | 27            | 1                | 58               | 10               |
| clip-gated         | 21            | 6                | 7                | 26               |
| event-boost        | 28            | 0                | 52               | 14               |

Zero false alerts for the RBF model comes with only six red detections. Likewise, the clip gate suppresses useful positive evidence along with false alerts. Neither is a quality improvement at these boundaries.

The semantic-contrast improvement is only one negative. A descriptive bootstrap resampling the eleven threads gives a 95% interval of **−2.47 to +2.34 false alerts per 1,000 negatives** for its difference from baseline. That interval includes a regression; it also does not account for adaptive research or undiscovered families.

## Nested selection results

| Selection procedure                       | Red positives | Yellow positives | False red alerts | Yellow negatives |
| ----------------------------------------- | ------------- | ---------------- | ---------------- | ---------------- |
| Fixed event-hybrid                        | 29            | 0                | 58               | 42               |
| Baseline-only threshold selection         | 29            | 0                | 55               | 39               |
| Round-one model and threshold selection   | 25            | 4                | 46               | 45               |
| Both rounds model and threshold selection | 26            | 3                | 46               | 45               |

The selector's inner recall constraint did not generalize reliably to outer threads. Its final choice using all development out-of-fold predictions is still **event-hybrid at 0.8**. Lower false-alert counts do not justify presenting the selected alternatives as better detectors.

For the combined-round selector, the per-thread red counts are:

| Thread    | Positives / negatives | Baseline detected / false red | Selected detected / false red |
| --------- | --------------------- | ----------------------------- | ----------------------------- |
| 336185346 | 2 / 535               | 2 / 8                         | 2 / 6                         |
| 336243339 | 2 / 0                 | 2 / 0                         | 2 / 0                         |
| 336272252 | 2 / 333               | 2 / 1                         | 2 / 1                         |
| 336291305 | 4 / 313               | 4 / 2                         | 3 / 1                         |
| 336466927 | 1 / 339               | 1 / 5                         | 1 / 5                         |
| 336603044 | 0 / 412               | 0 / 2                         | 0 / 2                         |
| 336679639 | 0 / 260               | 0 / 10                        | 0 / 9                         |
| 336723954 | 1 / 682               | 1 / 12                        | 1 / 12                        |
| 336991612 | 2 / 212               | 2 / 8                         | 1 / 7                         |
| 337355133 | 2 / 108               | 2 / 3                         | 2 / 3                         |
| 337690329 | 14 / 272              | 13 / 7                        | 12 / 0                        |

The last row excludes content groups owned by earlier threads. It is not the separate 295-reviewed-negative full-thread view in the [earlier report](RETRAINING_20261025.md).

## Exploratory combination of rules and ML

Error analysis exposed complementary behavior. For `17929502589060439332.mp4`, event-hybrid proposes **21.4 seconds** but scores it **0.00551**, despite a roughly 46 dB jump and a 0.35-second event. Its standardized 0.6-second and 1.0-second level features contribute about −2.78 and −2.01 to the logit: quiet audio after the burst strongly suppresses the result. This is a duration-handling problem in this example, not a missing proposal. The current rules give this clip a red alert.

That motivated one additional fixed operation: **`score = max(current_rule_score, model_score)`**, preserving 0.6/0.8 boundaries. No fusion thresholds or coefficients were fitted. All thirteen combinations were evaluated, rather than reporting only the favorable one:

| ML component combined with fixed rules | Red positives | Yellow positives | False red alerts | Yellow negatives |
| -------------------------------------- | ------------- | ---------------- | ---------------- | ---------------- |
| event-hybrid                           | 30            | 0                | 61               | 70               |
| strong-ridge                           | 30            | 0                | 67               | 91               |
| semantic-contrast                      | 30            | 0                | 60               | 74               |
| hard-negatives                         | 28            | 2                | 36               | 71               |
| mil-hybrid                             | 30            | 0                | 63               | 73               |
| **mil-linear-softmax**                 | **29**        | **1**            | **30**           | **81**           |
| additive-spline                        | 29            | 1                | 45               | 69               |
| rbf-event                              | 25            | 4                | 13               | 58               |
| semantic-taxonomy                      | 29            | 1                | 57               | 93               |
| semantic-mil                           | 29            | 1                | 60               | 96               |
| semantic-boost                         | 29            | 0                | 60               | 50               |
| clip-gated                             | 27            | 2                | 18               | 67               |
| event-boost                            | 29            | 0                | 54               | 55               |

These rows have **held-out ML components and fixed development rules**, not fully held-out ensembles. The rules-plus-linear-softmax candidate keeps the same total red recall as event-hybrid, but changes which positive is yellow: the quieter `17929314751662663260.webm` receives approximately **0.721**, versus event-hybrid's **0.833**. The short-burst recording above moves from below yellow to red. All thirty receive at least yellow in this replay.

There are two costs to make explicit. First, negative yellow warnings increase substantially; this is partly a redistribution of warning severity. Second, compared with the **current production rules themselves**, which replay at 25 red positives and 13 false red alerts, the combination increases red detections to 29 and false red alerts to 30. Event-hybrid was the requested quality benchmark; it is not the current production detector.

The saved [fusion policy](models/rules-mil-20261025/policy.json) references the exact frozen ML snapshot and userscript checksum. It is a candidate for prospective comparison, not a production recommendation based on proven generalization.

## Browser decoding check

All 27 saved original-media browser fixtures were verified by media and PCM hashes and re-extracted through the same feature/encoder pipeline. They contain fourteen positives, nine explicitly named hard negatives, and four unknown cases. Models exclude thread 337690329 entirely. This is Python ML inference on browser-decoded PCM, not a JavaScript ML implementation.

| Detector                      | Red / yellow positives out of 14 | False red / yellow among 9 hard negatives |
| ----------------------------- | -------------------------------- | ----------------------------------------- |
| Current production rules      | 10 / 3                           | 5 / 4                                     |
| Event-hybrid                  | 13 / 0                           | 6 / 1                                     |
| Semantic contrast             | 12 / 1                           | 6 / 1                                     |
| Linear-softmax MIL            | 12 / 1                           | 0 / 6                                     |
| Rules plus linear-softmax MIL | 13 / 1                           | 5 / 4                                     |

The semantic-contrast candidate loses a red detection under browser decoding, despite retaining all 29 baseline red detections in the full retained-WAV comparison. The fusion has useful complementary coverage here, but the nine hand-picked negatives cannot estimate a production false-positive rate. Full per-file scores, proposal times, and decoder changes are in the [decoder audit](models/event-research-20261025/decoder-audit.json) and [fusion evaluation](models/rules-mil-20261025/evaluation-summary.json).

## Recommendation and saved evidence

Keep production unchanged and score future reviewed threads with the frozen rules-plus-MIL candidate **before retraining**. Include event-hybrid as the reference and report red, yellow, and low positives separately. Do not interpret any score as a calibrated probability or estimate deployment precision from the enriched positive corpus.

The next feature-level research target is duration handling: a brief loud event should not be suppressed simply because a longer averaging window includes subsequent quiet audio. More human event timings would distinguish correctly located screamers from unrelated loud passages and allow better duration supervision. The existing workflow remains intact: thread-level positive filenames first, safe labels only inside the reviewed snapshot, exact timings later.

Tracked artifacts:

- [Thirteen frozen final models and provenance](models/event-research-20261025/manifest.json), with the executable joblib payload beside it.
- [Both rounds, per-thread metrics, held-out positives and false alerts](models/event-research-20261025/results-summary.json).
- [Decoder and baseline audit](models/event-research-20261025/decoder-audit.json).
- [Frozen exploratory fusion policy](models/rules-mil-20261025/policy.json) and [all thirteen fusion comparisons](models/rules-mil-20261025/evaluation-summary.json).

The ignored full runs are `research/artifacts/improve-events-20261025/` and `research/artifacts/improve-semantics-20261025/`. Every saved fit was reloaded and compared with its original scores within 1e-12. Published models were also reloaded and scored across all **4,014** non-visual audio groups. Final training-fit scores remain diagnostics, not validation. Original models, labels, event timings, and `spokoyno.user.js` are unchanged.

## Reproduction

Use the existing corpus snapshot and feature/context/event caches. Training uses `.venv` with the recorded scikit-learn 1.9.0; encoder extraction uses `.venv-audio`. Limit BLAS/OpenMP threads when running two training jobs. Every command refuses to overwrite its output snapshot.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/improve_events.py \
  --output-dir research/artifacts/event-study-rerun --jobs 2

.venv-audio/bin/python research/export_yamnet_head.py \
  --output-dir research/artifacts/event-study-head-rerun

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/improve_event_semantics.py \
  --source research/artifacts/event-study-rerun \
  --head research/artifacts/event-study-head-rerun \
  --output-dir research/artifacts/event-semantics-rerun --jobs 2 --trust-models
```

For future frozen scoring, first rebuild corpus feature/context/event caches through the existing workflow, then generate fresh retained-WAV rule scores and ML scores:

```sh
node research/evaluate_userscript.cjs \
  --baseline 5f32be88c3dfe0ee0601252c34cb68562b2ec79d \
  --output research/artifacts/future-rule-scores.json --workers 4

OPENBLAS_NUM_THREADS=2 .venv/bin/python research/event_research_artifacts.py score \
  --snapshot research/models/event-research-20261025 \
  --output research/artifacts/future-event-scores.json --trust-models

OPENBLAS_NUM_THREADS=2 .venv/bin/python research/score_event_fusion.py score \
  --model-scores research/artifacts/future-event-scores.json \
  --heuristic-scores research/artifacts/future-rule-scores.json \
  --output research/artifacts/future-fusion-scores.json
```

The fusion scorer rejects a changed userscript reference, different corpus snapshots, missing components, failed analyses, and invalid scores. Model snapshots use executable pickle; `--trust-models` is required and should only be used for trusted artifacts. The frozen fusion intentionally refuses results from a later userscript version, so a new rule version requires a separately versioned policy.

Validation passed: **57 Python tests and 19 JavaScript tests**, Ruff checks and formatting, userscript syntax, Markdown formatting, and Git whitespace checks. The research tests cover the MIL gradient, grouped selection boundaries, preservation of each inner baseline detection, missing semantic context, serialization checks, explicit pickle trust, and fusion failure behavior. Both thirteen-row result tables were checked against their saved metrics.
