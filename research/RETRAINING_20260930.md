# Retraining and new algorithms — 2026-09-30

**Research only: ten versioned shadow candidates fitted; no browser promotion, threshold retuning, label edits, or replacement of historical models.** The previous frozen-model evidence remains in [thread 336991612](reviews/336991612.md).

## Outcome

The existing annotated-event models detect all 14 held-out positive recordings at the fixed 0.8 red threshold, with 51 (physical) or 53 (hybrid) false alerts among 3,074 negatives. They have the strongest recall in this comparison, not sufficiently low false-alert rates to silently replace production. The experimental weak-label event model detects 13/14 with 47 false alerts.

New tree algorithms do not provide a clear win: shallow histogram gradient boosting detects 9/14 with 17 false alerts; Extra Trees detects 5/14 with 3. Rich logistic reduces false alerts to 6 but detects only 7/14. Retraining often produces excellent **training** results without comparable held-out performance.

## Data and evaluation protocol

- 14 positive and 3,074 negative exact-audio groups; 519 unlabeled/ambiguous and the visual-only group are excluded from binary targets. No-audio attachments are not successful negative classifications.
- Nine source-thread folds, including negative-only threads. Every labeled audio group appears in the aggregate test results exactly once, assigned to its earliest source thread. All copies from the held-out thread are excluded from training, even when their scoring ownership belongs to an earlier fold.
- The existing whole-recording shared-audio links are excluded transitively. Unreliable short annotated-fingerprint matches are not used. Undetected transformed/related audio can still leak, so this is not guaranteed family-independent validation.
- All candidates use the same grouped partitions. Imputation, scaling, PCA, classifier fitting, and latent-event selection see training rows only. Histogram boosting explicitly disables its automatic random early-stopping validation split.
- Ten fixed configurations were declared before fitting in the saved run plan; no grid search or threshold selection used the held-out scores. Rich models use 151 current physical features (the older export had 150); compact models use 48.
- Report both the existing uncalibrated 0.6/0.8 display policy and the historical style of a cutoff strictly above the maximum fitted-training negative score. The latter guarantees zero training false positives only, not zero deployment errors.
- Source-held-out predictions are model-development diagnostics: feature design has seen this corpus, model comparison/selection uses these results, and there is no outer nested model-selection test or untouched prospective test. Fourteen positives cannot establish broad reliability.
- Final models are additionally fitted on all eligible labeled data, with scores explicitly identified as final-fit diagnostics. They are never substituted for held-out results.
- All seven earlier saved variants are represented: the numerically equivalent compact logistic shadow/whole-clip physical formulations are consolidated into one candidate; forest, embedding, hybrid, and both event families are retained. This gives six existing configurations plus four new ones.

## Held-out results

All rows below use identical source/family exclusions and denominators. Yellow-only positives are not counted as red detections. The final two columns use a separate fold-specific training-negative-max cutoff, not the 0.8 policy.

| Configuration                              | Red positives / 14 | Red false alerts / 3,074 | Yellow-only negatives | Training-max cutoff: positives / 14 | Training-max cutoff: false alerts |
| ------------------------------------------ | -----------------: | -----------------------: | --------------------: | ----------------------------------: | --------------------------------: |
| Compact logistic / whole-clip physical     |                  8 |                       16 |                    20 |                                   3 |                                 1 |
| Random forest, depth 3                     |                  0 |                        1 |                     1 |                                   4 |                                 2 |
| YAMNet embeddings + logistic               |                  1 |                      196 |                   426 |                                   0 |                                 1 |
| Physical + YAMNet logistic                 |                  8 |                       15 |                    20 |                                   4 |                                 1 |
| Annotated-event physical                   |                 14 |                       51 |                    44 |                                   6 |                                 2 |
| Annotated-event hybrid                     |                 14 |                       53 |                    34 |                                   6 |                                 1 |
| Extra Trees, depth 3 (new)                 |                  5 |                        3 |                     6 |                                   2 |                                 1 |
| Histogram gradient boosting, depth 2 (new) |                  9 |                       17 |                     4 |                                   1 |                                 1 |
| Rich logistic (new feature configuration)  |                  7 |                        6 |                    11 |                                   5 |                                 2 |
| Weak-label event MIL (new)                 |                 13 |                       47 |                    43 |                                   7 |                                 3 |

The refitted forest’s fixed-0.8 held-out result is **0/14**, despite 11/14 positives and zero red false alerts on its final training fit. At fold-specific training-max cutoffs it detects 4/14 with two false alerts. This illustrates why a low false-positive count or a good all-data fit must not be treated as reliable recall. Score scales can change after retraining; the 0.8 UI convention is not model calibration.

## What learned from the new labels?

Ten original positive recordings have timings; four distinct positive groups do not. Clip-level models can use all 14 labels immediately. Original annotated-event models train on 10 timed positive recordings plus all 3,074 negatives: the four untimed positives are **not** listed as actually used training clips. They still receive predictions.

The new `event-mil-physical` candidate starts from the annotated-event physical fit. It then performs three fixed updates, selecting the currently highest-scoring automatic proposal in each **training-only untimed positive clip** as a latent positive. Other windows in those clips remain unknown, not negative. Each clip is weighted before event-class balancing. This assumes at least one proposal contains the event; that assumption may fail. The four final latent selections are recorded as model choices, never written to `corpus/events.json` or presented as human timing annotations.

On the final fit, the new label information raises compact-logistic scores for `17907067660803027078.mp4` and `17907090692410007259.webm` to about 0.9975 and 0.9892. But with their entire thread excluded, their scores are only 0.7179 and 0.2662. The model has fitted those examples; this is not evidence it will detect the next new screamer.

### Latest thread entirely excluded from training

This separate view evaluates all four positives and 285 negatives in thread 336991612 with the entire thread and its copies excluded. It includes groups owned by earlier folds, so these numbers **must not be added** to aggregate held-out totals. The single ambiguous clip is omitted from accuracy.

| Configuration                              | Red positives / 4 | Red false alerts / 285 |
| ------------------------------------------ | ----------------: | ---------------------: |
| Compact logistic / whole-clip physical     |                 2 |                      3 |
| Random forest, depth 3                     |                 1 |                      0 |
| YAMNet embeddings + logistic               |                 0 |                     20 |
| Physical + YAMNet logistic                 |                 2 |                      4 |
| Annotated-event physical                   |                 4 |                     11 |
| Annotated-event hybrid                     |                 4 |                     11 |
| Extra Trees, depth 3 (new)                 |                 1 |                      1 |
| Histogram gradient boosting, depth 2 (new) |                 3 |                      7 |
| Rich logistic (new feature configuration)  |                 1 |                      0 |
| Weak-label event MIL (new)                 |                 4 |                      9 |

For context, the unchanged handcrafted 5.8.2 retained-WAV replay gives 2/4 red positives, two yellow positives, and 3/285 red false alerts on this thread. This is a frozen-rule reference, not a newly trained fold model or independent proof of generalization. Its parameters and feature design have prior development exposure.

The two current browser yellow-only screamers are red for both annotated-event models and the weak-label event model even with the whole latest thread excluded. The weak-label event model scores `17907090692410007259.webm` about 0.9810 and `17907126689410376090.webm` about 0.9907 in that held-out-thread view, but incurs nine false alerts instead of the handcrafted detector’s three. All ML comparisons here use retained FFmpeg audio; no new ML browser decoding/feature-parity claim is made.

### Final fits: not validation

| Configuration                              | Red positives / 14 | Red false alerts / 3,074 | Actual training clips |
| ------------------------------------------ | -----------------: | -----------------------: | --------------------: |
| Compact logistic / whole-clip physical     |                 14 |                       12 |                  3088 |
| Random forest, depth 3                     |                 11 |                        0 |                  3088 |
| YAMNet embeddings + logistic               |                  3 |                      212 |                  3088 |
| Physical + YAMNet logistic                 |                 14 |                       11 |                  3088 |
| Annotated-event physical                   |                 14 |                       50 |                  3084 |
| Annotated-event hybrid                     |                 14 |                       47 |                  3084 |
| Extra Trees, depth 3 (new)                 |                 14 |                        3 |                  3088 |
| Histogram gradient boosting, depth 2 (new) |                 14 |                        9 |                  3088 |
| Rich logistic (new feature configuration)  |                 14 |                        2 |                  3088 |
| Weak-label event MIL (new)                 |                 14 |                       44 |                  3088 |

## Fixed configuration details

- Compact/rich logistic: median imputation, standard scaling, balanced logistic regression, C=0.05, liblinear; 48 or 151 features.
- Random forest: 400 trees, depth 3, minimum leaf size 3, 35% feature subsampling, balanced-subsample weights.
- Extra Trees: 400 trees, depth 3, minimum leaf size 3, 35% feature subsampling, balanced weights.
- Histogram gradient boosting: depth 2 / at most 4 leaves, 100 iterations, learning rate 0.05, minimum leaf size 10, L2=10, balanced weights, early stopping disabled.
- Existing whole-clip embedding/hybrid and event families reuse the original frozen YAMNet context and fold-local PCA-8/logistic implementations. No encoder fine-tuning.
- Tabular seed: 20260930. Existing PCA/context/event functions retain their fixed seed 42. These are single-seed experiments, not seed-robustness estimates.
- Weak-label event MIL uses physical features only and three latent-selection updates after its supervised initialization; no manual boundary or test-time label is used.

## Saved models, scoring, and reproducibility

All ten final fitted models and their manifests are tracked under `research/models/retrained-20260930/`, alongside `results-summary.json` with aggregate/fold metrics and every positive held-out prediction. They are new versioned research snapshots, not replacements for `shadow-model.json`, `challenger-model.json`, or the earlier context/event artifacts.

The complete plan, all 90 fold fits, full held-out predictions, and final-fit predictions are retained locally in ignored `research/artifacts/retrain-20260930/`. Source/feature/label/event/encoder/model hashes and training membership are recorded. Output directories refuse overwrites.

Joblib files are executable pickle: **load only trusted artifacts**. The scorer requires `--trust-models`, checks model hashes and the recorded scikit-learn version, and aligns physical columns to the fitted feature schema. A checksum does not make an untrusted pickle safe. The saved environment uses scikit-learn 1.9.0; exact versions are recorded in manifests.

```sh
# Fresh retraining run; rebuild feature/context/event datasets first if labels change.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python research/retrain_models.py \
  --output-dir research/artifacts/retrain-FRESH --jobs 2

# Export trusted final fits as a separate, immutable research snapshot.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python research/score_retrained.py publish \
  --run research/artifacts/retrain-FRESH \
  --output research/models/retrained-FRESH --trust-models

# Score future reviewed data without refitting; choose a fresh output filename.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python research/score_retrained.py score \
  --run research/models/retrained-20260930 \
  --output research/artifacts/retrained-scores-FRESH.json --trust-models
```

Every saved fit was reloaded and its scores compared to the in-memory estimator. Published final fits are also rescored and compared with the original run. Historical model artifacts, the userscript, labels, and human timing annotations remain unchanged.

Validation passed: 44 Python tests, 19 JavaScript tests, Ruff, formatting,
userscript syntax, and diff checks. The new tests cover duplicate/family
exclusion, negative-only folds, training-only latent labels, feature-column
alignment, unknown-label metric exclusion, empty feature columns, explicit
pickle trust, and refusal to overwrite an existing run.

| Evidence                 | SHA-256                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Training runner          | `8e8f86ea6d75f3bee353eca2a45debd890c2218f251ed56096dcf6083b3f98e7` |
| Full local results       | `f25b14053cef8ce4a2606eb0c26b87a51682337766d21e12fc5edd82f14b5e68` |
| Local pre-fit plan       | `0dcb02c2ba1934f19f450aef0aaaa62951a59b6ae3397cd050f0db305887d9fd` |
| Published result summary | `2ca630d04a25dfbcd9ee874e330e4bbb3c3a02f5e97b10a15685c0d6dbf91066` |

## Next decision

Keep production unchanged. The event-based candidates are worth further prospective testing because they detect the two currently yellow-only examples, but their false-alert cost is substantial. Exact timings for the four untimed positive groups would let the supervised event models learn from them directly and avoid latent-window assumptions. Continue collecting fully reviewed threads, freeze these predictions before subsequent fitting, and evaluate browser/codec robustness before any production port.

Methodological references: [scikit-learn leakage precautions](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), and [histogram gradient boosting parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
