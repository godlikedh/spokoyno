# Retraining — 2026-10-25

**Ten new versioned shadow models were trained and published; production remains 5.8.2.** No original or September model weights, detector thresholds, or human event annotations were replaced. Frozen pre-training predictions and browser results are in [thread 337690329](reviews/337690329.md).

## Outcome

There is still no clear production replacement. The event-hybrid model has the strongest grouped held-out red recall, **29/30**, but **58 false alerts among 3,466 negatives**. Rich logistic reduces that to **11 false alerts**, while detecting only **19/30**. The forest detects **6/30** at the fixed 0.8 boundary, despite **25/30** on its final training fit. A low false-alert count alone is not success.

The new thread is valuable prospective evidence for the previously frozen snapshots: the September forest gives **0/14 red alerts**, versus 13/14 for both September annotated-event models with seven false alerts among 295 reviewed negatives. Favorable results on the prior two-positive thread did not generalize to this batch.

## Data and evaluation protocol

- **30 positive and 3,466 negative exact-audio groups**; 518 unlabeled groups and the one visual-only group are not binary targets. There are 35 positive attachment paths, not 35 independent recordings.
- The newest thread contributes 14 new positive hashes. Only videos reviewed through **2026-10-25 21:00:55 Moscow time** receive inferred negative labels. The long unwatched clip, later paths, and the three borderline cases remain unknown. One ambiguous content group is quarantined across three older negative aliases; historical review metadata is preserved.
- **Ten timed positive groups, twenty untimed.** Clip models and experimental weak-label MIL can use all 30. Supervised event physical/hybrid models actually train on 10 timed positive groups plus 3,466 negatives (3,476 clips), not all 3,496 eligible clips. They can still score untimed positives.
- Eleven source-thread folds, including negative-only folds; all exact copies from the held-out thread are excluded from fitting. Each labeled content group appears exactly once in pooled held-out predictions, assigned to its earliest source thread.
- Previously established whole-audio family links are excluded transitively, using the same frozen identity-link snapshot as September. A full expanded whole-recording identity scan was attempted in an isolated artifact directory but stopped due excessive runtime/memory; no partial results were used. **New transformed/related copies may still overlap folds.** These results are not guaranteed family-independent validation.
- Ten fixed configurations and the existing seed **20260930** were retained; no grid search or threshold tuning against this thread. Feature extraction, the pinned YAMNet encoder, and candidate generation are unchanged. Preprocessing and weak-label event selection are fitted on training folds only.
- Shared display boundaries remain yellow >=0.6 and red >=0.8. Saved training-negative-max cutoffs are a separate diagnostic policy, not calibrated probabilities or a zero-false-positive guarantee.
- All ML and whole-corpus scripted comparisons use FFmpeg-retained WAVs. The named browser tests show decoding can change tiers; no ML browser-parity or production-port claim is made.
- Source-held-out comparisons are development diagnostics: feature/model design has seen the corpus, the comparison is used for model selection, and there is no untouched outer model-selection test. Final-fit results are never substituted for held-out validation.

## Grouped held-out results

Every row below uses **30 positives and 3,466 negatives**. Yellow columns exclude red. The two final columns use each fold's separately saved training-negative-max cutoff (strictly greater than), not the 0.8 policy.

| Configuration             | Red positives / 30 | Yellow positives | Red false alerts / 3466 | Yellow negatives | Train-max detected | Train-max false alerts |
| ------------------------- | ------------------ | ---------------- | ----------------------- | ---------------- | ------------------ | ---------------------- |
| logistic-compact          | 18                 | 5                | 19                      | 32               | 13                 | 1                      |
| forest-rich-depth3        | 6                  | 2                | 3                       | 9                | 6                  | 2                      |
| clip-embedding            | 5                  | 12               | 208                     | 484              | 0                  | 1                      |
| clip-hybrid               | 19                 | 3                | 23                      | 23               | 13                 | 2                      |
| event-physical            | 28                 | 1                | 57                      | 53               | 16                 | 2                      |
| event-hybrid              | 29                 | 0                | 58                      | 42               | 16                 | 1                      |
| extra-trees-rich-depth3   | 8                  | 7                | 4                       | 18               | 6                  | 1                      |
| hist-gradient-rich-depth2 | 14                 | 1                | 18                      | 9                | 7                  | 4                      |
| logistic-rich             | 19                 | 1                | 11                      | 8                | 16                 | 3                      |
| event-mil-physical        | 26                 | 3                | 59                      | 48               | 14                 | 1                      |

## Latest reviewed thread, completely excluded from fitting

This separate view includes the 14 positives and 295 reviewed negative audio groups from thread 337690329, including copies owned by earlier folds. It overlaps pooled held-out rows and must not be added to those totals.

The runner's full-source-thread artifact additionally contains one after-cutoff attachment whose audio is already an earlier known negative (296 negatives); the table below filters to the explicit reviewed snapshot. The excluded extra negative is not a new human review.

| Configuration             | Red positives / 14 | Yellow positives | Red false alerts / 295 | Yellow negatives |
| ------------------------- | ------------------ | ---------------- | ---------------------- | ---------------- |
| logistic-compact          | 10                 | 1                | 1                      | 4                |
| forest-rich-depth3        | 0                  | 0                | 0                      | 0                |
| clip-embedding            | 3                  | 7                | 24                     | 55               |
| clip-hybrid               | 11                 | 1                | 1                      | 4                |
| event-physical            | 12                 | 1                | 6                      | 7                |
| event-hybrid              | 13                 | 0                | 7                      | 6                |
| extra-trees-rich-depth3   | 1                  | 2                | 0                      | 0                |
| hist-gradient-rich-depth2 | 3                  | 0                | 1                      | 1                |
| logistic-rich             | 10                 | 0                | 1                      | 0                |
| event-mil-physical        | 13                 | 0                | 7                      | 4                |

These fits exclude the new thread and its copies, so they do **not** demonstrate learning from its 14 positives. They test whether training on the other reviewed data transfers to this thread. Their training data includes the October 13 batch, unlike the September frozen snapshots. The next future thread is needed to assess the final fits after learning from today's labels.

For context, unchanged 5.8.2 retained-WAV replay gives **11/14 red, two yellow, one low, and 4/295 red false alerts** on this reviewed thread. Actual original-media browser checks give **10/14 red, three yellow, one low**, and five red errors among the nine explicitly named false positives. The named negative subset is not the full-thread browser false-positive rate.

## Final fits: learning diagnostics, not validation

These fits can be used for future frozen scoring, but their own training results are optimistic. The twenty untimed positives are not used to supervise the two annotated-event models. MIL uses automatically selected training-only latent windows, never saved as human annotations.

| Configuration             | Red positives / 30 | Red false alerts / 3466 | Yellow negatives | Actually used training clips |
| ------------------------- | ------------------ | ----------------------- | ---------------- | ---------------------------- |
| logistic-compact          | 30                 | 18                      | 25               | 3496                         |
| forest-rich-depth3        | 25                 | 1                       | 6                | 3496                         |
| clip-embedding            | 12                 | 222                     | 499              | 3496                         |
| clip-hybrid               | 30                 | 17                      | 23               | 3496                         |
| event-physical            | 28                 | 56                      | 47               | 3476                         |
| event-hybrid              | 29                 | 54                      | 42               | 3476                         |
| extra-trees-rich-depth3   | 27                 | 2                       | 25               | 3496                         |
| hist-gradient-rich-depth2 | 30                 | 16                      | 5                | 3496                         |
| logistic-rich             | 30                 | 4                       | 11               | 3496                         |
| event-mil-physical        | 30                 | 57                      | 49               | 3496                         |

For example, rich logistic fits all 30 positives with only four red false alerts, yet its grouped held-out result is 19/30 with 11 false alerts. That gap is why this update does not switch the browser to ML. The low-volume positive remains a useful hard example; correct event timings are needed to tell whether an event model is finding the actual sound rather than another loud passage.

## Implementation and saved evidence

The retraining runner previously hard-coded thread 336991612 as its latest-thread evaluation and described the dataset as fourteen positives. It now accepts `--evaluation-thread`, defaults to the newest labeled source-thread fold, records the selected thread in results/plan, and derives timing/data counts dynamically. `--families` makes the chosen identity-link snapshot explicit; this run uses the unchanged prior snapshot. Two regression tests cover thread selection, rejection of a missing fold, and full-thread scoring of copies owned by earlier folds. No estimator or score algorithm was changed.

- Immutable full run: `research/artifacts/retrain-20261025-known-families/`, including its pre-fit plan, 110 fold fits, ten final fits, full predictions, and per-fit manifests.
- Tracked final snapshots: `research/models/retrained-20261025/`, ten model/manifest pairs plus `results-summary.json`. The summary retains held-out positive predictions, fold metrics, and provenance.
- Published-model scoring: `research/artifacts/retrained-337690329-20261025-postfit.json`.
- Frozen pre-training snapshots and original-media evidence: [thread review](reviews/337690329.md#artifacts-and-checks).

Every saved fit was reloaded during training and compared to its in-memory scores. All ten published models were then reloaded and rescored across all **4,014 non-visual audio groups**; scores match the original final fits within 1e-12 tolerance. The 32 protected historical model/script/annotation hashes remain unchanged. No ground-truth label or event boundaries were inferred for the unwatched clip; all tested models keep it below yellow.

Model joblib files are executable pickle. Only load trusted repository snapshots; the scorer requires explicit `--trust-models`, validates hashes, and requires the recorded scikit-learn version. Raw media and full research artifacts remain local/ignored and require backup.

## Reproduction

After restoring the same snapshot/labels and rebuilding features, context, and event data, use a fresh run directory:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2   .venv/bin/python research/retrain_models.py   --output-dir research/artifacts/retrain-20261025-rerun   --evaluation-thread 337690329   --families research/artifacts/fingerprints/evaluation.json --jobs 2

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2   .venv/bin/python research/score_retrained.py publish   --run research/artifacts/retrain-20261025-rerun   --output research/models/retrained-20261025-rerun --trust-models

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2   .venv/bin/python research/score_retrained.py score   --run research/models/retrained-20261025   --output research/artifacts/retrained-20261025-rerun-scores.json --trust-models
```

The run and scorer refuse to overwrite existing output snapshots. Candidate definitions and the original protocol are documented in [the September report](RETRAINING_20260930.md#fixed-configuration-details).

| Evidence                 | SHA-256                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Runner                   | `80bb904b6a4bc11bebb6bdd794238c18de629d03cfaf0a34d97ecc1a0dbd3065` |
| Labels                   | `2024f8dec02d86307081626275614a23d9e3923be4efd79723bafe48ceef4247` |
| Human event timings      | `ece77d2bceb330becd6e8c49ce16b2a84d0089381b76687f3ef81c2195caa911` |
| Known family-link source | `e5292e6ab25d2b6a5b33d27203fed37f960f73e617e6c8b767b7fec28bc3c016` |
| Pre-fit plan             | `1c59b9ef6110114fcd62356699104fb74916a2b9f9f49fd88ee3efe3c112997b` |
| Full run results         | `d31ef0079651eefd47f160e992af479442668b7075a520fc13f5dbf956143700` |
| Published summary        | `9306c4d9e23305fc1a94eb5de9ea1128d2a24b4bd0ed67dfc776087df6999212` |
| Published-model replay   | `a6812d25c2285dd213b59aaed5d87f910b51294134a5b263b99890443ca440a3` |

Validation: 46 Python tests and 19 JavaScript tests passed, along with Ruff checks/formatting, userscript syntax, human-edited Markdown/label formatting, and Git whitespace checks. Snapshot/cutoff identities, ambiguous aliases, exclusion of unknown clips from every saved fit, all retained WAV hashes, and fold membership were audited.

Keep production unchanged. Continue freezing predictions on future threads before fitting, and supply event timings for the twenty untimed positive recordings when convenient. A production ML port would additionally need feature/decoder parity and independent future-thread checks.
