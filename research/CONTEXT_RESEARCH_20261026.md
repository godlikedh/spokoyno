# Hard negatives and sound-context research — 2026-10-26

**Nine additional configurations did not beat the previous best ML model overall.** This continuation tested new training supervision, 32 new features, and nonlinear combinations of those features. It completed **603 model fits across 67 distinct training partitions**. Some controls improved, but none justified replacing the previous `synthetic-mil-20pct` research baseline or releasing a new detector.

The [machine-readable comparison](models/context-comparison-20261026/comparison.json) includes every configuration, per-thread results, codec score changes, and descriptive thread-bootstrap intervals. These are further adaptive experiments on the existing development corpus, not new independent validation. The [preceding study](ROBUST_AUDIO_RESEARCH_20261026.md) describes the codec/delay generator and common feature pipeline.

## Held-out results

The evaluation contains **30 positive and 3,466 negative original audio groups**. Source threads, duplicate memberships, known transitive audio families, and every synthetic parent are excluded from the corresponding training folds. Yellow remains 0.6 and red remains 0.8. Scores are risk evidence, not calibrated probabilities.

| Configuration                         | Original red positives / 30 | Original red negatives / 3,466 | AAC red positives / 30 | Opus red positives / 30 | MP3 red positives / 30 |
| ------------------------------------- | --------------------------: | -----------------------------: | ---------------------: | ----------------------: | ---------------------: |
| Previous best ML: synthetic MIL       |                          30 |                             34 |                     25 |                      28 |                     28 |
| Hard-negative mixed                   |                          27 |                             79 |                     26 |                      25 |                     27 |
| Hard-negative maximum                 |                          26 |                             57 |                     24 |                      26 |                     26 |
| Matching all-window control           |                          29 |                            158 |                     29 |                      29 |                     29 |
| Context hybrid                        |                          28 |                             49 |                     25 |                      28 |                     28 |
| Context lower-order pooling           |                          23 |                             20 |                     22 |                      24 |                     25 |
| Context, real augmentation only       |                          24 |                             31 |                     21 |                      24 |                     24 |
| Content-boundary control              |                          28 |                             51 |                     24 |                      28 |                     28 |
| Context trees                         |                          27 |                             77 |                     26 |                      26 |                     27 |
| Matching trees without context inputs |                          27 |                             79 |                     26 |                      27 |                     27 |

The previous model's codec results use the corrected bounded codec recipes from the preceding study on all thirty positives. Its full negative corpus was evaluated only for original files; its codec stress negatives are a selected nine-clip subset. Therefore, the baseline row does not establish a comparable full-corpus codec false-positive rate.

Hard-negative mining **halved false red alerts from 158 to 79** against its matching three-round tree control, but lost two additional positive recordings. Putting all negative weight on a single event lowered false alerts further to 57, while losing another positive. This is a real change in the recall/false-alert tradeoff, not an overall improvement.

The context lower-order pooling model caught **30/30 originals at yellow-or-red with 90 negative warnings**, compared with 98 for the previous baseline. However, only 23 positives were red, and one AAC positive fell below yellow. Moving the threshold after inspecting those results would not establish a better detector.

Automatic model selection used only inner-fold original and training-codec predictions. Its outer results were 27/30 positives and 79 false alerts for mining, 28/30 and 52 for linear context, and 27/30 and 77 for context trees. Selecting the nicest fixed row after seeing all outer results would exaggerate the evidence. Snapshot defaults are selections within each experiment, not production recommendations.

## What was changed and why

### Training on the events that cause false alarms

The previous trees distributed each normal recording's training weight uniformly across its proposed events, while inference emphasizes its highest-scoring events. A single false event can therefore dominate the recording's output despite receiving little training weight.

The new [mining trainer](train_hard_negative_mil.py) compares three predetermined allocations: uniform weight, 75% on the three highest-scoring events plus 25% uniform, and 100% on the highest-scoring event. Every allocation preserves the normal recording's total weight. It uses three fixed rounds; the all-window control uses the same number of rounds, tree size, regularization, and data. Mining sees only the eligible training recordings.

Positive recordings select a likely event from windows that are not known negative, retain every strongly labeled positive window, and use known-negative windows as auxiliary supervision. Initial assignments come from a model fitted to the exact same training partition. The trees have seven leaves, depth three, 120 boosting iterations, and no clipping-fraction input. No filename-specific rule was introduced.

### New sound-context features

The [context extractor](context_features.py) adds **32 label-independent measurements** to the existing 68 acoustic measurements:

- Six embedding distances and norm contrasts compare before, during, and after the candidate event.
- Twenty-five semantic measurements describe sound-category scores, their changes, and whether music/singing persists or vocal alarm/noise appears.
- One content-boundary flag identifies events in the first second of audible content. Its reference is the trimmed content boundary, so adding leading digital zeros does not change it.

The semantic inputs use twenty fixed classifier columns from the pinned YAMNet model, grouped into speech, music, vocal alarm, singing, laughter, noise, and silence. The group called “vocal alarm” includes shouting, screaming, crying, moaning, groaning, grunting, gasping, and panting. These are broad pretrained sound categories, **not screamer labels**. YAMNet was not retrained. Its category scores and embeddings are documented in the [official YAMNet tutorial](https://www.tensorflow.org/hub/tutorials/yamnet), with categories defined by the [AudioSet ontology](https://research.google.com/audioset/ontology/index.html).

The linear study compares full context features, lower-order recording pooling, removal of synthetic mixtures, and a control retaining only the content-boundary flag among the new features. All use the same training-only PCA8 projection and disable near-clipping fraction. Removing synthetic data reduced false alerts but lost recall: 24/30 versus 28/30 for full context. That does not prove the generated examples are realistic; it measures their contribution within this particular training setup.

The final pair of tree models tests interactions among context and acoustic features. Both use identical context-seeded positive assignments, hard-negative supervision, PCA, and hyperparameters. One disables all 32 context inputs. This isolates their direct use by the tree; the common seed already uses context. Original false alerts changed from 79 to 77, with both catching 27/30. Context features were useful in some cases but did not solve the task.

## Data, leakage checks, and remaining failures

This continuation reuses the frozen **25,794 recording variants and 348,112 event windows** from the preceding study. It does not count more copies as more independent evidence. There are still thirty distinct positive audio groups in this evaluation, only ten with human event timings, and 518 unknown groups excluded from supervised fitting and accuracy. Human labels and intervals were unchanged.

All three studies use the same parent eligibility and source-balanced weighting. Preprocessing and PCA fit only original training windows. Evaluation AAC80, Opus48, and MP3 profiles enter neither fitting nor model selection. Each study has an immutable candidate declaration; the context-tree pair was declared after seeing the linear context results and before seeing either its own or the mining study's final results. This ordering is recorded rather than presented as prospective validation.

The new heads still fail on real examples. The context hybrid scores `17929291025992177046.webm` **0.5592 on the source-excluded original and 0.2783 after AAC**. The context tree scores it 0.6372 and 0.3857. The same linear model fitted on all labels gives it 0.9325; that training score is explicitly not evidence of generalization. `17884274747240014140.mp4` is another context-hybrid miss at 0.6666 original and 0.5521 AAC.

Across the original and all three evaluation codecs, the context hybrid keeps only 25/30 positives red in every version, and 28/30 yellow-or-red in every version. Hard-negative mixed also keeps 25/30 red. Pure zero padding is fixed structurally; lossy-codec stability is still incomplete. No threshold was retuned to hide these failures.

Repeated experimentation on the same thirty positives can overfit the research decisions even when individual fits obey all exclusions. Known-family grouping cannot account for undiscovered shared sound families. The results reject these configurations; they do not establish that no better architecture is possible. A strong future result needs confirmation on subsequently received original threads.

## Runtime, artifacts, and reproduction

The Python and JavaScript inference paths now support the frozen context head. The standalone inference command returns continuous scores, tiers, and proposed event times. Errors remain errors rather than safe scores. The userscript stays at 5.8.2 because none of the candidates passes the quality requirement.

The selected YAMNet classifier columns match the official graph within **6.36e-8**. A numerical Python/JavaScript test checks all 32 context measurements, including silence, temporal contrasts, and boundary rounding. Full context inference in a muted isolated browser on 27 preserved original-media fixtures matches Python within **0.00000255**, with no tier changes. The fixtures contain fourteen positives, nine selected hard negatives, and four unknowns; this is a runtime check, not a representative false-positive estimate.

Both context tree heads match Python scores exactly in the same browser check. The three physical-only mining heads match within 1.12e-16 in Node JavaScript on those browser-decoded fixtures. That mining check does not launch another browser. The seven exported linear/mining models reproduce all 25,794 saved final-fit scores within 1.73e-15. Repository checks pass: 82 Python tests, 19 JavaScript tests, Ruff, and Prettier.

**The context trees failed the stricter full-matrix export check.** Training and portable inference perform PCA matrix operations in a different order. Input differences up to 6.13e-14 crossed a few exact tree-threshold ties. The context head changed four recording scores by at most 0.000778; its control changed fifteen by at most 0.009261. No tier changed in this final-fit comparison, but exact model reproduction failed. The export check was not relaxed. These two models are published as evaluation evidence only, with a [reproducible diagnostic](audit_tree_preprocessing.py), rather than loadable inference snapshots. Their table scores come from the frozen training/evaluation implementation; portable equivalence over every fold has not been established. The 27-fixture browser result alone would have missed this failure.

Each completed audit verifies all 67 saved partitions and recomputes the 121 outer/inner source/family split definitions. It checks 195 delay comparisons on thirty positives and nine negatives, including 0.013, 0.125, 1.75, 11, and 30 seconds. Acoustic features, encoder inputs, and the added context features remain exactly unchanged; displayed event times shift by the expected offset. These are repeated checks on 39 sources, not 195 independent examples.

The published artifacts preserve every configuration's held-out positives and false warnings, codec stress results, and provenance hashes. Seven models have portable snapshots; the rejected context-tree export has evidence only:

- [Hard-negative models](models/hard-negative-audio-20261026/manifest.json)
- [Linear context models](models/context-audio-20261026/manifest.json)
- [Context tree results and failed-export evidence](models/context-trees-20261026/manifest.json)
- [Combined comparison and runtime evidence](models/context-comparison-20261026/manifest.json)

Raw media, generated matrices, and per-fold fit caches remain local. Context matrices reuse the existing embedding/owner/target files as read-only hardlinks. Published transforms embed the selected classifier weights and include their license. Existing model snapshots and hash-frozen extractors were not modified.

Use new output directories when reproducing. First export the frozen semantic columns and assemble context features:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-audio/bin/python research/export_context_head.py \
  --output-dir research/artifacts/context-head-rerun
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/assemble_context_data.py \
  --base research/artifacts/robust-matrix-20261026 \
  --head-dir research/artifacts/context-head-rerun --output-dir research/artifacts/context-matrix-rerun
```

Run the mining and linear context experiments; the context-tree experiment depends on the completed linear fits:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_hard_negative_mil.py \
  --data-dir research/artifacts/robust-matrix-20261026 \
  --seed-run research/artifacts/codec-stable-study-20261026 \
  --output-dir research/artifacts/hard-negative-study-rerun --jobs 2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_context_models.py \
  --data-dir research/artifacts/context-matrix-rerun \
  --output-dir research/artifacts/context-study-rerun --jobs 2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python research/train_context_trees.py \
  --data-dir research/artifacts/context-matrix-rerun \
  --seed-run research/artifacts/context-study-rerun \
  --output-dir research/artifacts/context-tree-study-rerun --jobs 2
```

For each run, use `audit_robust_models.py` in `.venv-audio`, then `export_robust_models.py` in `.venv`, with its matching matrix and fresh output directories. The context-tree export is expected to reject the numerical mismatch described above; `audit_tree_preprocessing.py --data-dir ... --run ... --output ...` reproduces its diagnosis. `summarize_robust.py` accepts repeated `--run` arguments. Browser reproduction uses `prepare_robust_browser.py` and `check_robust_browser.cjs`, with the pinned encoder and TensorFlow.js dependencies described in the preceding report. `check_robust_node.cjs` checks the physical-only mining heads on preserved browser PCM.

The actual local paths use `20261026` instead of `rerun`. The completed linear context audit is `context-audit-20261026-v2`; the first invocation stopped before inference because it used the non-TensorFlow Python environment. No model or data changes were made for the rerun.
