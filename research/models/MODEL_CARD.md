# Spokoyno shadow-model card

This model is an offline research/shadow scorer. It does not replace the v5.7 production warning rule and its output is not a calibrated probability.

## Dataset

- 881 logical labeled audio tracks
- 859 exact-content groups after deduplication
- 10 positive and 849 negative groups
- feature dataset SHA-256: `213b043b0ac6a8950b210acb1a522af646f03974c3afb20856844d1689316c66`
- visual-only and unlabeled clips excluded from training
- folds hold out one positive-bearing thread at a time
- ranking metrics pool score-minus-training-threshold margins so separately fitted fold scores share a conservative reference

## Grouped experiment

| model                 | features |   TP |    FP | average precision | ROC AUC |
| --------------------- | -------: | ---: | ----: | ----------------: | ------: |
| logistic-compact-c005 |       48 | 3/10 | 0/849 |             0.642 |   0.895 |
| logistic-compact-c02  |       48 | 4/10 | 1/849 |             0.643 |   0.923 |
| forest-compact-depth2 |       48 | 6/10 | 3/849 |             0.622 |   0.996 |
| forest-compact-depth3 |       48 | 6/10 | 3/849 |             0.613 |   0.995 |
| forest-rich-depth3    |      150 | 7/10 | 2/849 |             0.627 |   0.996 |

## Held-out errors

- `logistic-compact-c005` — false positives: none; missed positives: 17883659327260384359.webm, 17884174673280229863.mp4, 17884274747240014140.mp4, 17885024222391568056.mp4, 17885460222141837902.mp4, 17885474189950590427.webm, 17885487431220276835.mp4.
- `logistic-compact-c02` — false positives: 17885468877441353320.mp4; missed positives: 17883659327260384359.webm, 17884174673280229863.mp4, 17884274747240014140.mp4, 17885024222391568056.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-compact-depth2` — false positives: 17882990553793679533.webm, 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-compact-depth3` — false positives: 17882990553793679533.webm, 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-rich-depth3` — false positives: 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885474189950590427.webm.

## Exported candidate

`logistic-compact-c005` was selected from models with at most 0 out-of-fold false positives, then by held-out true positives, average precision, and simplicity.
The exported threshold is the largest fitted-training negative score plus a 1e-9 serialization margin, and the decision operator is strictly greater-than. That guarantees zero training false positives only; it is not a validated production threshold.
`forest-rich-depth3` is also exported as the higher-recall challenger; its grouped errors prevent it from satisfying the conservative selection constraint.
On the final all-data fit, the conservative model marks 5/10 positives and 0/849 negatives; the challenger marks 9/10 positives and 0/849 negatives. These are training results, not validation.

## Strongest fitted features

- `logistic-compact-c005`: event_1_near_clip_fraction, global_loud_fraction, event_1_spectral_shape_distance, event_1_persistence_1s, event_1_loudness_area_db, event_1_band_1000_2000_jump_db, event_1_event_1_0s_db, global_dynamic_range_db, start_spectral_centroid_hz, start_event_db, event_1_robust_z, event_1_band_2000_4000_jump_db, event_1_duration_plus6_s, event_1_band_4000_8000_jump_db, global_derivative_db
- `forest-rich-depth3`: event_1_event_0_3s_db, event_1_event_level_db, global_p95_db, global_p99_db, event_1_event_0_6s_db, event_1_persistence_1s, event_2_baseline_1_0s_db, event_2_event_0_1s_db, event_1_event_1_0s_db, event_1_event_0_1s_db, event_2_event_0_3s_db, event_1_duration_s, global_raw_clip_fraction, global_loud_fraction, event_2_event_level_db

## Limitations

All feature design has seen the current ten positives, several source threads contain positives without fully labeled negatives, and the selected model is chosen using these same grouped diagnostics. Future prospectively labeled threads are required before promotion.
Synthetic gain/codec variants may be used for invariance tests but must never be counted as independent examples.
