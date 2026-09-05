# Spokoyno shadow-model card

This model is an offline research/shadow scorer. It does not replace the v5.7 production warning rule and its output is not a calibrated probability.

## Dataset

- 1228 logical labeled audio tracks
- 1191 exact-content groups after deduplication
- 10 positive and 1181 negative groups
- feature dataset SHA-256: `506b5000bb53a8b84ed27193dc085897ae2455ad8954553f3a8cb1033efbc239`
- visual-only and any unlabeled clips excluded from training
- folds hold out one positive-bearing thread at a time
- ranking metrics pool score-minus-training-threshold margins so separately fitted fold scores share a conservative reference

## Grouped experiment

| model                 | features |   TP |     FP | average precision | ROC AUC |
| --------------------- | -------: | ---: | -----: | ----------------: | ------: |
| logistic-compact-c005 |       48 | 4/10 | 0/1181 |             0.594 |   0.894 |
| logistic-compact-c02  |       48 | 4/10 | 3/1181 |             0.497 |   0.929 |
| forest-compact-depth2 |       48 | 6/10 | 2/1181 |             0.602 |   0.997 |
| forest-compact-depth3 |       48 | 6/10 | 3/1181 |             0.655 |   0.997 |
| forest-rich-depth3    |      150 | 7/10 | 2/1181 |             0.649 |   0.997 |

## Held-out errors

- `logistic-compact-c005` — false positives: none; missed positives: 17883659327260384359.webm, 17884174673280229863.mp4, 17884274747240014140.mp4, 17885024222391568056.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `logistic-compact-c02` — false positives: 17881818515883672838.mp4, 17884642798500870389.mp4, 17885468877441353320.mp4; missed positives: 17883659327260384359.webm, 17884174673280229863.mp4, 17884274747240014140.mp4, 17885024222391568056.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-compact-depth2` — false positives: 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-compact-depth3` — false positives: 17882990553793679533.webm, 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885460222141837902.mp4, 17885474189950590427.webm.
- `forest-rich-depth3` — false positives: 17885464539550063156.webm, 17885470130240080562.webm; missed positives: 17884174673280229863.mp4, 17884274747240014140.mp4, 17885474189950590427.webm.

## Exported candidate

`logistic-compact-c005` was selected from models with at most 0 out-of-fold false positives, then by held-out true positives, average precision, and simplicity.
The exported threshold is the largest fitted-training negative score plus a 1e-9 serialization margin, and the decision operator is strictly greater-than. That guarantees zero training false positives only; it is not a validated production threshold.
`forest-rich-depth3` is also exported as the higher-recall challenger; its grouped errors prevent it from satisfying the conservative selection constraint.
On the final all-data fit, the conservative model marks 5/10 positives and 0/1181 negatives; the challenger marks 10/10 positives and 0/1181 negatives. These are training results, not validation.

## Strongest fitted features

- `logistic-compact-c005`: event_1_near_clip_fraction, global_loud_fraction, event_1_loudness_area_db, event_1_spectral_shape_distance, event_1_persistence_1s, event_1_band_1000_2000_jump_db, global_dynamic_range_db, event_1_event_1_0s_db, event_1_robust_z, start_spectral_centroid_hz, duration_s, event_1_duration_plus6_s, start_event_db, global_derivative_db, global_raw_peak_dbfs
- `forest-rich-depth3`: event_1_event_0_3s_db, global_p95_db, global_p99_db, event_1_event_level_db, event_1_event_0_6s_db, event_2_event_0_1s_db, global_raw_clip_fraction, event_1_persistence_1s, event_1_event_0_1s_db, event_2_baseline_1_0s_db, event_1_event_1_0s_db, global_loud_fraction, event_3_baseline_1_0s_db, event_2_event_level_db, event_1_duration_s

## Limitations

All feature design has seen the current ten positives, several source threads contain positives without fully labeled negatives, and the selected model is chosen using these same grouped diagnostics. Future prospectively labeled threads are required before promotion.
Synthetic gain/codec variants may be used for invariance tests but must never be counted as independent examples.
