# Evaluation of MAD, merged calibration, attack time, and roughness

## Decision

- **Proposal 1: reject as a scoring rule; retain MAD and robust z as telemetry.** Baseline variability is a useful quantity to record, but neither the literal multiplier nor a bounded version improved this corpus.
- **Proposal 2: accept the calibration principle, not the claim that `max` is intrinsically wrong.** Spokoyno now applies one threshold to the maximum of the two structurally eligible branch scores. A normal result displays the stronger raw branch, while a warning displays the eligible branch that actually triggered it; the number is explicitly a heuristic risk score, not a probability.
- **Proposal 3: reject as a scoring rule; retain the 10 ms attack estimate as telemetry.** The proposed fast-attack separation is contradicted by both confirmed positives and reviewed negatives.
- **Proposal 4: agree that Arnal-style roughness should not be added.** Spokoyno never implemented it. Existing spectral flux is onset spectrum change, not 30–150 Hz envelope modulation.

## Method

All 539 audio-bearing files from thread `336185346` and both supplemental positives were decoded again from the mirrors to float32 PCM at 16 kHz. Float PCM matters: an intermediate integer FLAC fixture clipped decoder overshoots and was discarded. The 11 no-audio negatives retain score zero. This gives four confirmed positives and 548 user-reviewed/provisional negatives.

For reproducibility, negative content hashes were assigned before feature inspection to a deterministic 80/20 diagnostic split: 437 development and 111 validation clips, including no-audio files. This is **not a pristine external validation set**: the corpus had already influenced earlier detector design and the positive count is still four.

MAD telemetry uses:

```text
MAD = median(abs(L - median(L)))
robustScale = max(2 dB, 1.4826 * MAD)
robustZ = jumpDb / robustScale
```

The 1.4826 factor makes MAD comparable to standard deviation for Gaussian data. The 2 dB floor replaces the proposed epsilon: an epsilon makes a perfectly steady or floor-clipped baseline produce an effectively infinite multiplier.

Attack telemetry uses 10 ms K-weighted windows around the selected coarse event. It measures from the last window at or below −30 dB to the first window within 3 dB of the following 500 ms peak. It is `unavailable` when the signal never crosses −30 dB in the lookback.

## Measurements

| rule                                           | confirmed positives | development false positives | diagnostic-validation false positives | all-negative FPR |
| ---------------------------------------------- | ------------------: | --------------------------: | ------------------------------------: | ---------------: |
| v5.3 before                                    |                 4/4 |                       0/437 |                                 0/111 |            0/548 |
| literal `oldScore * z`, existing event         |                 3/4 |                       4/437 |                                 1/111 |   5/548 (0.912%) |
| literal `oldScore * z`, reselect event using z |                 4/4 |                       2/437 |                                 1/111 |   3/548 (0.547%) |
| v5.4 final merged threshold                    |                 4/4 |                       0/437 |                                 0/111 |            0/548 |

The final warning set is intentionally unchanged. Refactoring the mathematically equivalent two branch comparisons into one post-merge comparison makes the calibration unit explicit without claiming unsupported improvement.

### Confirmed positives

| file                        |               transition score | baseline MAD dB | robust z |   attack ms | final                  |
| --------------------------- | -----------------------------: | --------------: | -------: | ----------: | ---------------------- |
| `17883557324650588814.webm` |                          0.912 |            0.64 |    31.61 |          60 | warning                |
| `17883557325462786367.mp4`  |                          0.891 |            2.18 |     7.40 |         180 | warning                |
| `17883629069140053716.mp4`  |                          0.882 |           35.51 |     0.70 | unavailable | warning                |
| `17883659327260384359.webm` | 0.685 transition / 0.904 start |            0.00 |    44.17 |          30 | warning via loud-start |

The third positive is the decisive counterexample to a MAD multiplier. At the selected event its three-second history is bimodal, spanning very quiet material and already-loud windows. Its 35.51 dB MAD drives `z` below one. Multiplication at the existing event lowers its score to 0.622 and misses it.

Allowing z to reselect the event recovers that positive, but creates three false alarms:

| false positive under literal z reselection | reviewed content                | raw transition score | robust z | multiplied score |
| ------------------------------------------ | ------------------------------- | -------------------: | -------: | ---------------: |
| `17881699236063425442.mp4`                 | legitimate loud human screaming |                0.731 |    42.98 |            1.000 |
| `17881887457541053716.mp4`                 | legitimate phonk drop           |                0.722 |     2.59 |            1.000 |
| `17882978897452723012.webm`                | legitimate video                |                0.680 |     2.33 |            1.000 |

A bounded experimental multiplier did not solve the selection problem: it promoted reviewed poor-quality loud talking (`17882990553793679533.webm`) to 0.849 while reducing confirmed positive #3 to 0.512.

### Attack-time counterexamples

The measured confirmed-positive attacks are 60 ms, 180 ms, unavailable, and 30 ms. The reviewed loud-human-scream negative is 80 ms, the phonk drop is 20 ms, and the movie explosion is 30 ms. No attack threshold separates the target. A splice can be fast, but so can percussion, compression, an explosion, or an ordinary edit; a real jump-scare soundtrack can also ramp or have a baseline above −30 dB.

## Combined calibration and statistical limit

For each branch, structural gates are applied first. The single decision score is then:

```text
transitionDecision = transitionEligible ? transitionScore : 0
startDecision      = startEligible ? startScore : 0
decisionScore      = max(transitionDecision, startDecision)
warning            = decisionScore >= 0.80
```

The threshold is evaluated against the merged detector. The approximation `1 - (1 - alpha)^2` only applies when both branch false-positive rates equal `alpha` and their errors are independent; neither condition is established here. `max` remains a valid ranking statistic when calibrated after merging.

Zero observed false positives among 548 negatives does not validate the requested 0.08% production FPR. Its one-sided 95% binomial upper bound is about 0.545%. Roughly 3,743 fresh negatives with zero false positives would be needed merely to put that upper bound below 0.08%. Likewise, 4/4 positives remains an uninformative recall estimate. Future labels, especially 30–50 positives and several thousand fresh negatives, are required before either diagnostic feature should affect warnings.

Arnal et al. studied roughness as a discriminator between screams and ordinary vocalizations using 30–150 Hz modulation energy. That is not the relevant boundary between edited screamers and legitimate clips that may also contain screams, music, or alarms: [Human Screams Occupy a Privileged Niche in the Communication Soundscape](https://doi.org/10.1016/j.cub.2015.06.043).

## v5.5 implementation-hardening regression

The browser audit found that v5.4 approximated 16 kHz by skipping device-rate samples. v5.5 instead asks Web Audio for a 16 kHz decode and uses `OfflineAudioContext` resampling when a browser returns another rate. Its spectral power also averages channel power instead of summing channels into a waveform, preventing antiphase stereo from disappearing from spectral evidence.

The loud-start search now covers only the first second of possible event-window starts. A proposed 500 ms limit failed confirmed positive #4: that file is silent for roughly 0.6 s and its loud event begins around 1.1 s. The one-second boundary covers the region before the transition detector obtains its minimum reliable history without retaining v5.4's search through approximately 2.3 s.

The complete 541-track labeled/provisional regression after these changes produced:

| set                                      | warnings | result                                                  |
| ---------------------------------------- | -------: | ------------------------------------------------------- |
| four confirmed positives                 |      4/4 | all retained                                            |
| 537 audio-bearing provisional negatives  |    0/537 | no false warnings                                       |
| 11 no-audio provisional negatives        |     0/11 | no warning                                              |
| 349 separately collected unlabeled clips |    0/349 | no warnings at the time; not counted as known negatives |

The confirmed-positive merged scores are 0.888, 0.891, 0.882, and 0.902. The largest eligible negative score is 0.730. In the separate unlabeled batch, the largest raw display score is 0.458. These measurements preserve the existing statistical caveat: four positives and 548 provisional negatives are not enough to establish production recall or a sub-0.08% false-positive rate.

## v5.6 follow-up: two labels from the unlabeled batch

The user subsequently confirmed two of those 349 clips as screamers. This is exactly why the batch was never counted as negative ground truth. v5.5 missed both:

| file                       | event dB | jump dB | duration | near-clip | exact/nearby flux | spectral distance | v5.5 score |
| -------------------------- | -------: | ------: | -------: | --------: | ----------------: | ----------------: | ---------: |
| `17884174673280229863.mp4` |    -3.33 |   16.54 |   0.90 s |     1.33% |       0.124/0.339 |             0.869 |      0.455 |
| `17884274747240014140.mp4` |    -2.36 |   12.08 |   0.45 s |    43.82% |       0.054/0.401 |             0.444 |      0.216 |

The first demonstrates a 50 ms boundary-alignment failure: the exact onset flux is weak, while the ±100 ms maximum and baseline-to-event spectral change are strong. The second demonstrates a fixed-contrast failure: a short event can be dangerous with a 12 dB jump when nearly half of its samples are already near clipping. Neither supports globally lowering the transition threshold or replacing exact flux with nearby maximum flux. Doing the latter promotes the reviewed legitimate human scream and poor-quality loud talking to warnings.

v5.6 therefore adds two narrow short-burst rescue paths, one requiring spectral distance and one requiring severe clipping. Both require a short near-full-scale transition plus nearby spectral flux, and both retain upper duration bounds. Those bounds preserve the reviewed 1.55 s human scream, 1.9 s movie explosion, and 3 s phonk drop as normal. The post-tuning regression is:

| set                                      | warnings | result                                      |
| ---------------------------------------- | -------: | ------------------------------------------- |
| six confirmed positives                  |      6/6 | all retained                                |
| 537 audio-bearing provisional negatives  |    0/537 | no false warnings                           |
| 11 no-audio provisional negatives        |     0/11 | no warning                                  |
| 347 remaining explicitly unlabeled clips |    0/347 | no warnings; not counted as known negatives |

Positive #5 scores 0.855 through the spectral-burst path and positive #6 scores 0.862 through the clipped-burst path. This is not held-out validation: both new positives directly motivated the new rules. The six-positive Wilson interval remains too broad for a useful recall claim, and the unchanged 548-negative set remains too small to demonstrate the requested production FPR.

## v5.7 follow-up: thread 336291305

The next fully reviewed thread supplied four audio-target screamers, one visual-only monster jump-scare, and 342 negatives. Eight of its 347 attachments contain no audio stream. The visual-only example is retained under a separate label and deliberately excluded from audio recall.

v5.6 caught two of the four audio examples. The two misses were qualitatively different:

| file                       | event dB | jump dB | duration | near-clip | nearby flux | spectral distance | v5.6 score |
| -------------------------- | -------: | ------: | -------: | --------: | ----------: | ----------------: | ---------: |
| `17885024222391568056.mp4` |    -2.93 |   36.34 |   3.00 s |     2.89% |       0.500 |             0.866 |      0.728 |
| `17885460222141837902.mp4` |    -3.63 |   61.49 |   0.30 s |    21.79% |       0.424 |             0.559 |      0.792 |

The first is a sustained replacement of the prior spectrum whose exact-boundary flux is weak. The second is a slower but extremely high-contrast short burst. v5.7 adds separate, tightly gated rescue paths for those structures. Neither path fires on the accumulated negative or unlabeled corpus. The merged post-tuning regression is:

| set                                  | warnings | result                                      |
| ------------------------------------ | -------: | ------------------------------------------- |
| ten confirmed audio-target positives |    10/10 | all retained                                |
| 871 audio-bearing labeled negatives  |    0/871 | no false warnings                           |
| 19 no-audio labeled negatives        |     0/19 | no warning                                  |
| 347 explicitly unlabeled audio clips |    0/347 | no warnings; not counted as known negatives |
| one confirmed visual-only screamer   |      0/1 | normal by design                            |

The new positive scores are 0.883, 0.881, 0.812, and 0.900. The explicit loud-phonk negative scores 0.615. As before, the new rules were designed after inspecting their motivating positives; this is a regression check, not an unbiased recall estimate.
