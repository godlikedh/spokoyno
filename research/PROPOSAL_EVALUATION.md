# Evaluation of MAD, merged calibration, attack time, and roughness

## Decision

- **Proposal 1: reject as a scoring rule; retain MAD and robust z as telemetry.** Baseline variability is a useful quantity to record, but neither the literal multiplier nor a bounded version improved this corpus.
- **Proposal 2: accept the calibration principle, not the claim that `max` is intrinsically wrong.** Spokoyno now applies one threshold to the maximum of the two structurally eligible branch scores. The displayed number is explicitly a heuristic risk score, not a probability.
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

| rule | confirmed positives | development false positives | diagnostic-validation false positives | all-negative FPR |
|---|---:|---:|---:|---:|
| v5.3 before | 4/4 | 0/437 | 0/111 | 0/548 |
| literal `oldScore * z`, existing event | 3/4 | 4/437 | 1/111 | 5/548 (0.912%) |
| literal `oldScore * z`, reselect event using z | 4/4 | 2/437 | 1/111 | 3/548 (0.547%) |
| v5.4 final merged threshold | 4/4 | 0/437 | 0/111 | 0/548 |

The final warning set is intentionally unchanged. Refactoring the mathematically equivalent two branch comparisons into one post-merge comparison makes the calibration unit explicit without claiming unsupported improvement.

### Confirmed positives

| file | transition score | baseline MAD dB | robust z | attack ms | final |
|---|---:|---:|---:|---:|---|
| `17883557324650588814.webm` | 0.912 | 0.64 | 31.61 | 60 | warning |
| `17883557325462786367.mp4` | 0.891 | 2.18 | 7.40 | 180 | warning |
| `17883629069140053716.mp4` | 0.882 | 35.51 | 0.70 | unavailable | warning |
| `17883659327260384359.webm` | 0.685 transition / 0.904 start | 0.00 | 44.17 | 30 | warning via loud-start |

The third positive is the decisive counterexample to a MAD multiplier. At the selected event its three-second history is bimodal, spanning very quiet material and already-loud windows. Its 35.51 dB MAD drives `z` below one. Multiplication at the existing event lowers its score to 0.622 and misses it.

Allowing z to reselect the event recovers that positive, but creates three false alarms:

| false positive under literal z reselection | reviewed content | raw transition score | robust z | multiplied score |
|---|---|---:|---:|---:|
| `17881699236063425442.mp4` | legitimate loud human screaming | 0.731 | 42.98 | 1.000 |
| `17881887457541053716.mp4` | legitimate phonk drop | 0.722 | 2.59 | 1.000 |
| `17882978897452723012.webm` | legitimate video | 0.680 | 2.33 | 1.000 |

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
