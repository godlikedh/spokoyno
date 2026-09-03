# Spokoyno screamer-detector research

## Corpus and labels

The 2ch API response for thread `336185346` contained 550 video attachments: 417 MP4 and 133 WebM files, 548 unique MD5 values, about 3.47 GiB, and about 8.5 hours of media. Every attachment was downloaded from `2ch.su` with mirror fallback available. The hostname is not part of media identity; the dataset retains `/b/src/336185346/<file>` as the key.

FFmpeg decoded audio from 539 files: 413 AAC, 93 Vorbis, 31 Opus, and 2 MP3 tracks. Eleven files had no audio stream. The two user-confirmed screamers are positives; following the user's clarification and manual review, the other 548 attachments are high-confidence provisional negatives. Two later confirmed screamers from thread `336243339` were added as supplemental positives. One is a conventional transition event; the other begins its dangerous audio about 0.6 s into the file, before the transition path can obtain its required one-second baseline. Two clips in thread `336272252` were subsequently confirmed as screamers. The labeled evaluation set therefore contains six confirmed positives and 548 provisional negatives, plus 347 explicitly unlabeled comparison clips.

The complete original-thread table is in [thread-336185346.md](thread-336185346.md). Machine-readable versions, including every requested feature, are [thread-336185346.csv](thread-336185346.csv) and [thread-336185346.json](thread-336185346.json). The later positives are in [supplemental-positives.md](supplemental-positives.md), [supplemental-positives.csv](supplemental-positives.csv), and [supplemental-positives.json](supplemental-positives.json). Only the two supplied files—not their entire source thread—were analyzed for the supplemental set. A later external-review proposal study is documented separately in [PROPOSAL_EVALUATION.md](PROPOSAL_EVALUATION.md).

Thread `336272252` was retained as a separate corpus batch, and all 349 video attachments had decodable audio. The user later labeled `17884174673280229863.mp4` and `17884274747240014140.mp4` as screamers; the remaining 347 clips are still unlabeled rather than presumed negative. Its [table](thread-336272252.md), [CSV](thread-336272252.csv), and [JSON](thread-336272252.json) preserve that distinction, so the unlabeled remainder is excluded from the reported 548-negative evaluation.

## Method

FFmpeg decoded the complete first audio track to stereo float PCM at 16 kHz. Analysis used 50 ms non-overlapping windows. Per-track metrics include raw RMS/peak, approximate perceptual loudness, P10/P25/P50/P75/P90/P95/P99, crest factor, peak-to-median range, threshold occupancy, sample near-clipping occupancy, spectral flux, three coarse spectral bands, and maximum changes at 50/100/250/500/1000 ms.

The perceptual approximation is a 70 Hz high-pass followed by a +4 dB high shelf at 1.5 kHz, with the conventional -0.691 dB offset. It is intentionally described as “K-like” rather than LUFS: there is no BS.1770 channel weighting or program gating. With the otherwise identical score on raw RMS, confirmed positive #2 ranked fifth behind three provisional negatives (78.9%); with the K-like timeline it ranked second (89.1%). The weighting therefore helped this corpus, although local event structure, persistence, and onset flux remain the main safeguards.

For each possible event:

- Baseline: median of up to the previous 3 s, ending 100 ms before the candidate; at least 1 s of history is required.
- Event level: P25 of the next 300 ms. This means at least most of the six windows must be loud, rejecting an isolated click.
- Jump: event level minus baseline.
- Duration: contiguous windows above `max(-13 dB, baseline + 9 dB)`, bridging one 50 ms dip and capped at 3 s.
- Onset spectral flux: positive change in the normalized power spectrum between the candidate window and the immediately preceding 50 ms window.
- Near clipping: fraction of event samples at or above -1 dBFS. Lossy decoder overshoot can exceed 0 dBFS, so this is supporting evidence, not a hard condition.

The supplemental immediate screamer exposed a blind spot: a baseline-relative detector cannot evaluate a dangerous event that begins in the first second. Spokoyno now runs a second, independent loud-start path over event-window starts in the first second. It finds the loudest robust 300 ms opening block, measures persistence from the actual onset, and inspects up to one second of clipping. Six short FFTs provide spectral flatness, while first-difference energy provides a cheap high-frequency-brightness estimate. These spectral features are supporting evidence for an already near-full-scale sustained opening; they can never flag a merely high-frequency but quiet sound by themselves.

## Findings

| file                      | label                | mode           | median dB | peak dBFS | event s | baseline dB | event dB | jump dB | duration s | near-clip % | spectral evidence           | score |
| ------------------------- | -------------------- | -------------- | --------: | --------: | ------: | ----------: | -------: | ------: | ---------: | ----------: | --------------------------- | ----: |
| 17883557324650588814.webm | confirmed #1         | transition     |    -43.96 |    +24.73 |   15.85 |      -45.03 |   +18.18 |  +63.21 |       1.95 |       98.09 | flux 0.281                  | 88.8% |
| 17883659327260384359.webm | confirmed #4         | loud start     |    -90.00 |     +2.81 |    0.60 |         n/a |    -1.21 |     n/a |       3.00 |       28.28 | flat 0.176; bright -1.77 dB | 90.2% |
| 17883557325462786367.mp4  | confirmed #2         | transition     |    -23.37 |     +2.74 |    7.95 |      -24.53 |    -0.66 |  +23.87 |       2.45 |       52.76 | flux 0.410                  | 89.1% |
| 17883629069140053716.mp4  | confirmed #3         | transition     |     -4.15 |     +1.72 |    6.70 |      -39.56 |    -2.47 |  +37.10 |       3.00 |        4.51 | flux 0.313                  | 88.2% |
| 17884274747240014140.mp4  | confirmed #6         | clipped burst  |    -14.00 |     +5.16 |    9.55 |      -14.44 |    -2.36 |  +12.08 |       0.45 |       43.82 | nearby flux 0.401           | 86.2% |
| 17884174673280229863.mp4  | confirmed #5         | spectral burst |    -18.28 |     +1.55 |    3.00 |      -19.87 |    -3.33 |  +16.54 |       0.90 |        1.33 | shape 0.869; flux 0.339     | 85.5% |
| 17882984411551286220.webm | provisional negative | normal         |    -32.10 |     +0.50 |   76.65 |      -43.70 |    -6.35 |  +37.35 |       0.40 |        2.97 | flux 0.384                  | 77.4% |
| 17881699236063425442.mp4  | reviewed negative    | normal         |    -12.73 |     +2.68 |   13.30 |      -58.10 |    -0.60 |  +57.50 |       1.55 |       55.16 | flux 0.138                  | 73.0% |
| 17881887457541053716.mp4  | reviewed negative    | normal         |     -8.64 |     +1.90 |   16.05 |      -25.51 |    -5.76 |  +19.75 |       3.00 |       27.45 | flux 0.448                  | 72.2% |
| 17881481520942199957.mp4  | provisional negative | normal         |    -35.48 |     +0.38 |   18.10 |      -62.70 |    -7.81 |  +54.89 |       0.50 |        7.74 | flux 0.276                  | 66.6% |
| 17882990553793679533.webm | reviewed negative    | normal         |    -12.26 |     +2.68 |   65.90 |      -37.88 |    -3.98 |  +33.90 |       0.55 |       30.86 | flux 0.091                  | 65.5% |

Positives #1–#3 are not merely loud: each has a sustained near-full-scale section after an ordinary or quiet local baseline and a changed onset spectrum. Positive #1 is extreme, with decoded lossy PCM overshooting nominal full scale heavily. Positive #4 is different: it has no usable pre-event baseline, but its opening combines a -1.21 dB robust level, three seconds of persistence, 28.28% near-clipping, high spectral flatness, and unusually strong high-frequency energy. Positive #5 is a short end burst whose exact boundary flux is modest, but the maximum flux within ±100 ms is 0.339 and its baseline-to-event spectral distance is 0.869; its high-frequency band rises by about 63 dB. Positive #6 is not a high-frequency example: its decisive evidence is a 450 ms event with 43.82% near-clipping after a 12.08 dB rise.

For the general transition path, spectral flatness, zero-crossing rate, direct low/mid/high band jumps, brightness change, global crest factor, and maximum derivative did not add stable separation beyond the local envelope plus normalized onset flux. Full-spectrum shape distance is used only by the conservative short-spectral-burst rescue path introduced after positive #5. Flatness and brightness are used only by the loud-start path, where no baseline comparison is possible.

## Old detector

On the original thread, the old rule detected both positives but also flagged 36 of 537 decodable provisional negatives. On those provisional labels that is 100% recall, 5.3% precision, and a 6.7% false-positive rate. It missed supplemental positive #3 and both newly labeled short-burst positives (`old_score=2` for all three), while detecting immediate positive #4 (`old_score=4`). Its whole-file peak and median mix unrelated parts of a clip, its “peak” is actually maximum 100 ms RMS, and its one-second baseline allows short impulses or scene cuts to supply independent score points. A file can receive three points without containing one coherent dangerous event.

Old false-positive filenames (all have `old_classification=suspicious` in the CSV):

`17882984411551286220.webm`, `17881481520942199957.mp4`, `17883361674593313438.mp4`, `17882989063041961052.webm`, `17881818515883672838.mp4`, `17882962082822601629.mp4`, `17882991086611750790.webm`, `17882997637612360215.webm`, `17881465969503404529.mp4`, `17881525425650729167.mp4`, `17881898673292643739.webm`, `17881476531723610876.mp4`, `17881842513483638014.mp4`, `17881869854811113740.mp4`, `17882986903210194666.webm`, `17881844799460675138.mp4`, `17881525427333106028.mp4`, `17883360944451558085.mp4`, `17882990552741630557.webm`, `17882990553482999670.webm`, `17882993665621506103.webm`, `17883011960512230584.mp4`, `17883249492281720284.mp4`, `17883364449832636829.mp4`, `17881806178130843016.mp4`, `17881476139173281009.mp4`, `17881514246931196290.mp4`, `17881518395660582444.webm`, `17881910378980818126.webm`, `17882548075240997522.mp4`, `17882518681650361236.mp4`, `17881514247692206034.mp4`, `17881833404270949457.mp4`, `17883362821053253349.mp4`, `17881476138422688186.mp4`, `17883395202170259937.mp4`.

## Final score and decision

Let `S(x) = 1 / (1 + exp(-x))`. The original quiet-to-loud transition path is:

```text
loud     = S((eventDb + 10.5) / 2.5)
jump     = S((jumpDb - 13) / 3.5)
duration = S((durationSeconds - 0.18) / 0.09)
quiet    = S((-baselineDb - 15) / 5)
clip     = S((nearClipFraction - 0.005) / 0.012)
flux     = S((spectralFlux - 0.22) / 0.08)

transitionConfidence = loud^0.9 * jump^1.25 * duration^0.65
                       * (0.87 + 0.10*quiet + 0.03*clip)
                       * (0.65 + 0.35*flux)
```

It is structurally eligible only when all of these hold:

```text
eventDb >= -6
jumpDb >= 14
duration >= 0.15 s
```

The independent loud-start path is:

```text
startLoud       = S((startEventDb + 6) / 2)
startDuration   = S((startDurationSeconds - 0.35) / 0.15)
startClip       = S((startNearClipFraction - 0.01) / 0.03)
startNoise      = S((spectralFlatness - 0.025) / 0.025)
startBrightness = S((brightnessDb + 8) / 2.5)

startConfidence = startLoud^1.1 * startDuration^0.7
                  * (0.68 + 0.12*startClip + 0.14*startNoise + 0.06*startBrightness)
```

It is structurally eligible when `startEventDb >= -3`, duration is at least 0.50 s, and at least one of these independent spectral/damage signals holds: flatness at least 0.04, brightness at least -5 dB, or near-clipping at least 8%.

Two conservative rescue paths handle short edited bursts that the general transition score deliberately suppresses:

```text
short spectral burst:
  eventDb >= -4, jumpDb >= 16, 0.30 <= duration <= 1.05 s,
  max spectral flux within +/-100 ms >= 0.30,
  baseline-to-event Hellinger spectral distance >= 0.80

short clipped burst:
  eventDb >= -3, jumpDb >= 10, 0.25 <= duration <= 0.55 s,
  event near-clipping >= 35%,
  max spectral flux within +/-100 ms >= 0.30
```

The upper duration bounds are intentional: the closest reviewed loud-human-scream negative lasts 1.55 s, the phonk drop lasts at least 3 s, and the movie explosion lasts 1.9 s. Those clips remain on the conservative general path. An eligible rescue begins at 0.80 and receives only small bounded margin bonuses, producing risk 85.5/100 for positive #5 and 86.2/100 for positive #6. These values were tuned after seeing those examples and must not be interpreted as calibrated probabilities.

The ineligible branch scores are zeroed, all eligible scores are merged with `max`, and the single merged score is compared with the 0.80 warning threshold. This makes post-merge calibration explicit. For a normal result, the UI displays the stronger raw general branch so clips that miss one hard gate do not misleadingly collapse to zero. For a warning, it displays the eligible branch that actually triggered the warning, keeping the shown event and score consistent. Scores below 10/100 retain decimal precision and positive values below 0.01/100 display as `<0.01`. Analysis schema v6 forces older tab results to be recomputed from cached media without downloading it again.

Baseline MAD, MAD-normalized jump, and a local 10 ms attack estimate are recorded as diagnostics but do not affect the score. Full-corpus testing found that literal MAD multiplication created false positives and that attack time did not separate screamers from music drops or explosions; see [PROPOSAL_EVALUATION.md](PROPOSAL_EVALUATION.md).

The hard gates keep both continuous scores honest. A large rise from digital silence to a moderate sound, a short click, a high-frequency but quiet sound, or ordinary loud music cannot be promoted by unrelated bonuses alone.

The combined rule flags all six confirmed positives and no provisional negatives. On the full original thread it still flags exactly the original two. The immediate positive scores 90.2% on the start path; the highest raw start score among the 548 original comparison files is 68.1% and fails the absolute-level gate. No rescue rule fires among the 347 still-unlabeled clips. This is a post-label calibration result, not an estimate of real-world accuracy: six positive examples are still far too few, and positives #5/#6 were used to design their rescue paths.

## Difficult negatives and labeling priorities

- `17882984411551286220.webm` (77.4%) was manually reviewed as a TV-show clip with no heard screamer. Its short 37 dB transition settles at only -6.35 dB for 0.40 s, narrowly failing the absolute event gate.
- `17881699236063425442.mp4` (73.0% combined; 68.1% start path) was manually reviewed as legitimate but genuinely loud screaming. It is the closest loud-start negative, with -3.29 dB opening audio and 20.2% near-clipping, but its low 0.016 flatness and -6.88 dB brightness keep it well below the start threshold.
- `17881887457541053716.mp4` (72.2%) was manually identified as legitimate phonk with a loud drop. It is loud overall (median -8.64 dB), while its local jump is only 19.75 dB.
- `17882990553793679533.webm` (65.5%) reaches -3.98 dB after a 33.9 dB jump, but lasts 0.55 s and has very low onset flux (0.091).
- `17881481520942199957.mp4` (66.6%) and several old false positives make large jumps from near-silence but settle below the -6 dB event gate.

The user also reviewed the remaining original top-15 candidates as legitimate material: mic artifacts, poor-quality loud speech, rage screaming, an explosion, laughter, voice chat, and screaming girls. These labels reinforce the decision to model an unexpected hazardous transition—or an exceptionally damaging start—rather than treating “contains a scream” as the target. There are no provisional false positives from the new 80% decision rule.

## Browser implementation

The userscript uses one analysis worker-in-practice: a promise queue with concurrency 1. Downloads retain their existing concurrency, but downloaded `Blob` objects are not retained in the analysis backlog. After `cache.put()`, the one-slot worker reads each blob from CacheStorage only when its turn begins, so media is never downloaded again merely for analysis. Results remain in debounced, serialized `GM_getTab`/`GM_saveTab` state for the tab lifetime.

CacheStorage durability is reinforced without changing storage technology: Spokoyno checks origin persistence, offers a user-initiated persistence request, maintains small lease metadata inside each tab cache, gives absent tabs a one-hour orphan grace period, reconciles the physical cache after `pageshow`/focus/visibility resume, and requeues missing entries. Downloads validate HTTP completion and MP4/WebM/Ogg signatures before caching; interrupted downloads retry with mirror fallback and capped exponential backoff. Cache writes, reconciliation, and clearing are serialized, and clearing aborts active requests before waiting for in-flight work. A Tampermonkey menu command reports persistence, origin usage/quota, physical and indexed entry counts, origin changes, and queue health.

Community reports remain a separate boolean axis. The script scans `.post__message` text for `scream` or `скрим`, associates each mention with the nearest explicit `.post-reply-link[data-num]`, and ignores simple English/Russian negations. The report badge links back to the reporting reply and includes its text in a tooltip. Either a positive audio decision or a community report outlines the attachment red; report text never changes the audio percentage.

`decodeAudioData()` is the only broadly available way to obtain the entire PCM track faster than real time without shipping a demuxer/decoder. v5.5+ decodes at 16 kHz where supported and uses `OfflineAudioContext` for anti-aliased 16 kHz resampling otherwise; it no longer skips device-rate samples. The browser scans fixed 800-sample windows, yields during long timeline, candidate, and spectral-context scans, stores only small per-window arrays, and retains no PCM after the current result. Channel powers are combined independently for spectral evidence so antiphase stereo does not disappear. Radix-2 FFTs are limited to six opening windows plus at most two seconds of context around the winning transition; there is no full-track spectrogram allocation.

The v5.6 offline parity regression analyzed all 890 retained audio tracks. Confirmed positives #1–#6 score 88.8%, 89.1%, 88.2%, 90.2%, 85.5%, and 86.2%, retaining all six warnings. No warning occurs among 537 audio-bearing provisional negatives, and the 11 no-audio negatives remain nonwarnings. The two new warnings are exactly the newly labeled clips; no warning occurs among the remaining 347 explicitly unlabeled tracks. Because the rescue rules were selected after inspecting positives #5/#6, this is a regression result rather than held-out validation.

There are unavoidable limitations:

- Codec/container support differs by browser. A WebM/Opus or MP4/AAC file that `<video>` can play may still be rejected by `decodeAudioData()`. The script reports that as `audio analysis failed`, never as safe.
- `MediaElementAudioSourceNode` would analyze what a `<video>` plays, but only in real time and with autoplay/lifecycle constraints. `OfflineAudioContext` can render a buffer graph quickly, but it does not solve container demuxing before an `AudioBuffer` exists.
- WebCodecs accepts encoded chunks; using it for these MP4/WebM files also requires a container demuxer and has uneven browser availability. A large ffmpeg/WASM dependency is not justified here.
- `AudioBuffer` holds uncompressed float PCM, so a long clip can temporarily require hundreds of MiB even though Spokoyno avoids additional full-track copies. Sequential analysis bounds concurrency but cannot remove that decoder cost.
- CacheStorage is origin-specific. Canonical mirror keys deduplicate `2ch.org`, `2ch.su`, and `2ch.life` only inside the current origin's physical CacheStorage; the script does not claim cross-origin CacheStorage sharing.

## Reproduction

The extraction program is [analyze_audio.py](analyze_audio.py). [cache_audio.py](cache_audio.py) can rebuild ignored `../corpus/audio/` as an audio-only float32 corpus, [../corpus/labels.json](../corpus/labels.json) preserves the labels, and [evaluate_proposals.py](evaluate_proposals.py) reproduces the proposal experiment. They require FFmpeg, NumPy, and SciPy. The complete userscript is [../spokoyno.user.js](../spokoyno.user.js).
