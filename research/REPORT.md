# Spokoyno screamer-detector research

## Corpus and labels

The 2ch API response for thread `336185346` contained 550 video attachments: 417 MP4 and 133 WebM files, 548 unique MD5 values, about 3.47 GiB, and about 8.5 hours of media. Every attachment was downloaded from `2ch.su` with mirror fallback available. The hostname is not part of media identity; the dataset retains `/b/src/336185346/<file>` as the key.

FFmpeg decoded audio from 539 files: 413 AAC, 93 Vorbis, 31 Opus, and 2 MP3 tracks. Eleven files had no audio stream. The two user-confirmed screamers are positives; following the user's clarification, the other 548 attachments are provisional negatives.

The complete sorted table is in [thread-336185346.md](thread-336185346.md). Machine-readable versions, including every requested feature, are [thread-336185346.csv](thread-336185346.csv) and [thread-336185346.json](thread-336185346.json).

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

## Findings

| file | label | median dB | peak dBFS | event s | baseline dB | event dB | jump dB | duration s | near-clip % | onset flux | score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17883557324650588814.webm | confirmed #1 | -43.96 | +24.73 | 15.85 | -45.03 | +18.18 | +63.21 | 1.95 | 98.09 | 0.308 | 91.2% |
| 17883557325462786367.mp4 | confirmed #2 | -23.37 | +2.74 | 7.95 | -24.53 | -0.66 | +23.87 | 2.45 | 52.76 | 0.410 | 89.1% |
| 17882984411551286220.webm | provisional negative | -32.10 | +0.50 | 76.65 | -43.70 | -6.35 | +37.35 | 0.40 | 2.97 | 0.384 | 77.4% |
| 17881699236063425442.mp4 | provisional negative | -12.73 | +2.68 | 13.30 | -58.10 | -0.60 | +57.50 | 1.55 | 55.16 | 0.138 | 73.0% |
| 17881887457541053716.mp4 | provisional negative | -8.64 | +1.90 | 16.05 | -25.51 | -5.76 | +19.75 | 3.00 | 27.45 | 0.448 | 72.2% |
| 17881481520942199957.mp4 | provisional negative | -35.48 | +0.38 | 18.10 | -62.70 | -7.81 | +54.89 | 0.50 | 7.74 | 0.276 | 66.6% |
| 17882990553793679533.webm | provisional negative | -12.26 | +2.68 | 65.90 | -37.88 | -3.98 | +33.90 | 0.55 | 30.86 | 0.091 | 65.5% |

The positives are not merely loud. Each has a sustained, near-full-scale section after an ordinary/quiet local baseline and a changed onset spectrum. Positive #1 is extreme: its decoded lossy PCM overshoots nominal full scale heavily. Positive #2 is the more useful training example because its absolute levels overlap hard negatives; persistence and onset spectral change provide the separation.

Spectral flatness, zero-crossing rate, low/mid/high band jumps, brightness change, global crest factor, and maximum derivative were retained in the research dataset but did not add stable separation beyond the local envelope plus normalized onset flux. They are not part of the browser score. This avoids paying for complex features that the corpus did not justify.

## Old detector

The old rule detected both positives but also flagged 36 of 537 decodable provisional negatives. On these provisional labels that is 100% recall, 5.3% precision, and a 6.7% false-positive rate. Its whole-file peak and median mix unrelated parts of a clip, its “peak” is actually maximum 100 ms RMS, and its one-second baseline allows short impulses or scene cuts to supply independent score points. A file can receive three points without containing one coherent dangerous event.

Old false-positive filenames (all have `old_classification=suspicious` in the CSV):

`17882984411551286220.webm`, `17881481520942199957.mp4`, `17883361674593313438.mp4`, `17882989063041961052.webm`, `17881818515883672838.mp4`, `17882962082822601629.mp4`, `17882991086611750790.webm`, `17882997637612360215.webm`, `17881465969503404529.mp4`, `17881525425650729167.mp4`, `17881898673292643739.webm`, `17881476531723610876.mp4`, `17881842513483638014.mp4`, `17881869854811113740.mp4`, `17882986903210194666.webm`, `17881844799460675138.mp4`, `17881525427333106028.mp4`, `17883360944451558085.mp4`, `17882990552741630557.webm`, `17882990553482999670.webm`, `17882993665621506103.webm`, `17883011960512230584.mp4`, `17883249492281720284.mp4`, `17883364449832636829.mp4`, `17881806178130843016.mp4`, `17881476139173281009.mp4`, `17881514246931196290.mp4`, `17881518395660582444.webm`, `17881910378980818126.webm`, `17882548075240997522.mp4`, `17882518681650361236.mp4`, `17881514247692206034.mp4`, `17881833404270949457.mp4`, `17883362821053253349.mp4`, `17881476138422688186.mp4`, `17883395202170259937.mp4`.

## Final score and decision

Let `S(x) = 1 / (1 + exp(-x))`. Components are:

```text
loud     = S((eventDb + 10.5) / 2.5)
jump     = S((jumpDb - 13) / 3.5)
duration = S((durationSeconds - 0.18) / 0.09)
quiet    = S((-baselineDb - 15) / 5)
clip     = S((nearClipFraction - 0.005) / 0.012)
flux     = S((spectralFlux - 0.22) / 0.08)

confidence = loud^0.9 * jump^1.25 * duration^0.65
             * (0.87 + 0.10*quiet + 0.03*clip)
             * (0.65 + 0.35*flux)
```

The badge says `SCREAMER` only when all of these hold:

```text
confidence >= 0.80
eventDb >= -6
jumpDb >= 14
duration >= 0.15 s
```

The hard gates keep the continuous score honest: a large rise from digital silence to a moderate sound, or a spectral change in already-loud material, cannot be promoted by unrelated bonuses.

This rule flags exactly the two confirmed positives in this corpus. That is a calibration result, not an estimate of real-world accuracy: there are only two positive examples, and the closest provisional negative is only 2.6 score points below the threshold.

## Difficult negatives and labeling priorities

- `17882984411551286220.webm` (77.4%) is the closest negative. It has a short 37 dB broadband transition, but the sustained event is -6.35 dB and only 0.40 s. It should be the first manual re-check.
- `17881699236063425442.mp4` (73.0%) is the hardest envelope-only case: -58.1 to -0.6 dB for 1.55 s with 55% near-clipping. Its onset flux is only 0.138, consistent with an existing sound becoming louder rather than a new scream/noise spectrum appearing.
- `17881887457541053716.mp4` (72.2%) is loud overall (median -8.64 dB). Its local jump is only 19.75 dB, so the detector does not confuse high program loudness with surprise as readily.
- `17882990553793679533.webm` (65.5%) reaches -3.98 dB after a 33.9 dB jump, but lasts 0.55 s and has very low onset flux (0.091).
- `17881481520942199957.mp4` (66.6%) and several old false positives make large jumps from near-silence but settle below the -6 dB event gate.

There are no provisional false positives from the new 80% decision rule and no additional files above threshold. The files above are still useful active-learning candidates because their audio structures are closest to the positives.

## Browser implementation

The userscript uses one analysis worker-in-practice: a promise queue with concurrency 1. Downloads retain their existing concurrency. A just-downloaded `Blob` is passed directly to analysis; after reload the same blob is read from CacheStorage. It is never downloaded again for analysis. Results remain in `GM_getTab`/`GM_saveTab` state for the tab lifetime.

`decodeAudioData()` is the only broadly available way to obtain the entire PCM track faster than real time without shipping a demuxer/decoder. The browser implementation scans the whole timeline at an effective rate near 16 kHz, yields to the page every 100 windows, stores only small per-window arrays, and retains no PCM after the current result. Its K-like filters run during that scan. A radix-2 FFT is calculated only for the winning event and the preceding 50 ms, avoiding full-track spectrogram memory and CPU.

A direct Node harness ran the final JavaScript signal path against FFmpeg-decoded 48 kHz PCM. It scored the confirmed positives 87.5% and 87.2%; the four closest tested negatives scored 75.3%, 71.5%, 70.8%, and 64.8%. Event times and classifications matched the offline extractor. Small score differences come from resampling and FFT-bin resolution, which is why the 80% decision threshold is shared but displayed confidence is allowed to differ slightly.

There are unavoidable limitations:

- Codec/container support differs by browser. A WebM/Opus or MP4/AAC file that `<video>` can play may still be rejected by `decodeAudioData()`. The script reports that as `audio analysis failed`, never as safe.
- `MediaElementAudioSourceNode` would analyze what a `<video>` plays, but only in real time and with autoplay/lifecycle constraints. `OfflineAudioContext` can render a buffer graph quickly, but it does not solve container demuxing before an `AudioBuffer` exists.
- WebCodecs accepts encoded chunks; using it for these MP4/WebM files also requires a container demuxer and has uneven browser availability. A large ffmpeg/WASM dependency is not justified here.
- `AudioBuffer` holds uncompressed float PCM, so a long clip can temporarily require hundreds of MiB even though Spokoyno avoids additional full-track copies. Sequential analysis bounds concurrency but cannot remove that decoder cost.
- CacheStorage is origin-specific. Canonical mirror keys deduplicate `2ch.org`, `2ch.su`, and `2ch.life` only inside the current origin's physical CacheStorage; the script does not claim cross-origin CacheStorage sharing.

## Reproduction

The extraction program is [analyze_audio.py](analyze_audio.py). It requires FFmpeg, NumPy, and SciPy. The complete userscript is [../spokoyno.user.js](../spokoyno.user.js).
