# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, chooses the fastest 2ch mirror with fallback, keeps the existing download-speed monitor, and adds screamer-risk signals beside each attachment.

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey.

The empirical detector study is in [research/REPORT.md](research/REPORT.md). The [proposal evaluation](research/PROPOSAL_EVALUATION.md) tests MAD-normalized contrast, merged calibration, 10 ms attack time, and temporal roughness. The [complete thread dataset](research/thread-336185346.md) covers all 550 original video attachments; [supplemental-positives.md](research/supplemental-positives.md) records two later confirmed screamers, including an immediate loud-start case.

All media analysis happens locally in the browser. A decode failure is displayed as unknown, never safe.

The reusable local research corpus lives in ignored `corpus/audio/` as 16 kHz stereo float32 WAV audio only. Files retain the source media name plus `.audio.wav`, while the versioned label policy and the user's manual hard-negative reviews are in [corpus/labels.json](corpus/labels.json). No video payloads or corpus audio are committed.

The two warning axes are independent:

- Audio analysis shows a continuous heuristic risk score; it is not presented as a calibrated probability.
- A `🚩 reported` badge appears when a reply containing `scream` or `скрим` directly quotes the attachment's post. This is an unverified community report, not part of the audio score.

An attachment is outlined red when either axis warns. If a quoted post contains multiple videos, the report is conservatively applied to every video in that post.

Cache durability measures include an automatic persistent-storage request, cache reconciliation when a tab resumes, capped exponential download retries, and a one-hour grace period before an apparently orphaned tab cache can be removed. The Tampermonkey menu contains cache repair/diagnostics and a manual persistent-storage request. CacheStorage remains physically origin-specific, so changing a tab from one 2ch mirror hostname to another requires a new cache on that origin.
