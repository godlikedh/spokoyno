# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. On thread pages only, it preloads actual video attachments into a tab-local cache, chooses the fastest 2ch mirror with fallback, keeps the existing download-speed monitor, and adds screamer-risk signals beside each attachment.

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey.

The empirical detector study is in [research/REPORT.md](research/REPORT.md). The [proposal evaluation](research/PROPOSAL_EVALUATION.md) tests MAD-normalized contrast, merged calibration, 10 ms attack time, and temporal roughness. The [labeled/provisional dataset](research/thread-336185346.md) covers 550 original video attachments, the [second corpus batch](research/thread-336272252.md) contains two confirmed screamers plus 347 clips for future review, and [supplemental-positives.md](research/supplemental-positives.md) records two other later screamers, including an immediate loud-start case.

All media analysis happens locally in the browser. Audio is decoded/resampled by Web Audio to a real 16 kHz analysis rate before the fixed-window features are calculated; raw device-rate decimation is not used. A decode, cache, or download failure is displayed as unknown, never safe.

The reusable local research corpus lives in ignored `corpus/audio/` as 16 kHz stereo float32 WAV audio only. It currently contains 890 tracks: 6 confirmed-positive tracks, 537 provisional-negative tracks, and 347 explicitly unlabeled tracks. Eleven no-audio provisional-negative attachments exist only in the research manifest. Files retain the source media name plus `.audio.wav`, while the versioned label policy and the user's manual hard-negative reviews are in [corpus/labels.json](corpus/labels.json). No video payloads or corpus audio are committed.

The two warning axes are independent:

- Audio analysis shows a continuous heuristic risk score; it is not presented as a calibrated probability. Sub-percent scores retain decimal precision instead of being collapsed to `0/100` or `1/100`.
- A `🚩 reported` badge appears when a positive `scream`/`скрим` mention can be associated with a nearby quoted post. Simple negations such as “not a screamer” and “не скример” are excluded. This is an unverified community report, not part of the audio score.

An attachment is outlined red when either axis warns. If a quoted post contains multiple videos, the report is conservatively applied to every video in that post.

Cache durability measures include a persistence-status check, cache reconciliation when a tab resumes, validated mirror responses, capped exponential download retries, and a one-hour grace period before an apparently orphaned tab cache can be removed. The Tampermonkey menu contains cache repair/diagnostics and a user-initiated persistent-storage request. CacheStorage remains physically origin-specific, so changing a tab from one 2ch mirror hostname to another requires a new cache on that origin; cleanup also tracks the active origin explicitly.
