# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. On thread pages only, it preloads actual video attachments into a tab-local cache, chooses the fastest 2ch mirror with fallback, keeps the existing download-speed monitor, and adds screamer-risk signals beside each attachment.

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey.

The empirical detector study is in [research/REPORT.md](research/REPORT.md). The [proposal evaluation](research/PROPOSAL_EVALUATION.md) tests MAD-normalized contrast, merged calibration, 10 ms attack time, and temporal roughness. The [original labeled/provisional dataset](research/thread-336185346.md) covers 550 attachments, the [second corpus batch](research/thread-336272252.md) contains two confirmed screamers plus 347 clips for future review, the [new labeled batch](research/thread-336291305.md) adds four audio screamers, one visual-only screamer, and 342 negatives, and [supplemental-positives.md](research/supplemental-positives.md) records two other screamers including an immediate loud-start case.

All media analysis happens locally in the browser. Audio is decoded/resampled by Web Audio to a real 16 kHz analysis rate before the fixed-window features are calculated; raw device-rate decimation is not used. A decode, cache, or download failure is displayed as unknown, never safe.

The reusable local research corpus lives in ignored `corpus/audio/` as 16 kHz stereo float32 WAV audio only. It currently has 1,229 logical track names: 10 confirmed audio screamers, one visual-only screamer, 871 reviewed/provisional negative audio tracks, and 347 explicitly unlabeled tracks. Nineteen no-audio negative attachments exist only in the research manifests. Exact media hashes are reused before download, then identical WAV payloads are deduplicated with hard links; the 1,229 names currently occupy 1,192 physical payloads. Files retain the source media name plus `.audio.wav`, while the versioned label policy and manual reviews are in [corpus/labels.json](corpus/labels.json). No video payloads or corpus audio are committed.

The two warning axes are independent:

- Audio analysis shows a continuous heuristic risk score; it is not presented as a calibrated probability. Sub-percent scores retain decimal precision instead of being collapsed to `0/100` or `1/100`.
- A `🚩 reported` badge appears when a positive `scream`/`скрим` mention can be associated with a nearby quoted post. Simple negations such as “not a screamer” and “не скример” are excluded. This is an unverified community report, not part of the audio score.

An attachment is outlined red when either axis warns. If a quoted post contains multiple videos, the report is conservatively applied to every video in that post.

Cache durability measures include a persistence-status check, cache reconciliation when a tab resumes, validated mirror responses, capped exponential download retries, and a one-hour grace period before an apparently orphaned tab cache can be removed. The Tampermonkey menu contains cache repair/diagnostics and a user-initiated persistent-storage request. CacheStorage remains physically origin-specific, so changing a tab from one 2ch mirror hostname to another requires a new cache on that origin; cleanup also tracks the active origin explicitly.
