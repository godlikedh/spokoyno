# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, chooses the fastest 2ch mirror with fallback, keeps the existing download-speed monitor, and adds an event-based screamer-risk badge beside each attachment.

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey.

The empirical detector study is in [research/REPORT.md](research/REPORT.md). The [complete thread dataset](research/thread-336185346.md) covers all 550 original video attachments; [supplemental-positives.md](research/supplemental-positives.md) records two later confirmed screamers, including an immediate loud-start case.

All media analysis happens locally in the browser. A decode failure is displayed as unknown, never safe.
