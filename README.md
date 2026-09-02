# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, chooses the fastest 2ch mirror with fallback, keeps the existing download-speed monitor, and adds an event-based screamer-risk badge beside each attachment.

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey.

The empirical detector study for thread `336185346` is in [research/REPORT.md](research/REPORT.md). The [complete dataset table](research/thread-336185346.md), [CSV](research/thread-336185346.csv), and [JSON](research/thread-336185346.json) cover all 550 video attachments.

All media analysis happens locally in the browser. A decode failure is displayed as unknown, never safe.
