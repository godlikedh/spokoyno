# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments, selects the fastest mirror with fallback, shows download speed, estimates audio jump-scare risk, and adds a separate warning when replies report a screamer.

## Install

Open the [userscript installation link](https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js) ([CDN alternative](https://cdn.jsdelivr.net/gh/godlikedh/spokoyno@main/spokoyno.user.js)) and choose **Install** or **Update** in Tampermonkey. Version **6.0.1** supports `2ch.org`, `2ch.su`, and `2ch.life`.

The runtime is bundled into the userscript. On first analysis it downloads approximately 19 MB of model data, verifies SHA-256 hashes, and caches it in the browser. jsDelivr is the primary host; GitHub is the fallback. Later page loads reuse the cache. No Python or local server is needed. All audio analysis stays in the browser.

Scores below 0.6 are low risk, 0.6–below 0.8 are yellow **MAYBE**, and 0.8 or higher are red alerts. These are evidence scores, not calibrated probabilities. Community reports produce independent warnings. Decode, download, and model failures, and audio shorter than 300 ms, remain **unknown**. If model downloads fail, restore access and use Tampermonkey's screamer-results reset menu to retry.

Audio analysis uses reusable workers with `CONCURRENCY = 6`; a serialized main-thread fallback is available. Each worker retains its own model, so increasing concurrency uses more memory. Caches belong to each mirror's origin; clearing site data removes them.

## Repository

| Path                                       | Contents                                                          |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `spokoyno.user.js`                         | Installable userscript, including the generated ML runtime        |
| `ml/*.cjs`                                 | Production feature extraction and inference sources               |
| `ml/assets/`                               | Pinned release model, encoder, runtime, and licenses              |
| `corpus/labels.json`, `corpus/events.json` | Human labels and event timings                                    |
| `research/`                                | Training, evaluation, export code, and tests                      |
| `research/models/`                         | Final model snapshots, checksums, licenses, and aggregate results |
| `research/artifacts/`                      | Ignored local intermediates and caches                            |

The shipped model is **synthetic-mil-20pct**. Its thread/family-excluded development evaluation detected **30/30 positives**, with **34 false red alerts among 3,466 negatives**. Repeated experiments on this corpus limit what those numbers establish about new threads. Codec, gain, and timing sensitivity remain. Other models are retained as offline results. See the [training guide and final results](research/README.md).

Keep final results and the code needed to reproduce them in Git. Put run plans, per-record predictions, audit dumps, review notes, browser fixtures, and reencodes under ignored `research/artifacts/`. Historical reports remain available in Git history. Model manifests retain training membership and inference checksums because scorers use them.

The local corpus is also ignored: `corpus/audio/`, `corpus/index.json`, and `corpus/manifests/`. **Back it up together.** Expired media may be impossible to download again. Identical corpus files share storage through hard links; their distinct paths preserve index references.

## Development

Use Node 22 or newer:

```sh
npm ci
npm test
npm run format:check
```

Python setup and training commands are in the [training guide](research/README.md). Checks:

```sh
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
.venv/bin/ruff check research
.venv/bin/ruff format --check research
```

Edit production sources in `ml/`, then [rebuild and pin the release assets](research/README.md#browser-release). The marked runtime section of the userscript is generated.
