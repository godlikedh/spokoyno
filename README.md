# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments, selects the fastest mirror with fallback, shows download speed, estimates audio jump-scare risk, and adds a separate warning when replies report a screamer.

## Install

Open the [userscript installation link](https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js) ([CDN alternative](https://cdn.jsdelivr.net/gh/godlikedh/spokoyno@main/spokoyno.user.js)) and choose **Install** or **Update** in Tampermonkey. Version **6.0.1** supports `2ch.org`, `2ch.su`, and `2ch.life`.

The runtime is bundled into the userscript. On first analysis it downloads approximately 19 MB of model data, verifies SHA-256 hashes, and caches it in the browser. jsDelivr is the primary host; GitHub is the fallback. Later page loads reuse the cache. No Python or local server is needed. All audio analysis stays in the browser.

Scores below 0.6 are low risk, 0.6–below 0.8 are yellow **MAYBE**, and 0.8 or higher are red alerts. These are evidence scores, not calibrated probabilities. Community reports produce independent warnings. Decode, download, and model failures, and audio shorter than 300 ms, remain **unknown**. If model downloads fail, restore access and use Tampermonkey's screamer-results reset menu to retry.

Audio analysis uses reusable workers with `CONCURRENCY = 6`; a serialized main-thread fallback is available. Each worker retains its own model, so increasing concurrency uses more memory. Caches belong to each mirror's origin; clearing site data removes them.

## Repository

| Path                   | Purpose                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `spokoyno.user.js`     | Installable userscript                                             |
| `ml/`                  | Browser inference source and the shipped model in `ml/assets/`     |
| `training/`            | Local dataset workflow, features, model families, and augmentation |
| `training/results.csv` | Compact comparison of completed experiments                        |
| `tools/`               | Browser builds, exports, and integration checks                    |
| `tests/`               | Synthetic unit tests and optional local-corpus regression          |

The shipped model is **synthetic-mil-20pct**. Its source-thread/family-excluded development evaluation detected **30/30 positives**, with **34 false red alerts among 3,466 negatives**. Repeated experimentation on this corpus limits what those numbers establish about new threads; codec, gain, and timing sensitivity remain. See the [training workflow and experiment results](training/README.md).

## Local training

The two normal workflows are:

```sh
# Import the reviewed snapshot, label the named screamers, and retrain.
.venv/bin/python -m training add https://2ch.life/b/res/THREAD.html \
  --screamer FIRST.mp4 --screamer SECOND.webm --train

# Add user-provided timings, then compare models using the updated annotations.
.venv/bin/python -m training timings FIRST.mp4 --interval 7.76:10.42
.venv/bin/python -m training train
```

Use `python -m training labels` to list every confirmed screamer, its timings, and whether its audio is available. Setup and optional feature/augmentation experiments are described in the [training guide](training/README.md).

**The entire corpus stays local**, including labels, timings, audio, saved thread snapshots, and the index. Dead thread URLs cannot reconstruct it. Back up `corpus/` together; restore that backup or collect your own reviewed dataset to train. Historical model snapshots remain local in `research/models/`. New runs and generated caches go under ignored `artifacts/`.

Git contains the shipped model and a compact results comparison. Run plans, per-record predictions, generated media, and historical models are local working data.

## Development

Use Node 22 or newer:

```sh
npm ci
npm test
npm run format:check
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/ruff check training tools tests
.venv/bin/ruff format --check training tools tests
```

Tests use synthetic fixtures; the optional corpus regression skips when local audio is absent. Rebuild the browser runtime with `python -m tools.build_browser_ml`. It uses the committed model and npm dependencies, so it works without the local corpus or historical training runs. See [release instructions](training/README.md#browser-release) before publishing new assets.
