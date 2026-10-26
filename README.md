# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, selects the fastest 2ch mirror with fallback, shows download speed, estimates audio jump-scare risk, and adds a separate warning when replies report a screamer.

## Install

Open the [userscript installation link](https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js) ([CDN alternative](https://cdn.jsdelivr.net/gh/godlikedh/spokoyno@main/spokoyno.user.js)) and choose **Install** or **Update** in Tampermonkey, then reload the thread. Version **6.0.1** uses the frozen **synthetic-mil-20pct** ML model. It supports `2ch.org`, `2ch.su`, and `2ch.life`.

The executable ML runtime is bundled into the userscript. On first analysis, it downloads approximately 19 MB of model data from jsDelivr, with GitHub as a fallback, verifies SHA-256 hashes, and stores it in a persistent browser cache. Later page loads can reuse this cache without contacting either model host. No Python, local server, or separate model setup is needed. A score badge's tooltip identifies the model. Existing downloaded videos are retained; the update recomputes analysis results, including failed 6.0.0 attempts.

If both model hosts fail on first use, videos stay marked **unknown** and the tooltip identifies the failed downloads. After restoring access, use Tampermonkey's screamer-results reset menu to retry. Clearing site data removes that site's model cache. See the [6.0.1 delivery fix](research/BROWSER_MODEL_DELIVERY_20261027.md) for details.

All analysis stays in the browser. Decode, cache, and download failures are displayed as unknown rather than safe. CacheStorage is origin-specific, so caches are not physically shared between mirror hostnames.

## Research

- [Detector study](research/REPORT.md)
- [External proposal evaluation](research/PROPOSAL_EVALUATION.md)
- [ML playground](research/ML_PLAYGROUND.md)
- [Pretrained audio context, timing annotations, and known-screamer matching](research/AUDIO_EXPERIMENTS.md)
- [Current shadow-model card](research/models/MODEL_CARD.md)
- [Event-level experiments and score-tier results](research/EVENT_RESULTS.md)
- [Browser-decoding stability and full-corpus model comparison](research/SCORING_COMPARISON_20260921.md)
- [Reviewed-thread results and new detector misses (2026-09-30)](research/reviews/336991612.md)
- [Retrained models and new algorithm comparison (2026-09-30)](research/RETRAINING_20260930.md)
- [New reviewed thread and frozen browser/ML predictions (2026-10-13)](research/reviews/337355133.md)
- [Partial thread review, browser scores, and long-clip timecodes (2026-10-25)](research/reviews/337690329.md)
- [Retraining with 30 positive recordings (2026-10-25)](research/RETRAINING_20261025.md)
- [Event-hybrid research, nested validation, and exploratory rules plus MIL](research/EVENT_RESEARCH_20261025.md)
- [New acoustic features, synthetic training, and a stronger frozen candidate](research/ACOUSTIC_RESEARCH_20261025.md)
- [Codec/delay augmentation, synthetic recordings, and production-runtime research](research/ROBUST_AUDIO_RESEARCH_20261026.md)
- [Hard-negative training, 32 new sound-context features, and controlled comparisons](research/CONTEXT_RESEARCH_20261026.md)
- [Browser ML release, numerical parity, and installation verification](research/BROWSER_ML_RELEASE_20261026.md)
- [Model download recovery and bundled-runtime fix (6.0.1)](research/BROWSER_MODEL_DELIVERY_20261027.md)
- [Versioned label policy](corpus/labels.json)

The audio corpus and generated research artifacts are intentionally local. `corpus/audio/` contains 16 kHz WAV files, while `corpus/index.json` maps them to canonical media paths and hashes. Back up the entire ignored `corpus/` directory: expired imageboard media cannot necessarily be reconstructed from a fresh clone.

The userscript displays the model's continuous 0–1 evidence score directly: below 0.6 is low risk, 0.6–below 0.8 is a yellow **MAYBE**, and 0.8 or higher is a red alert. There is no heuristic cap or blending with the old rules. Very small nonzero scores use scientific notation. Scores are not calibrated probabilities. Decode/download/model failures and audio shorter than 300 ms remain unknown. Community reports remain separate red warnings and do not modify the audio score. Proposed event times are approximate candidates, not verified scream boundaries.

The selected model's existing thread/family-excluded development evaluation detects 30/30 positives with 34 false red alerts among 3,466 negatives. Repeated research on this corpus limits what these numbers prove about new threads. Known sensitivity to codecs, level changes, and added delays remains; this release ports the measured best candidate without retraining or claiming those problems are solved. See the release report for browser verification and the earlier studies for model comparisons.

Audio analysis reuses the existing `CONCURRENCY` setting (currently 6), independently of the download pool. Reusable local Web Workers run the feature extractor and YAMNet/ML inference; the browser handles audio decoding. Each worker retains its own initialized model. If workers are unavailable or blocked, the same model runs through a serialized main-thread fallback, which can briefly affect responsiveness. More parallel jobs use more memory. The panel shows active and queued audio jobs, and cache diagnostics report the analysis limit and worker/fallback mode.

The labeling workflow is: supply a reviewed thread and its screamer filenames; download its video attachments and label the other attachments in that snapshot as non-screamers. Exact sound timings can be supplied later. New posts outside the reviewed snapshot remain unlabeled. See the [workflow commands](research/ML_PLAYGROUND.md#1-extend-the-corpus).

Only `synthetic-mil-20pct` is shipped in the browser. Other research models and audio fingerprints remain offline experiments.

## Development checks

Run `npm ci` and `npm test` with Node 22 or newer. The optional retained-corpus regression skips when local audio is unavailable; synthetic-signal ML, worker scheduling, score tiers, and error behavior still run. Python research checks use `research/requirements-dev.txt` and `python -m unittest discover -s research -p 'test_*.py'`. Run `npm run format:check` and `ruff check research` for formatting/lint checks.

Production sources are in `ml/*.cjs`; `ml/assets/` contains generated, pinned release resources and the TensorFlow.js/YAMNet Apache licenses. `research/pin_browser_ml.py FULL_ASSET_COMMIT_SHA` embeds the verified runtime into the marked generated section of `spokoyno.user.js` and pins its model-data URLs. Edit the sources and rebuild instead of editing that generated section. The release reports describe building and verification.
