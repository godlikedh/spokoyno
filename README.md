# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, selects the fastest 2ch mirror with fallback, shows download speed, estimates audio jump-scare risk, and adds a separate warning when replies report a screamer.

## Install

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey. It supports `2ch.org`, `2ch.su`, and `2ch.life`.

All analysis stays in the browser. Decode, cache, and download failures are displayed as unknown rather than safe. CacheStorage is origin-specific, so caches are not physically shared between mirror hostnames.

## Research

- [Detector study](research/REPORT.md)
- [External proposal evaluation](research/PROPOSAL_EVALUATION.md)
- [ML playground](research/ML_PLAYGROUND.md)
- [Pretrained audio context, timing annotations, and known-screamer matching](research/AUDIO_EXPERIMENTS.md)
- [Current shadow-model card](research/models/MODEL_CARD.md)
- [Event-level experiments and score-tier results](research/EVENT_RESULTS.md)
- [Versioned label policy](corpus/labels.json)

The audio corpus and generated research artifacts are intentionally local. `corpus/audio/` contains 16 kHz WAV files, while `corpus/index.json` maps them to canonical media paths and hashes. Back up the entire ignored `corpus/` directory: expired imageboard media cannot necessarily be reconstructed from a fresh clone.

The v5.8 userscript preserves the v5.7 detector's red decisions and exposes one 0–1 decision score: below 0.6 is low risk, 0.6–below 0.8 is a yellow **MAYBE**, and 0.8 or higher is a red alert. This is heuristic evidence, not a calibrated probability. Decode/download failures remain unknown. Community reports remain separate red warnings and do not modify the audio score. Old audio-analysis results are invalidated on upgrade; downloaded media is retained.

The labeling workflow is: supply a reviewed thread and its screamer filenames; download its video attachments and label the other attachments in that snapshot as non-screamers. Exact sound timings can be supplied later. New posts outside the reviewed snapshot remain unlabeled. See the [workflow commands](research/ML_PLAYGROUND.md#1-extend-the-corpus).

Exported ML models and audio fingerprints remain offline shadow experiments. They are not embedded into the userscript or allowed to create browser warnings.
