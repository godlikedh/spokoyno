# Spokoyno

Spokoyno is a Tampermonkey companion for 2ch video threads. It preloads video attachments into a tab-local cache, selects the fastest 2ch mirror with fallback, shows download speed, estimates audio jump-scare risk, and adds a separate warning when replies report a screamer.

## Install

Install [spokoyno.user.js](spokoyno.user.js) in Tampermonkey. It supports `2ch.org`, `2ch.su`, and `2ch.life`.

All analysis stays in the browser. Decode, cache, and download failures are displayed as unknown rather than safe. CacheStorage is origin-specific, so caches are not physically shared between mirror hostnames.

## Research

- [Detector study](research/REPORT.md)
- [External proposal evaluation](research/PROPOSAL_EVALUATION.md)
- [ML playground](research/ML_PLAYGROUND.md)
- [Current shadow-model card](research/models/MODEL_CARD.md)
- [Versioned label policy](corpus/labels.json)

The audio corpus and generated research artifacts are intentionally local. `corpus/audio/` contains 16 kHz WAV files, while `corpus/index.json` maps them to canonical media paths and hashes. Back up the entire ignored `corpus/` directory: expired imageboard media cannot necessarily be reconstructed from a fresh clone.

Production remains on the v5.7 event detector. Exported models are experimental shadow scorers and do not change userscript warnings.
