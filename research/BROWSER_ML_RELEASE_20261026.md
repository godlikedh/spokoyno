# Browser ML release — 6.0.0

Version 6.0.0 replaces the userscript's handcrafted detector with the frozen **synthetic-mil-20pct** model from `models/acoustic-mil-20261025`. This is an implementation and deployment change, not a new training run. Human labels, frozen Python extractors, model coefficients, and evaluation splits are unchanged.

Install or update [spokoyno.user.js](https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js) through Tampermonkey, then reload the thread. The model is included through cached resources; there is no local service to configure.

## Browser behavior

The browser now computes the original event proposals, all 76 acoustic features, frozen YAMNet embeddings, the model's eight PCA context components, and its linear-softmax MIL recording score. The JavaScript port preserves raw PCM overshoot, the original channel/crop selection, opening-event behavior, and exact-length Hilbert-envelope processing. It does not substitute the later robust/context research pipeline, which has different features.

Scores are used directly: yellow at 0.6–below 0.8 and red at 0.8 or above. The tooltip identifies the model and lists its strongest proposed event times. Community reports remain independent. Analysis version 11 invalidates old heuristic scores while keeping downloaded media. Missing/corrupt resources, model failures, and clips below the 300 ms window are unknown, not safe.

Inference runs in reusable local workers using the existing `CONCURRENCY = 6`. Each worker initializes once, retains its own encoder, and uses one WASM thread. Cancellation terminates the worker and discards stale results. If worker infrastructure is unavailable, the same model runs through one shared, serialized main-thread fallback. This fallback can briefly affect UI responsiveness. Six workers and their decoded audio require more memory than one; this release does not establish a maximum memory bound for arbitrarily long attachments.

The unique runtime/model resources total **20,530,192 bytes**. Tampermonkey caches them at installation; both the main sandbox and worker use those resources. URLs are pinned to an immutable Git commit, with Tampermonkey SHA-256 integrity fragments and additional runtime checks. TFJS 4.22.0 and YAMNet Apache licenses are included in `ml/assets/`. Audio stays local.

## Verification

The compact [release evidence](reviews/browser-ml-20261026.json) records hashes and results. Large per-recording artifacts remain local.

| Check                                  | Result                                                                                           | Scope                                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Full retained corpus                   | 4,014 recordings; zero event-window, encoder-patch, or score-tier differences                    | JavaScript proposals, features and classifier versus frozen Python; classifier uses saved embeddings                         |
| Maximum corpus feature difference      | 0.00008708                                                                                       | Absolute numeric error across the 76 features                                                                                |
| Maximum corpus score difference        | 0.00000009248                                                                                    | Both thread-excluded and final-fit classifiers                                                                               |
| Complete userscript in Brave           | 27 preserved browser-PCM fixtures; maximum score difference below 0.000000569; zero tier changes | Fresh JavaScript CNN inference in six actual Blob workers; GM resource API shim                                              |
| Browser lifecycle/error checks         | Passed                                                                                           | Worker reuse/cancellation, same-model fallback, corrupt resource stays unknown, old scores invalidated                       |
| Actual Tampermonkey 5.5.1 installation | Passed; score difference 0.000000002664                                                          | Fresh isolated Brave profile; actual `@require`, `GM_getResourceText`, cached resources, and worker inference on one fixture |

The real Tampermonkey check installed the complete script with only local test URLs and an instrumentation hook changed. The model/runtime bytes and their integrity hashes were the release bytes. It did not modify the user's personal browser profile. This supplements the broader browser fixture run rather than claiming all 27 cases were tested through Tampermonkey itself.

All **85 Python tests and 16 Node tests** passed, with no skips on this machine, along with Prettier and Ruff checks. In the final 27-fixture browser run, per-recording inference took a median 481 ms and a maximum 1,179 ms with six concurrent slots. These timings include model/worker startup when needed but exclude downloading and decoding; they are measurements on this machine, not a cross-device performance guarantee.

An early corpus port check caught two roughness discrepancies at the exact 200 Hz cutoff in truncated bursts. Matching NumPy's frequency-grid arithmetic fixed them; the final entire-corpus run passes. Dedicated NumPy/SciPy tests cover this boundary, the Hilbert transform, resampling, and arbitrary-length FFTs. Packaging tests verify source/resource hashes and the concatenated runtime.

The 27-fixture browser check measures inference on preserved browser-decoded PCM, including worker startup and scoring, rather than fresh media downloads/decoding. Full-corpus verification does not rerun the CNN 4,014 times: it checks its input patches byte-for-byte, uses the frozen embeddings for classifier parity, and independently exercises the complete encoder on the browser fixtures. Neither type of parity check establishes generalization accuracy.

## Accuracy and remaining limits

The unchanged thread/family-excluded development predictions reproduce **30/30 red positive detections**, **34/3,466 false red alerts**, and **98/3,466 negative yellow-or-red warnings**. The 518 unknown groups are excluded. The shipped model is the final fit on available labeled data; rescoring those same recordings with it is not held-out evaluation.

These are the existing candidate's results, not a new independent test. Repeated model selection on this development corpus creates selection bias, and 30 positives remain a small sample. The earlier AAC stress test detected only 23/30 at red and 26/30 at either warning tier. Delay and level sensitivity also remain. This release makes the user's selected best measured model usable in the browser; it does not claim codec/delay invariance or perfect future recall. See [the acoustic study](ACOUSTIC_RESEARCH_20261025.md) and [later robustness research](ROBUST_AUDIO_RESEARCH_20261026.md) for comparisons and failed alternatives.

## Reproduction and release

From the repository root:

```sh
npm ci
npm test
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
npm run format:check
.venv/bin/ruff check research
.venv/bin/ruff format --check research
```

For full parity checks, use the existing local corpus and frozen research artifacts. Fixtures verify source, feature and model provenance; output paths must be new. Browser checks require Brave at the path used by the existing replay tools and permission to launch an isolated muted browser.

```sh
.venv/bin/python research/prepare_browser_ml.py --output-dir research/artifacts/ml-corpus-release
node research/check_browser_ml_features.cjs research/artifacts/ml-corpus-release research/artifacts/ml-corpus-release.json 4
.venv/bin/python research/prepare_browser_ml.py --browser --output-dir research/artifacts/ml-browser-release
node research/check_ml_userscript_browser.cjs research/artifacts/ml-browser-release research/artifacts/ml-browser-release.json
```

`research/build_browser_ml.py` packages the verified acoustic snapshot, the existing portable YAMNet export (`robust-portable-encoder-20261026-v3`), its original YAMNet license, and the pinned npm dependencies. The retained research artifacts are prerequisites for rebuilding; installation and regular Node tests use the committed assets directly. The builder verifies original extractor and encoder hashes and exports only the selected model.

After changing a production source, rebuild the assets, verify parity, and commit them first. Run `research/pin_browser_ml.py FULL_ASSET_COMMIT_SHA`, format the userscript, verify the final script, then commit it. This pins dependencies without a circular commit hash. Preserve that asset commit in history when merging. Verify the remote pinned resources before publishing the installation link.
