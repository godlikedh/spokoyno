# Model delivery fix — 6.0.1

The reported Chrome error, `@resource: can't load ... spokoynoMLRuntime ... Failed to fetch`, occurs while Tampermonkey retrieves a dependency, before inference. Tampermonkey distinguishes this error from a checksum rejection. The same installation worked in the user's Mac/Brave environment; the exact network, permissions, or extension state responsible on the other PC is unknown. A successful local installation alone did not establish that dependency delivery would work there.

Version 6.0.1 removes all mandatory `@resource` and `@require` downloads. The identical executable runtime is embedded in a generated function inside the userscript. Workers execute that bundled function, and the main-thread fallback calls the same function directly. There is no downloaded JavaScript evaluation or replacement of the trained model.

Four model-data files are downloaded lazily using `GM_xmlhttpRequest`, with explicit `@connect` permissions for jsDelivr and GitHub. Both URLs pin the existing asset commit. Every download and cached value must match the existing SHA-256 hash; network failures, HTTP errors, and invalid hashes advance to the alternate host. Verified data is stored in a separate CacheStorage cache and shared by concurrent jobs in a page. Storage failure does not discard an otherwise verified download. Browser/site data clearing can require another download.

A 45-second timer bounds each host attempt and aborts silent requests. This is enforced by the script because Tampermonkey documents that Chrome's fetch-backed requests can ignore the native timeout, and anonymous requests select that mode. [Official GM request documentation](https://www.tampermonkey.net/documentation.php?ext=d2&q=GM_xmlhttpRequest)

If neither host supplies verified data, the result stays unknown with an actionable error. Resetting screamer results retries failed downloads. Analysis version 12 automatically retries results from 6.0.0 after upgrading, while retaining video downloads. The model, features, thresholds, and six-worker concurrency are unchanged.

## Verification

- The published jsDelivr copies of all four pinned data files returned HTTP 200 and matched their frozen hashes, including the 17.2 MB base64 weights file.
- Node tests cover primary-host network/HTTP/checksum failures, corrupt cache repair, offline cache reuse on a new page, cache-write rejection, silent timeout/abort, shared concurrent downloads, and all-host failure followed by retry.
- The full userscript browser check exercises all 27 preserved PCM fixtures with the primary host deliberately failing. Six actual workers reproduce the original scores with maximum error below 0.000000569 and zero tier changes. Offline cache reuse, cancellation, main-thread fallback, and corrupt downloads also pass. This run uses a GM HTTP shim; the actual Tampermonkey check is separate.
- Packaging tests prove the embedded runtime is byte-identical to the previously verified runtime, and frozen model/source hashes remain unchanged. The 4,014-recording numerical parity study from 6.0.0 remains applicable; it is not presented as a new inference run.
- All 85 Python tests and 26 Node tests passed with no skips locally. The actual Tampermonkey 5.5.1 check below also passed: failed primary connection, alternate download, offline page reload, and main-thread fallback all produced the same score.

The automated real-extension check is `research/check_ml_tampermonkey.cjs`: it creates a temporary Brave profile, installs the actual script with local test URLs/hooks, forcibly breaks the primary connection, verifies fallback, reloads with both model hosts offline, verifies persistent cache reuse, and runs the main-thread model too. It closes its browser without interactive stdin commands and does not change the personal profile. `SPOKOYNO_BROWSER` and `SPOKOYNO_TM_EXTENSION` can select another compatible Chromium binary and unpacked Tampermonkey installation. These checks do not claim reproduction of the other PC's exact configuration.

```sh
npm test
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
node research/check_ml_userscript_browser.cjs research/artifacts/browser-ml-fixtures-20261026 research/artifacts/ml-delivery-browser-new.json
node research/check_ml_tampermonkey.cjs research/artifacts/browser-ml-fixtures-20261026 research/artifacts/ml-delivery-tampermonkey-new.json
```

Compact release results and hashes are in [the delivery evidence](reviews/browser-model-delivery-20261027.json). The model's previously documented accuracy and codec/delay limitations are unchanged.
