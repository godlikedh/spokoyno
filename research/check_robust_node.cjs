// End-to-end portable physical-only heads on verified browser-decoded PCM.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { extract } = require('./robust_features.cjs');
const { predict } = require('./robust_model.cjs');
const [directory, encoderManifest, output] = process.argv.slice(2);
if (!output || fs.existsSync(output)) throw new Error('expected fixtures, DSP manifest, new output.json');
const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures.json')));
const dsp = JSON.parse(fs.readFileSync(encoderManifest));
const tier = (s) => (s >= 0.8 ? 2 : s >= 0.6 ? 1 : 0);
const results = [];
let maximum = 0;
for (const fixture of fixtures) {
  const bytes = fs.readFileSync(path.join(directory, fixture.pcm));
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), fixture.pcm_sha256);
  const pcm = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const bundle = JSON.parse(fs.readFileSync(path.join(directory, fixture.models)));
  const start = performance.now();
  const data = extract(pcm, fixture.channels, dsp.dsp_sos, false);
  assert.deepEqual(data.names, bundle.transform.names);
  const predictions = {};
  for (const [name, model] of Object.entries(bundle.models)) {
    assert.equal(model.audio, false, 'this check requires physical-only heads');
    const actual = predict(bundle.transform, model, data.physical, null).score;
    const expected = fixture.expected[name].score;
    const error = Math.abs(actual - expected);
    maximum = Math.max(maximum, error);
    assert(error < 0.001, `${fixture.file} ${name} diverged: ${error}`);
    assert.equal(tier(actual), tier(expected));
    predictions[name] = actual;
  }
  results.push({ file: fixture.file, milliseconds: performance.now() - start, predictions });
}
fs.writeFileSync(
  output,
  JSON.stringify(
    {
      fixtures: fixtures.length,
      maximum_score_error: maximum,
      execution: 'Node JavaScript on preserved browser PCM; this check does not launch a browser',
      results
    },
    null,
    2
  ) + '\n',
  { flag: 'wx' }
);
console.log(JSON.stringify({ fixtures: fixtures.length, maximum_score_error: maximum }));
