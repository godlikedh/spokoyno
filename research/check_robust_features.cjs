#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { extract } = require('./robust_features.cjs');
const directory = process.argv[2];
const manifest = JSON.parse(fs.readFileSync(process.argv[3]));
const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures.json')));
function floats(name) {
  const bytes = fs.readFileSync(path.join(directory, name));
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
let maximum = 0,
  compared = 0,
  failures = [],
  patchFailures = [],
  windowFailures = [],
  milliseconds = 0;
for (const fixture of fixtures) {
  const start = performance.now();
  const actual = extract(floats(fixture.pcm), fixture.channels, manifest.dsp_sos);
  milliseconds += performance.now() - start;
  assert.deepEqual(actual.names, fixture.names);
  if (JSON.stringify(actual.windows) !== JSON.stringify(fixture.windows)) windowFailures.push(fixture.file);
  const expectedPatches = floats(fixture.patches);
  let offset = 0,
    patchMax = 0;
  for (const patch of actual.patches)
    for (const v of patch) patchMax = Math.max(patchMax, Math.abs(v - expectedPatches[offset++]));
  if (offset !== expectedPatches.length || patchMax !== 0)
    patchFailures.push({ file: fixture.file, error: patchMax, samples: offset, expected: expectedPatches.length });
  if (actual.physical.length !== fixture.physical.length) {
    failures.push({ file: fixture.file, windows: actual.physical.length, expected: fixture.physical.length });
    continue;
  }
  for (let r = 0; r < actual.physical.length; r++)
    for (let c = 0; c < actual.names.length; c++) {
      const a = actual.physical[r][c],
        e = fixture.physical[r][c];
      if (e === null && Number.isNaN(a)) continue;
      const error = e === null || !Number.isFinite(a) ? Infinity : Math.abs(a - e);
      maximum = Math.max(maximum, error);
      compared++;
      if (error > 0.001)
        failures.push({ file: fixture.file, row: r, feature: actual.names[c], expected: e, actual: a, error });
    }
}
console.log(
  JSON.stringify(
    {
      fixtures: fixtures.length,
      values: compared,
      milliseconds,
      maximum_absolute_error: maximum,
      windowFailures,
      patchFailures,
      failures: failures.slice(0, 40),
      failure_count: failures.length
    },
    null,
    2
  )
);
assert.equal(failures.length, 0, 'feature parity failed');
assert.equal(patchFailures.length, 0, 'encoder patches differ');
assert.equal(windowFailures.length, 0, 'event times differ');
