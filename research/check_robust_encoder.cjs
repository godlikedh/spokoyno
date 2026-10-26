#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { logMel, FrozenEncoder } = require('./robust_encoder.cjs');

async function main() {
  const directory = process.argv[2],
    dependencies = path.resolve(process.argv[3]);
  const tf = require(path.join(dependencies, '@tensorflow/tfjs'));
  const wasm = require(path.join(dependencies, '@tensorflow/tfjs-backend-wasm'));
  wasm.setWasmPaths(path.join(dependencies, '@tensorflow/tfjs-backend-wasm/dist/'));
  await tf.setBackend('wasm');
  await tf.ready();
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'model.json')));
  const bytes = fs.readFileSync(path.join(directory, 'weights.bin'));
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), manifest.weights_sha256);
  const floats = (name) => {
    const b = fs.readFileSync(path.join(directory, name));
    return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  };
  const weights = floats('weights.bin'),
    raw = floats('fixture-patches.bin');
  const patches = Array.from({ length: 3 }, (_, i) => raw.subarray(i * 15600, (i + 1) * 15600));
  const mel = Float32Array.from(patches.flatMap((p) => [...logMel(p, manifest, weights)]));
  const comparison = (actual, expected) => {
    assert.equal(actual.length, expected.length);
    let max = 0,
      sq = 0;
    for (let i = 0; i < actual.length; i++) {
      assert(Number.isFinite(actual[i]));
      const diff = actual[i] - expected[i];
      max = Math.max(max, Math.abs(diff));
      sq += diff * diff;
    }
    return { maximum_absolute_error: max, rmse: Math.sqrt(sq / actual.length) };
  };
  const baseline = tf.memory().numTensors;
  const encoder = new FrozenEncoder(tf, manifest, weights);
  const started = performance.now();
  const embeddings = await encoder.encode(patches);
  const elapsed = performance.now() - started;
  const result = {
    backend: tf.getBackend(),
    milliseconds_for_three_patches: elapsed,
    logmel: comparison(mel, floats('fixture-logmel.bin')),
    embeddings: comparison(embeddings, floats('fixture-embeddings.bin'))
  };
  encoder.dispose();
  console.log(JSON.stringify(result, null, 2));
  assert.equal(tf.memory().numTensors, baseline, 'encoder leaked tensors');
  // Float64 JS FFT vs TensorFlow float32 FFT differs most in nearly empty
  // sine-wave bins after logarithm. Check both worst bin and aggregate error.
  assert(result.logmel.maximum_absolute_error < 0.002 && result.logmel.rmse < 0.0002, 'frontend diverged');
  assert(result.embeddings.maximum_absolute_error < 0.001, 'embedding diverged');
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
