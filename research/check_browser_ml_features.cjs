'use strict';
const fs = require('node:fs'),
  path = require('node:path'),
  crypto = require('node:crypto');
const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { extract } = require('../ml/features.cjs'),
  { predict } = require('../ml/model.cjs');
const { readWav } = require('./userscript_harness.cjs');
const tier = (s) => (s >= 0.8 ? 2 : s >= 0.6 ? 1 : 0);
if (!isMainThread) {
  const directory = workerData.directory,
    modelBundle = JSON.parse(fs.readFileSync(path.join(__dirname, '../ml/assets/model.json')));
  const final = JSON.parse(fs.readFileSync(path.join(directory, 'model-final.json'))),
    models = new Map();
  const fd = fs.openSync(path.join(directory, 'embeddings.bin'), 'r');
  parentPort.on('message', (fixture) => {
    try {
      if (!models.has(fixture.model))
        models.set(fixture.model, JSON.parse(fs.readFileSync(path.join(directory, fixture.model))));
      if (crypto.createHash('sha256').update(fs.readFileSync(fixture.source)).digest('hex') !== fixture.source_sha256)
        throw new Error('PCM checksum changed');
      const channels = readWav(fixture.source),
        pcm = new Float32Array(channels[0].length * channels.length);
      for (let c = 0; c < channels.length; c++)
        for (let i = 0; i < channels[c].length; i++) pcm[i * channels.length + c] = channels[c][i];
      const start = performance.now(),
        data = extract(pcm, channels.length, modelBundle.dsp);
      const bytes = Buffer.alloc(fixture.windows.length * 3072 * 4);
      fs.readSync(fd, bytes, 0, bytes.length, fixture.embedding_offset);
      const audio = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
      const errors = [];
      if (JSON.stringify(data.names) !== JSON.stringify(fixture.names)) throw new Error('feature names differ');
      const windowError = JSON.stringify(data.windows) !== JSON.stringify(fixture.windows);
      const hash = crypto.createHash('sha256');
      for (const patch of data.patches) hash.update(new Uint8Array(patch.buffer, patch.byteOffset, patch.byteLength));
      const patchError = hash.digest('hex') !== fixture.patch_sha256;
      let maximum = 0;
      for (let r = 0; r < data.physical.length; r++)
        for (let c = 0; c < data.names.length; c++) {
          const actual = data.physical[r][c],
            expected = fixture.physical[r]?.[c];
          if (expected === null && Number.isNaN(actual)) continue;
          const error = expected === null || !Number.isFinite(actual) ? Infinity : Math.abs(expected - actual);
          maximum = Math.max(maximum, error);
          if (error > 0.001) errors.push({ row: r, feature: data.names[c], expected, actual, error });
        }
      const scored = {};
      let scoreError = 0,
        tierChanges = 0;
      for (const [name, model] of [
        ['heldout', models.get(fixture.model)],
        ['final', final]
      ]) {
        const score = predict(model, data.physical, audio, data.names).score;
        scored[name] = score;
        scoreError = Math.max(scoreError, Math.abs(score - fixture.expected[name]));
        tierChanges += tier(score) !== tier(fixture.expected[name]) ? 1 : 0;
      }
      parentPort.postMessage({
        index: fixture.index,
        file: fixture.file,
        label: fixture.label,
        maximum,
        windowError,
        patchError,
        errors,
        scoreError,
        tierChanges,
        scores: scored,
        milliseconds: performance.now() - start
      });
    } catch (error) {
      parentPort.postMessage({ index: fixture.index, file: fixture.file, error: String(error.stack || error) });
    }
  });
} else {
  const [directory, output, jobs = '4'] = process.argv.slice(2);
  if (!output || fs.existsSync(output)) throw new Error('pass fixtures directory, new output.json, optional jobs');
  const fixtures = fs
    .readFileSync(path.join(directory, 'fixtures.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((s) => JSON.parse(s));
  let next = 0,
    finished = 0;
  const results = [];
  const workers = Array.from(
    { length: Math.min(+jobs, fixtures.length) },
    () => new Worker(__filename, { workerData: { directory } })
  );
  for (const worker of workers) {
    worker.on('error', (error) => {
      console.error(error);
      process.exit(1);
    });
    worker.on('message', (result) => {
      results.push(result);
      finished++;
      if (finished % 100 === 0 || result.error)
        console.log(`verified ${finished}/${fixtures.length}${result.error ? ' ' + result.error : ''}`);
      if (next < fixtures.length) worker.postMessage(fixtures[next++]);
      else worker.terminate();
      if (finished === fixtures.length) {
        results.sort((a, b) => a.index - b.index);
        const failures = results.filter(
          (r) => r.error || r.errors.length || r.patchError || r.windowError || r.scoreError > 1e-5 || r.tierChanges
        );
        const summary = {
          fixtures: results.length,
          maximum_feature_error: Math.max(...results.map((r) => r.maximum || 0)),
          maximum_score_error: Math.max(...results.map((r) => r.scoreError || 0)),
          failures: failures.length,
          window_failures: results.filter((r) => r.windowError).length,
          patch_failures: results.filter((r) => r.patchError).length,
          tier_changes: results.reduce((s, r) => s + (r.tierChanges || 0), 0)
        };
        fs.writeFileSync(output, JSON.stringify({ summary, results }, null, 2) + '\n', { flag: 'wx' });
        console.log(JSON.stringify(summary));
        if (failures.length) {
          console.log(JSON.stringify(failures.slice(0, 3), null, 2));
          process.exitCode = 1;
        }
      }
    });
    worker.postMessage(fixtures[next++]);
  }
}
