const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');
const { ROOT, detector, analyze, readWav } = require('./userscript_harness.cjs');
const { summarize } = require('./evaluate_userscript.cjs');
const { floatWav } = require('./browser_replay.cjs');

test('corpus metrics exclude unknown labels and keep failed analysis unknown', () => {
  const rows = [
    { label: 'positive', result: { status: 'ok', riskTier: 'alert' } },
    { label: 'positive', result: { status: 'ok', riskTier: 'maybe' } },
    { label: 'positive', result: { status: 'decode-error' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'alert' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'maybe' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'low' } },
    { label: 'unlabeled', result: { status: 'ok', riskTier: 'alert' } }
  ];
  assert.deepEqual(summarize(rows, 'result'), {
    positive: 3,
    negative: 3,
    positiveRed: 1,
    positiveYellow: 1,
    negativeRed: 1,
    negativeYellow: 1,
    unknown: 1
  });
});

test('the fixed pre-336991612 positive regression set retains its red alerts', async (t) => {
  // Freeze the earlier accuracy expectations instead of asserting that every
  // future positive label is already detected. New misses belong in evaluations.
  const files = [
    '17883557324650588814.webm',
    '17883557325462786367.mp4',
    '17883629069140053716.mp4',
    '17883659327260384359.webm',
    '17884174673280229863.mp4',
    '17884274747240014140.mp4',
    '17885024222391568056.mp4',
    '17885460222141837902.mp4',
    '17885474189950590427.webm',
    '17885487431220276835.mp4',
    '17898262773380676776.webm'
  ];
  const paths = files.map((file) => path.join(ROOT, 'corpus/audio', `${file}.audio.wav`));
  if (paths.some((file) => !fs.existsSync(file))) return t.skip('Retained audio corpus is local-only');
  const engine = detector();
  for (const file of paths) {
    const result = await analyze(engine, readWav(file));
    assert.equal(result.status, 'ok', file);
    assert.equal(result.riskTier, 'alert', file);
  }
});

test('browser replay retains float overshoot and stereo layout without normalization', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'spokoyno-wave-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const pcm = Buffer.alloc(16);
  [20, -20, 0.25, -0.5].forEach((value, index) => pcm.writeFloatLE(value, index * 4));
  const file = path.join(directory, 'test.wav');
  fs.writeFileSync(file, floatWav(pcm, 16000, 2));
  const channels = readWav(file);
  assert.deepEqual(Array.from(channels[0]), [20, 0.25]);
  assert.deepEqual(Array.from(channels[1]), [-20, -0.5]);
});
