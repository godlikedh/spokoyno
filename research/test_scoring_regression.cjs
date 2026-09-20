const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { test } = require('node:test');
const { ROOT, detector, analyze, readWav } = require('./userscript_harness.cjs');
const { summarize } = require('./evaluate_userscript.cjs');
const { floatWav } = require('./browser_replay.cjs');

test('corpus metrics exclude unknown labels and keep failed analysis unknown', () => {
  const rows = [
    { label: 'positive', result: { status: 'ok', riskTier: 'alert' } },
    { label: 'positive', result: { status: 'decode-error' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'alert' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'maybe' } },
    { label: 'negative', result: { status: 'ok', riskTier: 'low' } },
    { label: 'unlabeled', result: { status: 'ok', riskTier: 'alert' } }
  ];
  assert.deepEqual(summarize(rows, 'result'), {
    positive: 2,
    negative: 3,
    positiveRed: 1,
    positiveYellow: 0,
    negativeRed: 1,
    negativeYellow: 1,
    unknown: 1
  });
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

test('transition evidence uses bounded nearby spectral change, not only the selected frame', async () => {
  let seed = 42;
  const audio = Float32Array.from({ length: 80000 }, (_, i) => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return i < 32000 ? 0.002 * Math.sin(i * 0.1) : 0.95 * ((seed >>> 0) / 2147483648 - 1);
  });
  const result = await analyze(detector(), [audio, audio]);
  const sigmoid = (value) => 1 / (1 + Math.exp(-value));
  const base =
    sigmoid((result.eventDb + 10.5) / 2.5) ** 0.9 *
    sigmoid((result.jumpDb - 13) / 3.5) ** 1.25 *
    sigmoid((result.eventDuration - 0.18) / 0.09) ** 0.65 *
    (0.87 +
      0.1 * sigmoid((-result.baselineDb - 15) / 5) +
      0.03 * sigmoid((result.eventNearClipPct / 100 - 0.005) / 0.012));
  assert.ok(result.spectralFluxNear > result.spectralFlux);
  assert.ok(
    Math.abs(result.transitionConfidence - base * (0.65 + 0.35 * sigmoid((result.spectralFluxNear - 0.22) / 0.08))) <
      1e-12
  );
});

test('reported browser-decoded screamer alerts across sub-window timing shifts', async (t) => {
  const file = path.join(ROOT, 'research/artifacts/browser-nearby-20260921.json.wav');
  if (!fs.existsSync(file)) return t.skip('Original-media browser regression audio is local-only');
  const audio = readWav(file);
  const baseline = detector(
    execFileSync('git', ['show', '5f32be88c3dfe0ee0601252c34cb68562b2ec79d:spokoyno.user.js'], {
      cwd: ROOT,
      encoding: 'utf8'
    })
  );
  const old = await analyze(baseline, audio);
  assert.equal(old.riskTier, 'maybe');
  assert.ok(Math.abs(old.score - 0.6664) < 0.001);
  for (const shift of [-400, -200, 0, 200, 400]) {
    const shifted = audio.map((channel) => {
      if (shift <= 0) return channel.subarray(-shift);
      const result = new Float32Array(channel.length + shift);
      result.set(channel, shift);
      return result;
    });
    const current = await analyze(detector(), shifted);
    assert.equal(current.riskTier, 'alert', `shift ${shift / 16}ms: ${current.score}`);
  }
});
