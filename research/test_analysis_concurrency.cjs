const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { Worker: NodeWorker } = require('node:worker_threads');
const { ROOT, SOURCE, detector, analyze, readWav } = require('./userscript_harness.cjs');

const turn = () => new Promise((resolve) => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function queueHarness(options = {}) {
  const source = SOURCE.slice(
    SOURCE.indexOf('  function queueScreamerAnalysis'),
    SOURCE.indexOf('  function discover')
  );
  return new Function(
    'options',
    `
    const CONCURRENCY = options.limit || 3;
    const analysisQueued = new Map(), analysisQueue = [], activeAnalyses = new Map(), analysisWorkers = new Map();
    const analysisResults = new Map(), cached = new Set(), seenMedia = new Map(), published = [], retries = [];
    let analysisGeneration = 0, tabStateDirty = false;
    const screamerKey = url => url;
    const cache = options.cache || {match: async key => ({blob: async () => ({key}), headers: {get: () => 'video/webm'}})};
    const analyzeScreamer = options.analyze || (async () => ({status:'ok', score:.2}));
    const identifyMediaBlob = options.identify || (async () => {});
    const withCacheLock = options.lock || (fn => fn());
    const publishScreamerResult = (sk, result) => {published.push({sk,result}); analysisResults.set(sk,result);};
    const enqueueMedia = (...args) => retries.push(args);
    const pump = () => {}, scheduleUI = () => {}, stopAnalysisWorker = () => {};
    const flushTabState = async () => {};
    const console = {warn: () => {}};
    ${source}
    return {enqueue: queueScreamerAnalysis, reset: resetAnalysisQueue, active: activeAnalyses,
      pending: analysisQueued, queue: analysisQueue, published, cached, seenMedia, retries};
  `
  )(options);
}

test('analysis uses the configured concurrency and suppresses active/queued duplicates', async () => {
  for (const limit of [1, 3, 6]) {
    const started = [],
      jobs = [];
    const harness = queueHarness({
      limit,
      analyze: async (blob) => {
        started.push(blob.key);
        const job = deferred();
        jobs.push(job);
        return job.promise;
      }
    });
    for (let i = 0; i < 9; i++) {
      harness.enqueue(String(i), String(i));
      harness.enqueue(String(i), String(i));
    }
    await turn();
    assert.equal(started.length, limit);
    assert.equal(harness.active.size, limit);
    harness.enqueue('0', '0');
    for (let i = 0; i < 9; i++) {
      jobs[i].resolve({ status: 'ok', score: 0.2 });
      await turn();
      assert.ok(harness.active.size <= limit);
    }
    assert.equal(started.length, 9);
    assert.equal(new Set(started).size, 9);
    assert.equal(harness.published.length, 9);
    assert.equal(harness.active.size, 0);
    assert.equal(harness.pending.size, 0);
  }
});

test('reset discards stale results without exceeding the limit or deleting a new duplicate marker', async () => {
  const jobs = [];
  const harness = queueHarness({
    limit: 1,
    analyze: async () => {
      const job = deferred();
      jobs.push(job);
      return job.promise;
    }
  });
  harness.enqueue('clip', 'clip');
  await turn();
  harness.reset();
  harness.enqueue('clip', 'clip');
  assert.equal(harness.active.size, 1);
  assert.equal(harness.queue.length, 1);
  jobs[0].resolve({ status: 'ok', score: 0.9 });
  await turn();
  assert.equal(harness.published.length, 0);
  assert.equal(jobs.length, 2);
  assert.ok(harness.pending.has('clip'));
  harness.enqueue('clip', 'clip');
  jobs[1].resolve({ status: 'ok', score: 0.3 });
  await turn();
  assert.equal(harness.published.length, 1);
  assert.equal(harness.published[0].result.score, 0.3);
  assert.equal(harness.pending.size, 0);
});

test('a rejected analysis releases its slot and keeps the failed result unknown', async () => {
  const harness = queueHarness({
    limit: 1,
    analyze: async (blob) => {
      if (blob.key === 'bad') throw Error('decode failed');
      return { status: 'ok', score: 0.1 };
    }
  });
  harness.enqueue('bad', 'bad');
  harness.enqueue('good', 'good');
  await turn();
  assert.deepEqual(
    harness.published.map((r) => r.result.status),
    ['decode-error', 'ok']
  );
  assert.equal(harness.active.size, 0);
  assert.equal(harness.pending.size, 0);
});

test('cache misses retry download instead of publishing a safe score', async () => {
  const harness = queueHarness({ cache: { match: async () => null } });
  harness.cached.add('clip');
  harness.seenMedia.set('clip', 'clip');
  harness.enqueue('clip', 'clip');
  await turn();
  assert.equal(harness.published.length, 0);
  assert.equal(harness.retries.length, 1);
  assert.equal(harness.cached.size, 0);
  assert.equal(harness.pending.size, 0);
});

test('reset while waiting for the cache lock cannot delete new-generation media', async () => {
  let deleted = 0,
    release;
  const harness = queueHarness({
    cache: {
      match: async () => ({ blob: async () => ({}), headers: { get: () => '' } }),
      delete: async () => {
        deleted++;
      }
    },
    identify: async () => {
      throw Error('not media');
    },
    lock: (fn) =>
      new Promise((resolve) => {
        release = () => resolve(fn());
      })
  });
  harness.enqueue('clip', 'clip');
  await turn();
  assert.equal(typeof release, 'function');
  harness.reset();
  release();
  await turn();
  assert.equal(deleted, 0);
  assert.equal(harness.published.length, 0);
  assert.equal(harness.active.size, 0);
});

function signal() {
  const mono = Float32Array.from({ length: 64000 }, (_, i) => (i < 32000 ? 0.002 : 0.7) * Math.sin(i * i * 0.017));
  return [mono, Float32Array.from(mono, (value) => -value)];
}
function audioBuffer(channels) {
  return {
    sampleRate: 16000,
    length: channels[0].length,
    duration: channels[0].length / 16000,
    numberOfChannels: channels.length,
    getChannelData: (channel) => channels[channel]
  };
}

test(
  'actual generated worker program matches main-thread inference and reuses workers',
  { timeout: 10000 },
  async (t) => {
    let created = 0;
    class BrowserWorker {
      constructor() {
        created++;
        this.thread = new NodeWorker(
          `const {parentPort}=require('node:worker_threads');
        const self={postMessage: data=>parentPort.postMessage(data)};
        ${runtime.workerSource()}
        parentPort.on('message',data=>self.onmessage({data}));`,
          { eval: true }
        );
        this.thread.on('message', (data) => this.onmessage?.({ data }));
        this.thread.on('error', (error) => this.onerror?.({ message: error.message }));
      }
      postMessage(data, transfer) {
        this.thread.postMessage(data, transfer);
      }
      terminate() {
        this.thread.terminate();
      }
    }
    const runtime = detector(SOURCE, { Worker: BrowserWorker });
    t.after(() => {
      for (const slot of [...runtime.workers.keys()]) runtime.stopWorker(slot);
    });
    const channels = signal(),
      buffer = audioBuffer(channels);
    const expected = await analyze(runtime, channels);
    const actual = await Promise.all([
      runtime.workerAnalyze(buffer, 0, () => true),
      runtime.workerAnalyze(buffer, 1, () => true)
    ]);
    for (const result of actual) assert.deepEqual(result, expected);
    assert.equal(runtime.workers.size, 2, 'worker failures must not silently pass through fallback');
    assert.ok(channels[0].byteLength > 0, 'transferring worker copies must not detach AudioBuffer storage');
    assert.deepEqual(await runtime.workerAnalyze(buffer, 0, () => true), expected);
    assert.equal(created, 2);

    // The retained corpus is intentionally local; exercise real positives when available.
    const indexPath = path.join(ROOT, 'corpus/index.json');
    if (fs.existsSync(indexPath)) {
      const labels = JSON.parse(fs.readFileSync(path.join(ROOT, 'corpus/labels.json'), 'utf8'));
      const rows = JSON.parse(fs.readFileSync(indexPath, 'utf8')).items;
      for (const row of rows.filter((item) => item.path in labels.confirmed_positives)) {
        const recording = readWav(path.join(ROOT, 'corpus/audio', row.audio_file));
        const direct = await analyze(runtime, recording);
        const threaded = await runtime.workerAnalyze(audioBuffer(recording), 0, () => true);
        assert.deepEqual(threaded, direct, row.file);
        assert.equal(threaded.riskTier, 'alert', row.file);
      }
      assert.equal(created, 2);
      assert.equal(runtime.workers.size, 2, 'real recordings must also execute in workers without fallback');
    }
  }
);

test('blocked workers fall back to identical inference, not a safe default', async () => {
  class BlockedWorker {
    constructor() {
      throw Error('worker blocked by CSP');
    }
  }
  const runtime = detector(SOURCE, { Worker: BlockedWorker }),
    channels = signal();
  assert.deepEqual(await runtime.workerAnalyze(audioBuffer(channels), 0, () => true), await analyze(runtime, channels));
  assert.equal(runtime.workers.size, 0);
});

test('worker cancellation settles the job and releases its timeout/resources', async () => {
  let terminated = false,
    current = true;
  class SilentWorker {
    postMessage() {}
    terminate() {
      terminated = true;
    }
  }
  const runtime = detector(SOURCE, { Worker: SilentWorker });
  const job = runtime.workerAnalyze(audioBuffer(signal()), 0, () => current);
  current = false;
  runtime.stopWorker(0);
  assert.equal(await job, null);
  assert.equal(terminated, true);
  assert.equal(runtime.workers.size, 0);
});
