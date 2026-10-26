const assert = require('node:assert/strict');
const { test } = require('node:test');
const { SOURCE, detector, resourceRequest } = require('./userscript_harness.cjs');

function storage() {
  const entries = new Map();
  const cache = {
    match: async (key) => entries.get(key)?.clone(),
    put: async (key, value) => entries.set(key, value.clone()),
    delete: async (key) => entries.delete(key)
  };
  return { entries, cache, caches: { open: async () => cache } };
}
const primary = (url) => new URL(url).hostname === 'cdn.jsdelivr.net';

test('model downloads fall back on network, HTTP, and integrity failures', async (t) => {
  for (const failure of ['network', 'http', 'checksum'])
    await t.test(failure, async () => {
      const attempts = [];
      const request = (details) => {
        attempts.push(details.url);
        if (!primary(details.url)) return resourceRequest(details);
        queueMicrotask(() =>
          failure === 'network'
            ? details.onerror()
            : details.onload({
                status: failure === 'http' ? 503 : 200,
                responseText: 'damaged'
              })
        );
        return { abort() {} };
      };
      const value = await detector(SOURCE, { request }).loadResources();
      assert.equal(JSON.parse(value.model).name, 'synthetic-mil-20pct');
      assert.equal(attempts.filter(primary).length, 4);
      assert.equal(attempts.filter((url) => !primary(url)).length, 4);
    });
});

test('verified model cache survives a new page with both hosts unavailable', async () => {
  const local = storage();
  const first = await detector(SOURCE, local).loadResources();
  let requests = 0;
  const second = await detector(SOURCE, {
    ...local,
    request: () => {
      requests++;
      throw Error('offline');
    }
  }).loadResources();
  assert.deepEqual(second, first);
  assert.equal(requests, 0);
  assert.equal(local.entries.size, 4);
});

test('corrupt cached data is discarded and replaced with verified data', async () => {
  const local = storage();
  const first = await detector(SOURCE, local).loadResources();
  const key = [...local.entries.keys()].find((key) => key.includes('/weights.bin.b64?'));
  local.entries.set(key, new Response('corrupt cache'));
  const requests = [];
  const second = await detector(SOURCE, {
    ...local,
    request: (d) => {
      requests.push(d.url);
      return resourceRequest(d);
    }
  }).loadResources();
  assert.deepEqual(second, first);
  assert.equal(requests.length, 1);
  assert.equal(await (await local.cache.match(key)).text(), first.weights);
});

test('downloaded data is usable when persistent storage rejects writes', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const local = storage();
  local.cache.put = async () => {
    throw Error('quota exceeded');
  };
  const value = await detector(SOURCE, local).loadResources();
  assert.equal(JSON.parse(value.model).name, 'synthetic-mil-20pct');
});

test('a silent GM request times out, aborts, and falls back without native timeout callbacks', async () => {
  let aborted = 0;
  const request = (details) =>
    primary(details.url)
      ? {
          abort() {
            aborted++;
            details.onabort();
            details.onload({ status: 200, responseText: 'late response' });
          }
        }
      : resourceRequest(details);
  const source = SOURCE.replace('ML_RESOURCE_TIMEOUT = 45_000', 'ML_RESOURCE_TIMEOUT = 10');
  assert.notEqual(source, SOURCE);
  const value = await detector(source, { request }).loadResources();
  assert.equal(JSON.parse(value.model).name, 'synthetic-mil-20pct');
  assert.equal(aborted, 4);
});

test('concurrent analysis shares model downloads', async () => {
  let requests = 0;
  const engine = detector(SOURCE, {
    request: (d) => {
      requests++;
      return resourceRequest(d);
    }
  });
  const values = await Promise.all(Array.from({ length: 6 }, () => engine.loadResources()));
  assert.equal(requests, 4);
  assert(values.every((value) => value === values[0]));
});

test('complete download failure stays unknown and a retry can recover', async () => {
  let offline = true;
  const request = (details) => {
    if (!offline) return resourceRequest(details);
    queueMicrotask(() => details.onerror());
    return { abort() {} };
  };
  const engine = detector(SOURCE, { request });
  const pcm = new Float32Array(16000);
  const result = await engine.workerAnalyze(
    { numberOfChannels: 1, length: pcm.length, sampleRate: 16000, getChannelData: () => pcm },
    null
  );
  assert.equal(result.status, 'model-error');
  assert.equal(result.score, undefined);
  assert.match(result.message, /cdn\.jsdelivr\.net/);
  assert.match(result.message, /raw\.githubusercontent\.com/);
  offline = false;
  assert.equal(JSON.parse((await engine.loadResources()).model).name, 'synthetic-mil-20pct');
});
