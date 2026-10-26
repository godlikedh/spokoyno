// Run the new ML pipeline in a muted browser with an isolated temporary profile.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const assert = require('node:assert/strict');

async function evaluateFixture(index) {
  const fixture = await (await fetch(`/fixture/${index}`)).json();
  const bytes = await (await fetch(`/pcm/${index}`)).arrayBuffer();
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  if (digest !== fixture.pcm_sha256) throw new Error('PCM checksum mismatch');
  const { transform, models } = await (await fetch(`/models/${index}`)).json();
  const start = performance.now();
  const data = SpokoynoFeatures.extract(new Float32Array(bytes), fixture.channels, encoderManifest.dsp_sos);
  if (JSON.stringify(data.names) !== JSON.stringify(transform.context_features?.base_names || transform.names))
    throw new Error('schema mismatch');
  const physicalMs = performance.now() - start;
  const audio = new Float32Array(data.patches.length * 1024);
  const baseline = tf.memory().numTensors;
  for (let i = 0; i < data.patches.length; i += 12)
    audio.set(await frozenEncoder.encode(data.patches.slice(i, i + 12)), i * 1024);
  if (tf.memory().numTensors !== baseline) throw new Error('encoder leaked tensors');
  const predictions = {};
  for (const [name, model] of Object.entries(models)) {
    const result = SpokoynoModel.predict(transform, model, data.physical, audio, data.contentPositions);
    let best = 0;
    for (let i = 1; i < result.events.length; i++) if (result.events[i] > result.events[best]) best = i;
    predictions[name] = { score: result.score, event_at_s: data.windows[best].start_s };
  }
  return {
    file: fixture.file,
    predictions,
    feature_ms: physicalMs,
    total_ms: performance.now() - start,
    event_count: data.physical.length,
    backend: tf.getBackend(),
    user_agent: navigator.userAgent
  };
}

async function main() {
  const [directory, encoderDir, dependencyDir, output] = process.argv.slice(2);
  if (!output || fs.existsSync(output)) throw new Error('pass fixtures, encoder, node_modules, new output.json');
  const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures.json')));
  const tfRoot = path.resolve(dependencyDir, '@tensorflow');
  const fixed = {
    '/tf.js': path.join(tfRoot, 'tfjs/dist/tf.min.js'),
    '/wasm.js': path.join(tfRoot, 'tfjs-backend-wasm/dist/tf-backend-wasm.min.js'),
    '/encoder.json': path.join(encoderDir, 'model.json'),
    '/weights.bin': path.join(encoderDir, 'weights.bin'),
    ...Object.fromEntries(
      ['robust_encoder.cjs', 'robust_features.cjs', 'robust_context.cjs', 'robust_model.cjs'].map((name) => [
        '/' + name,
        path.join(__dirname, name)
      ])
    )
  };
  for (const name of ['tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm', 'tfjs-backend-wasm-threaded-simd.wasm'])
    fixed['/' + name] = path.join(tfRoot, 'tfjs-backend-wasm/dist', name);
  const server = http.createServer((request, response) => {
    if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end(
        '<!doctype html><title>Local ML parity</title>' +
          [
            '/tf.js',
            '/wasm.js',
            '/robust_encoder.cjs',
            '/robust_features.cjs',
            '/robust_context.cjs',
            '/robust_model.cjs'
          ]
            .map((src) => `<script src="${src}"></script>`)
            .join('')
      );
      return;
    }
    const match = request.url.match(/^\/(fixture|pcm|models)\/(\d+)$/);
    let file = fixed[request.url];
    if (match && fixtures[+match[2]]) {
      const fixture = fixtures[+match[2]];
      if (match[1] === 'fixture') {
        response.end(JSON.stringify(fixture));
        return;
      }
      file = path.join(directory, fixture[match[1]]);
    }
    if (!file) {
      response.writeHead(404);
      response.end();
      return;
    }
    response.setHeader(
      'Content-Type',
      file.endsWith('.wasm')
        ? 'application/wasm'
        : file.endsWith('.js') || file.endsWith('.cjs')
          ? 'text/javascript'
          : 'application/octet-stream'
    );
    fs.createReadStream(file).pipe(response);
  });
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'spokoyno-robust-'));
  let child,
    socket,
    stderr = '';
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    child = spawn(
      '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
      [
        '--headless',
        '--disable-gpu',
        '--mute-audio',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-background-networking',
        '--disable-extensions',
        '--remote-debugging-port=0',
        `--user-data-dir=${profile}`,
        'about:blank'
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] }
    );
    child.stderr.on('data', (b) => {
      stderr = (stderr + b).slice(-5000);
    });
    let spawnError;
    child.on('error', (error) => {
      spawnError = error;
    });
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; !fs.existsSync(portFile); i++) {
      if (spawnError) throw spawnError;
      if (i > 100 || child.exitCode !== null || child.signalCode !== null)
        throw new Error(`browser startup failed: ${stderr}`);
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = reject;
    });
    let sequence = 0;
    const pending = new Map();
    socket.onmessage = (event) => {
      const value = JSON.parse(event.data),
        task = pending.get(value.id);
      if (task) {
        pending.delete(value.id);
        clearTimeout(task.timer);
        value.error ? task.reject(new Error(JSON.stringify(value.error))) : task.resolve(value.result);
      }
    };
    const rpc = (method, params = {}) =>
      new Promise((resolve, reject) => {
        const id = ++sequence,
          timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error(`${method} timed out`));
          }, 60000);
        pending.set(id, { resolve, reject, timer });
        socket.send(JSON.stringify({ id, method, params }));
      });
    const evaluate = async (expression) => {
      const result = await rpc('Runtime.evaluate', {
        expression,
        awaitPromise: true,
        returnByValue: true,
        timeout: 55000
      });
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    await rpc('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
    for (let i = 0; !(await evaluate('typeof SpokoynoModel !== "undefined"')); i++) {
      if (i > 100) throw new Error('scripts did not load');
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    await evaluate(`(async () => {
      tf.wasm.setWasmPaths('/'); tf.wasm.setThreadsCount(1); await tf.setBackend('wasm'); await tf.ready();
      globalThis.encoderManifest = await (await fetch('/encoder.json')).json();
      const bytes = await (await fetch('/weights.bin')).arrayBuffer();
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(b => b.toString(16).padStart(2, '0')).join('');
      if (digest !== encoderManifest.weights_sha256) throw new Error('weight checksum mismatch');
      globalThis.frozenEncoder = new SpokoynoEncoder.FrozenEncoder(tf, encoderManifest, new Float32Array(bytes));
      return true;
    })()`);
    const results = [];
    let maximum = 0;
    const tierChanges = [];
    const tier = (s) => (s >= 0.8 ? 'alert' : s >= 0.6 ? 'maybe' : 'low');
    for (let i = 0; i < fixtures.length; i++) {
      const result = await evaluate(`(${evaluateFixture.toString()})(${i})`);
      for (const [name, p] of Object.entries(result.predictions)) {
        const expected = fixtures[i].expected[name];
        const error = Math.abs(p.score - expected.score);
        maximum = Math.max(maximum, error);
        assert(error < 0.001, `${result.file} ${name}: Python/browser score difference ${error}`);
        if (tier(p.score) !== tier(expected.score)) tierChanges.push({ file: result.file, name, expected, actual: p });
      }
      results.push(result);
      console.log(`browser ${i + 1}/${fixtures.length}: ${result.total_ms.toFixed(0)} ms`);
    }
    await evaluate('frozenEncoder.dispose(); true');
    const payload = {
      fixtures: fixtures.length,
      maximum_score_error: maximum,
      tier_changes: tierChanges,
      sources: Object.fromEntries(
        Object.entries(fixed).map(([name, p]) => [
          name,
          crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')
        ])
      ),
      results
    };
    fs.writeFileSync(output, JSON.stringify(payload, null, 2) + '\n', { flag: 'wx' });
    assert.equal(tierChanges.length, 0, 'browser changed warning tiers');
    console.log(JSON.stringify({ maximum_score_error: maximum, tier_changes: tierChanges.length }));
    await rpc('Browser.close');
  } finally {
    socket?.close();
    if (child && child.exitCode === null) child.kill('SIGTERM');
    server.closeAllConnections();
    server.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
