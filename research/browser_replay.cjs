// Original-media regression through a real, muted Chromium-family decoder.
// Uses an isolated temporary profile; never opens the user's browser profile.
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn, execFileSync } = require('node:child_process');
const { ROOT } = require('./userscript_harness.cjs');
const hash = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

async function browserReplay() {
  function engine(source) {
    const constants = source.slice(
      source.indexOf('  const ANALYSIS_VERSION'),
      source.indexOf('\n\n  if (!THREAD_PATH_RE')
    );
    const core = source.slice(source.indexOf('  const dbPower'), source.indexOf('  function queueScreamerAnalysis'));
    return new Function(`${constants}\nconst yieldMain = async () => {};
      const analysisWorkers = new Map(); let analysisWorkerUrl = null, analysisWorkersUnavailable = false;
      let decoderContext = null;
      ${core}
      return {analyzeScreamer, getDecoderContext, resampleForAnalysis, stopAnalysisWorker};`)();
  }
  const before = engine(await (await fetch('/baseline.js')).text());
  const after = engine(await (await fetch('/current.js')).text());
  const media = await (await fetch('/media')).blob();
  try {
    const baseline = await before.analyzeScreamer(media);
    const current = await after.analyzeScreamer(media);
    const worker = await after.analyzeScreamer(media, 0);
    const decoded = await after.resampleForAnalysis(
      await after.getDecoderContext().decodeAudioData(await media.arrayBuffer())
    );
    const channels = decoded.numberOfChannels;
    const pcm = new Float32Array(decoded.length * channels);
    let peak = 0;
    for (let c = 0; c < channels; c++) {
      const values = decoded.getChannelData(c);
      for (let i = 0; i < values.length; i++) {
        pcm[i * channels + c] = values[i];
        peak = Math.max(peak, Math.abs(values[i]));
      }
    }
    const response = await fetch('/decoded', { method: 'POST', body: pcm.buffer });
    if (!response.ok) throw Error('Failed to preserve browser-decoded samples');
    return {
      userAgent: navigator.userAgent,
      before: baseline,
      after: current,
      worker,
      decoded: { sampleRate: decoded.sampleRate, channels, frames: decoded.length, duration: decoded.duration, peak }
    };
  } finally {
    after.stopAnalysisWorker(0);
  }
}

function floatWav(pcm, rate, channels) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(3, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * channels * 4, 28);
  header.writeUInt16LE(channels * 4, 32);
  header.writeUInt16LE(32, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

async function main() {
  const args = process.argv.slice(2);
  const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
  const media = option('--media', null),
    output = option('--output', null),
    ref = option('--baseline', null);
  if (!media || !output || !ref) throw Error('Required: --media FILE --output FILE.json --baseline GIT_REV');
  const decodedOutput = `${output}.wav`;
  if (fs.existsSync(output) || fs.existsSync(decodedOutput)) throw Error('Refusing to overwrite a browser snapshot');
  const binary = option('--browser', '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser');
  const source = fs.readFileSync(path.join(ROOT, 'spokoyno.user.js'));
  const baseline = execFileSync('git', ['show', `${ref}:spokoyno.user.js`], { cwd: ROOT });
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'spokoyno-browser-'));
  const mediaBytes = fs.readFileSync(media);
  let child,
    socket,
    decodedPcm,
    stderr = '';
  const server = http.createServer((request, response) => {
    const data = { '/baseline.js': baseline, '/current.js': source, '/media': mediaBytes };
    if (request.method === 'POST' && request.url === '/decoded') {
      const chunks = [];
      let size = 0;
      request.on('data', (chunk) => {
        size += chunk.length;
        if (size > 512 * 1024 * 1024) request.destroy();
        else chunks.push(chunk);
      });
      request.on('end', () => {
        decodedPcm = Buffer.concat(chunks);
        response.end('ok');
      });
    } else if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><title>Muted local audio regression</title>');
    } else if (Object.hasOwn(data, request.url)) response.end(data[request.url]);
    else {
      response.statusCode = 404;
      response.end();
    }
  });
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    child = spawn(
      binary,
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
    child.stderr.on('data', (chunk) => {
      stderr = (stderr + chunk).slice(-10000);
    });
    let spawnError;
    child.on('error', (error) => {
      spawnError = error;
    });
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; !fs.existsSync(portFile); i++) {
      if (spawnError) throw spawnError;
      if (i > 100 || child.exitCode !== null || child.signalCode !== null)
        throw Error(`Browser startup failed: ${stderr}`);
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
    const targets = await (
      await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.timeout(10000) })
    ).json();
    socket = new WebSocket(targets.find((target) => target.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = reject;
    });
    let sequence = 0,
      loaded;
    const pending = new Map();
    socket.onmessage = (event) => {
      const value = JSON.parse(event.data);
      if (value.method === 'Page.loadEventFired' && loaded) loaded();
      const task = pending.get(value.id);
      if (task) {
        pending.delete(value.id);
        clearTimeout(task.timer);
        value.error ? task.reject(Error(JSON.stringify(value.error))) : task.resolve(value.result);
      }
    };
    const rpc = (method, params = {}) =>
      new Promise((resolve, reject) => {
        const id = ++sequence;
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(Error(`${method} timed out`));
        }, 60000);
        pending.set(id, { resolve, reject, timer });
        socket.send(JSON.stringify({ id, method, params }));
      });
    await rpc('Page.enable');
    const loading = new Promise((resolve) => {
      loaded = resolve;
    });
    await rpc('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
    await Promise.race([
      loading,
      new Promise((_, reject) => {
        const timer = setTimeout(() => reject(Error('Page load timed out')), 15000);
        timer.unref();
      })
    ]);
    const evaluated = await rpc('Runtime.evaluate', {
      expression: `(${browserReplay.toString()})()`,
      awaitPromise: true,
      returnByValue: true,
      timeout: 55000
    });
    if (evaluated.exceptionDetails) throw Error(JSON.stringify(evaluated.exceptionDetails));
    const result = evaluated.result.value;
    if (!decodedPcm || decodedPcm.length !== result.decoded.frames * result.decoded.channels * 4)
      throw Error('Incomplete browser PCM');
    const wav = floatWav(decodedPcm, result.decoded.sampleRate, result.decoded.channels);
    const payload = {
      schema: 1,
      created_at: new Date().toISOString(),
      source: media,
      source_sha256: hash(mediaBytes),
      baseline_ref: ref,
      baseline_sha256: hash(baseline),
      current_sha256: hash(source),
      decoded_audio: decodedOutput,
      decoded_audio_sha256: hash(wav),
      ...result
    };
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(decodedOutput, wav, { flag: 'wx' });
    fs.writeFileSync(output, JSON.stringify(payload, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify(payload, null, 2));
  } finally {
    if (socket) socket.close();
    if (child && child.exitCode === null) child.kill('SIGTERM');
    server.closeAllConnections();
    server.close();
  }
}

if (require.main === module)
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
module.exports = { floatWav };
