// Executes the complete bundled userscript, download failover and cache in an isolated browser.
'use strict';
const fs = require('node:fs'),
  path = require('node:path'),
  os = require('node:os'),
  http = require('node:http'),
  crypto = require('node:crypto');
const { spawn } = require('node:child_process'),
  assert = require('node:assert/strict');
const { readWav } = require('./userscript_harness.cjs');
async function evaluateFixture(index, slot = 0) {
  const fixture = await (await fetch(`/fixture/${index}`)).json();
  const pcm = new Float32Array(await (await fetch(`/pcm/${index}`)).arrayBuffer());
  const channels = Array.from({ length: fixture.channels }, (_, c) =>
    Float32Array.from({ length: pcm.length / fixture.channels }, (_, i) => pcm[i * fixture.channels + c])
  );
  const buffer = {
    sampleRate: 16000,
    length: channels[0].length,
    numberOfChannels: channels.length,
    duration: channels[0].length / 16000,
    getChannelData: (c) => channels[c]
  };
  const start = performance.now();
  const result = await mlTest.analyze(buffer, slot, () => true);
  if (result.status !== 'ok') throw new Error(JSON.stringify(result));
  const badge = document.createElement('span');
  mlTest.render(badge, result);
  if (!badge.title.includes('synthetic-mil-20pct')) throw new Error('badge does not identify ML');
  return {
    file: fixture.file,
    score: result.score,
    tier: result.riskTier,
    milliseconds: performance.now() - start,
    badge: badge.textContent
  };
}
async function main() {
  const [directory, output] = process.argv.slice(2);
  if (!output || fs.existsSync(output)) throw new Error('pass fixtures and new output.json');
  const fixtures = fs
    .readFileSync(path.join(directory, 'fixtures.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((s) => JSON.parse(s));
  const source = fs.readFileSync(path.join(__dirname, '../spokoyno.user.js'), 'utf8');
  const resources = { model: 'model.json', weights: 'weights.bin.b64', wasm: 'tf.wasm.b64', simd: 'tf-simd.wasm.b64' };
  const hook = `globalThis.mlTest={analyze:analyzeDecodedAudio,render:renderScreamerResult,workers:analysisWorkers,stop:stopAnalysisWorker,reset:resetAnalysisQueue,load:loadMLResources,results:analysisResults,state,concurrency:CONCURRENCY};`;
  const instrumented = source.replace(/\}\)\(\);\s*$/, hook + '\n})();');
  const fixed = Object.fromEntries(
    Object.values(resources).map((name) => ['/assets/' + name, path.join(__dirname, '../ml/assets', name)])
  );
  const server = http.createServer((request, response) => {
    if (request.url === '/b/res/1.html') {
      response.setHeader('Content-Type', 'text/html; charset=utf-8');
      response.end('<!doctype html><html><body></body></html>');
      return;
    }
    if (request.url === '/script.js') {
      response.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      response.end(instrumented);
      return;
    }
    const match = request.url.match(/^\/(fixture|pcm)\/(\d+)$/);
    if (match && fixtures[+match[2]]) {
      const fixture = fixtures[+match[2]];
      if (match[1] === 'fixture') {
        response.end(JSON.stringify(fixture));
        return;
      }
      const channels = readWav(fixture.source),
        pcm = new Float32Array(channels[0].length * channels.length);
      for (let c = 0; c < channels.length; c++)
        for (let i = 0; i < channels[c].length; i++) pcm[i * channels.length + c] = channels[c][i];
      response.end(Buffer.from(pcm.buffer));
      return;
    }
    const file = fixed[request.url];
    if (!file) {
      response.writeHead(404);
      response.end();
      return;
    }
    response.setHeader(
      'Content-Type',
      file.endsWith('.js') ? 'text/javascript; charset=utf-8' : 'text/plain; charset=utf-8'
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
    await rpc('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/b/res/1.html` });
    for (let i = 0; !(await evaluate('document.readyState === "complete"')); i++) {
      if (i > 100) throw new Error('runtime did not load');
      await new Promise((r) => setTimeout(r, 100));
    }
    await evaluate(`(async()=>{
      globalThis.modelRequests=[];globalThis.modelNetworkMode='fallback';
      globalThis.GM_getTab=cb=>cb({tm2chMediaV4:{token:'test',cacheName:'tm2ch-media-v4:test',screamer:{old:{analysisVersion:10,status:'ok',score:0}}}});
      globalThis.GM_saveTab=(tab,cb)=>cb?.();globalThis.GM_getTabs=cb=>cb({});
      globalThis.GM_registerMenuCommand=()=>{};globalThis.GM_openInTab=()=>{};
      globalThis.GM_xmlhttpRequest=d=>{
        const url=new URL(d.url),file=url.pathname.split('/').pop();
        if(!${JSON.stringify(Object.values(resources))}.includes(file))throw Error('unexpected media network request');
        modelRequests.push(d.url);
        const controller=new AbortController();
        if(modelNetworkMode==='offline'||(modelNetworkMode==='fallback'&&url.hostname==='cdn.jsdelivr.net'))queueMicrotask(()=>d.onerror());
        else if(modelNetworkMode==='corrupt')queueMicrotask(()=>d.onload({status:200,responseText:'damaged model'}));
        else fetch('/assets/'+file,{signal:controller.signal}).then(async r=>d.onload({status:r.status,responseText:await r.text()})).catch(()=>d.onerror());
        return {abort:()=>controller.abort()};
      };
      await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src='/script.js';s.onload=resolve;s.onerror=reject;document.body.append(s)});
      return true;
    })()`);
    for (let i = 0; !(await evaluate('typeof mlTest!=="undefined"')); i++) {
      if (i > 100) throw new Error('userscript did not initialize');
      await new Promise((r) => setTimeout(r, 100));
    }
    assert.equal(await evaluate('mlTest.results.size'), 0, 'old heuristic scores were reused');
    const results = [];
    let maximum = 0;
    const tierChanges = [];
    const tier = (s) => (s >= 0.8 ? 'alert' : s >= 0.6 ? 'maybe' : 'low');
    for (let start = 0; start < fixtures.length; start += 6) {
      const batch = await evaluate(
        `Promise.all(${JSON.stringify(Array.from({ length: Math.min(6, fixtures.length - start) }, (_, i) => start + i))}.map((i,slot)=>(${evaluateFixture.toString()})(i,slot)))`
      );
      for (let j = 0; j < batch.length; j++) {
        const result = batch[j],
          expected = fixtures[start + j].expected.final;
        const error = Math.abs(result.score - expected);
        maximum = Math.max(maximum, error);
        assert(error < 0.001, `${result.file}: ${error}`);
        if (result.tier !== tier(expected)) tierChanges.push(result.file);
        results.push(result);
      }
      console.log(`userscript ${results.length}/${fixtures.length}`);
    }
    const workers = await evaluate('mlTest.workers.size');
    assert.equal(workers, 6, 'inference must run in reusable workers');
    const requests = await evaluate('modelRequests.length');
    assert.equal(requests, 8, 'four primary failures must use four fallback downloads, shared across workers');
    await evaluate(`(async()=>{mlTest.reset();modelNetworkMode='offline';await mlTest.load();return true})()`);
    assert.equal(await evaluate('modelRequests.length'), requests, 'verified cache must work offline');
    const cancelled = await evaluate(`(async()=>{
      for(const slot of [...mlTest.workers.keys()])mlTest.stop(slot);
      const values=new Float32Array(32000),buffer={sampleRate:16000,length:32000,numberOfChannels:1,getChannelData:()=>values};
      let current=true;const pending=mlTest.analyze(buffer,0,()=>current);
      for(let i=0;!mlTest.workers.has(0)&&i<100;i++)await new Promise(r=>setTimeout(r,1));
      current=false;mlTest.stop(0);return await pending;
    })()`);
    assert.equal(cancelled, null);
    await evaluate('globalThis.savedWorker=Worker;globalThis.Worker=undefined;true');
    const fallback = await evaluate(`(${evaluateFixture.toString()})(0,0)`);
    assert(Math.abs(fallback.score - results[0].score) < 1e-10, 'fallback differs');
    const corrupt = await evaluate(`(async()=>{
      mlTest.reset();await caches.delete('spokoyno-ml-v1');modelNetworkMode='corrupt';
      const v=new Float32Array(32000);const r=await mlTest.analyze({sampleRate:16000,length:32000,numberOfChannels:1,getChannelData:()=>v},0,()=>true);
      globalThis.Worker=savedWorker;return r;
    })()`);
    assert.equal(corrupt.status, 'model-error');
    assert.equal(corrupt.score, undefined);
    const payload = {
      fixtures: fixtures.length,
      maximum_score_error: maximum,
      tier_changes: tierChanges,
      workers,
      reused_workers: true,
      cancellation_passed: true,
      fallback,
      corrupt_resource: corrupt,
      old_cache_invalidated: true,
      primary_host_failure_recovered: true,
      verified_cache_works_offline: true,
      userscript_sha256: crypto.createHash('sha256').update(source).digest('hex'),
      resources,
      results,
      scope:
        'Complete bundled userscript and actual Blob workers; GM HTTP API shim forces primary-host failure, serves exact release bytes through fallback, and then verifies offline CacheStorage reuse.'
    };
    fs.writeFileSync(output, JSON.stringify(payload, null, 2) + '\n', { flag: 'wx' });
    assert.equal(tierChanges.length, 0);
    console.log(JSON.stringify({ maximum_score_error: maximum, tier_changes: tierChanges.length, workers }));
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
