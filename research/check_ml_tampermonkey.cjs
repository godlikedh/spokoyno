// One-command real Tampermonkey install, network-failure, offline-cache and fallback check.
// Uses an isolated profile and never reads personal browser state beyond the extension's code.
const fs = require('node:fs'),
  path = require('node:path'),
  os = require('node:os');
const http = require('node:http'),
  assert = require('node:assert/strict'),
  crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { readWav } = require('./userscript_harness.cjs');
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(predicate, description) {
  for (let i = 0; i < 150; i++) {
    const result = await predicate();
    if (result) return result;
    await pause(100);
  }
  throw Error(`Timed out: ${description}`);
}
async function main() {
  const [directory, output] = process.argv.slice(2);
  if (!output || fs.existsSync(output)) throw Error('pass fixture directory and new output.json');
  const root = path.join(__dirname, '..');
  const source = fs.readFileSync(path.join(root, 'spokoyno.user.js'), 'utf8');
  const fixture = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures.jsonl'), 'utf8').split('\n')[0]);
  const channels = readWav(fixture.source),
    pcm = new Float32Array(channels[0].length * channels.length);
  channels.forEach((a, c) =>
    a.forEach((v, i) => {
      pcm[i * channels.length + c] = v;
    })
  );
  const files = ['model.json', 'weights.bin.b64', 'tf.wasm.b64', 'tf-simd.wasm.b64'];
  let online = true;
  const requests = [];
  const server = http.createServer((req, res) => {
    if (req.url === '/b/res/1.html') {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      return res.end('<!doctype html><html><body><div id="result"></div></body></html>');
    }
    if (req.url === '/install.user.js') {
      const origin = `http://127.0.0.1:${server.address().port}`;
      let code = source
        .replace(/^\/\/ @match.*$/gm, '')
        .replace('// @run-at', `// @connect      127.0.0.1\n// @match        ${origin}/b/res/*.html\n// @run-at`);
      code = code.replace(
        /  const ML_ASSET_BASES = .*?;/s,
        `  const ML_ASSET_BASES = ${JSON.stringify([origin + '/primary/', origin + '/fallback/'])};`
      );
      const hook = `document.addEventListener('ml-test',async()=>{try{
        const p=new Float32Array(await(await fetch('/pcm')).arrayBuffer()),n=${channels.length};
        const a=Array.from({length:n},(_,c)=>Float32Array.from({length:p.length/n},(_,i)=>p[i*n+c]));
        const b={sampleRate:16000,length:a[0].length,numberOfChannels:n,getChannelData:i=>a[i]};
        const r=await analyzeDecodedAudio(b,document.documentElement.dataset.main==='yes'?null:0,()=>true);
        document.querySelector('#result').textContent=JSON.stringify({r,workers:analysisWorkers.size});
      }catch(e){document.querySelector('#result').textContent=JSON.stringify({error:String(e.stack||e)})}});
      document.documentElement.dataset.mlReady='yes';`;
      code = code.replace(/\}\)\(\);\s*$/, hook + '\n})();');
      res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      return res.end(code);
    }
    if (req.url === '/pcm') return res.end(Buffer.from(pcm.buffer));
    const match = req.url.match(/^\/(primary|fallback)\/([^/?]+)$/);
    if (match && files.includes(match[2])) {
      requests.push({ host: match[1], file: match[2], online });
      if (!online || match[1] === 'primary') return req.socket.destroy();
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      return fs.createReadStream(path.join(root, 'ml/assets', match[2])).pipe(res);
    }
    res.writeHead(404);
    res.end();
  });
  const defaultExtensions = path.join(
    os.homedir(),
    'Library/Application Support/BraveSoftware/Brave-Browser/Default/Extensions/dhdgffkkebhmkfjojejmpbldmpobfkfo'
  );
  const extension =
    process.env.SPOKOYNO_TM_EXTENSION ||
    path.join(
      defaultExtensions,
      fs
        .readdirSync(defaultExtensions)
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
        .at(-1)
    );
  const extensionVersion = JSON.parse(fs.readFileSync(path.join(extension, 'manifest.json'), 'utf8')).version;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'spokoyno-tm-delivery-'));
  const connections = new Map();
  let child,
    port,
    next = 0,
    stderr = '',
    page;
  const targets = async () => (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  async function rpc(id, method, params = {}) {
    if (!connections.has(id)) {
      const target = (await targets()).find((t) => t.id === id);
      if (!target) throw Error('browser target missing');
      const ws = new WebSocket(target.webSocketDebuggerUrl),
        pending = new Map();
      await new Promise((resolve, reject) => {
        ws.onopen = resolve;
        ws.onerror = reject;
      });
      ws.onmessage = ({ data }) => {
        const r = JSON.parse(data),
          p = pending.get(r.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(r.id);
          r.error ? p.reject(Error(JSON.stringify(r.error))) : p.resolve(r.result);
        }
      };
      connections.set(id, { ws, pending });
    }
    const { ws, pending } = connections.get(id);
    return new Promise((resolve, reject) => {
      const id = ++next,
        timer = setTimeout(() => {
          pending.delete(id);
          reject(Error(method + ' timed out'));
        }, 20000);
      pending.set(id, { resolve, reject, timer });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async function evaluate(id, expression) {
    const r = await rpc(id, 'Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
      timeout: 18000
    });
    if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  }
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    child = spawn(
      process.env.SPOKOYNO_BROWSER || '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
      [
        '--headless',
        '--disable-gpu',
        '--mute-audio',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-background-networking',
        '--remote-debugging-port=0',
        `--user-data-dir=${profile}`,
        `--disable-extensions-except=${extension}`,
        `--load-extension=${extension}`,
        'about:blank'
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] }
    );
    child.stderr.on('data', (b) => {
      stderr = (stderr + b).slice(-3000);
    });
    child.on('error', (e) => {
      stderr = String(e);
    });
    const portFile = path.join(profile, 'DevToolsActivePort');
    await until(() => {
      if (child.exitCode !== null) throw Error(stderr);
      return fs.existsSync(portFile);
    }, 'browser startup');
    port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
    page = (await targets()).find((t) => t.type === 'page' && t.url === 'about:blank').id;
    await rpc(page, 'Page.navigate', { url: 'chrome://extensions/' });
    await until(() => evaluate(page, 'typeof chrome.developerPrivate!=="undefined"'), 'extension settings');
    await until(
      () =>
        evaluate(
          page,
          `chrome.developerPrivate.getExtensionsInfo({}).then(a=>a.some(e=>e.id==='dhdgffkkebhmkfjojejmpbldmpobfkfo'))`
        ),
      'extension load'
    );
    await evaluate(
      page,
      `chrome.developerPrivate.updateExtensionConfiguration({extensionId:'dhdgffkkebhmkfjojejmpbldmpobfkfo',userScriptsAccess:true}).then(()=>true)`
    );
    // A fresh installation registers its URL-detection rules asynchronously.
    await until(
      async () => (await targets()).some((t) => t.url.startsWith('https://www.tampermonkey.net/index.php')),
      'Tampermonkey first-run initialization'
    );
    await rpc(page, 'Page.navigate', { url: origin + '/install.user.js' });
    const install = await until(
      async () =>
        (await targets()).find((t) => t.url.startsWith('chrome-extension://dhdgffkkebhmkfjojejmpbldmpobfkfo/ask.html')),
      'installation dialog'
    );
    await until(
      () =>
        evaluate(
          install.id,
          `[...document.querySelectorAll('input[type=button]')].some(e=>/^(Install|Установить)$/.test(e.value))`
        ),
      'install button'
    );
    await evaluate(
      install.id,
      `[...document.querySelectorAll('input[type=button]')].find(e=>/^(Install|Установить)$/.test(e.value)).click();true`
    );
    await until(async () => !(await targets()).some((t) => t.id === install.id), 'installation completion');
    const run = async (mainThread = false) => {
      await rpc(page, 'Page.navigate', { url: origin + '/b/res/1.html' });
      await until(() => evaluate(page, `document.documentElement?.dataset.mlReady==='yes'`), 'userscript startup');
      await evaluate(
        page,
        `document.documentElement.dataset.main='${mainThread ? 'yes' : 'no'}';document.dispatchEvent(new Event('ml-test'));true`
      );
      const value = await until(() => evaluate(page, `document.querySelector('#result')?.textContent`), 'ML inference');
      const result = JSON.parse(value);
      assert.equal(result.r?.status, 'ok', value);
      assert.equal(result.r.model, 'synthetic-mil-20pct');
      assert.equal(result.workers, mainThread ? 0 : 1);
      assert(Math.abs(result.r.score - fixture.expected.final) < 0.001);
      return { score: result.r.score, workers: result.workers };
    };
    const cold = await run();
    assert.equal(new Set(requests.filter((r) => r.host === 'primary').map((r) => r.file)).size, 4);
    assert.equal(requests.filter((r) => r.host === 'fallback').length, 4);
    console.log('Tampermonkey cold load recovered from failed primary host');
    online = false;
    const count = requests.length;
    const offline = await run(),
      fallback = await run(true);
    assert.equal(requests.length, count, 'offline reload must use verified persistent cache');
    assert.equal(offline.score, cold.score);
    assert.equal(fallback.score, cold.score);
    const evidence = {
      tampermonkey: extensionVersion,
      isolated_profile: true,
      actual_GM_requests: true,
      primary_network_failure_recovered: true,
      offline_reload_passed: true,
      main_thread_fallback_passed: true,
      cold,
      offline,
      fallback,
      requests,
      userscript_sha256: crypto.createHash('sha256').update(source).digest('hex'),
      scope:
        'Actual Tampermonkey with local test URLs and hooks; model/runtime bytes unchanged. No personal profile modified.'
    };
    fs.writeFileSync(output, JSON.stringify(evidence, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify(evidence));
  } catch (error) {
    if (port) console.error(JSON.stringify((await targets()).map(({ type, url }) => ({ type, url }))));
    if (page) console.error(await evaluate(page, 'document.body?.innerText.slice(0,1200)').catch(() => ''));
    throw error;
  } finally {
    if (page) await rpc(page, 'Browser.close').catch(() => {});
    for (const { ws, pending } of connections.values()) {
      for (const p of pending.values()) clearTimeout(p.timer);
      ws.close();
    }
    if (child && child.exitCode === null) child.kill('SIGTERM');
    server.closeAllConnections();
    server.close();
  }
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
