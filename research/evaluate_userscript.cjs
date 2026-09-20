// Replay a fixed baseline and working userscript on every exact-audio group.
// WAV replay is not a substitute for browser decoding of the original media.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { execFileSync } = require('node:child_process');
const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { ROOT, SOURCE, detector, readWav, analyze } = require('./userscript_harness.cjs');
const digest = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

function summarize(rows, field) {
  const counts = {
    positive: 0,
    negative: 0,
    positiveRed: 0,
    positiveYellow: 0,
    negativeRed: 0,
    negativeYellow: 0,
    unknown: 0
  };
  for (const row of rows) {
    if (!['positive', 'negative'].includes(row.label)) continue;
    counts[row.label]++;
    const result = row[field];
    if (result.status !== 'ok') counts.unknown++;
    else if (result.riskTier === 'alert') counts[`${row.label}Red`]++;
    else if (result.riskTier === 'maybe') counts[`${row.label}Yellow`]++;
  }
  return counts;
}

async function runWorker() {
  const before = detector(workerData.baseline),
    after = detector(workerData.current);
  parentPort.on('message', async (row) => {
    try {
      const filename = path.join(ROOT, 'corpus/audio', `${row.file}.audio.wav`);
      if (digest(fs.readFileSync(filename)) !== row.audio_sha256) throw Error(`Audio hash mismatch: ${row.path}`);
      const channels = readWav(filename);
      const result = { ...row, before: await analyze(before, channels), after: await analyze(after, channels) };
      parentPort.postMessage({ result });
    } catch (error) {
      parentPort.postMessage({ error: String(error.stack || error), path: row.path });
    }
  });
}

async function main() {
  const args = process.argv.slice(2);
  const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
  const ref = option('--baseline', null);
  if (!ref) throw Error('An explicit --baseline Git revision is required');
  const output = option('--output', 'research/artifacts/userscript-evaluation.json');
  if (fs.existsSync(output)) throw Error(`Refusing to overwrite frozen results: ${output}`);
  const workerCount = Number(option('--workers', '4'));
  if (!Number.isInteger(workerCount) || workerCount < 1 || workerCount > 16) throw Error('Invalid worker count');
  const baseline = execFileSync('git', ['show', `${ref}:spokoyno.user.js`], { cwd: ROOT, encoding: 'utf8' });
  const featuresPath = path.join(ROOT, 'research/artifacts/features-v1.json');
  const features = JSON.parse(fs.readFileSync(featuresPath, 'utf8'));
  // Resolve labels through the same Python policy as extraction, rather than
  // accidentally evaluating stale labels after a later thread review.
  const currentLabels = JSON.parse(
    execFileSync(
      path.join(ROOT, '.venv/bin/python'),
      [
        '-c',
        'import json,sys; from pathlib import Path; sys.path.insert(0,"research"); from dataset import label_for; labels=json.loads(Path("corpus/labels.json").read_text()); rows=json.loads(Path("research/artifacts/features-v1.json").read_text())["rows"]; print(json.dumps({r["path"]:label_for(r["path"],labels) for r in rows}))'
      ],
      { cwd: ROOT, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }
    )
  );
  const groups = new Map();
  for (const row of features.rows) {
    if (row.label !== currentLabels[row.path]) throw Error(`Stale feature labels: ${row.path}; rerun extraction`);
    if (row.label === 'visual-only') continue;
    const group = groups.get(row.audio_sha256) || { ...row, paths: [], threads: [], labels: [] };
    delete group.features;
    group.paths.push(row.path);
    if (!group.threads.includes(row.thread)) group.threads.push(row.thread);
    if (['positive', 'negative'].includes(row.label) && !group.labels.includes(row.label)) group.labels.push(row.label);
    groups.set(row.audio_sha256, group);
  }
  const tasks = [...groups.values()].map(({ labels, ...row }) => {
    if (labels.length > 1) throw Error(`Conflicting labels: ${row.audio_sha256}`);
    return { ...row, label: labels[0] || 'unlabeled' };
  });
  const workers = [],
    results = [];
  let next = 0;
  try {
    await Promise.all(
      Array.from(
        { length: workerCount },
        () =>
          new Promise((resolve, reject) => {
            const worker = new Worker(__filename, { workerData: { baseline, current: SOURCE } });
            workers.push(worker);
            worker.on('error', reject);
            const submit = () => (next < tasks.length ? worker.postMessage(tasks[next++]) : resolve());
            worker.on('message', ({ result, error }) => {
              if (error) return reject(Error(error));
              results.push(result);
              if (results.length % 100 === 0 || results.length === tasks.length)
                console.log(`replay ${results.length}/${tasks.length}`);
              submit();
            });
            submit();
          })
      )
    );
  } finally {
    await Promise.all(workers.map((worker) => worker.terminate()));
  }
  results.sort((a, b) => a.path.localeCompare(b.path));
  const summary = { before: summarize(results, 'before'), after: summarize(results, 'after') };
  const payload = {
    schema: 1,
    created_at: new Date().toISOString(),
    baseline_ref: ref,
    baseline_sha256: digest(baseline),
    current_sha256: digest(SOURCE),
    features_sha256: digest(fs.readFileSync(featuresPath)),
    labels_sha256: digest(fs.readFileSync(path.join(ROOT, 'corpus/labels.json'))),
    warning:
      'Development-corpus regression on FFmpeg-decoded WAVs. Unknown labels excluded from accuracy. Exact-audio groups counted once. Not original-media browser decoding or independent validation.',
    summary,
    rows: results
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(payload, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ summary, output }, null, 2));
}

if (isMainThread && require.main === module)
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
else if (!isMainThread) runWorker();
module.exports = { summarize };
