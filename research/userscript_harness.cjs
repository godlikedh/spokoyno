// Executes the actual userscript detector against retained WAVs without a browser or playback.
const fs = require('node:fs');
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..');
const SOURCE = fs.readFileSync(path.join(ROOT, 'spokoyno.user.js'), 'utf8');

function detector(source = SOURCE, environment = {}) {
  const constants = source.slice(
    source.indexOf('  const ANALYSIS_VERSION'),
    source.indexOf('\n\n  if (!THREAD_PATH_RE')
  );
  const core = source.slice(source.indexOf('  const dbPower'), source.indexOf('  function queueScreamerAnalysis'));
  if (!constants || !core) throw Error('Userscript extraction markers changed');
  return new Function(
    'environment',
    `${constants}\nconst yieldMain = async () => {};
    const Worker = environment.Worker;
    const analysisWorkers = new Map(); let analysisWorkerUrl = null, analysisWorkersUnavailable = false;
    let decoderContext = { decodeAudioData: async value => value };
    ${core}
    return {analyzeScreamer,
      score: typeof screamerScore === 'function' ? screamerScore : null,
      tier: typeof screamerTier === 'function' ? screamerTier : null,
      formatScore: typeof formatScreamerScore === 'function' ? formatScreamerScore : null,
      continuousScore: typeof continuousScreamerScore === 'function' ? continuousScreamerScore : null,
      workerSource: typeof makeAnalysisWorkerSource === 'function' ? makeAnalysisWorkerSource : null,
      workerAnalyze: typeof analyzeDecodedAudio === 'function' ? analyzeDecodedAudio : null,
      stopWorker: typeof stopAnalysisWorker === 'function' ? stopAnalysisWorker : null,
      workers: analysisWorkers};`
  )(environment);
}

function readWav(filename) {
  const raw = fs.readFileSync(filename);
  let channels, rate, format, bits, data;
  if (raw.toString('ascii', 0, 4) !== 'RIFF') throw Error('Expected RIFF');
  for (let offset = 12; offset + 8 <= raw.length; ) {
    const tag = raw.toString('ascii', offset, offset + 4);
    const length = raw.readUInt32LE(offset + 4),
      start = offset + 8;
    if (start + length > raw.length) throw Error('Truncated WAV');
    if (tag === 'fmt ') {
      format = raw.readUInt16LE(start);
      channels = raw.readUInt16LE(start + 2);
      rate = raw.readUInt32LE(start + 4);
      bits = raw.readUInt16LE(start + 14);
      if (format === 65534) format = raw.readUInt16LE(start + 24);
    }
    if (tag === 'data') data = raw.subarray(start, start + length);
    offset = start + length + (length % 2);
  }
  if (format !== 3 || bits !== 32 || rate !== 16000 || !data || !channels)
    throw Error('Expected retained 16 kHz float32 WAV');
  const length = data.length / (4 * channels);
  const arrays = Array.from({ length: channels }, () => new Float32Array(length));
  for (let i = 0; i < length; i++)
    for (let c = 0; c < channels; c++) arrays[c][i] = data.readFloatLE((i * channels + c) * 4);
  return arrays;
}

async function analyze(engine, arrays) {
  const buffer = {
    sampleRate: 16000,
    numberOfChannels: arrays.length,
    length: arrays[0].length,
    duration: arrays[0].length / 16000,
    getChannelData: (c) => arrays[c]
  };
  return engine.analyzeScreamer({ arrayBuffer: async () => buffer });
}

module.exports = { ROOT, SOURCE, detector, readWav, analyze };

if (require.main === module)
  (async () => {
    const { execFileSync } = require('node:child_process');
    const old = detector(execFileSync('git', ['show', 'HEAD:spokoyno.user.js'], { cwd: ROOT, encoding: 'utf8' }));
    const current = detector();
    const rows = JSON.parse(fs.readFileSync(path.join(ROOT, 'research/artifacts/features-v1.json'), 'utf8')).rows;
    const seen = new Set(),
      results = [],
      summary = {
        positive: 0,
        negative: 0,
        alertPositive: 0,
        alertNegative: 0,
        maybePositive: 0,
        maybeNegative: 0,
        changedRedDecisions: 0,
        changedScores: 0,
        formerlyZeroNowPositive: 0
      };
    for (const row of rows) {
      if (seen.has(row.audio_sha256) || !['positive', 'negative'].includes(row.label)) continue;
      seen.add(row.audio_sha256);
      const arrays = readWav(path.join(ROOT, 'corpus/audio', `${row.file}.audio.wav`));
      const before = await analyze(old, arrays),
        after = await analyze(current, arrays);
      summary[row.label]++;
      if (after.riskTier === 'alert') summary[row.label === 'positive' ? 'alertPositive' : 'alertNegative']++;
      if (after.riskTier === 'maybe') summary[row.label === 'positive' ? 'maybePositive' : 'maybeNegative']++;
      if (before.suspicious !== after.suspicious || before.decisionScore !== after.decisionScore)
        summary.changedRedDecisions++;
      if (before.score !== after.score) summary.changedScores++;
      if (before.score === 0 && after.score > 0) summary.formerlyZeroNowPositive++;
      results.push({
        path: row.path,
        label: row.label,
        score: after.score,
        riskTier: after.riskTier,
        previousAlert: before.suspicious
      });
      if (results.length % 100 === 0) console.log(JSON.stringify({ processed: results.length, summary }));
    }
    const target = path.join(ROOT, 'research/artifacts/userscript-parallel-v1.json');
    fs.writeFileSync(target, JSON.stringify({ summary, rows: results }, null, 2) + '\n');
    console.log(JSON.stringify({ summary, output: target }));
    if (summary.changedRedDecisions) process.exitCode = 1;
  })().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
