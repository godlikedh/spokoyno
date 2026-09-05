const assert = require('node:assert/strict');
const { test } = require('node:test');
const { SOURCE, detector, analyze } = require('./userscript_harness.cjs');
const engine = detector();

test('risk tiers use precise inclusive boundaries and keep invalid results unknown', () => {
  for (const [score, tier] of [
    [0, 'low'],
    [0.599999, 'low'],
    [0.6, 'maybe'],
    [0.799999, 'maybe'],
    [0.8, 'alert'],
    [1, 'alert'],
    [null, 'unknown'],
    [NaN, 'unknown'],
    [-1, 'unknown'],
    [2, 'unknown']
  ])
    assert.equal(engine.tier(score), tier);
  for (const result of [
    null,
    { status: 'decode-error', score: 0 },
    { status: 'ok', score: NaN },
    { status: 'ok', confidence: 1 }
  ])
    assert.equal(engine.score(result), null);
  assert.equal(engine.score({ status: 'ok', decisionScore: 0.65, displayRisk: 0.99 }), 0.65);
});

function badge() {
  return {
    dataset: {},
    setAttribute(name, value) {
      this[name] = value;
    }
  };
}
const renderSource = SOURCE.slice(
  SOURCE.indexOf('  function renderScreamerResult'),
  SOURCE.indexOf('  function refreshAttachmentRisk')
);
const render = new Function(
  'screamerScore',
  'screamerTier',
  `const SCREAMER_CONFIDENCE=.8, SCREAMER_MAYBE_CONFIDENCE=.6; const formatTime = x => x.toFixed(1); const formatRiskPoints = x => String(x); ${renderSource}; return renderScreamerResult;`
)(engine.score, engine.tier);
function fixture(score) {
  return Object.assign(
    Object.fromEntries(
      [
        'jumpDb',
        'eventDb',
        'eventDuration',
        'eventPeakDb',
        'eventNearClipPct',
        'spectralFlux',
        'startSpectralFlatness',
        'startBrightnessDb',
        'medianDb',
        'peakDb'
      ].map((key) => [key, 0])
    ),
    { status: 'ok', score, decisionScore: score, suspicious: false, detectionMode: 'transition' }
  );
}

test('badge shows yellow, red, low, and unknown without score/flag disagreement', () => {
  const b = badge();
  render(b, fixture(0.6));
  assert.equal(b.dataset.state, 'maybe');
  assert.match(b.textContent, /MAYBE.*0\.600/);
  render(b, fixture(0.8));
  assert.equal(b.dataset.state, 'bad');
  assert.match(b.textContent, /SCREAMER.*0\.800/);
  render(b, { ...fixture(0.799999), suspicious: true });
  assert.equal(b.dataset.state, 'maybe');
  assert.match(b.textContent, /0\.799/);
  render(b, fixture(0.2));
  assert.equal(b.dataset.state, 'ok');
  assert.match(b.textContent, /low risk/);
  render(b, { status: 'decode-error' });
  assert.equal(b.dataset.state, 'error');
  assert.match(b.title, /unknown, not safe/);
  render(b, null);
  assert.equal(b.dataset.state, '');
  assert.match(b.textContent, /analyzing/);
  render(b, { status: 'ok' });
  assert.equal(b.dataset.state, 'error');
  assert.ok(b['aria-label']);
});

test('community reports keep their independent red outline and resets remove yellow', () => {
  const analyses = new Map(),
    reports = new Map(),
    classes = new Set();
  const figure = {
    isConnected: true,
    dataset: {},
    classList: {
      toggle(name, on) {
        if (on) classes.add(name);
        else classes.delete(name);
      }
    }
  };
  const figures = new Map([['clip', new Set([figure])]]);
  const source = SOURCE.slice(
    SOURCE.indexOf('  function refreshAttachmentRisk'),
    SOURCE.indexOf('  function rememberAttachment')
  );
  const refresh = new Function(
    'analysisResults',
    'communityReports',
    'attachmentFigures',
    'screamerScore',
    'screamerTier',
    `${source}; return refreshAttachmentRisk;`
  )(analyses, reports, figures, engine.score, engine.tier);
  analyses.set('clip', fixture(0.7));
  refresh('clip');
  assert.ok(classes.has('tm2ch-risk-maybe'));
  assert.ok(!classes.has('tm2ch-risk'));
  reports.set('clip', {});
  refresh('clip');
  assert.ok(classes.has('tm2ch-risk'));
  assert.ok(!classes.has('tm2ch-risk-maybe'));
  assert.equal(figure.dataset.tm2chScore, '0.7');
  reports.clear();
  analyses.clear();
  refresh('clip');
  assert.equal(classes.size, 0);
  assert.equal(figure.dataset.tm2chRiskTier, 'unknown');
  assert.equal(figure.dataset.tm2chScore, '');
});

test('real detector emits a numeric score and silence stays low', async () => {
  const result = await analyze(engine, [new Float32Array(32000), new Float32Array(32000)]);
  assert.equal(result.status, 'ok');
  assert.ok(Number.isFinite(result.score));
  assert.equal(result.score, result.decisionScore);
  assert.equal(result.riskTier, 'low');
  assert.equal(result.suspicious, false);
});
