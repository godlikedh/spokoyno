/* Frozen regularized MIL head. Scores express risk evidence, not probability. */
'use strict';
(() => {
  const features = typeof require === 'function' ? require('./robust_features.cjs') : globalThis.SpokoynoFeatures;
  const { FrozenEncoder } =
    typeof require === 'function' ? require('./robust_encoder.cjs') : globalThis.SpokoynoEncoder;
  function prepare(transform, physical, audio) {
    const spec = transform.physical;
    const result = physical.map((v, i) => ((Number.isNaN(v) ? spec.median[i] : v) - spec.mean[i]) / spec.scale[i]);
    if (audio) {
      const spec = transform.audio;
      for (let i = 0; i < spec.components.length; i++) {
        let value = 0;
        for (let j = 0; j < audio.length; j++) value += (audio[j] - spec.pca_mean[j]) * spec.components[i][j];
        result.push((value - spec.mean[i]) / spec.scale[i]);
      }
    }
    return result.map((v) => Math.max(-8, Math.min(8, v)));
  }
  function predict(transform, model, physical, audio, contentPositions) {
    if (transform.context_features && physical[0].length !== transform.physical.mean.length) {
      const context = typeof require === 'function' ? require('./robust_context.cjs') : globalThis.SpokoynoContext;
      const extra = context.extract(audio, contentPositions, transform.context_features.head);
      if (JSON.stringify(extra.names) !== JSON.stringify(transform.context_features.names))
        throw new Error('context schema differs');
      physical = physical.map((row, i) => [...row, ...extra.rows[i]]);
    }
    const events = physical.map((row, i) => {
      const x = prepare(transform, row, model.audio ? audio.subarray(i * 3072, (i + 1) * 3072) : null);
      let value = model.intercept;
      if ((model.kind || 'linear') === 'linear') {
        if (x.length !== model.coefficient.length) throw new Error('model dimensions differ');
        for (let j = 0; j < x.length; j++) value += x[j] * model.coefficient[j];
      } else if (model.kind === 'histogram-boosting') {
        for (const tree of model.trees) {
          let node = tree[0];
          while (!node.leaf) node = tree[x[node.feature] <= node.threshold ? node.left : node.right];
          value += node.value;
        }
      } else throw new Error('unsupported model head');
      return 1 / (1 + Math.exp(-value));
    });
    let numerator = 0,
      denominator = 0;
    for (const value of events) {
      const p = Math.max(value, 1e-15);
      numerator += p ** model.power;
      denominator += p ** (model.power - 1);
    }
    const score = numerator / Math.max(denominator, 1e-300);
    if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('invalid model output');
    return { score, events };
  }
  class Detector {
    constructor(tf, encoderManifest, encoderWeights, transform, model, name) {
      this.encoder = model.audio ? new FrozenEncoder(tf, encoderManifest, encoderWeights) : null;
      this.sos = encoderManifest.dsp_sos;
      this.transform = transform;
      this.model = model;
      this.name = name;
    }
    async score(samples, channels) {
      const extracted = features.extract(samples, channels, this.sos, !!this.encoder);
      if (extracted.silent) return { score: 0, risk_tier: 'low', events: [], silent: true, model: this.name };
      if (
        JSON.stringify(extracted.names) !==
        JSON.stringify(this.transform.context_features?.base_names || this.transform.names)
      )
        throw new Error('feature names differ');
      let audio = null;
      if (this.encoder) {
        audio = new Float32Array(extracted.patches.length * 1024);
        for (let i = 0; i < extracted.patches.length; i += 12) {
          audio.set(await this.encoder.encode(extracted.patches.slice(i, i + 12)), i * 1024);
        }
      }
      const result = predict(this.transform, this.model, extracted.physical, audio, extracted.contentPositions);
      const events = result.events
        .map((score, i) => ({ ...extracted.windows[i], score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
      return {
        score: result.score,
        risk_tier: result.score >= 0.8 ? 'alert' : result.score >= 0.6 ? 'maybe' : 'low',
        events,
        silent: false,
        model: this.name,
        analyzed_events: extracted.physical.length
      };
    }
    dispose() {
      this.encoder?.dispose();
    }
  }
  if (typeof module !== 'undefined') module.exports = { prepare, predict, Detector };
  else globalThis.SpokoynoModel = { prepare, predict, Detector };
})();
