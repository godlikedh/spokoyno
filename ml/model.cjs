/* Frozen synthetic-mil-20pct inference. No retraining, heuristic blending, or score cap. */
'use strict';
(() => {
  const features = typeof require === 'function' ? require('./features.cjs') : globalThis.SpokoynoFeatures;
  const { FrozenEncoder } = typeof require === 'function' ? require('./encoder.cjs') : globalThis.SpokoynoEncoder;
  function predict(model, physical, audio, names) {
    const old = model.old_names.map((n) => names.indexOf(n)),
      extra = model.extra_names.map((n) => names.indexOf(n));
    if ([...old, ...extra].some((i) => i < 0) || audio.length !== physical.length * 3072)
      throw new Error('ML feature schema mismatch');
    const scale = (values, spec) =>
      values.map((v, i) => ((Number.isNaN(v) ? spec.median[i] : v) - spec.mean[i]) / spec.scale[i]);
    const events = physical.map((row, index) => {
      const x = [
        ...scale(
          old.map((i) => row[i]),
          model.preprocessor.physical
        ),
        ...scale(
          extra.map((i) => row[i]),
          model.extra_scale
        )
      ];
      const spec = model.preprocessor.audio,
        values = audio.subarray(index * 3072, (index + 1) * 3072);
      for (let i = 0; i < spec.components.length; i++) {
        let v = 0;
        for (let j = 0; j < values.length; j++) v += (values[j] - spec.pca_mean[j]) * spec.components[i][j];
        x.push((v - spec.mean[i]) / spec.scale[i]);
      }
      if (model.columns.length !== model.coefficient.length) throw new Error('ML coefficient dimensions mismatch');
      let value = model.intercept;
      for (let i = 0; i < model.columns.length; i++) value += x[model.columns[i]] * model.coefficient[i];
      return 1 / (1 + Math.exp(-value));
    });
    let sum = 0,
      squared = 0;
    for (const p of events) {
      sum += p;
      squared += p * p;
    }
    const score = squared / Math.max(sum, 1e-12);
    if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('invalid ML score');
    return { score, events };
  }
  class Detector {
    constructor(tf, bundle, weights) {
      if (bundle.schema !== 1 || bundle.name !== 'synthetic-mil-20pct' || bundle.model.pooling !== 'linear-softmax')
        throw new Error('unsupported ML model');
      this.bundle = bundle;
      this.encoder = new FrozenEncoder(tf, bundle.encoder, weights);
    }
    async score(channels, sampleRate = 16000) {
      if (
        sampleRate !== 16000 ||
        ![1, 2].includes(channels.length) ||
        channels.some((c) => c.length !== channels[0].length)
      )
        throw new Error('ML needs 16 kHz mono/stereo audio');
      const length = channels[0].length;
      if (length < 4800)
        return { status: 'too-short', message: 'Audio is shorter than the model’s 300 ms analysis window' };
      const audio = new Float32Array(length * channels.length);
      for (let c = 0; c < channels.length; c++)
        for (let i = 0; i < length; i++) audio[i * channels.length + c] = channels[c][i];
      const data = features.extract(audio, channels.length, this.bundle.dsp);
      const embeddings = await this.encoder.encode(data.patches);
      const context = new Float32Array(data.physical.length * 3072);
      data.patchIndices.forEach((indices, row) =>
        indices.forEach((patch, part) => {
          if (patch >= 0) context.set(embeddings.subarray(patch * 1024, (patch + 1) * 1024), row * 3072 + part * 1024);
        })
      );
      const output = predict(this.bundle.model, data.physical, context, data.names);
      const events = output.events.map((score, i) => ({ ...data.windows[i], score })).sort((a, b) => b.score - a.score);
      return {
        status: 'ok',
        model: this.bundle.name,
        score: output.score,
        riskTier: output.score >= 0.8 ? 'alert' : output.score >= 0.6 ? 'maybe' : 'low',
        suspicious: output.score >= 0.8,
        detectionMode: 'ml',
        eventAt: events[0].start_s,
        events,
        duration: length / sampleRate
      };
    }
    dispose() {
      this.encoder.dispose();
    }
  }
  function decodeBase64(text) {
    const raw = atob(text.trim()),
      bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return bytes;
  }
  async function digest(bytes) {
    return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
      .map((v) => v.toString(16).padStart(2, '0'))
      .join('');
  }
  async function initialize(tf, resources) {
    const bundle = JSON.parse(resources.model);
    const weights = decodeBase64(resources.weights);
    if ((await digest(weights)) !== bundle.encoder.weights_sha256) throw new Error('ML encoder checksum mismatch');
    const urls = [];
    try {
      for (const encoded of [resources.wasm, resources.simd])
        urls.push(URL.createObjectURL(new Blob([decodeBase64(encoded)], { type: 'application/wasm' })));
      tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false);
      tf.wasm.setThreadsCount(1);
      tf.wasm.setWasmPaths(
        {
          'tfjs-backend-wasm.wasm': urls[0],
          'tfjs-backend-wasm-simd.wasm': urls[1],
          'tfjs-backend-wasm-threaded-simd.wasm': urls[1]
        },
        true
      );
      if (!(await tf.setBackend('wasm'))) throw new Error('ML WebAssembly backend unavailable');
      await tf.ready();
      return new Detector(tf, bundle, new Float32Array(weights.buffer));
    } finally {
      urls.forEach((url) => URL.revokeObjectURL(url));
    }
  }
  const api = { predict, Detector, initialize, digest };
  if (typeof module !== 'undefined') module.exports = api;
  else globalThis.SpokoynoML = api;
})();
