/* Portable frozen YAMNet embedding runtime. TensorFlow.js is supplied by caller.
 * No fitting, media upload, classifier head, or score calibration occurs here.
 */
'use strict';
(() => {
  function fftMagnitude(input, size) {
    const re = new Float64Array(size);
    const im = new Float64Array(size);
    re.set(input);
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) [re[i], re[j]] = [re[j], re[i]];
    }
    for (let n = 2; n <= size; n <<= 1) {
      const angle = (-2 * Math.PI) / n;
      const cr = Math.cos(angle),
        ci = Math.sin(angle);
      for (let lo = 0; lo < size; lo += n) {
        let wr = 1,
          wi = 0;
        for (let k = 0; k < n / 2; k++) {
          const a = lo + k,
            b = a + n / 2;
          const tr = wr * re[b] - wi * im[b];
          const ti = wr * im[b] + wi * re[b];
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
          const next = wr * cr - wi * ci;
          wi = wr * ci + wi * cr;
          wr = next;
        }
      }
    }
    return Float64Array.from(re.subarray(0, size / 2 + 1), (v, i) => Math.hypot(v, im[i]));
  }

  function logMel(patch, manifest, weights) {
    if (patch.length !== 15600) throw new Error('expected 15600-sample patch');
    const view = (name) => {
      const spec = manifest.tensors[name];
      return weights.subarray(spec.offset, spec.offset + spec.length);
    };
    const hann = view('frontend/hann'),
      mel = view('frontend/mel');
    const result = new Float32Array(96 * 64),
      frame = new Float32Array(400);
    for (let t = 0; t < 96; t++) {
      for (let i = 0; i < 400; i++) frame[i] = patch[t * 160 + i] * hann[i];
      const magnitude = fftMagnitude(frame, 512);
      for (let m = 0; m < 64; m++) {
        let value = 0;
        for (let f = 0; f < 257; f++) value += magnitude[f] * mel[f * 64 + m];
        result[t * 64 + m] = Math.log(value + manifest.frontend.log_offset);
      }
    }
    return result;
  }

  class FrozenEncoder {
    constructor(tf, manifest, weights) {
      if (manifest.schema !== 1 || weights.length * 4 !== 12936768) throw new Error('unsupported frozen encoder');
      this.tf = tf;
      this.manifest = manifest;
      this.weights = weights;
      this.tensors = {};
      for (const [name, spec] of Object.entries(manifest.tensors)) {
        if (name.startsWith('frontend/')) continue;
        this.tensors[name] = tf.tensor(weights.subarray(spec.offset, spec.offset + spec.length), spec.shape);
      }
    }
    async encode(patches) {
      if (!patches.length) return new Float32Array();
      const tf = this.tf;
      const features = new Float32Array(patches.length * 96 * 64);
      patches.forEach((patch, i) => features.set(logMel(patch, this.manifest, this.weights), i * 96 * 64));
      let current = tf.tensor(features, [patches.length, 96, 64]);
      try {
        for (const op of this.manifest.operations) {
          const cfg = op.config;
          const w = Object.fromEntries(Object.entries(op.weights).map(([k, name]) => [k, this.tensors[name]]));
          const next = tf.tidy(() => {
            switch (op.kind) {
              case 'Reshape':
                return current.reshape([patches.length, ...cfg.target_shape]);
              case 'Conv2D':
                return tf.conv2d(current, w.kernel, cfg.strides, cfg.padding);
              case 'DepthwiseConv2D':
                return tf.depthwiseConv2d(current, w.depthwise_kernel, cfg.strides, cfg.padding);
              case 'BatchNormalization':
                return tf.batchNorm(current, w.moving_mean, w.moving_variance, w.beta, w.gamma, cfg.epsilon);
              case 'ReLU':
                return tf.relu(current);
              case 'GlobalAveragePooling2D':
                return tf.mean(current, [1, 2]);
              default:
                throw new Error(`unsupported encoder operation ${op.kind}`);
            }
          });
          current.dispose();
          current = next;
        }
        return new Float32Array(await current.data());
      } finally {
        current.dispose();
      }
    }
    dispose() {
      Object.values(this.tensors).forEach((tensor) => tensor.dispose());
    }
  }

  if (typeof module !== 'undefined') module.exports = { fftMagnitude, logMel, FrozenEncoder };
  else globalThis.SpokoynoEncoder = { fftMagnitude, logMel, FrozenEncoder };
})();
