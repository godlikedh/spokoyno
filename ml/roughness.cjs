/* Exact-length Hilbert envelope and 500 Hz resampling used by the frozen model. */
'use strict';
(() => {
  function radix(re, im, inverse = false) {
    const size = re.length;
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let n = 2; n <= size; n *= 2) {
      const angle = ((inverse ? 2 : -2) * Math.PI) / n,
        cr = Math.cos(angle),
        ci = Math.sin(angle);
      for (let lo = 0; lo < size; lo += n) {
        let wr = 1,
          wi = 0;
        for (let k = 0; k < n / 2; k++) {
          const a = lo + k,
            b = a + n / 2;
          const tr = wr * re[b] - wi * im[b],
            ti = wr * im[b] + wi * re[b];
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
    if (inverse)
      for (let i = 0; i < size; i++) {
        re[i] /= size;
        im[i] /= size;
      }
  }
  const plans = new Map();
  function transform(real, imaginary = new Float64Array(real.length), inverse = false) {
    const n = real.length;
    if (!n) throw new Error('empty FFT');
    if (inverse) {
      const [r, i] = transform(
        real,
        Float64Array.from(imaginary, (v) => -v)
      );
      return [r.map((v) => v / n), i.map((v) => -v / n)];
    }
    if ((n & (n - 1)) === 0) {
      const re = Float64Array.from(real),
        im = Float64Array.from(imaginary);
      radix(re, im);
      return [re, im];
    }
    let plan = plans.get(n);
    if (!plan) {
      let m = 1;
      while (m < 2 * n - 1) m *= 2;
      const cos = new Float64Array(n),
        sin = new Float64Array(n),
        br = new Float64Array(m),
        bi = new Float64Array(m);
      for (let j = 0; j < n; j++) {
        const angle = (Math.PI * ((j * j) % (2 * n))) / n;
        cos[j] = Math.cos(angle);
        sin[j] = Math.sin(angle);
        br[j] = cos[j];
        bi[j] = sin[j];
        if (j) {
          br[m - j] = cos[j];
          bi[m - j] = sin[j];
        }
      }
      radix(br, bi);
      plan = { m, cos, sin, br, bi };
      if (plans.size >= 4) plans.delete(plans.keys().next().value);
      plans.set(n, plan);
    }
    const { m, cos, sin, br, bi } = plan,
      ar = new Float64Array(m),
      ai = new Float64Array(m);
    for (let j = 0; j < n; j++) {
      ar[j] = real[j] * cos[j] + imaginary[j] * sin[j];
      ai[j] = imaginary[j] * cos[j] - real[j] * sin[j];
    }
    radix(ar, ai);
    for (let j = 0; j < m; j++) {
      const r = ar[j] * br[j] - ai[j] * bi[j];
      ai[j] = ar[j] * bi[j] + ai[j] * br[j];
      ar[j] = r;
    }
    radix(ar, ai, true);
    const re = new Float64Array(n),
      im = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      re[j] = ar[j] * cos[j] + ai[j] * sin[j];
      im[j] = ai[j] * cos[j] - ar[j] * sin[j];
    }
    return [re, im];
  }
  function modulation(audio, channels, lo, hi, dsp) {
    const length = hi - lo;
    if (length < 3200) return [NaN, NaN];
    const energies = new Float64Array(channels);
    for (let i = lo; i < hi; i += 16) for (let c = 0; c < channels; c++) energies[c] += audio[i * channels + c] ** 2;
    let channel = 0;
    for (let c = 1; c < channels; c++) if (energies[c] > energies[channel]) channel = c;
    const samples = new Float64Array(length),
      states = dsp.roughness_sos.map(() => [0, 0]);
    for (let i = 0; i < length; i++) {
      let value = audio[(lo + i) * channels + channel];
      for (let j = 0; j < states.length; j++) {
        const s = states[j],
          b = dsp.roughness_sos[j],
          next = b[0] * value + s[0];
        s[0] = b[1] * value - b[4] * next + s[1];
        s[1] = b[2] * value - b[5] * next;
        value = next;
      }
      samples[i] = value;
    }
    const [re, im] = transform(samples);
    for (let k = 0; k < length; k++) {
      const factor = k === 0 || (length % 2 === 0 && k === length / 2) ? 1 : k < (length + 1) / 2 ? 2 : 0;
      re[k] *= factor;
      im[k] *= factor;
    }
    const [hr, hiValues] = transform(re, im, true);
    const envelope = hr.map((v, i) => Math.hypot(v, hiValues[i]));
    const down = new Float64Array(Math.ceil(length / 32)),
      fir = dsp.envelope_fir;
    for (let i = 0; i < down.length; i++)
      for (let j = 0; j < fir.length; j++) {
        const at = i * 32 + 320 - j;
        if (at >= 0 && at < length) down[i] += fir[j] * envelope[at];
      }
    const mean = down.reduce((a, b) => a + b, 0) / down.length;
    if (mean < 1e-7) return [0, 0];
    const centered = down.map((v) => v - mean),
      variance = centered.reduce((s, v) => s + v * v, 0) / down.length;
    const [fr, fi] = transform(
      centered.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (down.length - 1))))
    );
    let total = 0,
      rough = 0;
    for (let k = 0; k <= Math.floor(down.length / 2); k++) {
      const frequency = k * (1 / (down.length * (1 / 500))),
        p = fr[k] ** 2 + fi[k] ** 2;
      if (frequency >= 4 && frequency <= 200) total += p;
      if (frequency >= 30 && frequency <= 150) rough += p;
    }
    return [rough / Math.max(total, 1e-20), Math.sqrt(variance) / mean];
  }
  const api = { transform, modulation };
  if (typeof module !== 'undefined') module.exports = api;
  else globalThis.SpokoynoRoughness = api;
})();
