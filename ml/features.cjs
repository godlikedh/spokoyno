/* Frozen acoustic_features.py / event_dataset.py feature pipeline at 16 kHz.
 * Inputs are interleaved Float32 PCM; no metadata or class labels are features.
 */
'use strict';
(() => {
  const { fftMagnitude } = typeof require === 'function' ? require('./encoder.cjs') : globalThis.SpokoynoEncoder;
  const RATE = 16000,
    EDGES = [70, 250, 500, 1000, 2000, 4000, 8000];
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => (a.length ? sum(a) / a.length : NaN);
  const db = (x) => 10 * Math.log10(Math.max(x, 1e-9));
  const power = (x) => 10 ** (x / 10);
  const f32 = Math.fround;
  const range = (a, b, step = 1) =>
    Array.from({ length: Math.max(0, Math.ceil((b - a) / step)) }, (_, i) => a + i * step);
  function argmax(a) {
    let best = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
    return best;
  }
  function percentile(a, q) {
    if (!a.length) return -90;
    const v = Array.from(a).sort((a, b) => a - b),
      at = ((v.length - 1) * q) / 100;
    const lo = Math.floor(at),
      hi = Math.ceil(at);
    return v[lo] + (v[hi] - v[lo]) * (at - lo);
  }
  function timelines(audio, channels, sos) {
    const count = Math.floor(audio.length / channels / 800);
    const coarse = new Float32Array(count),
      fine = new Float32Array(count * 5),
      rawClip = new Float32Array(count);
    const states = Array.from({ length: channels }, () => sos.map(() => [0, 0]));
    const threshold = 10 ** (-1 / 20);
    for (let frame = 0; frame < count; frame++) {
      let coarsePower = 0,
        clips = 0;
      for (let small = 0; small < 5; small++) {
        let finePower = 0;
        for (let n = 0; n < 160; n++)
          for (let c = 0; c < channels; c++) {
            let x = audio[((frame * 5 + small) * 160 + n) * channels + c];
            clips += Math.abs(x) >= threshold ? 1 : 0;
            for (let s = 0; s < sos.length; s++) {
              const b = sos[s],
                state = states[c][s],
                y = b[0] * x + state[0];
              state[0] = b[1] * x - b[4] * y + state[1];
              state[1] = b[2] * x - b[5] * y;
              x = y;
            }
            x = f32(x);
            finePower += x * x;
          }
        coarsePower += finePower;
        fine[frame * 5 + small] = Math.max(-90, -0.691 + db(finePower / (160 * channels)));
      }
      coarse[frame] = Math.max(-90, -0.691 + db(coarsePower / (800 * channels)));
      rawClip[frame] = clips / (800 * channels);
    }
    return { coarse, fine, rawClip };
  }
  class Spectra {
    constructor(audio, channels) {
      this.audio = audio;
      this.channels = channels;
      this.cache = new Map();
      this.hann = Float32Array.from(range(0, 800), (i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 799));
    }
    frame(index) {
      if (this.cache.has(index)) return this.cache.get(index);
      let result = new Float64Array(513).fill(1 / 513);
      if ((index + 1) * 800 * this.channels <= this.audio.length) {
        result.fill(1e-12);
        for (let c = 0; c < this.channels; c++) {
          const frame = Float32Array.from(this.hann, (w, i) => this.audio[(index * 800 + i) * this.channels + c] * w);
          const magnitude = fftMagnitude(frame, 1024);
          for (let k = 0; k < 513; k++) result[k] += magnitude[k] ** 2 / this.channels;
        }
        const total = sum(result);
        result = result.map((x) => x / total);
      }
      this.cache.set(index, result);
      return result;
    }
    aggregate(indices) {
      const profiles = indices.filter((i) => i >= 0).map((i) => this.frame(i));
      if (!profiles.length) return new Float64Array(513).fill(1 / 513);
      const result = new Float64Array(513);
      for (const p of profiles) for (let i = 0; i < 513; i++) result[i] += p[i];
      const total = sum(result);
      return result.map((x) => x / total);
    }
    pair(baseline, event) {
      const result = {
        spectral_shape_distance: distance(baseline, event),
        spectral_flatness_delta:
          Math.exp(mean(event.map((x) => Math.log(x + 1e-12)))) / mean(event) -
          Math.exp(mean(baseline.map((x) => Math.log(x + 1e-12)))) / mean(baseline),
        spectral_centroid_delta_hz: sum(event.map((v, i) => ((i * RATE) / 1024) * (v - baseline[i])))
      };
      for (let b = 0; b < EDGES.length - 1; b++) {
        let e = 0,
          v = 0;
        for (let i = 0; i < 513; i++)
          if ((i * RATE) / 1024 >= EDGES[b] && (i * RATE) / 1024 < EDGES[b + 1]) {
            e += event[i];
            v += baseline[i];
          }
        result[`band_${EDGES[b]}_${EDGES[b + 1]}_jump_db`] = 10 * Math.log10((e + 1e-9) / (v + 1e-9));
      }
      return result;
    }
  }
  function distance(a, b) {
    return Math.sqrt(sum(a.map((v, i) => (Math.sqrt(v) - Math.sqrt(b[i])) ** 2)) / 2);
  }
  function duration(level, start, threshold) {
    let last = start - 1,
      misses = 0;
    for (let i = start; i < Math.min(level.length, start + 120); i++) {
      if (level[i] >= threshold) {
        last = i;
        misses = 0;
      } else if (++misses >= 2) break;
    }
    return Math.max(0, (last - start + 1) * 0.05);
  }
  function differences(a) {
    return Float64Array.from(a.slice(1), (v, i) => f32(v - a[i]));
  }
  function baseFeatures(level, fine, raw, spectra, i) {
    const hist = level.slice(Math.max(0, i - 60), Math.max(0, i - 2));
    const baseline = percentile(hist, 50),
      event = percentile(level.slice(i, i + 6), 25);
    const mad = percentile(
        hist.map((v) => Math.abs(v - baseline)),
        50
      ),
      jump = event - baseline;
    const result = {
      event_level_db: event,
      jump_db: jump,
      baseline_mad_db: mad,
      robust_z: jump / Math.max(2, 1.4826 * mad)
    };
    const guard = Math.max(0, i - 2);
    for (const [seconds, count] of [
      ['0_5', 10],
      ['1_0', 20],
      ['3_0', 60],
      ['6_0', 120]
    ]) {
      const h = level.slice(Math.max(0, guard - count), guard),
        b = percentile(h, 50);
      result[`baseline_${seconds}s_db`] = b;
      result[`baseline_${seconds}s_mad_db`] = h.length
        ? percentile(
            h.map((v) => Math.abs(v - b)),
            50
          )
        : 0;
      result[`jump_vs_${seconds}s_db`] = event - b;
    }
    for (const [seconds, count] of [
      ['0_1', 2],
      ['0_3', 6],
      ['0_6', 12],
      ['1_0', 20]
    ])
      result[`event_${seconds}s_db`] = percentile(level.slice(i, i + count), 25);
    const threshold = Math.max(-13, baseline + 9);
    result.duration_s = duration(level, i, threshold);
    result.duration_plus6_s = duration(level, i, baseline + 6);
    result.duration_plus12_s = duration(level, i, baseline + 12);
    const following = level.slice(i, i + 20),
      tail = level.slice(i, i + 60);
    result.persistence_1s = following.length ? mean(following.map((v) => (v >= threshold ? 1 : 0))) : 0;
    result.loudness_area_db = mean(tail.map((v) => Math.max(0, v - baseline)));
    const h = level.slice(Math.max(0, i - 60), i),
      diff = differences(h);
    result.baseline_iqr_db = percentile(h, 75) - percentile(h, 25);
    result.baseline_derivative_db = diff.length ? percentile(diff.map(Math.abs), 50) : 0;
    result.prior_abrupt_changes = sum(diff.map((v) => (v >= 8 ? 1 : 0)));
    result.near_clip_fraction = mean(
      raw.slice(i, Math.min(raw.length, i + Math.max(6, Math.round(Math.max(result.duration_s, 0.3) / 0.05))))
    );
    Object.assign(
      result,
      spectra.pair(
        spectra.aggregate(range(Math.max(0, i - 42), Math.max(0, i - 2), 2)),
        spectra.aggregate(range(i, Math.min(level.length, i + 6)))
      )
    );
    const flux = (k) => sum(spectra.frame(k).map((v, j) => Math.max(0, v - spectra.frame(Math.max(0, k - 1))[j])));
    result.spectral_flux = flux(i);
    result.nearby_spectral_flux = Math.max(
      0,
      ...range(i - 2, i + 3)
        .filter((k) => k >= 1 && k < level.length)
        .map(flux)
    );
    return result;
  }
  function relativeAttack(fine, point) {
    const lo = Math.max(0, point - 10),
      hi = Math.min(fine.length, point + 31);
    if (hi <= point) return NaN;
    const peak = point + argmax(fine.slice(point, hi));
    const bg = power(percentile(fine.slice(Math.max(0, point - 100), Math.max(0, point - 5)), 50));
    const pk = power(fine[peak]);
    if (pk <= bg * 1.01) return 0;
    const values = fine.slice(lo, peak + 1).map(power),
      low = bg + 0.2 * (pk - bg),
      high = bg + 0.8 * (pk - bg);
    let start = 0;
    values.forEach((v, i) => {
      if (v < low) start = i;
    });
    const cross = values.slice(start).findIndex((v) => v >= high);
    return cross < 0 ? NaN : cross * 10;
  }
  function burst(fine, point) {
    const baseline = percentile(fine.slice(Math.max(0, point - 300), Math.max(0, point - 5)), 50);
    const end = Math.min(fine.length, point + 200),
      early = fine.slice(point, Math.min(end, point + 30));
    const peak = point + argmax(early),
      peakDb = fine[peak],
      threshold = Math.max(baseline + 6, peakDb - 10);
    let left = peak,
      right = peak + 1;
    while (left > Math.max(0, point - 10) && fine[left - 1] >= threshold) left--;
    while (right < end && fine[right] >= threshold) right++;
    const active = fine.slice(left, right),
      tail = fine.slice(right + 5, right + 35),
      following = fine.slice(point, point + 100);
    const result = {
      burst_peak_10ms_db: peakDb,
      burst_peak_jump_db: peakDb - baseline,
      burst_relative_attack_ms: relativeAttack(fine, point),
      burst_active_duration_s: (right - left) / 100,
      burst_active_median_db: percentile(active, 50),
      burst_active_energy_db: db(mean(active.map(power))),
      burst_tail_drop_db: tail.length ? peakDb - percentile(tail, 50) : NaN,
      burst_peak_to_1s_median_db: peakDb - percentile(following, 50),
      burst_active_fraction_1s: mean(following.map((v) => (v >= threshold ? 1 : 0)))
    };
    const segment = fine.slice(Math.max(0, point - 5), Math.min(fine.length, point + 40)).map(power);
    for (const length of [5, 10, 20]) {
      const avg =
        segment.length >= length
          ? range(0, segment.length - length + 1).map((i) => mean(segment.slice(i, i + length)))
          : segment;
      result[`burst_peak_${length * 10}ms_db`] = db(Math.max(...avg));
    }
    return result;
  }
  function rhythm(fine, point, spectra, coarse) {
    const history = fine.slice(Math.max(0, point - 800), point),
      onset = differences(history).map((v) => Math.max(v, 0));
    for (let i = 0; i < onset.length; i++)
      onset[i] = f32(onset[i] / (1 + Math.exp(-Math.max(-30, Math.min(30, (history[i + 1] + 25) / 6)))));
    let best = 0;
    for (let lag = 20; lag < Math.min(151, Math.floor(onset.length / 2)); lag++) {
      let aa = 0,
        bb = 0,
        ab = 0;
      for (let i = 0; i + lag < onset.length; i++) {
        aa += onset[i] ** 2;
        bb += onset[i + lag] ** 2;
        ab += onset[i] * onset[i + lag];
      }
      const denominator = Math.sqrt(aa * bb);
      if (denominator > 1e-8) best = Math.max(best, ab / denominator);
    }
    const current = fine.slice(Math.max(0, point - 5), Math.min(fine.length, point + 20)),
      diff = differences(current);
    const rise = diff.length ? Math.max(...diff) : 0,
      previous = onset.length ? percentile(onset, 95) : 0;
    const event = spectra.aggregate(range(coarse, Math.min(coarse + 2, Math.floor(fine.length / 5))));
    const profiles = range(Math.max(0, coarse - 160), Math.max(0, coarse - 10), 5).map((i) =>
      distance(event, spectra.frame(i))
    );
    return {
      rhythm_onset_periodicity: onset.length >= 60 ? best : NaN,
      rhythm_onset_surprise_ratio: Math.min(30, rise / Math.max(1, previous)),
      rhythm_previous_onset_p95_db: previous,
      rhythm_previous_spectral_distance_min: profiles.length ? Math.min(...profiles) : NaN
    };
  }
  function bands(audio, channels, at) {
    const start = Math.max(0, Math.round((at - 1) * RATE)),
      stop = Math.min(audio.length / channels, Math.round((at + 0.4) * RATE));
    const window = Float64Array.from(range(0, 512), (i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 511));
    const norm = 512 * sum(window.map((v) => v * v));
    const before = EDGES.slice(1).map(() => []),
      during = EDGES.slice(1).map(() => []);
    for (let pos = start; pos + 512 <= stop; pos += 160) {
      const time = (pos + 256) / RATE,
        target = time < at - 0.05 ? before : time >= at && time < at + 0.3 ? during : null;
      if (!target) continue;
      const pow = new Float64Array(257);
      for (let c = 0; c < channels; c++) {
        const magnitude = fftMagnitude(
          window.map((w, i) => w * audio[(pos + i) * channels + c]),
          512
        );
        for (let k = 0; k < 257; k++) pow[k] += (magnitude[k] ** 2 / channels / norm) * (k > 0 && k < 256 ? 2 : 1);
      }
      for (let b = 0; b < 6; b++) {
        let value = 0;
        for (let k = 0; k < 257; k++)
          if ((k * RATE) / 512 >= EDGES[b] && (k * RATE) / 512 < EDGES[b + 1]) value += pow[k];
        target[b].push(db(value));
      }
    }
    const result = {};
    for (let b = 0; b < 6; b++) {
      const peak = stop - start < 512 ? NaN : percentile(during[b], 90);
      result[`bands_event_db_${EDGES[b]}_${EDGES[b + 1]}`] = peak;
      result[`bands_jump_db_${EDGES[b]}_${EDGES[b + 1]}`] = before[b].length ? peak - percentile(before[b], 50) : NaN;
    }
    return result;
  }
  // Frozen v1 proposal selection: three separated transition candidates and opening.
  const sigmoid = (v) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v))));
  const roundEven = (v) => {
    const lo = Math.floor(v);
    return v - lo === 0.5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(v);
  };
  function median32(values) {
    const a = Array.from(values).sort((x, y) => x - y),
      m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : f32(f32(a[m - 1] + a[m]) / 2);
  }
  function proposals(level) {
    const candidates = [];
    for (let i = 22; i < level.length - 5; i++) {
      const history = level.slice(Math.max(0, i - 60), i - 2),
        baseline = median32(history);
      const mad = median32(history.map((v) => Math.abs(v - baseline)));
      const event = percentile(level.slice(i, i + 6), 25),
        jump = event - baseline;
      const score =
        sigmoid((event + 10.5) / 2.5) ** 0.9 *
        sigmoid((jump - 13) / 3.5) ** 1.25 *
        (0.65 + 0.35 * sigmoid((jump / Math.max(2, 1.4826 * mad) - 6) / 2));
      candidates.push({ i, score, baseline });
    }
    candidates.sort((a, b) => b.score - a.score || a.i - b.i);
    const selected = [];
    for (const row of candidates) {
      if (selected.some((p) => Math.abs(p.index - row.i) < 20)) continue;
      selected.push({ index: row.i, slot: selected.length, baseline: row.baseline });
      if (selected.length === 3) break;
    }
    const limit = Math.min(Math.max(0, level.length - 6), 20);
    const opening = argmax(range(0, limit + 1).map((i) => percentile(level.slice(i, i + 6), 25)));
    selected.push({ index: opening, slot: 3 });
    return selected;
  }
  function attack(fine, index) {
    const event = index * 5,
      lo = Math.max(0, event - 100),
      hi = Math.min(fine.length, event + 100);
    if (hi <= event) return 2000;
    const peak = event + argmax(fine.slice(event, hi));
    let origin = -1;
    for (let i = lo; i < peak; i++) if (fine[i] <= -30) origin = i;
    if (origin < 0) return 2000;
    const target = f32(fine[peak] - 3);
    for (let i = origin + 1; i <= peak; i++) if (fine[i] >= target) return (i - origin) * 10;
    return 2000;
  }
  function extract(audio, channels, dsp, includePatches = true) {
    if (!(audio instanceof Float32Array) || ![1, 2].includes(channels) || audio.length % channels)
      throw new Error('expected mono/stereo float32 PCM');
    const length = audio.length / channels;
    if (length < 4800) throw new Error('audio is shorter than 300 ms');
    for (const value of audio) if (!Number.isFinite(value)) throw new Error('nonfinite audio');
    const { coarse, fine, rawClip } = timelines(audio, channels, dsp.sos),
      spectra = new Spectra(audio, channels);
    const points = proposals(coarse),
      rows = [],
      windows = [],
      patches = [],
      patchIndices = [];
    const energy = new Float64Array(channels);
    for (let n = 0; n < length; n += 16) for (let c = 0; c < channels; c++) energy[c] += audio[n * channels + c] ** 2;
    const channel = argmax(energy);
    const { modulation } = typeof require === 'function' ? require('./roughness.cjs') : globalThis.SpokoynoRoughness;
    function crop(time) {
      const start = roundEven(time * RATE),
        patch = new Float32Array(15600);
      let peak = 1;
      for (let i = 0; i < patch.length; i++)
        if (start + i >= 0 && start + i < length) {
          const value = audio[(start + i) * channels + channel];
          patch[i] = value;
          peak = Math.max(peak, Math.abs(value));
        }
      for (let i = 0; i < patch.length; i++) patch[i] = patch[i] / peak;
      return patch;
    }
    for (const point of points) {
      const i = point.index,
        at = i * 0.05,
        fp = Math.min(fine.length - 1, roundEven(at * 100));
      const base = baseFeatures(coarse, fine, rawClip, spectra, i);
      base.history_s = Math.min(at, 6);
      base.attack_ms = attack(fine, i);
      base.is_opening = point.slot === 3 ? 1 : 0;
      if (i <= 2)
        for (const name of Object.keys(base))
          if (
            name.includes('baseline') ||
            name.includes('jump') ||
            ['robust_z', 'spectral_shape_distance'].includes(name)
          )
            base[name] = NaN;
      const [br, bd] = modulation(
        audio,
        channels,
        Math.max(0, roundEven((at - 0.7) * RATE)),
        Math.max(0, roundEven((at - 0.1) * RATE)),
        dsp
      );
      const [er, ed] = modulation(
        audio,
        channels,
        roundEven(at * RATE),
        Math.min(length, roundEven((at + 0.6) * RATE)),
        dsp
      );
      const row = {
        ...base,
        ...burst(fine, fp),
        ...rhythm(fine, fp, spectra, i),
        ...bands(audio, channels, at),
        roughness_before_ratio: br,
        roughness_event_ratio: er,
        roughness_ratio_delta: er - br,
        roughness_event_depth: ed,
        roughness_depth_delta: ed - bd
      };
      rows.push(row);
      windows.push({ slot: point.slot, start_s: at, end_s: Math.min(length / RATE, at + 0.3) });
      if (includePatches) {
        if (point.slot === 3) {
          patchIndices.push([-1, patches.length, -1]);
          patches.push(crop(at));
        } else {
          // Context duration uses the original candidate median, before physical-event reconstruction.
          const d = duration(coarse, i, Math.max(-13, point.baseline + 9));
          patchIndices.push([patches.length, patches.length + 1, patches.length + 2]);
          for (const time of [at - 0.1 - 0.975, at - 0.1, at + Math.max(0.3, Math.min(3, d))]) patches.push(crop(time));
        }
      }
    }
    const names = Object.keys(rows[0]).sort();
    return { names, physical: rows.map((r) => names.map((n) => r[n])), windows, patches, patchIndices };
  }
  const api = { extract, percentile, proposals };
  if (typeof module !== 'undefined') module.exports = api;
  else globalThis.SpokoynoFeatures = api;
})();
