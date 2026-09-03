#!/usr/bin/env python3
"""Extract full-track, event-oriented audio features from the supplied 2ch thread."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt

RATE = 16_000
WINDOW_S = 0.05
WINDOW = round(RATE * WINDOW_S)
FLOOR_DB = -90.0
POSITIVE_NAMES = {
    "17883557324650588814.webm": "confirmed positive #1",
    "17883557325462786367.mp4": "confirmed positive #2",
    "17883629069140053716.mp4": "confirmed positive #3",
    "17883659327260384359.webm": "confirmed positive #4 (immediate)",
    "17884174673280229863.mp4": "confirmed positive #5 (short spectral burst)",
    "17884274747240014140.mp4": "confirmed positive #6 (short clipped burst)",
}


def db_power(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-9))


def db_amp(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 10 ** (FLOOR_DB / 20)))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def high_shelf_sos(
    fs: int, f0: float = 1500.0, gain_db: float = 4.0, q: float = 1 / math.sqrt(2)
) -> np.ndarray:
    """RBJ high-shelf biquad, used with a high-pass as a cheap K-like weighting."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    c = math.cos(w0)
    beta = 2 * math.sqrt(a) * alpha
    b0 = a * ((a + 1) + (a - 1) * c + beta)
    b1 = -2 * a * ((a - 1) + (a + 1) * c)
    b2 = a * ((a + 1) + (a - 1) * c - beta)
    a0 = (a + 1) - (a - 1) * c + beta
    a1 = 2 * ((a - 1) - (a + 1) * c)
    a2 = (a + 1) - (a - 1) * c - beta
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def decode(path: Path) -> tuple[np.ndarray | None, str | None]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        if "does not contain any stream" in p.stderr.decode("utf-8", "replace"):
            return np.empty((0, 2), dtype=np.float32), None
        return (
            None,
            p.stderr.decode("utf-8", "replace").strip()
            or f"ffmpeg exit {p.returncode}",
        )
    if not p.stdout:
        return np.empty((0, 2), dtype=np.float32), None
    x = np.frombuffer(p.stdout, dtype="<f4")
    x = x[: len(x) // 2 * 2].reshape(-1, 2)
    return x, None


def rolling_event_arrays(
    level: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(level)
    baseline = np.full(n, np.nan)
    event = np.full(n, np.nan)
    baseline_mad = np.full(n, np.nan)
    for i in range(n):
        # Ignore the immediately preceding 100 ms so the baseline is not polluted by onset ramps.
        lo, hi = max(0, i - 60), i - 2
        if hi - lo >= 20:
            history = level[lo:hi]
            baseline[i] = np.median(history)
            baseline_mad[i] = np.median(np.abs(history - baseline[i]))
        if i + 6 <= n:
            # P25 over 300 ms rejects clicks and single-window spikes.
            event[i] = np.percentile(level[i : i + 6], 25)
    return baseline, event, baseline_mad


def duration_above(level: np.ndarray, start: int, threshold: float) -> float:
    last_above = start - 1
    # Bridge one 50 ms dip, but stop after 3 s; a screamer event need not occupy the whole tail.
    misses = 0
    for index in range(start, min(len(level), start + 60)):
        if level[index] >= threshold:
            last_above = index
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return max(0.0, (last_above - start + 1) * WINDOW_S)


def max_changes(level: np.ndarray) -> dict[str, float]:
    out = {}
    smooth = np.convolve(level, np.ones(2) / 2, mode="same")
    for seconds in (0.05, 0.1, 0.25, 0.5, 1.0):
        d = max(1, round(seconds / WINDOW_S))
        out[f"change_{int(seconds * 1000)}ms_db"] = (
            float(np.max(smooth[d:] - smooth[:-d])) if len(level) > d else 0.0
        )
    return out


def attack_time(fine_level: np.ndarray, coarse_index: int) -> float | None:
    """Measure the last -30 dB-to-near-peak rise on 10 ms K-weighted windows."""
    event_at = coarse_index * 5
    lo = max(0, event_at - 50)
    peak_lo = max(lo, event_at)
    peak_hi = min(len(fine_level), event_at + 50)
    if peak_hi <= peak_lo:
        return None
    peak = peak_lo + int(np.argmax(fine_level[peak_lo:peak_hi]))
    below = np.flatnonzero(fine_level[lo:peak] <= -30.0)
    if not len(below):
        return None
    last_below = lo + int(below[-1])
    target = float(fine_level[peak] - 3.0)
    reached = np.flatnonzero(fine_level[last_below + 1 : peak + 1] >= target)
    if not len(reached):
        return None
    return float((int(reached[0]) + 1) * 10)


def analyze(path: Path, meta: dict) -> dict:
    source_name = Path(meta["path"]).name
    row = {
        "file": source_name,
        "path": meta["path"],
        "md5": meta.get("md5", ""),
        "api_duration_s": meta.get("duration_secs", meta.get("api_duration_s")),
        "api_size_kib": meta.get("size", meta.get("api_size_kib")),
        "label": POSITIVE_NAMES.get(source_name, meta.get("label", "unlabeled")),
    }
    samples, error = decode(path)
    if error:
        return row | {"status": "decode-error", "error": error}
    if samples is None or not len(samples):
        return row | {
            "status": "no-audio",
            "score": 0.0,
            "old_score": 0,
            "classification": "no audio",
        }

    frame_count = len(samples)
    usable = frame_count // WINDOW * WINDOW
    if not usable:
        return row | {
            "status": "no-audio",
            "score": 0.0,
            "old_score": 0,
            "classification": "no audio",
        }
    x = samples[:usable]
    frames = x.reshape(-1, WINDOW, 2)
    raw_energy = np.mean(frames.astype(np.float64) ** 2, axis=(1, 2))
    raw_level = np.maximum(FLOOR_DB, db_power(raw_energy))
    frame_peak = np.max(np.abs(frames), axis=(1, 2))
    frame_peak_db = db_amp(frame_peak)

    hp = butter(2, 70, btype="highpass", fs=RATE, output="sos")
    shelf = high_shelf_sos(RATE)
    weighted = np.empty_like(x)
    for ch in range(2):
        weighted[:, ch] = sosfilt(shelf, sosfilt(hp, x[:, ch]))
    wframes = weighted.reshape(-1, WINDOW, 2)
    weighted_energy = np.mean(wframes.astype(np.float64) ** 2, axis=(1, 2))
    # Same offset as BS.1770's gated loudness convention; this is an approximation, not true LUFS.
    level = np.maximum(FLOOR_DB, -0.691 + db_power(weighted_energy))
    fine_window = round(RATE * 0.01)
    fine_usable = len(weighted) // fine_window * fine_window
    fine_frames = weighted[:fine_usable].reshape(-1, fine_window, 2)
    fine_level = np.maximum(
        FLOOR_DB,
        -0.691 + db_power(np.mean(fine_frames.astype(np.float64) ** 2, axis=(1, 2))),
    )
    del weighted, wframes, fine_frames

    # Coarse spectral bands are deliberately limited to interpretable broadband-change evidence.
    spectral_energy = np.mean(frames.astype(np.float64) ** 2, axis=(1, 2))
    diff_energy = np.mean(np.diff(frames, axis=1).astype(np.float64) ** 2, axis=(1, 2))
    brightness_db = db_power(diff_energy / np.maximum(spectral_energy, 1e-12))
    zcr = np.mean(
        np.signbit(frames[:, 1:, :]) != np.signbit(frames[:, :-1, :]), axis=(1, 2)
    )
    fft_size = 1 << (WINDOW - 1).bit_length()
    fft = np.fft.rfft(frames * np.hanning(WINDOW)[None, :, None], n=fft_size, axis=1)
    power = np.mean(np.abs(fft) ** 2 + 1e-12, axis=2)
    freqs = np.fft.rfftfreq(fft_size, 1 / RATE)
    keep = freqs <= 7800
    power, freqs = power[:, keep], freqs[keep]
    bands = []
    for low, high in ((80, 500), (500, 3000), (3000, 7800)):
        bands.append(
            db_power(np.mean(power[:, (freqs >= low) & (freqs < high)], axis=1))
        )
    band_db = np.stack(bands, axis=1)
    norm_spec = power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-12)
    spectral_flux = np.zeros(len(level))
    if len(level) > 1:
        spectral_flux[1:] = np.sqrt(
            np.sum(np.maximum(norm_spec[1:] - norm_spec[:-1], 0) ** 2, axis=1)
        )
    spectral_flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    flux_near = maximum_filter1d(spectral_flux, size=5, mode="nearest")

    # Separate safety path for a near-full-scale first event without a usable long baseline.
    start_limit = min(max(0, len(level) - 6), round(1.0 / WINDOW_S))
    start_events = np.array(
        [np.percentile(level[i : i + 6], 25) for i in range(start_limit + 1)]
    )
    start_best = int(np.argmax(start_events))
    start_level = float(start_events[start_best])
    start_target = max(-6.0, start_level - 3.0)
    start_candidates = np.flatnonzero(start_events >= start_target)
    start_preview = int(start_candidates[0]) if len(start_candidates) else start_best
    # The 300 ms look-ahead can become suspicious before the sound itself begins. Report and
    # measure from the first actually loud 50 ms window instead of that preview window.
    onset_threshold = max(-9.0, start_level - 6.0)
    onset_candidates = np.flatnonzero(
        level[start_preview : min(len(level), start_preview + 6)] >= onset_threshold
    )
    start_at = start_preview + (
        int(onset_candidates[0]) if len(onset_candidates) else 0
    )
    start_duration = duration_above(level, start_at, -9.0)
    start_end = min(len(level), start_at + 20)
    start_sample_end = min(len(x), start_end * WINDOW)
    start_samples = np.abs(x[start_at * WINDOW : start_sample_end])
    start_clip = (
        float(np.mean(start_samples >= 10 ** (-1 / 20))) if len(start_samples) else 0.0
    )
    start_flatness = float(
        np.median(spectral_flatness[start_at : min(len(level), start_at + 6)])
    )
    start_brightness = float(
        np.median(brightness_db[start_at : min(len(level), start_at + 6)])
    )
    start_loud_component = float(sigmoid((start_level + 6.0) / 2.0))
    start_duration_component = float(sigmoid((start_duration - 0.35) / 0.15))
    start_clip_component = float(sigmoid((start_clip - 0.01) / 0.03))
    start_noise_component = float(sigmoid((start_flatness - 0.025) / 0.025))
    start_brightness_component = float(sigmoid((start_brightness + 8.0) / 2.5))
    start_score = start_loud_component**1.1 * start_duration_component**0.7
    start_score *= (
        0.68
        + 0.12 * start_clip_component
        + 0.14 * start_noise_component
        + 0.06 * start_brightness_component
    )

    baseline, event, baseline_mad = rolling_event_arrays(level)
    jump = event - baseline
    valid = np.isfinite(jump)
    has_transition = bool(np.any(valid))
    if not has_transition:
        best = 0
        baseline = np.where(np.isfinite(baseline), baseline, np.median(level))
        event = np.where(np.isfinite(event), event, level)
        fallback_mad = np.median(np.abs(level - np.median(level)))
        baseline_mad = np.where(np.isfinite(baseline_mad), baseline_mad, fallback_mad)
        jump = event - baseline
    else:
        # Event score: loud + large contrast + sustained; quiet history and clipping are modest evidence.
        base_score = (
            sigmoid((event + 10.5) / 2.5) ** 0.9 * sigmoid((jump - 13.0) / 3.5) ** 1.25
        )
        base_score[~valid] = -1
        best = int(np.argmax(base_score))

    base = float(baseline[best])
    event_level = float(event[best])
    event_jump = float(jump[best])
    event_mad = float(baseline_mad[best])
    robust_scale = max(2.0, 1.4826 * event_mad)
    robust_z = event_jump / robust_scale
    event_attack_ms = attack_time(fine_level, best)
    raw_base = float(np.median(raw_level[max(0, best - 60) : max(0, best - 2)]))
    raw_event = float(np.percentile(raw_level[best : best + 6], 25))
    look_end = min(len(level), best + 10)
    event_peak_db = float(np.max(frame_peak_db[best:look_end]))
    threshold = max(-13.0, base + 9.0)
    event_duration = duration_above(level, best, threshold)
    persistence = float(np.mean(level[best:look_end] >= threshold))
    sample_end = min(len(x), (best + 10) * WINDOW)
    event_samples = np.abs(x[best * WINDOW : sample_end])
    event_near_clip = (
        float(np.mean(event_samples >= 10 ** (-1 / 20))) if len(event_samples) else 0.0
    )
    band_lo, band_hi = max(0, best - 40), max(0, best - 2)
    if band_hi - band_lo >= 10:
        band_jump = band_db[best] - np.median(band_db[band_lo:band_hi], axis=0)
        base_shape = np.mean(norm_spec[band_lo:band_hi], axis=0)
        event_shape = np.mean(norm_spec[best : min(len(level), best + 6)], axis=0)
        spectral_shape_distance = float(
            np.sqrt(np.sum((np.sqrt(event_shape) - np.sqrt(base_shape)) ** 2))
            / math.sqrt(2)
        )
    else:
        band_jump = np.zeros(3)
        spectral_shape_distance = 0.0
    broadband_jump = float(np.min(band_jump))

    loud_component = float(sigmoid((event_level + 10.5) / 2.5))
    jump_component = float(sigmoid((event_jump - 13.0) / 3.5))
    duration_component = float(sigmoid((event_duration - 0.18) / 0.09))
    quiet_component = float(sigmoid((-base - 15.0) / 5.0))
    broadband_component = float(sigmoid((broadband_jump - 5.0) / 4.0))
    clip_component = float(sigmoid((event_near_clip - 0.005) / 0.012))
    flux_component = float(sigmoid((spectral_flux[best] - 0.22) / 0.08))
    transition_score = (
        (loud_component**0.9) * (jump_component**1.25) * (duration_component**0.65)
    )
    transition_score *= 0.87 + 0.10 * quiet_component + 0.03 * clip_component
    transition_score *= 0.65 + 0.35 * flux_component
    transition_score = float(np.clip(transition_score, 0, 1)) if has_transition else 0.0
    transition_eligible = (
        has_transition
        and event_level >= -6
        and event_jump >= 14
        and event_duration >= 0.15
    )
    spectral_burst_eligible = (
        has_transition
        and event_level >= -4
        and event_jump >= 16
        and 0.3 <= event_duration <= 1.05
        and flux_near[best] >= 0.3
        and spectral_shape_distance >= 0.8
    )
    clipped_burst_eligible = (
        has_transition
        and event_level >= -3
        and event_jump >= 10
        and 0.25 <= event_duration <= 0.55
        and event_near_clip >= 0.35
        and flux_near[best] >= 0.3
    )
    spectral_burst_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((spectral_shape_distance - 0.85) / 0.08))
            + 0.04 * float(sigmoid((event_jump - 18) / 3))
            + 0.04 * float(sigmoid((event_level + 3) / 1.5)),
        )
        if spectral_burst_eligible
        else 0.0
    )
    clipped_burst_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((event_near_clip - 0.4) / 0.08))
            + 0.04 * float(sigmoid((event_jump - 12) / 2))
            + 0.04 * float(sigmoid((event_level + 2) / 1.2)),
        )
        if clipped_burst_eligible
        else 0.0
    )
    rescue_score = max(spectral_burst_score, clipped_burst_score)
    rescue_mode = (
        "short-clipped-burst"
        if clipped_burst_score > spectral_burst_score
        else "short-spectral-burst"
    )
    start_spectral_evidence = (
        start_flatness >= 0.04 or start_brightness >= -5.0 or start_clip >= 0.08
    )
    start_eligible = (
        start_level >= -3.0 and start_duration >= 0.50 and start_spectral_evidence
    )
    transition_decision_score = transition_score if transition_eligible else 0.0
    start_decision_score = start_score if start_eligible else 0.0
    decision_score = float(
        max(transition_decision_score, start_decision_score, rescue_score)
    )
    suspicious = decision_score >= 0.80
    if rescue_score == decision_score and rescue_score > 0:
        decision_mode = rescue_mode
    elif start_decision_score > transition_decision_score:
        decision_mode = "loud-start"
    else:
        decision_mode = "transition"
    risk_mode = "loud-start" if start_score > transition_score else "transition"
    detection_mode = decision_mode if suspicious else risk_mode
    confidence = float(
        start_score
        if detection_mode == "loud-start"
        else rescue_score
        if detection_mode in ("short-spectral-burst", "short-clipped-burst")
        else transition_score
    )

    # Reproduce the old detector's ~256 samples/channel per 100 ms window.
    old_win = round(RATE * 0.1)
    old_values = []
    for start in range(0, len(x), old_win):
        end = min(len(x), start + old_win)
        step = max(1, (end - start) // 256)
        old_values.append(db_power(np.mean(x[start:end:step].astype(np.float64) ** 2)))
    old_level = np.asarray(old_values)
    old_median = float(np.median(old_level))
    old_peak = float(np.max(old_level))
    old_jump, old_at = -math.inf, 0
    for i in range(1, len(old_level)):
        j = float(old_level[i] - np.median(old_level[max(0, i - 10) : i]))
        if j > old_jump:
            old_jump, old_at = j, i
    old_dynamic = old_peak - old_median
    old_score = (
        int(old_peak > -8)
        + int(old_dynamic >= 14)
        + int(old_jump >= 12)
        + int(old_median < -18 and old_peak > -7)
    )

    percentiles = {
        f"p{p}_db": float(np.percentile(level, p)) for p in (10, 25, 50, 75, 90, 95, 99)
    }
    abs_samples = np.abs(x)
    rms_all_db = float(db_power(np.mean(x.astype(np.float64) ** 2)))
    peak_db = float(db_amp(np.max(abs_samples)))
    row.update(
        {
            "status": "ok",
            "duration_s": frame_count / RATE,
            **percentiles,
            "raw_median_dbfs": float(np.median(raw_level)),
            "rms_dbfs": rms_all_db,
            "peak_dbfs": peak_db,
            "crest_db": peak_db - rms_all_db,
            "peak_to_median_db": float(np.max(level) - np.median(level)),
            "event_at_s": best * WINDOW_S,
            "baseline_db": base,
            "event_db": event_level,
            "jump_db": event_jump,
            "baseline_mad_db": event_mad,
            "robust_scale_db": robust_scale,
            "robust_z": robust_z,
            "attack_ms": event_attack_ms,
            "raw_baseline_dbfs": raw_base,
            "raw_event_dbfs": raw_event,
            "raw_jump_db": raw_event - raw_base,
            "event_duration_s": event_duration,
            "event_persistence": persistence,
            "event_peak_dbfs": event_peak_db,
            "event_near_clip_pct": event_near_clip * 100,
            "band_low_jump_db": float(band_jump[0]),
            "band_mid_jump_db": float(band_jump[1]),
            "band_high_jump_db": float(band_jump[2]),
            "broadband_jump_db": broadband_jump,
            "spectral_flux_at_event": float(spectral_flux[best]),
            "spectral_flux_near_event": float(flux_near[best]),
            "max_spectral_flux": float(np.max(spectral_flux)),
            "spectral_shape_distance": spectral_shape_distance,
            "event_spectral_flatness": float(
                np.median(spectral_flatness[best : min(len(level), best + 6)])
            ),
            "event_brightness_db": float(
                np.median(brightness_db[best : min(len(level), best + 6)])
            ),
            "baseline_brightness_db": (
                float(np.median(brightness_db[band_lo:band_hi]))
                if band_hi > band_lo
                else 0.0
            ),
            "brightness_change_db": (
                float(
                    np.median(brightness_db[best : min(len(level), best + 6)])
                    - np.median(brightness_db[band_lo:band_hi])
                )
                if band_hi > band_lo
                else 0.0
            ),
            "event_zcr": float(np.median(zcr[best : min(len(level), best + 6)])),
            "start_at_s": start_at * WINDOW_S,
            "start_event_db": start_level,
            "start_duration_s": start_duration,
            "start_near_clip_pct": start_clip * 100,
            "start_spectral_flatness": start_flatness,
            "start_brightness_db": start_brightness,
            "start_score": float(start_score),
            **max_changes(level),
            **{
                f"sample_above_{d}db_pct": float(
                    np.mean(abs_samples >= 10 ** (d / 20)) * 100
                )
                for d in (-3, -6, -9, -12)
            },
            **{
                f"window_above_{d}db_pct": float(np.mean(level >= d) * 100)
                for d in (-3, -6, -9, -12)
            },
            "old_median_db": old_median,
            "old_peak_window_db": old_peak,
            "old_dynamic_range_db": old_dynamic,
            "old_max_jump_db": old_jump,
            "old_jump_at_s": old_at * 0.1,
            "old_score": old_score,
            "old_classification": "suspicious" if old_score >= 3 else "normal",
            "transition_score": transition_score,
            "rescue_score": rescue_score,
            "rescue_mode": rescue_mode if rescue_score else None,
            "decision_score": decision_score,
            "score": confidence,
            "suspicious": suspicious,
            "detection_mode": detection_mode,
            "decision_mode": decision_mode,
            "risk_mode": risk_mode,
            "classification": "suspicious" if suspicious else "normal",
        }
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread-json", type=Path, required=True)
    ap.add_argument("--media-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--title", default="Thread 336185346 — complete audio dataset")
    ap.add_argument(
        "--description",
        default="Sorted by the event detector's suspicion score. `unlabeled` rows are treated as provisional negatives for evaluation.",
    )
    args = ap.parse_args()
    data = json.loads(args.thread_json.read_text())
    if isinstance(data, list):
        files = data
    else:
        files = [
            f
            for p in data["threads"][0]["posts"]
            for f in (p.get("files") or [])
            if Path(f.get("path", "")).suffix.lower()
            in {".mp4", ".webm", ".m4v", ".mov", ".ogv"}
        ]
    if args.limit:
        files = files[: args.limit]
    rows = []
    for index, meta in enumerate(files, 1):
        path = args.media_dir / f"{Path(meta['path']).name}.audio.wav"
        print(f"[{index}/{len(files)}] {path.name}", file=sys.stderr, flush=True)
        if not path.exists() and meta.get("status") == "no-audio":
            rows.append(meta)
        else:
            rows.append(analyze(path, meta))
    rows.sort(key=lambda r: float(r.get("score", -1)), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    fields = sorted({k for row in rows for k in row})
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        ("file", "file"),
        ("label", "label/status"),
        ("duration_s", "duration s"),
        ("p50_db", "median dB"),
        ("peak_dbfs", "peak dBFS"),
        ("event_at_s", "event s"),
        ("baseline_db", "baseline dB"),
        ("event_db", "event dB"),
        ("jump_db", "jump dB"),
        ("baseline_mad_db", "baseline MAD dB"),
        ("robust_z", "robust z"),
        ("attack_ms", "attack ms"),
        ("event_duration_s", "event duration s"),
        ("event_near_clip_pct", "near-clip %"),
        ("spectral_flux_at_event", "onset flux"),
        ("spectral_flux_near_event", "nearby flux"),
        ("spectral_shape_distance", "spectral distance"),
        ("transition_score", "transition score"),
        ("start_score", "start score"),
        ("rescue_score", "short-burst score"),
        ("decision_score", "decision score"),
        ("score", "risk score"),
        ("detection_mode", "mode"),
        ("classification", "new"),
        ("old_score", "old score"),
    ]
    with args.output.with_suffix(".md").open("w") as f:
        f.write(f"# {args.title}\n\n")
        f.write(f"{args.description}\n\n")
        f.write("| " + " | ".join(title for _, title in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            values = []
            for key, _ in columns:
                value = row.get(key, "")
                if key == "label" and row.get("status") == "no-audio":
                    value = "no audio"
                elif isinstance(value, float):
                    value = (
                        f"{value:.3f}"
                        if key
                        in {
                            "score",
                            "transition_score",
                            "start_score",
                            "rescue_score",
                            "spectral_flux_at_event",
                            "spectral_flux_near_event",
                            "spectral_shape_distance",
                        }
                        else f"{value:.2f}"
                    )
                values.append(str(value).replace("|", "\\|"))
            f.write("| " + " | ".join(values) + " |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
