#!/usr/bin/env python3
"""Audio-only v2 features and short-event proposals, independent of labels.

Preserves the v1 proposals and records which extra proposals were introduced.
Roughness is a modulation-energy proxy, not a perceptual roughness calibration.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
from event_dataset import atomic_npz, window_label
from extract_audio_context import sha256
from extract_features import (
    BAND_EDGES,
    RATE,
    WINDOW_S,
    Spectra,
    db_power,
    event_features,
    percentile,
    read_audio,
    timelines,
)
from scipy.signal import butter, hilbert, resample_poly, sosfilt
from train_models import atomic_json

FAMILIES = ("burst", "rhythm", "bands", "roughness")


def relative_attack(fine, point):
    lo, hi = max(0, point - 10), min(len(fine), point + 31)
    if hi <= point:
        return np.nan
    peak = point + int(np.argmax(fine[point:hi]))
    background = percentile(fine[max(0, point - 100) : max(0, point - 5)], 50)
    base_power, peak_power = 10 ** (background / 10), 10 ** (fine[peak] / 10)
    if peak_power <= base_power * 1.01:
        return 0.0
    power = 10 ** (fine[lo : peak + 1] / 10)
    low, high = base_power + np.asarray([0.2, 0.8]) * (peak_power - base_power)
    below = np.flatnonzero(power < low)
    start = int(below[-1]) if len(below) else 0
    crossing = np.flatnonzero(power[start:] >= high)
    return float(crossing[0] * 10) if len(crossing) else np.nan


def burst_features(fine, point):
    history = fine[max(0, point - 300) : max(0, point - 5)]
    baseline = percentile(history, 50)
    end = min(len(fine), point + 200)
    early = fine[point : min(end, point + 30)]
    peak = point + int(np.argmax(early)) if len(early) else point
    peak_db = float(fine[peak])
    threshold = max(baseline + 6, peak_db - 10)
    left, right = peak, peak + 1
    while left > max(0, point - 10) and fine[left - 1] >= threshold:
        left -= 1
    while right < end and fine[right] >= threshold:
        right += 1
    active = fine[left:right]
    tail = fine[min(len(fine), right + 5) : min(len(fine), right + 35)]
    following = fine[point : min(len(fine), point + 100)]
    result = {
        "burst_peak_10ms_db": peak_db,
        "burst_peak_jump_db": peak_db - baseline,
        "burst_relative_attack_ms": relative_attack(fine, point),
        "burst_active_duration_s": (right - left) / 100,
        "burst_active_median_db": percentile(active, 50),
        "burst_active_energy_db": float(db_power(np.mean(10 ** (active / 10)))),
        "burst_tail_drop_db": peak_db - percentile(tail, 50) if len(tail) else np.nan,
        "burst_peak_to_1s_median_db": peak_db - percentile(following, 50),
        "burst_active_fraction_1s": float(np.mean(following >= threshold)),
    }
    for length in (5, 10, 20):
        segment = fine[max(0, point - 5) : min(len(fine), point + 40)]
        power = 10 ** (segment / 10)
        averaged = (
            np.convolve(power, np.ones(length) / length, mode="valid")
            if len(power) >= length
            else power
        )
        result[f"burst_peak_{length * 10}ms_db"] = float(db_power(max(averaged)))
    return result


def rhythm_features(fine, point, spectra, coarse):
    history = fine[max(0, point - 800) : point]
    onset = np.maximum(np.diff(history), 0)
    if len(onset):
        onset *= 1 / (1 + np.exp(-np.clip((history[1:] + 25) / 6, -30, 30)))
    best = 0.0
    for lag in range(20, min(151, len(onset) // 2)):
        a, b = onset[:-lag], onset[lag:]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 1e-8:
            best = max(best, float(np.dot(a, b) / denominator))
    current = fine[max(0, point - 5) : min(len(fine), point + 20)]
    rise = float(max(np.diff(current), default=0))
    previous = percentile(onset, 95) if len(onset) else 0.0
    event = spectra.aggregate(range(coarse, min(coarse + 2, len(fine) // 5)))
    profiles = [
        spectra.frame(i) for i in range(max(0, coarse - 160), max(0, coarse - 10), 5)
    ]
    recurrence = min(
        (
            float(np.linalg.norm(np.sqrt(event) - np.sqrt(p)) / np.sqrt(2))
            for p in profiles
        ),
        default=np.nan,
    )
    return {
        "rhythm_onset_periodicity": best if len(onset) >= 60 else np.nan,
        "rhythm_onset_surprise_ratio": min(30.0, rise / max(1.0, previous)),
        "rhythm_previous_onset_p95_db": previous,
        "rhythm_previous_spectral_distance_min": recurrence,
    }


def band_features(samples, at):
    start, stop = (
        max(0, round((at - 1.0) * RATE)),
        min(len(samples), round((at + 0.4) * RATE)),
    )
    chunk = np.asarray(samples[start:stop], dtype=np.float64)
    if len(chunk) < 512:
        return {
            f"bands_{kind}_{lo}_{hi}": np.nan
            for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True)
            for kind in ("event_db", "jump_db")
        }
    frames = np.lib.stride_tricks.sliding_window_view(chunk, 512, axis=0)[::160]
    window = np.hanning(512)
    fft = np.fft.rfft(frames * window, axis=-1)
    power = np.mean(np.abs(fft) ** 2, axis=1) / (512 * np.sum(window**2))
    power[:, 1:-1] *= 2
    time = start / RATE + (np.arange(len(power)) * 160 + 256) / RATE
    before, during = (time < at - 0.05), ((time >= at) & (time < at + 0.3))
    freq = np.fft.rfftfreq(512, 1 / RATE)
    result = {}
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
        values = db_power(power[:, (freq >= lo) & (freq < hi)].sum(axis=1))
        peak = percentile(values[during], 90)
        result[f"bands_event_db_{lo}_{hi}"] = peak
        result[f"bands_jump_db_{lo}_{hi}"] = (
            peak - percentile(values[before], 50) if np.any(before) else np.nan
        )
    return result


def modulation(samples):
    if len(samples) < RATE // 5:
        return np.nan, np.nan
    # Envelopes sampled at 500 Hz can represent the 30..150 Hz range; the
    # original 10/50 ms level timelines cannot represent all of this range.
    energy = np.mean(np.square(samples[::16], dtype=np.float64), axis=0)
    mono = np.asarray(samples[:, int(np.argmax(energy))], dtype=np.float64)
    filtered = sosfilt(butter(2, 250, btype="highpass", fs=RATE, output="sos"), mono)
    envelope = resample_poly(np.abs(hilbert(filtered)), 1, 32)
    mean = float(np.mean(envelope))
    if mean < 1e-7:
        return 0.0, 0.0
    centered = envelope - mean
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequency = np.fft.rfftfreq(len(centered), 1 / 500)
    total = power[(frequency >= 4) & (frequency <= 200)].sum()
    rough = power[(frequency >= 30) & (frequency <= 150)].sum() / max(total, 1e-20)
    return float(rough), float(np.std(envelope) / mean)


def roughness_features(samples, at):
    before = samples[
        max(0, round((at - 0.7) * RATE)) : max(0, round((at - 0.1) * RATE))
    ]
    during = samples[round(at * RATE) : min(len(samples), round((at + 0.6) * RATE))]
    br, bd = modulation(before)
    er, ed = modulation(during)
    return {
        "roughness_before_ratio": br,
        "roughness_event_ratio": er,
        "roughness_ratio_delta": er - br,
        "roughness_event_depth": ed,
        "roughness_depth_delta": ed - bd,
    }


def short_proposals(fine, original, count=4):
    # Rank 50 ms peaks against the preceding 500 ms. O(N) preprocessing and
    # deterministic nonmaximum suppression, without any class/timing labels.
    power = 10 ** (fine.astype(np.float64) / 10)
    cumulative = np.r_[0.0, np.cumsum(power)]
    indices = np.arange(15, len(fine) - 5)
    if not len(indices):
        return []
    before_lo, before_hi = np.maximum(0, indices - 55), indices - 5
    before = (cumulative[before_hi] - cumulative[before_lo]) / (before_hi - before_lo)
    after = (cumulative[indices + 5] - cumulative[indices]) / 5
    level = db_power(after)
    jump = level - db_power(before)
    rank = np.maximum(jump, 0) / (1 + np.exp(-np.clip((level + 20) / 6, -30, 30)))
    selected = []
    for offset in np.argsort(-rank, kind="stable"):
        if rank[offset] <= 0:
            break
        at = round(int(indices[offset]) / 5) * WINDOW_S
        if any(abs(at - t) < 0.15 for t in original) or any(
            abs(at - t) < 0.5 for t in selected
        ):
            continue
        selected.append(at)
        if len(selected) == count:
            break
    return selected


def base_at(level, fine, stats, spectra, at, opening=False):
    index = round(at / WINDOW_S)
    history = level[max(0, index - 60) : max(0, index - 2)]
    baseline = percentile(history, 50)
    event = percentile(level[index : index + 6], 25)
    candidate = {
        "index": index,
        "baseline": baseline,
        "event": event,
        "jump": event - baseline,
        "mad": percentile(np.abs(history - baseline), 50) if len(history) else 0.0,
    }
    vector = event_features(candidate, level, fine, stats["raw_clip"], spectra)
    vector.pop("position_fraction")
    vector["is_opening"] = float(opening)
    if not len(history):
        for name in vector:
            if (
                "baseline" in name
                or "jump" in name
                or name in ("robust_z", "spectral_shape_distance")
            ):
                vector[name] = np.nan
    return vector


def extract(samples, original):
    level, fine, stats = timelines(samples)
    spectra = Spectra(samples)
    times = [w["start_s"] for w in original]
    points = [dict(w, legacy=True) for w in original]
    points += [
        {
            "start_s": t,
            "end_s": min(t + 0.3, len(samples) / RATE),
            "slot": 4 + i,
            "legacy": False,
        }
        for i, t in enumerate(short_proposals(fine, times))
    ]
    vectors = []
    for point in points:
        at = point["start_s"]
        fine_at = min(len(fine) - 1, round(at * 100))
        vector = base_at(level, fine, stats, spectra, at, point["slot"] == 3)
        vector.update(burst_features(fine, fine_at))
        vector.update(rhythm_features(fine, fine_at, spectra, round(at / WINDOW_S)))
        vector.update(band_features(samples, at))
        vector.update(roughness_features(samples, at))
        vectors.append(vector)
    return vectors, points


def task(clip, windows, audio_dir, cache, extractor_hash):
    source = audio_dir / f"{clip['file']}.audio.wav"
    if sha256(source) != clip["audio_sha256"]:
        raise ValueError(f"changed audio: {source}")
    key = hashlib.sha256(
        (
            extractor_hash + clip["audio_sha256"] + json.dumps(windows, sort_keys=True)
        ).encode()
    ).hexdigest()
    target = cache / f"{key}.json"
    if target.exists():
        return json.loads(target.read_text())
    vectors, points = extract(read_audio(source), windows)
    payload = {
        "vectors": [
            {n: float(v) if np.isfinite(v) else None for n, v in row.items()}
            for row in vectors
        ],
        "windows": points,
    }
    atomic_json(target, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise ValueError("refusing to overwrite extracted snapshot")
    base = json.loads((args.base / "manifest.json").read_text())
    extractor_hash = sha256(Path(__file__)) + sha256(
        Path("research/extract_features.py")
    )
    cache = args.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for clip in base["clips"]:
            lo = clip["first_window"]
            original = base["windows"][lo : lo + clip["window_count"]]
            futures.append(
                pool.submit(task, clip, original, args.audio_dir, cache, extractor_hash)
            )
        rows, clips, windows, targets, legacy_indices = [], [], [], [], []
        names = None
        for owner, (clip, future) in enumerate(
            zip(base["clips"], futures, strict=True)
        ):
            extracted = future.result()
            names = names or sorted(extracted["vectors"][0])
            clips.append(
                clip
                | {
                    "first_window": len(windows),
                    "window_count": len(extracted["windows"]),
                }
            )
            for offset, (vector, point) in enumerate(
                zip(extracted["vectors"], extracted["windows"], strict=True)
            ):
                if sorted(vector) != names:
                    raise ValueError("feature schema mismatch")
                rows.append([np.nan if vector[n] is None else vector[n] for n in names])
                windows.append(point | {"clip_index": owner})
                targets.append(
                    window_label(
                        clip["label"],
                        clip["annotation"],
                        point["start_s"],
                        point["end_s"],
                        clip["duration_s"],
                    )
                )
                legacy_indices.append(
                    clip["first_window"] + offset if point["legacy"] else -1
                )
            if (owner + 1) % 100 == 0 or owner + 1 == len(base["clips"]):
                print(
                    f"new acoustic features {owner + 1}/{len(base['clips'])}",
                    flush=True,
                )
    path = args.output_dir / "examples.npz"
    atomic_npz(
        path,
        physical=np.asarray(rows),
        targets=np.asarray(targets, np.int8),
        legacy_indices=np.asarray(legacy_indices),
    )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": 2,
            "source_sha256": sha256(Path(__file__)),
            "base_manifest_sha256": sha256(args.base / "manifest.json"),
            "matrix_sha256": sha256(path),
            "physical_names": names,
            "families": FAMILIES,
            "clips": clips,
            "windows": windows,
            "policy": "All original proposals plus up to four short-onset proposals; label-independent extraction. Original annotations are targets only.",
        },
    )


if __name__ == "__main__":
    main()
