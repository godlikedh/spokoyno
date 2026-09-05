#!/usr/bin/env python3
"""Extract deterministic clip/event features for Spokoyno's ML playground."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from dataset import label_for, load_labels, thread_group
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

RATE = 16_000
WINDOW = 800
FINE_WINDOW = 160
WINDOW_S = WINDOW / RATE
FLOOR_DB = -90.0
FEATURE_VERSION = 1
TOP_EVENTS = 3
BAND_EDGES = (70, 250, 500, 1000, 2000, 4000, 8000)


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def db_power(value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(value, 1e-9))


def high_shelf_sos(
    fs: int, frequency: float = 1500.0, gain_db: float = 4.0
) -> np.ndarray:
    q = 1 / math.sqrt(2)
    amplitude = 10 ** (gain_db / 40)
    omega = 2 * math.pi * frequency / fs
    alpha = math.sin(omega) / (2 * q)
    cosine = math.cos(omega)
    beta = 2 * math.sqrt(amplitude) * alpha
    b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + beta)
    b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
    b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - beta)
    a0 = (amplitude + 1) - (amplitude - 1) * cosine + beta
    a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cosine)
    a2 = (amplitude + 1) - (amplitude - 1) * cosine - beta
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def read_audio(path: Path) -> np.ndarray:
    rate, samples = wavfile.read(path, mmap=True)
    if rate != RATE:
        raise ValueError(f"expected {RATE} Hz, got {rate}")
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] not in (1, 2):
        raise ValueError(f"expected mono/stereo audio, got {samples.shape[1]} channels")
    if np.issubdtype(samples.dtype, np.integer):
        scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / scale
    return samples


def timelines(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    windows = len(samples) // WINDOW
    if windows < 6:
        raise ValueError("audio is shorter than 300 ms")
    channels = samples.shape[1]
    coarse = np.empty(windows, dtype=np.float32)
    fine = np.empty(windows * (WINDOW // FINE_WINDOW), dtype=np.float32)
    raw_peak = np.empty(windows, dtype=np.float32)
    raw_clip = np.empty(windows, dtype=np.float32)
    sos = np.vstack(
        (butter(2, 70, btype="highpass", fs=RATE, output="sos"), high_shelf_sos(RATE))
    )
    states = [np.zeros((len(sos), 2), dtype=np.float64) for _ in range(channels)]
    block_windows = 512
    for first in range(0, windows, block_windows):
        count = min(block_windows, windows - first)
        start, end = first * WINDOW, (first + count) * WINDOW
        block = np.asarray(samples[start:end], dtype=np.float32)
        weighted = np.empty_like(block)
        for channel in range(channels):
            weighted[:, channel], states[channel] = sosfilt(
                sos, block[:, channel], zi=states[channel]
            )
        frames = weighted.reshape(count, WINDOW, channels).astype(np.float64)
        coarse[first : first + count] = np.maximum(
            FLOOR_DB, -0.691 + db_power(np.mean(frames * frames, axis=(1, 2)))
        )
        fine_frames = weighted.reshape(-1, FINE_WINDOW, channels).astype(np.float64)
        fine[first * 5 : (first + count) * 5] = np.maximum(
            FLOOR_DB, -0.691 + db_power(np.mean(fine_frames * fine_frames, axis=(1, 2)))
        )
        raw = block.reshape(count, WINDOW, channels)
        raw_peak[first : first + count] = np.max(np.abs(raw), axis=(1, 2))
        raw_clip[first : first + count] = np.mean(
            np.abs(raw) >= 10 ** (-1 / 20), axis=(1, 2)
        )
    global_stats = {
        "duration_s": len(samples) / RATE,
        "global_raw_peak_dbfs": float(
            20 * np.log10(max(float(np.max(raw_peak)), 1e-9))
        ),
        "global_raw_clip_fraction": float(np.mean(raw_clip)),
    }
    return coarse, fine, global_stats | {"raw_clip": raw_clip}


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value)) if len(values) else FLOOR_DB


def duration_above(
    level: np.ndarray, start: int, threshold: float, limit: int = 120
) -> float:
    last = start - 1
    misses = 0
    for index in range(start, min(len(level), start + limit)):
        if level[index] >= threshold:
            last, misses = index, 0
        else:
            misses += 1
            if misses >= 2:
                break
    return max(0.0, (last - start + 1) * WINDOW_S)


def attack_ms(fine: np.ndarray, coarse_index: int) -> float:
    event = coarse_index * 5
    lo, hi = max(0, event - 100), min(len(fine), event + 100)
    if hi <= event:
        return 2000.0
    peak = event + int(np.argmax(fine[event:hi]))
    below = np.flatnonzero(fine[lo:peak] <= -30)
    if not len(below):
        return 2000.0
    origin = lo + int(below[-1])
    target = float(fine[peak] - 3)
    reached = np.flatnonzero(fine[origin + 1 : peak + 1] >= target)
    return float((int(reached[0]) + 1) * 10) if len(reached) else 2000.0


def candidate_events(level: np.ndarray, count: int = TOP_EVENTS) -> list[dict]:
    candidates = []
    for index in range(22, len(level) - 5):
        lo, hi = max(0, index - 60), index - 2
        history = level[lo:hi]
        baseline = float(np.median(history))
        mad = float(np.median(np.abs(history - baseline)))
        event = percentile(level[index : index + 6], 25)
        jump = event - baseline
        score = float(
            sigmoid((event + 10.5) / 2.5) ** 0.9
            * sigmoid((jump - 13) / 3.5) ** 1.25
            * (0.65 + 0.35 * sigmoid((jump / max(2, 1.4826 * mad) - 6) / 2))
        )
        candidates.append(
            {
                "index": index,
                "baseline": baseline,
                "mad": mad,
                "event": event,
                "jump": jump,
                "selection_score": score,
            }
        )
    selected = []
    for candidate in sorted(
        candidates, key=lambda row: row["selection_score"], reverse=True
    ):
        if any(abs(candidate["index"] - row["index"]) < 20 for row in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


class Spectra:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.cache: dict[int, np.ndarray] = {}
        self.frequencies = np.fft.rfftfreq(1024, 1 / RATE)
        self.window = np.hanning(WINDOW).astype(np.float32)

    def frame(self, index: int) -> np.ndarray:
        if index in self.cache:
            return self.cache[index]
        start = index * WINDOW
        frame = np.asarray(self.samples[start : start + WINDOW], dtype=np.float32)
        if len(frame) < WINDOW:
            result = np.full(513, 1 / 513, dtype=np.float64)
        else:
            spectrum = np.fft.rfft(frame * self.window[:, None], n=1024, axis=0)
            power = np.mean(np.abs(spectrum) ** 2, axis=1).astype(np.float64) + 1e-12
            result = power / np.sum(power)
        self.cache[index] = result
        return result

    def aggregate(self, indices: range | list[int]) -> np.ndarray:
        profiles = [self.frame(index) for index in indices if index >= 0]
        if not profiles:
            return np.full(513, 1 / 513, dtype=np.float64)
        result = np.mean(profiles, axis=0)
        return result / np.sum(result)

    def describe_pair(self, baseline: np.ndarray, event: np.ndarray) -> dict:
        shape = float(
            np.sqrt(np.sum((np.sqrt(event) - np.sqrt(baseline)) ** 2)) / math.sqrt(2)
        )
        event_flatness = float(np.exp(np.mean(np.log(event + 1e-12))) / np.mean(event))
        base_flatness = float(
            np.exp(np.mean(np.log(baseline + 1e-12))) / np.mean(baseline)
        )
        event_centroid = float(np.sum(self.frequencies * event))
        base_centroid = float(np.sum(self.frequencies * baseline))
        result = {
            "spectral_shape_distance": shape,
            "spectral_flatness_delta": event_flatness - base_flatness,
            "spectral_centroid_delta_hz": event_centroid - base_centroid,
        }
        for low, high in pairwise(BAND_EDGES):
            mask = (self.frequencies >= low) & (self.frequencies < high)
            result[f"band_{low}_{high}_jump_db"] = float(
                10
                * np.log10(
                    (np.sum(event[mask]) + 1e-9) / (np.sum(baseline[mask]) + 1e-9)
                )
            )
        return result


def event_features(
    candidate: dict,
    level: np.ndarray,
    fine: np.ndarray,
    raw_clip: np.ndarray,
    spectra: Spectra,
) -> dict:
    index = candidate["index"]
    result = {
        "position_fraction": index / max(1, len(level) - 1),
        "history_s": min(index * WINDOW_S, 6.0),
        "event_level_db": candidate["event"],
        "jump_db": candidate["jump"],
        "baseline_mad_db": candidate["mad"],
        "robust_z": candidate["jump"] / max(2, 1.4826 * candidate["mad"]),
        "attack_ms": attack_ms(fine, index),
    }
    guard = max(0, index - 2)
    baselines = {}
    for seconds, windows in ((0.5, 10), (1.0, 20), (3.0, 60), (6.0, 120)):
        history = level[max(0, guard - windows) : guard]
        baseline = percentile(history, 50)
        mad = percentile(np.abs(history - baseline), 50) if len(history) else 0.0
        key = str(seconds).replace(".", "_")
        result[f"baseline_{key}s_db"] = baseline
        result[f"baseline_{key}s_mad_db"] = mad
        result[f"jump_vs_{key}s_db"] = candidate["event"] - baseline
        baselines[seconds] = baseline
    for seconds, windows in ((0.1, 2), (0.3, 6), (0.6, 12), (1.0, 20)):
        key = str(seconds).replace(".", "_")
        result[f"event_{key}s_db"] = percentile(level[index : index + windows], 25)
    threshold = max(-13, candidate["baseline"] + 9)
    result["duration_s"] = duration_above(level, index, threshold)
    result["duration_plus6_s"] = duration_above(level, index, candidate["baseline"] + 6)
    result["duration_plus12_s"] = duration_above(
        level, index, candidate["baseline"] + 12
    )
    following = level[index : min(len(level), index + 20)]
    result["persistence_1s"] = (
        float(np.mean(following >= threshold)) if len(following) else 0.0
    )
    tail = level[index : min(len(level), index + 60)]
    result["loudness_area_db"] = float(
        np.mean(np.maximum(tail - candidate["baseline"], 0))
    )
    history = level[max(0, index - 60) : index]
    differences = np.diff(history)
    result["baseline_iqr_db"] = percentile(history, 75) - percentile(history, 25)
    result["baseline_derivative_db"] = (
        percentile(np.abs(differences), 50) if len(differences) else 0.0
    )
    result["prior_abrupt_changes"] = (
        float(np.sum(differences >= 8)) if len(differences) else 0.0
    )
    clip_end = min(
        len(raw_clip), index + max(6, round(max(result["duration_s"], 0.3) / WINDOW_S))
    )
    result["near_clip_fraction"] = (
        float(np.mean(raw_clip[index:clip_end])) if clip_end > index else 0.0
    )

    base_indices = list(range(max(0, index - 42), max(0, index - 2), 2))
    baseline_profile = spectra.aggregate(base_indices)
    event_profile = spectra.aggregate(range(index, min(len(level), index + 6)))
    result.update(spectra.describe_pair(baseline_profile, event_profile))
    previous = spectra.frame(max(0, index - 1))
    current = spectra.frame(index)
    result["spectral_flux"] = float(np.sum(np.maximum(current - previous, 0)))
    nearby = []
    for offset in range(-2, 3):
        point = index + offset
        if 1 <= point < len(level):
            nearby.append(
                float(
                    np.sum(
                        np.maximum(spectra.frame(point) - spectra.frame(point - 1), 0)
                    )
                )
            )
    result["nearby_spectral_flux"] = max(nearby, default=0.0)
    return result


def start_features(
    samples: np.ndarray, level: np.ndarray, raw_clip: np.ndarray, spectra: Spectra
) -> dict:
    limit = min(max(0, len(level) - 6), 20)
    events = [percentile(level[index : index + 6], 25) for index in range(limit + 1)]
    index = int(np.argmax(events))
    event = events[index]
    profile = spectra.aggregate(range(index, min(len(level), index + 6)))
    flatness = float(np.exp(np.mean(np.log(profile + 1e-12))) / np.mean(profile))
    centroid = float(np.sum(spectra.frequencies * profile))
    return {
        "start_position_s": index * WINDOW_S,
        "start_event_db": event,
        "start_duration_s": duration_above(level, index, max(-9, event - 6)),
        "start_near_clip_fraction": float(
            np.mean(raw_clip[index : min(len(raw_clip), index + 20)])
        ),
        "start_spectral_flatness": flatness,
        "start_spectral_centroid_hz": centroid,
    }


def extract_clip(path: Path) -> dict:
    samples = read_audio(path)
    level, fine, timeline_stats = timelines(samples)
    raw_clip = timeline_stats.pop("raw_clip")
    features = {
        **timeline_stats,
        "global_p10_db": percentile(level, 10),
        "global_p25_db": percentile(level, 25),
        "global_median_db": percentile(level, 50),
        "global_p75_db": percentile(level, 75),
        "global_p95_db": percentile(level, 95),
        "global_p99_db": percentile(level, 99),
        "global_dynamic_range_db": percentile(level, 99) - percentile(level, 50),
        "global_derivative_db": percentile(np.abs(np.diff(level)), 50),
        "global_loud_fraction": float(np.mean(level >= -9)),
        "global_quiet_fraction": float(np.mean(level <= -30)),
    }
    spectra = Spectra(samples)
    candidates = candidate_events(level)
    for number in range(TOP_EVENTS):
        prefix = f"event_{number + 1}_"
        if number < len(candidates):
            values = event_features(candidates[number], level, fine, raw_clip, spectra)
            features.update({prefix + key: value for key, value in values.items()})
        else:
            features[prefix + "missing"] = 1.0
    features.update(start_features(samples, level, raw_clip, spectra))
    return {key: float(value) for key, value in features.items()}


def extract_task(task: tuple[str, str]) -> tuple[str, dict | None, str | None]:
    digest, path = task
    try:
        return digest, extract_clip(Path(path)), None
    except Exception as error:  # noqa: BLE001 - isolate a bad file in a multi-hour batch
        return digest, None, f"{type(error).__name__}: {error}"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("corpus/index.json"))
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument(
        "--output", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    index = json.loads(args.index.read_text())
    labels = load_labels(args.labels)
    items = [item for item in index["items"] if item.get("status") == "ok"]
    existing = {}
    if args.output.exists() and not args.refresh:
        old = json.loads(args.output.read_text())
        if old.get("feature_version") == FEATURE_VERSION:
            existing = {
                row["audio_sha256"]: row["features"] for row in old.get("rows", [])
            }

    representatives = {}
    for item in items:
        representatives.setdefault(item["audio_sha256"], item)
    tasks = [
        (digest, str(args.audio_dir / item["audio_file"]))
        for digest, item in representatives.items()
        if digest not in existing
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    failures = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = [executor.submit(extract_task, task) for task in tasks]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            digest, features, error = future.result()
            if error:
                failures[digest] = error
                print(
                    f"[feature {number}/{len(futures)}] ERROR {error}", file=sys.stderr
                )
            elif number % 25 == 0 or number == len(futures):
                existing[digest] = features
                print(
                    f"[feature {number}/{len(futures)}] {digest[:12]}", file=sys.stderr
                )
            else:
                existing[digest] = features

    rows = []
    for item in items:
        features = existing.get(item["audio_sha256"])
        if features is None:
            continue
        rows.append(
            {
                "path": item["path"],
                "file": item["file"],
                "thread": thread_group(item["path"]),
                "label": label_for(item["path"], labels),
                "audio_sha256": item["audio_sha256"],
                "features": features,
            }
        )
    rows.sort(key=lambda row: row["path"])
    feature_names = sorted({name for row in rows for name in row["features"]})
    payload = {
        "schema": 1,
        "feature_version": FEATURE_VERSION,
        "sample_rate": RATE,
        "window_ms": int(WINDOW_S * 1000),
        "top_events": TOP_EVENTS,
        "feature_names": feature_names,
        "rows": rows,
        "failures": failures,
    }
    atomic_json(args.output, payload)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as target:
        fieldnames = ["path", "file", "thread", "label", "audio_sha256", *feature_names]
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**{key: row[key] for key in fieldnames[:5]}, **row["features"]}
            )
    print(
        f"wrote {len(rows)} logical rows, {len(existing)} unique feature vectors, "
        f"{len(feature_names)} features; {len(failures)} failures",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
