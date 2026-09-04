#!/usr/bin/env python3
"""Measure MAD-normalized transitions and 10 ms attack time on the cached corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from analyze_audio import (
    FLOOR_DB,
    RATE,
    WINDOW,
    WINDOW_S,
    db_power,
    decode,
    high_shelf_sos,
    sigmoid,
)

FINE_WINDOW = round(RATE * 0.01)


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p * 100))


def spectrum(samples: np.ndarray, coarse_index: int) -> np.ndarray | None:
    start = coarse_index * WINDOW
    if start < 0 or start >= len(samples):
        return None
    frame = samples[start : min(len(samples), start + WINDOW)]
    fft_size = 1 << (WINDOW - 1).bit_length()
    fft = np.fft.rfft(frame * np.hanning(len(frame))[:, None], n=fft_size, axis=0)
    power = np.mean(np.abs(fft) ** 2 + 1e-12, axis=1)
    return power[
        : np.searchsorted(np.fft.rfftfreq(fft_size, 1 / RATE), 7800, side="right")
    ]


def spectral_event_features(
    samples: np.ndarray, coarse_index: int, level_count: int
) -> tuple[float, float, float]:
    cache: dict[int, np.ndarray | None] = {}

    def profile(index: int) -> np.ndarray | None:
        if index < 0 or index >= level_count:
            return None
        if index not in cache:
            value = spectrum(samples, index)
            cache[index] = value / np.sum(value) if value is not None else None
        return cache[index]

    def flux(index: int) -> float:
        before, current = profile(index - 1), profile(index)
        if before is None or current is None:
            return 0.0
        return float(np.sqrt(np.sum(np.maximum(current - before, 0) ** 2)))

    onset = flux(coarse_index)
    nearby = max(
        [onset]
        + [
            flux(index)
            for index in range(
                max(1, coarse_index - 2), min(level_count - 1, coarse_index + 2) + 1
            )
        ]
    )
    baseline = [
        profile(index)
        for index in range(max(0, coarse_index - 40), max(0, coarse_index - 2))
    ]
    event = [
        profile(index)
        for index in range(coarse_index, min(level_count, coarse_index + 6))
    ]
    baseline = [value for value in baseline if value is not None]
    event = [value for value in event if value is not None]
    if not baseline or not event:
        return onset, nearby, 0.0
    baseline_mean = np.mean(baseline, axis=0)
    event_mean = np.mean(event, axis=0)
    shape = float(
        np.sqrt(np.sum((np.sqrt(event_mean) - np.sqrt(baseline_mean)) ** 2))
        / np.sqrt(2)
    )
    return onset, nearby, shape


def duration_above(levels: np.ndarray, start: int, threshold: float) -> float:
    last_above, misses = start - 1, 0
    for index in range(start, min(len(levels), start + 60)):
        if levels[index] >= threshold:
            last_above = index
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return max(0.0, (last_above - start + 1) * WINDOW_S)


def attack_time(fine_levels: np.ndarray, coarse_index: int, baseline: float) -> dict:
    event_at = coarse_index * 5
    lo, hi = max(0, event_at - 50), min(len(fine_levels), event_at + 80)
    if hi <= lo:
        return {
            "attack_ms": None,
            "attack_relative_ms": None,
            "attack_peak_db": FLOOR_DB,
        }
    # Limit peak search to 500 ms after the coarse candidate; the preceding 500 ms is only history.
    peak_lo, peak_hi = max(lo, event_at), min(hi, event_at + 50)
    peak = peak_lo + int(np.argmax(fine_levels[peak_lo:peak_hi]))
    peak_db = float(fine_levels[peak])

    def rise_from(threshold: float) -> float | None:
        below = np.flatnonzero(fine_levels[lo:peak] <= threshold)
        if not len(below):
            return None
        last_below = lo + int(below[-1])
        target = peak_db - 3.0
        reached = np.flatnonzero(fine_levels[last_below + 1 : peak + 1] >= target)
        if not len(reached):
            return None
        return float((int(reached[0]) + 1) * 10)

    relative_threshold = baseline + 0.2 * max(0.0, peak_db - baseline)
    return {
        "attack_ms": rise_from(-30.0),
        "attack_relative_ms": rise_from(relative_threshold),
        "attack_peak_db": peak_db,
    }


def finalize_transition(
    samples: np.ndarray, levels: np.ndarray, fine_levels: np.ndarray, candidate: dict
) -> dict:
    best = candidate["index"]
    baseline, event, jump = (
        candidate["baseline_db"],
        candidate["event_db"],
        candidate["jump_db"],
    )
    threshold = max(-13.0, baseline + 9.0)
    duration = duration_above(levels, best, threshold)
    look_end = min(len(levels), best + 10)
    event_samples = np.abs(
        samples[best * WINDOW : min(len(samples), look_end * WINDOW)]
    )
    near_clip = (
        float(np.mean(event_samples >= 10 ** (-1 / 20))) if len(event_samples) else 0.0
    )
    flux, nearby_flux, shape_distance = spectral_event_features(
        samples, best, len(levels)
    )
    loud_component = float(sigmoid((event + 10.5) / 2.5))
    jump_component = float(sigmoid((jump - 13.0) / 3.5))
    duration_component = float(sigmoid((duration - 0.18) / 0.09))
    quiet_component = float(sigmoid((-baseline - 15.0) / 5.0))
    clip_component = float(sigmoid((near_clip - 0.005) / 0.012))
    flux_component = float(sigmoid((flux - 0.22) / 0.08))
    score = loud_component**0.9 * jump_component**1.25 * duration_component**0.65
    score *= 0.87 + 0.10 * quiet_component + 0.03 * clip_component
    score *= 0.65 + 0.35 * flux_component
    if not candidate["has_history"]:
        score = 0.0
    spectral_burst_eligible = (
        candidate["has_history"]
        and event >= -4
        and jump >= 16
        and 0.3 <= duration <= 1.05
        and nearby_flux >= 0.3
        and shape_distance >= 0.8
    )
    clipped_burst_eligible = (
        candidate["has_history"]
        and event >= -3
        and jump >= 10
        and 0.25 <= duration <= 0.55
        and near_clip >= 0.35
        and nearby_flux >= 0.3
    )
    high_contrast_burst_eligible = (
        candidate["has_history"]
        and event >= -4
        and jump >= 30
        and 0.25 <= duration <= 0.5
        and near_clip >= 0.15
        and nearby_flux >= 0.35
        and shape_distance >= 0.5
    )
    sustained_spectral_takeover_eligible = (
        candidate["has_history"]
        and event >= -3.5
        and jump >= 25
        and duration >= 2
        and nearby_flux >= 0.4
        and shape_distance >= 0.8
    )
    spectral_burst_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((shape_distance - 0.85) / 0.08))
            + 0.04 * float(sigmoid((jump - 18) / 3))
            + 0.04 * float(sigmoid((event + 3) / 1.5)),
        )
        if spectral_burst_eligible
        else 0.0
    )
    clipped_burst_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((near_clip - 0.4) / 0.08))
            + 0.04 * float(sigmoid((jump - 12) / 2))
            + 0.04 * float(sigmoid((event + 2) / 1.2)),
        )
        if clipped_burst_eligible
        else 0.0
    )
    high_contrast_burst_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((near_clip - 0.2) / 0.08))
            + 0.04 * float(sigmoid((jump - 35) / 6))
            + 0.04 * float(sigmoid((event + 3.5) / 1.2)),
        )
        if high_contrast_burst_eligible
        else 0.0
    )
    sustained_spectral_takeover_score = (
        min(
            1.0,
            0.8
            + 0.04 * float(sigmoid((shape_distance - 0.85) / 0.08))
            + 0.04 * float(sigmoid((nearby_flux - 0.4) / 0.08))
            + 0.04 * float(sigmoid((jump - 30) / 6)),
        )
        if sustained_spectral_takeover_eligible
        else 0.0
    )
    rescue_mode, rescue_score = max(
        (
            ("short-spectral-burst", spectral_burst_score),
            ("short-clipped-burst", clipped_burst_score),
            ("short-high-contrast-burst", high_contrast_burst_score),
            ("sustained-spectral-takeover", sustained_spectral_takeover_score),
        ),
        key=lambda item: item[1],
    )
    return (
        candidate
        | {
            "event_at_s": best * WINDOW_S,
            "duration_s": duration,
            "near_clip_pct": near_clip * 100,
            "spectral_flux": flux,
            "nearby_spectral_flux": nearby_flux,
            "spectral_shape_distance": shape_distance,
            "score": float(np.clip(score, 0, 1)),
            "rescue_score": rescue_score,
            "rescue_mode": rescue_mode if rescue_score else None,
        }
        | attack_time(fine_levels, best, baseline)
    )


def select_transition(levels: np.ndarray, strategy: str) -> dict:
    candidates = []
    for i in range(len(levels) - 5):
        lo, hi = max(0, i - 60), i - 2
        if hi - lo < 20:
            continue
        history = levels[lo:hi]
        baseline = float(np.median(history))
        mad = float(np.median(np.abs(history - baseline)))
        event = percentile(levels[i : i + 6], 0.25)
        jump = event - baseline
        robust_scale = max(2.0, 1.4826 * mad)
        z = jump / robust_scale
        envelope = float(
            sigmoid((event + 10.5) / 2.5) ** 0.9 * sigmoid((jump - 13.0) / 3.5) ** 1.25
        )
        if strategy == "literal-z":
            selection_score = envelope * max(0.0, z)
        elif strategy == "bounded-z":
            selection_score = envelope * (0.65 + 0.35 * float(sigmoid((z - 6.0) / 2.0)))
        else:
            selection_score = envelope
        candidates.append(
            {
                "index": i,
                "baseline_db": baseline,
                "event_db": event,
                "jump_db": jump,
                "baseline_mad_db": mad,
                "robust_scale_db": robust_scale,
                "robust_z": z,
                "selection_score": selection_score,
                "has_history": True,
            }
        )
    if not candidates:
        baseline = float(np.median(levels))
        return {
            "index": 0,
            "baseline_db": baseline,
            "event_db": float(levels[0]),
            "jump_db": float(levels[0] - baseline),
            "baseline_mad_db": float(np.median(np.abs(levels - baseline))),
            "robust_scale_db": 2.0,
            "robust_z": float((levels[0] - baseline) / 2.0),
            "selection_score": 0.0,
            "has_history": False,
        }
    return max(candidates, key=lambda item: item["selection_score"])


def start_score(samples: np.ndarray, levels: np.ndarray) -> dict:
    limit = min(max(0, len(levels) - 6), round(1.0 / WINDOW_S))
    events = np.asarray([percentile(levels[i : i + 6], 0.25) for i in range(limit + 1)])
    best = int(np.argmax(events))
    level = float(events[best])
    target = max(-6.0, level - 3.0)
    candidates = np.flatnonzero(events >= target)
    preview = int(candidates[0]) if len(candidates) else best
    onset_threshold = max(-9.0, level - 6.0)
    onsets = np.flatnonzero(
        levels[preview : min(len(levels), preview + 6)] >= onset_threshold
    )
    start = preview + (int(onsets[0]) if len(onsets) else 0)
    duration = duration_above(levels, start, -9.0)
    clip_samples = np.abs(
        samples[start * WINDOW : min(len(samples), (start + 20) * WINDOW)]
    )
    near_clip = (
        float(np.mean(clip_samples >= 10 ** (-1 / 20))) if len(clip_samples) else 0.0
    )
    profiles = [spectrum(samples, i) for i in range(start, min(len(levels), start + 6))]
    profiles = [profile for profile in profiles if profile is not None]
    flatness = (
        float(np.median([np.exp(np.mean(np.log(p))) / np.mean(p) for p in profiles]))
        if profiles
        else 0.0
    )
    brightness = []
    for i in range(start, min(len(levels), start + 6)):
        frame = samples[i * WINDOW : min(len(samples), (i + 1) * WINDOW)]
        brightness.append(
            float(
                db_power(
                    np.mean(np.diff(frame, axis=0).astype(np.float64) ** 2)
                    / max(np.mean(frame.astype(np.float64) ** 2), 1e-12)
                )
            )
        )
    brightness_db = float(np.median(brightness)) if brightness else FLOOR_DB
    score = float(
        sigmoid((level + 6) / 2) ** 1.1 * sigmoid((duration - 0.35) / 0.15) ** 0.7
    )
    score *= (
        0.68
        + 0.12 * float(sigmoid((near_clip - 0.01) / 0.03))
        + 0.14 * float(sigmoid((flatness - 0.025) / 0.025))
        + 0.06 * float(sigmoid((brightness_db + 8) / 2.5))
    )
    evidence = flatness >= 0.04 or brightness_db >= -5.0 or near_clip >= 0.08
    return {
        "score": float(np.clip(score, 0, 1)),
        "event_db": level,
        "duration_s": duration,
        "near_clip_pct": near_clip * 100,
        "flatness": flatness,
        "brightness_db": brightness_db,
        "spectral_evidence": bool(evidence),
        "event_at_s": start * WINDOW_S,
    }


def analyze_one(task: tuple[dict, str]) -> dict:
    row, media_dir = task
    samples, error = decode(Path(media_dir) / f"{row['file']}.audio.wav")
    if error or samples is None or not len(samples):
        return {"file": row["file"], "error": error or "no audio"}
    usable = len(samples) // WINDOW * WINDOW
    samples = samples[:usable]
    hp, shelf = butter(2, 70, btype="highpass", fs=RATE, output="sos"), high_shelf_sos(
        RATE
    )
    weighted = np.empty_like(samples)
    for channel in range(2):
        weighted[:, channel] = sosfilt(shelf, sosfilt(hp, samples[:, channel]))
    frames = weighted.reshape(-1, WINDOW, 2)
    levels = np.maximum(
        FLOOR_DB,
        -0.691 + db_power(np.mean(frames.astype(np.float64) ** 2, axis=(1, 2))),
    )
    fine_usable = len(weighted) // FINE_WINDOW * FINE_WINDOW
    fine = weighted[:fine_usable].reshape(-1, FINE_WINDOW, 2)
    fine_levels = np.maximum(
        FLOOR_DB, -0.691 + db_power(np.mean(fine.astype(np.float64) ** 2, axis=(1, 2)))
    )
    del weighted, frames, fine
    transitions = {
        strategy: finalize_transition(
            samples, levels, fine_levels, select_transition(levels, strategy)
        )
        for strategy in ("current", "literal-z", "bounded-z")
    }
    start = start_score(samples, levels)
    current = transitions["current"]
    transition_eligible = (
        current["has_history"]
        and current["event_db"] >= -6
        and current["jump_db"] >= 14
        and current["duration_s"] >= 0.15
    )
    start_eligible = (
        start["event_db"] >= -3
        and start["duration_s"] >= 0.5
        and start["spectral_evidence"]
    )
    branch_scores = {
        "transition": current["score"] if transition_eligible else 0.0,
        "loud-start": start["score"] if start_eligible else 0.0,
        current["rescue_mode"] or "short-burst": current["rescue_score"],
    }
    decision_mode, decision_score = max(branch_scores.items(), key=lambda item: item[1])
    key = row.get("md5") or row["file"]
    split = (
        "validation"
        if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0
        else "development"
    )
    return {
        "file": row["file"],
        "label": row.get("label", "unlabeled"),
        "split": split,
        "duration_s": len(samples) / RATE,
        "transition": transitions,
        "start": start,
        "decision_mode": decision_mode,
        "decision_score": decision_score,
        "suspicious": decision_score >= 0.8,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2)
    )
    args = parser.parse_args()
    rows = []
    for source in args.inputs:
        rows.extend(
            row for row in json.loads(source.read_text()) if row.get("status") == "ok"
        )
    unique = {row["file"]: row for row in rows}
    tasks = [(row, args.media_dir) for row in unique.values()]
    results = []
    with multiprocessing.Pool(args.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(analyze_one, tasks), 1):
            results.append(result)
            print(f"[{index}/{len(tasks)}] {result['file']}", flush=True)
    results.sort(key=lambda row: row["file"])
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any("error" in row for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
