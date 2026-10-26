#!/usr/bin/env python3
"""Production-oriented bounded audio, translation-invariant anchoring and events.

No filenames, labels, video lengths or absolute positions enter feature vectors.
Codec transforms operate on bounded PCM, preserving channels and recording their
alignment. The earlier raw-overrange AAC experiment remains an immutable control.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
from acoustic_features import band_features, base_at, burst_features, rhythm_features
from extract_audio_context import PATCH_SAMPLES, crop
from extract_features import RATE, WINDOW_S, Spectra, db_power, timelines
from scipy.signal import correlate

VERSION = 1
PAD_SECONDS = 6
SILENCE_THRESHOLD = 1e-5
REMOVED = {"history_s", "is_opening", "attack_ms"}
CODECS = {
    "aac64": ("aac", 48000, "64k", "adts"),
    "aac96": ("aac", 48000, "96k", "adts"),
    "opus32": ("libopus", 48000, "32k", "ogg"),
    "opus64": ("libopus", 48000, "64k", "ogg"),
    # These profiles are reserved for evaluation, not training recipes.
    "aac80": ("aac", 48000, "80k", "adts"),
    "opus48": ("libopus", 48000, "48k", "ogg"),
    "mp3_128": ("libmp3lame", 44100, "128k", "mp3"),
    "aac16k96": ("aac", 16000, "96k", "adts"),
}


def waveform_hash(samples):
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def bounded(samples):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] not in (1, 2) or not len(samples):
        raise ValueError("expected nonempty mono/stereo audio")
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio contains nonfinite samples")
    return np.clip(samples, -1.0, 1.0)


def canonical(samples):
    samples = bounded(samples)
    activity = np.flatnonzero(np.max(np.abs(samples), axis=1) > SILENCE_THRESHOLD)
    first = int(activity[0]) if len(activity) else len(samples)
    # Removing leading digital silence gives exact invariance to any integer
    # number of prepended zero samples. Offset is retained for display only.
    content = (
        samples[first:]
        if len(activity)
        else np.zeros((RATE // 2, samples.shape[1]), np.float32)
    )
    result = np.pad(content, ((PAD_SECONDS * RATE, 0), (0, 0)))
    return result, {
        "trimmed_samples": first,
        "offset_s": first / RATE - PAD_SECONDS,
        "silent": not len(activity),
        "source_samples": len(samples),
    }


def alignment(source, decoded, expected=None):
    """Measure transform delay on an energetic 1 s segment, independently of labels."""
    channel = int(np.argmax(np.mean(source[::16].astype(float) ** 2, axis=0)))
    step = RATE // 4
    count = len(source) // step
    if count == 0:
        return 0, 0.0
    power = np.mean(
        source[: count * step, channel].reshape(count, step).astype(float) ** 2, axis=1
    )
    if max(power) < 1e-12:
        return 0, 0.0
    lo = max(min(RATE // 4, len(source) // 4), int(np.argmax(power)) * step - RATE // 2)
    hi = min(len(source), lo + RATE)
    ref = source[lo:hi, channel].astype(float)
    scan_lo, scan_hi = max(0, lo - RATE // 4), min(len(decoded), hi + RATE // 4)
    scan = decoded[scan_lo:scan_hi, channel].astype(float)
    if len(scan) < len(ref):
        return 0, 0.0
    cross = correlate(scan, ref, mode="valid", method="fft")
    energy = np.r_[0.0, np.cumsum(scan**2)]
    norm = np.sqrt(
        np.maximum(energy[len(ref) :] - energy[: -len(ref)], 1e-20)
        * max(np.sum(ref**2), 1e-20)
    )
    corr = cross / norm
    if expected is not None:
        lags = np.arange(len(corr)) + scan_lo - lo
        # Periodic music has many correlation peaks. Search near the codec's
        # documented framing delay rather than shifting human intervals by a beat.
        corr[np.abs(lags - expected) > RATE // 200] = -np.inf
    selected = int(np.argmax(corr))
    return scan_lo + selected - lo, float(corr[selected])


def transcode(samples, profile, align=False):
    source = bounded(samples)
    codec, rate, bitrate, container = CODECS[profile]
    channels = source.shape[1]
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-f",
        "f32le",
        "-ar",
        str(RATE),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-ar",
        str(rate),
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        "-f",
        container,
        "pipe:1",
    ]
    encoded = subprocess.run(
        command, input=source.tobytes(), capture_output=True, check=True
    ).stdout
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "pipe:1",
        ],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    output = np.frombuffer(decoded, np.float32).reshape(-1, channels)
    expected = (
        round(1024 * RATE / rate)
        if codec == "aac"
        else (0 if codec == "libopus" else round(1105 * RATE / rate))
    )
    delay, correlation = alignment(source, output, expected)
    if align:
        output = output[delay:] if delay >= 0 else np.pad(output, ((-delay, 0), (0, 0)))
        output = output[: len(source)]
        if len(output) < len(source):
            output = np.pad(output, ((0, len(source) - len(output)), (0, 0)))
    return output, {
        "codec": profile,
        "codec_delay_samples": delay,
        "annotation_delay_samples": 0 if align else delay,
        "alignment_correlation": correlation,
        "aligned": align,
        "encoded_bytes": len(encoded),
    }


def augment(samples, recipe):
    samples = bounded(
        np.asarray(samples, np.float32) * 10 ** (recipe.get("gain_db", 0) / 20)
    )
    leading = round(recipe.get("delay_s", 0) * RATE)
    samples = np.pad(samples, ((leading, 0), (0, 0)))
    metadata = {"codec": None, "codec_delay_samples": 0, "annotation_delay_samples": 0}
    if recipe.get("codec"):
        samples, metadata = transcode(
            samples, recipe["codec"], recipe.get("align", False)
        )
    metadata["annotation_delay_samples"] += leading
    metadata["recipe"] = recipe
    return samples, metadata


def proposals(level):
    first, last = PAD_SECONDS * 20, len(level) - 1
    if last <= first:
        return [first]
    indices = np.arange(first, last + 1)
    power = 10 ** (level.astype(float) / 10)
    integral = np.r_[0.0, np.cumsum(power)]
    before_hi = np.maximum(0, indices - 1)
    before_lo = np.maximum(0, indices - 60)
    background = db_power(
        (integral[before_hi] - integral[before_lo])
        / np.maximum(1, before_hi - before_lo)
    )
    ranks = []
    for width in (1, 6):
        hi = np.minimum(len(level), indices + width)
        event = db_power((integral[hi] - integral[indices]) / (hi - indices))
        ranks.append(
            np.maximum(event - background, 0)
            / (1 + np.exp(-np.clip((event + 24) / 6, -30, 30)))
        )
    selected = []
    for rank in ranks:
        added = 0
        for offset in np.argsort(-rank, kind="stable"):
            point = int(indices[offset])
            if any(abs(point - i) < 10 for i in selected):
                continue
            selected.append(point)
            added += 1
            if added == 6:
                break
    # Cover long videos: each 20 s block contributes its strongest transition.
    rank = np.maximum(ranks[0], ranks[1])
    for lo in range(0, len(indices), 400):
        point = int(indices[lo + np.argmax(rank[lo : lo + 400])])
        if all(abs(point - i) >= 3 for i in selected):
            selected.append(point)
    selected.append(first + int(np.argmax(level[first : min(len(level), first + 20)])))
    return sorted(set(selected))


def extract(samples, include_patches=True):
    audio, metadata = canonical(samples)
    level, fine, stats = timelines(audio)
    spectra = Spectra(audio)
    points = proposals(level)
    rows, windows, patches = [], [], []
    energies = np.mean(audio[::16].astype(float) ** 2, axis=0)
    mono = audio[:, int(np.argmax(energies))]
    for point in points:
        at = point * WINDOW_S
        row = base_at(level, fine, stats, spectra, at)
        row.update(burst_features(fine, min(len(fine) - 1, point * 5)))
        row.update(rhythm_features(fine, min(len(fine) - 1, point * 5), spectra, point))
        row.update(band_features(audio, at))
        for key in REMOVED:
            row.pop(key)
        rows.append(row)
        windows.append(
            {
                "start_s": max(0.0, at + metadata["offset_s"]),
                "end_s": min(len(samples) / RATE, at + metadata["offset_s"] + 0.3),
            }
        )
        if include_patches:
            starts = [
                at - 0.1 - PATCH_SAMPLES / RATE,
                at - 0.1,
                at + np.clip(row["duration_s"], 0.3, 3),
            ]
            patches.extend(crop(mono, start) for start in starts)
    names = sorted(rows[0])
    return {
        "physical": np.asarray([[r[n] for n in names] for r in rows]),
        "names": names,
        "windows": windows,
        "metadata": metadata,
        "patches": np.asarray(patches, np.float32) if include_patches else None,
    }


def recipe_key(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
