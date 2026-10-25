#!/usr/bin/env python3
"""Reproducible waveform augmentation; derivatives never become corpus labels.

Both parents must be in a fold's training set before a derivative is eligible.
Evaluation always uses real recordings. Controls use the same edit envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from acoustic_features import extract
from event_annotations import validate_intervals
from event_dataset import atomic_npz, physical_events, window_label
from extract_audio_context import Encoder, context_audio, ensure_encoder, sha256
from extract_features import RATE, extract_clip, read_audio
from scipy.io import wavfile
from train_models import atomic_json

SEED = 20261026


def eligible(clips, allowed):
    allowed = set(allowed)
    return np.asarray([set(c["parents"]) <= allowed for c in clips])


def mono(samples):
    return samples[:, np.argmax(np.mean(samples[::16].astype(float) ** 2, axis=0))]


def mix(background, foreground, onset, gain_db, contrast_db):
    """Replace a segment with a 5 ms crossfade; control applies the same fade.

    Foreground level and contrast are bounded by the available headroom. There
    is no clipping, codec change, or file-name input to the classifier.
    """
    bg, fg = background.astype(float).copy(), foreground.astype(float).copy()
    start = round(onset * RATE)
    end = start + len(fg)
    target = 10 ** (gain_db / 20)
    fg *= min(target / max(np.sqrt(np.mean(fg**2)), 1e-9), 0.98 / max(abs(fg)))
    rms = np.sqrt(np.mean(fg**2))
    background_rms = np.sqrt(np.mean(bg**2))
    bg *= min(
        rms / 10 ** (contrast_db / 20) / max(background_rms, 1e-9),
        0.98 / max(max(abs(bg)), 1e-9),
    )
    fade = min(round(0.005 * RATE), len(fg) // 2)
    envelope = np.ones(len(fg))
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    positive, control = bg.copy(), bg.copy()
    positive[start:end] = bg[start:end] * (1 - envelope) + fg * envelope
    # Identical cut/crossfade with background from the same interval: no novel
    # sound is invented and assigned a negative human label.
    control[start:end] = bg[start:end] * (1 - envelope) + bg[start:end] * envelope
    return positive.astype(np.float32), control.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = json.loads((args.base / "manifest.json").read_text())
    source = base["clips"]
    donors = [
        i for i, c in enumerate(source) if c["label"] == "positive" and c["annotation"]
    ]
    backgrounds = [
        i
        for i, c in enumerate(source)
        if c["label"] == "negative" and c["duration_s"] >= 12
    ]
    rng = np.random.default_rng(SEED)
    plan = {
        "seed": SEED,
        "donors": donors,
        "variants_per_donor": 12,
        "durations_s": [0.35, 0.75, 1.5],
        "foreground_rms_db": [-12, -6, -3],
        "contrasts_db": [10, 20, 30],
        "source_sha256": sha256(Path(__file__)),
        "base_manifest_sha256": sha256(args.base / "manifest.json"),
        "policy": "Training only; every parent must be eligible in each outer and inner fold; never count derivatives as independent validation examples.",
    }
    atomic_json(args.output_dir / "plan.json", plan)
    encoder_manifest = ensure_encoder(args.encoder_dir)
    encoder = Encoder(args.encoder_dir, threads=2)
    clips, windows, rows, embeddings, targets = [], [], [], [], []
    names = None
    for donor in donors:
        c = source[donor]
        audio = args.audio_dir / f"{c['file']}.audio.wav"
        if sha256(audio) != c["audio_sha256"]:
            raise ValueError("donor audio changed")
        donor_audio = mono(read_audio(audio))
        intervals = validate_intervals(c["annotation"]["intervals"], c["duration_s"])
        for variant in range(12):
            background = int(rng.choice(backgrounds))
            b = source[background]
            path = args.audio_dir / f"{b['file']}.audio.wav"
            if sha256(path) != b["audio_sha256"]:
                raise ValueError("background audio changed")
            bg = mono(read_audio(path))
            offset = int(rng.integers(0, max(1, len(bg) - 12 * RATE)))
            bg = bg[offset : offset + 12 * RATE]
            interval = intervals[variant % len(intervals)]
            error = c["annotation"].get("estimated_boundary_error_s", 0.05)
            lower, upper = interval["start_s"] + error, interval["end_s"] - error
            length = min(plan["durations_s"][variant % 3], upper - lower)
            at = float(rng.uniform(lower, max(lower, upper - length)))
            fg = donor_audio[round(at * RATE) : round((at + length) * RATE)]
            onset = float(rng.uniform(3, 8))
            positive, control = mix(
                bg,
                fg,
                onset,
                plan["foreground_rms_db"][variant % 3],
                plan["contrasts_db"][(variant // 3) % 3],
            )
            for label, samples in (("positive", positive), ("negative", control)):
                file = args.output_dir / f"{donor}-{variant}-{label}.wav"
                wavfile.write(file, RATE, samples)
                features = extract_clip(file)
                stereo = samples[:, None]
                _, original = physical_events(stereo, features)
                vectors, points = extract(stereo, original)
                vectors, points = vectors[: len(original)], points[: len(original)]
                patches, _ = context_audio(stereo, features)
                encoded = encoder(patches)
                annotation = {
                    "intervals": [{"start_s": onset, "end_s": onset + len(fg) / RATE}],
                    "estimated_boundary_error_s": 0.0,
                }
                owner = len(clips)
                clips.append(
                    {
                        "file": str(file),
                        "label": label,
                        "parents": [donor, background],
                        "parent_hashes": [c["audio_sha256"], b["audio_sha256"]],
                        "audio_sha256": sha256(file),
                        "first_window": len(windows),
                        "window_count": len(points),
                        "onset_s": onset,
                        "source_start_s": at,
                        "source_duration_s": len(fg) / RATE,
                        "background_start_s": offset / RATE,
                        "variant": variant,
                    }
                )
                names = names or sorted(vectors[0])
                for vector, point in zip(vectors, points, strict=True):
                    rows.append([vector[n] for n in names])
                    slot = point["slot"]
                    embeddings.append(
                        np.concatenate([np.zeros(1024), encoded[9], np.zeros(1024)])
                        if slot == 3
                        else encoded[slot * 3 : slot * 3 + 3].ravel()
                    )
                    windows.append(point | {"clip_index": owner})
                    targets.append(
                        window_label(
                            label,
                            annotation,
                            point["start_s"],
                            point["end_s"],
                            len(samples) / RATE,
                        )
                    )
        print(f"generated donor {c['file']}: {len(clips)} derivatives", flush=True)
    path = args.output_dir / "examples.npz"
    atomic_npz(
        path,
        physical=np.asarray(rows),
        embeddings=np.asarray(embeddings),
        targets=np.asarray(targets, np.int8),
    )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "plan_sha256": sha256(args.output_dir / "plan.json"),
            "matrix_sha256": sha256(path),
            "physical_names": names,
            "encoder": encoder_manifest,
            "clips": clips,
            "windows": windows,
        },
    )


if __name__ == "__main__":
    main()
