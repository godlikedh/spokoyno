#!/usr/bin/env python3
"""Frozen production-candidate inference on decoded audio or a media file.

This module performs no fitting and no heuristic/model blending. Silence is
explicitly recognized; decode/model failures are errors, never safe scores.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from extract_audio_context import Encoder, ensure_encoder, sha256
from extract_features import RATE, read_audio
from risk_score import risk_tier
from robust_audio import canonical, extract
from robust_tree import window_probabilities
from train_models import atomic_json
from train_robust_models import pool, prepare


def load_snapshot(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"model snapshot changed: {name}")
    transform = json.loads((directory / "transform.json").read_text())
    models = json.loads((directory / "models.json").read_text())
    for name, expected in manifest["extractor_sha256"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"extractor does not match frozen model: {name}")
    return manifest, transform, models


class Detector:
    def __init__(
        self, snapshot, model=None, encoder_dir=Path("research/artifacts/yamnet")
    ):
        self.manifest, self.transform, models = load_snapshot(snapshot)
        self.name = model or self.manifest["default_model"]
        self.model = models[self.name]
        self.encoder = None
        if self.model["audio"]:
            info = ensure_encoder(encoder_dir)
            if info != self.manifest["encoder"]:
                raise ValueError("encoder does not match frozen model")
            self.encoder = Encoder(encoder_dir, threads=3)

    def score(self, samples):
        _, canonical_meta = canonical(samples)
        if canonical_meta["silent"]:
            return {
                "score": 0.0,
                "risk_tier": "low",
                "events": [],
                "silent": True,
                "model": self.name,
            }
        extracted = extract(samples, include_patches=self.encoder is not None)
        if extracted["names"] != self.transform["names"]:
            raise ValueError("feature names differ from trained model")
        audio = None
        if self.encoder is not None:
            patches = extracted["patches"]
            audio = np.concatenate(
                [
                    self.encoder(patches[lo : lo + 96])
                    for lo in range(0, len(patches), 96)
                ]
            ).reshape(-1, 3072)
        x = prepare(self.transform, extracted["physical"], audio)
        p = window_probabilities(self.model, x)
        value = float(
            pool(p, np.zeros(len(p), dtype=int), 1, self.model["power"])[0][0]
        )
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid model output")
        order = np.argsort(-p, kind="stable")[:5]
        return {
            "score": value,
            "risk_tier": risk_tier(value),
            "model": self.name,
            "events": [extracted["windows"][i] | {"score": float(p[i])} for i in order],
            "analyzed_events": len(p),
            "silent": False,
            "score_kind": "uncalibrated risk evidence, not a probability",
            "time_kind": "machine proposals, not verified event boundaries",
        }


def decode_media(path, target):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_f32le",
            str(target),
        ],
        check=True,
        timeout=180,
    )
    return read_audio(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a previous prediction")
    detector = Detector(args.snapshot, args.model)
    with tempfile.TemporaryDirectory() as directory:
        samples = (
            read_audio(args.media)
            if args.media.suffix.lower() == ".wav"
            else decode_media(args.media, Path(directory) / "decoded.wav")
        )
        result = detector.score(samples)
    result.update(
        source=str(args.media),
        source_sha256=sha256(args.media),
        snapshot_sha256=sha256(args.snapshot / "manifest.json"),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
