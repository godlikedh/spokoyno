#!/usr/bin/env python3
"""Score the previous best ML model on the new, bounded codec protocol.

This is a diagnostic on 30 positives and 9 selected hard negatives, using saved
thread-excluded models. It does not retrain or change the old snapshot.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from audit_acoustics import encode, score
from event_dataset import physical_events
from extract_audio_context import Encoder, ensure_encoder, sha256
from extract_features import RATE, extract_clip, read_audio
from improve_events import training_key
from robust_audio import augment, bounded
from scipy.io import wavfile
from train_events import metrics
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("research/artifacts/improve-augmentation-20261025"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.run / "plan.json").read_text())
    real = json.loads(
        Path("research/artifacts/event-data-v1/manifest.json").read_text()
    )
    browser = json.loads(
        Path(
            "research/artifacts/browser-events-full-20261025-v2/manifest.json"
        ).read_text()
    )
    hard = {c["audio_sha256"] for c in browser["clips"] if c["label"] == "negative"}
    clips = [
        c
        for c in real["clips"]
        if c["label"] == "positive" or c["audio_sha256"] in hard
    ]
    models = {}
    for fold in plan["folds"]:
        fit = args.run / "fits" / training_key(fold["train"])
        meta = json.loads((fit / "manifest.json").read_text())
        if sha256(fit / "models.json") != meta["model_sha256"]:
            raise ValueError("previous model changed")
        models[fold["thread"]] = {
            "synthetic-mil-20pct": json.loads((fit / "models.json").read_text())[
                "synthetic-mil-20pct"
            ]
        }
    ensure_encoder(Path("research/artifacts/yamnet"))
    encoder = Encoder(Path("research/artifacts/yamnet"), threads=2)
    rows = []
    for index, clip in enumerate(clips):
        source = Path("corpus/audio") / (clip["file"] + ".audio.wav")
        if sha256(source) != clip["audio_sha256"]:
            raise ValueError("source changed")
        samples = read_audio(source)
        variants = {
            "raw-original": samples,
            "bounded-original": bounded(samples),
            "delay-11s": np.pad(samples, ((11 * RATE, 0), (0, 0))),
        }
        variants.update(
            {
                name: augment(samples, recipe)[0]
                for name, recipe in {
                    "aac80": {"codec": "aac80", "delay_s": 0.013},
                    "opus48": {"codec": "opus48", "delay_s": 0.125},
                    "mp3_128": {"codec": "mp3_128", "delay_s": 1.75},
                    "aac-aligned": {"codec": "aac80", "align": True},
                }.items()
            }
        )
        for name, transformed in variants.items():
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / "variant.wav"
                wavfile.write(file, RATE, transformed)
                features = extract_clip(file)
            _, windows = physical_events(transformed, features)
            embedded = encode(transformed, features, windows, encoder)
            prediction = score(
                models[clip["fold_thread"]], transformed, windows, embedded
            )["synthetic-mil-20pct"]
            rows.append(
                {
                    "path": clip["path"],
                    "label": clip["label"],
                    "variant": name,
                    **prediction,
                }
            )
        print(f"previous ML stress {index + 1}/{len(clips)}", flush=True)
    atomic_json(args.output_dir / "predictions.json", rows)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "model": "synthetic-mil-20pct",
            "metrics": {
                v: metrics([r for r in rows if r["variant"] == v]) for v in variants
            },
            "limitations": "Selected hard negatives are not a representative false-positive-rate sample; all transforms share the same 39 originals.",
        },
    )


if __name__ == "__main__":
    main()
