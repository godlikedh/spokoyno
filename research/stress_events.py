#!/usr/bin/env python3
"""Controlled robustness of held-out event models on positives and reviewed hard negatives."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from event_dataset import physical_events
from extract_audio_context import Encoder, context_audio, sha256
from extract_features import RATE, extract_clip, read_audio
from fingerprint_audio import aac_roundtrip
from risk_score import POLICY, risk_tier
from scipy.io import wavfile
from train_audio_context import predict
from train_models import atomic_json


def variants(samples: np.ndarray):
    yield "unchanged", samples
    yield "prepend-25ms", np.pad(samples, ((400, 0), (0, 0)))
    yield "gain-minus-3db", samples * 10 ** (-3 / 20)
    yield "gain-plus-3db", samples * 10 ** (3 / 20)
    # Retain channel layout; interleaving stereo as mono would alter duration/pitch.
    decoded = aac_roundtrip(samples)
    yield "aac-96k", decoded[:, None] if decoded.ndim == 1 else decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("research/artifacts/event-models-v1")
    )
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--physical-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    labels = json.loads(args.labels.read_text())
    if sha256(args.labels) != manifest["labels_sha256"]:
        raise ValueError("event dataset has stale labels")
    clips = [
        r
        for r in manifest["clips"]
        if r["label"] == "positive"
        or any(path in labels["reviewed_negatives"] for path in r["paths"])
    ]
    encoder = None if args.physical_only else Encoder(args.encoder_dir)
    modes = ("physical",) if args.physical_only else ("physical", "hybrid")
    results, models, model_hashes = [], {}, {}
    for clip in clips:
        for mode in modes:
            key = (mode, clip["fold_thread"])
            if key not in models:
                path = args.model_dir / f"{mode}-fold-{clip['fold_thread']}.json"
                models[key] = json.loads(path.read_text())
                model_hashes[str(path)] = sha256(path)
            if clip["audio_sha256"] in models[key]["training_audio_hashes"]:
                raise ValueError("stress source leaked into fold training")
        source = args.audio_dir / f"{clip['file']}.audio.wav"
        if sha256(source) != clip["audio_sha256"]:
            raise ValueError("stress audio changed")
        samples = read_audio(source)
        for name, transformed in variants(samples):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "variant.wav"
                wavfile.write(path, RATE, transformed)
                # Recompute proposals from the full modified recording, never shift
                # old proposals or use the human intervals to locate the event.
                features = extract_clip(path)
            vectors, windows = physical_events(transformed, features)
            physical = np.asarray(
                [[row[k] for k in manifest["physical_names"]] for row in vectors]
            )
            audio = np.zeros((len(windows), 3072))
            if encoder is not None:
                patches, _ = context_audio(transformed, features)
                embedded = encoder(patches)
                for i, window in enumerate(windows):
                    slot = window["slot"]
                    audio[i] = (
                        embedded[slot * 3 : slot * 3 + 3].reshape(-1)
                        if slot < 3
                        else np.concatenate(
                            [np.zeros(1024), embedded[9], np.zeros(1024)]
                        )
                    )
            for mode in modes:
                scores = predict(
                    models[(mode, clip["fold_thread"])]["model"], physical, audio
                )
                chosen = int(np.argmax(scores))
                score = float(scores[chosen])
                results.append(
                    {
                        "path": clip["path"],
                        "label": clip["label"],
                        "mode": mode,
                        "variant": name,
                        "score": score,
                        "risk_tier": risk_tier(score),
                        "event_at_s": windows[chosen]["start_s"],
                    }
                )
        print(f"stress {clip['file']}", flush=True)
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for name in (
            "unchanged",
            "prepend-25ms",
            "gain-minus-3db",
            "gain-plus-3db",
            "aac-96k",
        ):
            rows = [r for r in results if r["mode"] == mode and r["variant"] == name]
            summary[mode][name] = {
                "positive": sum(r["label"] == "positive" for r in rows),
                "positive_alerts": sum(
                    r["label"] == "positive" and r["risk_tier"] == "alert" for r in rows
                ),
                "hard_negative": sum(r["label"] == "negative" for r in rows),
                "hard_negative_alerts": sum(
                    r["label"] == "negative" and r["risk_tier"] == "alert" for r in rows
                ),
                "hard_negative_maybe": sum(
                    r["label"] == "negative" and r["risk_tier"] == "maybe" for r in rows
                ),
            }
    atomic_json(
        args.model_dir / "stress.json",
        {
            "schema": 1,
            "risk_policy": POLICY,
            "model_sha256": model_hashes,
            "warning": "Controlled stress on 10 known positives and explicitly reviewed hard negatives, not independent examples or population false-alarm estimates. Gain edits may alter perceived severity. No thresholds were tuned on these variants.",
            "summary": summary,
            "rows": results,
        },
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
