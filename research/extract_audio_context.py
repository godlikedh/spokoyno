#!/usr/bin/env python3
"""Extract frozen YAMNet embeddings around label-independent event candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from extract_features import RATE, WINDOW, read_audio
from train_models import atomic_json

SOURCE_REVISION = "d598fb8b23d9cd2fb26b5789b8242de3f494aca7"
SOURCE_ROOT = (
    f"https://raw.githubusercontent.com/tensorflow/models/{SOURCE_REVISION}"
    "/research/audioset/yamnet/"
)
WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
WEIGHTS_SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"
PATCH_SAMPLES = 15_600  # 96 STFT frames, including the last 25 ms analysis window.
CONTEXT_VERSION = 1
SLOTS = [
    f"event_{i}_{part}" for i in range(1, 4) for part in ("before", "event", "after")
]
SLOTS.append("opening")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_encoder(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["source_revision"] != SOURCE_REVISION:
        raise ValueError("encoder revision changed; use a new --encoder-dir")
    urls = {
        name: SOURCE_ROOT + name
        for name in ("yamnet.py", "features.py", "params.py", "yamnet_class_map.csv")
    }
    urls["LICENSE"] = (
        f"https://raw.githubusercontent.com/tensorflow/models/{SOURCE_REVISION}/LICENSE"
    )
    urls["yamnet.h5"] = WEIGHTS_URL
    hashes = {}
    for name, url in urls.items():
        target = directory / name
        if not target.exists():
            print(f"downloading {url}", file=sys.stderr, flush=True)
            temporary = target.with_suffix(target.suffix + ".download")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                while block := response.read(1024 * 1024):
                    out.write(block)
            os.replace(temporary, target)
        hashes[name] = sha256(target)
        if name == "yamnet.h5" and hashes[name] != WEIGHTS_SHA256:
            raise ValueError(
                "downloaded YAMNet weights do not match the pinned checksum"
            )
        if previous and hashes[name] != previous["sha256"][name]:
            raise ValueError(f"encoder file changed: {target}")
    manifest = {"source_revision": SOURCE_REVISION, "urls": urls, "sha256": hashes}
    atomic_json(manifest_path, manifest)
    return manifest


def crop(mono: np.ndarray, start_s: float) -> np.ndarray:
    """Fixed-duration context, with explicit padding outside the clip."""
    start = round(start_s * RATE)
    result = np.zeros(PATCH_SAMPLES, dtype=np.float32)
    lo, hi = max(0, start), min(len(mono), start + PATCH_SAMPLES)
    if hi > lo:
        result[lo - start : hi - start] = mono[lo:hi]
    # YAMNet expects [-1,1]. Preserve relative levels within each patch; raw
    # loudness and decoder overshoot remain in the separate physical features.
    result /= max(1.0, float(np.max(np.abs(result))))
    return result


def context_starts(features: dict, sample_count: int) -> list[float | None]:
    starts = []
    last_window = max(1, sample_count // WINDOW - 1)
    for number in range(1, 4):
        prefix = f"event_{number}_"
        position = features.get(prefix + "position_fraction")
        if position is None:
            starts.extend([None] * 3)
            continue
        at = position * last_window * WINDOW / RATE
        duration = np.clip(features.get(prefix + "duration_s", 0.3), 0.3, 3.0)
        starts.extend([at - 0.1 - PATCH_SAMPLES / RATE, at - 0.1, at + duration])
    starts.append(float(features["start_position_s"]))
    return starts


def context_audio(samples: np.ndarray, features: dict) -> tuple[np.ndarray, list]:
    # Select a real channel instead of averaging waveforms: antiphase stereo
    # must not cancel. This rule is fixed and never uses a class/event label.
    energies = np.mean(np.square(samples[::16], dtype=np.float64), axis=0)
    mono = samples[:, int(np.argmax(energies))]
    starts = context_starts(features, len(samples))
    patches = np.stack(
        [
            crop(mono, at) if at is not None else np.zeros(PATCH_SAMPLES, np.float32)
            for at in starts
        ]
    )
    return patches, starts


class Encoder:
    def __init__(self, directory: Path, threads: int = 4):
        import tensorflow as tf
        import tf_keras

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        sys.path.insert(0, str(directory.resolve()))
        params = importlib.import_module("params").Params()
        yamnet = importlib.import_module("yamnet")
        features = importlib.import_module("features")
        # Use the official feature frontend and network, with independent
        # patches batched through the CNN. No concatenation-boundary embeddings.
        patch_input = tf_keras.layers.Input(shape=(96, 64))
        scores, embeddings = yamnet.yamnet(patch_input, params)
        network = tf_keras.Model(patch_input, [scores, embeddings])
        network.load_weights(str(directory / "yamnet.h5"))

        @tf.function(input_signature=[tf.TensorSpec((None, PATCH_SAMPLES), tf.float32)])
        def encode(waveforms):
            patches = tf.map_fn(
                lambda waveform: features.waveform_to_log_mel_spectrogram_patches(
                    waveform, params
                )[1][0],
                waveforms,
                fn_output_signature=tf.TensorSpec((96, 64), tf.float32),
            )
            return network(patches, training=False)[1]

        self.encode = encode
        # Guard the batching adaptation against the unmodified official model.
        official = yamnet.yamnet_frames_model(params)
        official.load_weights(str(directory / "yamnet.h5"))
        rng = np.random.default_rng(42)
        fixture = rng.normal(0, 0.05, PATCH_SAMPLES).astype(np.float32)
        expected = official(fixture, training=False)[1].numpy()
        actual = self(fixture[None])
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        return self.encode(patches).numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/artifacts/audio-context-v1")
    )
    parser.add_argument(
        "--limit", type=int, help="smoke run over the first N unique clips"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    encoder_manifest = ensure_encoder(args.encoder_dir)
    payload = json.loads(args.features.read_text())
    if payload.get("feature_version") != 1:
        raise ValueError("context v1 requires physical feature version 1")
    unique = {}
    for row in payload["rows"]:
        if row["label"] != "visual-only":
            unique.setdefault(row["audio_sha256"], row)
    rows = list(unique.values())[: args.limit]
    encoder_id = hashlib.sha256(
        json.dumps(encoder_manifest, sort_keys=True).encode()
    ).hexdigest()
    cache = args.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    encoder = None
    vectors, metadata = [], []
    for number, row in enumerate(rows, 1):
        input_id = hashlib.sha256(
            json.dumps(
                {
                    "version": CONTEXT_VERSION,
                    "encoder": encoder_id,
                    "audio": row["audio_sha256"],
                    "features": row["features"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        target = cache / f"{input_id}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                vector = saved["embedding"]
        else:
            source = args.audio_dir / f"{row['file']}.audio.wav"
            if sha256(source) != row["audio_sha256"]:
                raise ValueError(
                    f"audio hash differs from extracted features: {source}"
                )
            samples = read_audio(source)
            patches, _ = context_audio(samples, row["features"])
            if encoder is None:
                encoder = Encoder(args.encoder_dir, args.threads)
            vector = encoder(patches).reshape(-1)
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as out:
                np.savez_compressed(out, embedding=vector)
            os.replace(temporary, target)
        if vector.shape != (len(SLOTS) * 1024,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding for {row['file']}")
        vectors.append(vector)
        metadata.append(
            {
                "audio_sha256": row["audio_sha256"],
                "path": row["path"],
                "input_sha256": input_id,
            }
        )
        if number % 25 == 0 or number == len(rows):
            print(f"context {number}/{len(rows)}", file=sys.stderr, flush=True)
    matrix_path = args.output_dir / "embeddings.npz"
    with matrix_path.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out, embeddings=np.asarray(vectors, dtype=np.float32))
    os.replace(matrix_path.with_suffix(".tmp"), matrix_path)
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": CONTEXT_VERSION,
            "encoder": encoder_manifest,
            "feature_dataset_sha256": sha256(args.features),
            "feature_version": payload["feature_version"],
            "slots": SLOTS,
            "embedding_width": 1024,
            "sample_rate": RATE,
            "patch_samples": PATCH_SAMPLES,
            "annotation_usage": "none; candidates do not use labels/timings",
            "matrix_sha256": sha256(matrix_path),
            "rows": metadata,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
