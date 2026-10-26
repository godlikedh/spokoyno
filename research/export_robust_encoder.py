#!/usr/bin/env python3
"""Export the pinned frozen YAMNet embedding network to simple portable tensors.

The browser runtime uses standard TensorFlow.js convolution operations. No model
conversion service or audio upload is involved. The classifier head is omitted.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from extract_audio_context import ensure_encoder, sha256
from extract_features import RATE, high_shelf_sos
from scipy.signal import butter
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    info = ensure_encoder(args.encoder_dir)
    import tensorflow as tf
    import tf_keras

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.encoder_dir.resolve()))
    params = importlib.import_module("params").Params()
    yamnet = importlib.import_module("yamnet")
    frontend = importlib.import_module("features")
    inputs = tf_keras.layers.Input(shape=(96, 64))
    outputs = yamnet.yamnet(inputs, params)
    network = tf_keras.Model(inputs, outputs)
    network.load_weights(str(args.encoder_dir / "yamnet.h5"))
    operations, tensors, chunks = [], {}, []
    cursor = 0

    def add(name, values):
        nonlocal cursor
        values = np.asarray(values, dtype="<f4")
        tensors[name] = {
            "offset": cursor,
            "shape": list(values.shape),
            "length": values.size,
        }
        chunks.append(values.tobytes())
        cursor += values.size
        return name

    for layer in network.layers:
        kind = type(layer).__name__
        if kind == "InputLayer":
            continue
        config = {
            key: value
            for key, value in layer.get_config().items()
            if key in ("target_shape", "strides", "padding", "epsilon", "axis")
        }
        names = {
            v.name.split(":")[0].split("/")[-1]: add(v.name, v.numpy())
            for v in layer.weights
        }
        operations.append(
            {"kind": kind, "name": layer.name, "config": config, "weights": names}
        )
        if kind == "GlobalAveragePooling2D":
            break
    mel = tf.signal.linear_to_mel_weight_matrix(
        64, 257, RATE, params.mel_min_hz, params.mel_max_hz
    ).numpy()
    add("frontend/mel", mel)
    add("frontend/hann", tf.signal.hann_window(400).numpy())
    weights = args.output_dir / "weights.bin"
    weights.write_bytes(b"".join(chunks))
    atomic_json(
        args.output_dir / "model.json",
        {
            "schema": 1,
            "encoder": info,
            "weights_sha256": sha256(weights),
            "tensors": tensors,
            "operations": operations,
            "frontend": {
                "frames": 96,
                "hop": 160,
                "fft": 512,
                "window": 400,
                "log_offset": params.log_offset,
            },
            "dsp_sos": np.vstack(
                (
                    butter(2, 70, btype="highpass", fs=RATE, output="sos"),
                    high_shelf_sos(RATE),
                )
            ).tolist(),
            "source_sha256": sha256(Path(__file__)),
        },
    )
    rng = np.random.default_rng(18)
    patches = np.stack(
        [
            np.zeros(15600),
            rng.normal(0, 0.1, 15600),
            0.5 * np.sin(2 * np.pi * 440 * np.arange(15600) / RATE),
        ]
    ).astype(np.float32)
    features = np.stack(
        [
            frontend.waveform_to_log_mel_spectrogram_patches(patch, params)[1].numpy()[
                0
            ]
            for patch in patches
        ]
    )
    embeddings = network(features, training=False)[1].numpy()
    patches.astype("<f4").tofile(args.output_dir / "fixture-patches.bin")
    features.astype("<f4").tofile(args.output_dir / "fixture-logmel.bin")
    embeddings.astype("<f4").tofile(args.output_dir / "fixture-embeddings.bin")
    print(
        json.dumps(
            {
                "operations": len(operations),
                "weight_bytes": weights.stat().st_size,
                "fixtures": len(patches),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
