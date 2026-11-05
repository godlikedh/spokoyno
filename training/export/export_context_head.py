#!/usr/bin/env python3
"""Export selected frozen AudioSet classifier columns and verify graph parity."""

import argparse
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.special import expit

from training.features.context_features import GROUPS
from training.features.extract_audio_context import ensure_encoder
from training.paths import artifact_path, source_path
from training.storage import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, default=artifact_path("yamnet"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=artifact_path("portable-encoder"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    info = ensure_encoder(args.encoder_dir)
    with (args.encoder_dir / "yamnet_class_map.csv").open() as source:
        classes = list(csv.DictReader(source))
    indices = sorted({i for group in GROUPS.values() for i in group})
    with h5py.File(args.encoder_dir / "yamnet.h5", "r") as model:
        kernel = model["logits/logits/kernel:0"][()][:, indices]
        bias = model["logits/logits/bias:0"][()][indices]
    head = {
        "indices": indices,
        "classes": [classes[i]["display_name"] for i in indices],
        "groups": GROUPS,
        "kernel": kernel.tolist(),
        "bias": bias.tolist(),
        "encoder": info,
    }
    atomic_json(args.output_dir / "head.json", head)
    shutil.copyfile(args.encoder_dir / "LICENSE", args.output_dir / "LICENSE")
    import tensorflow as tf
    import tf_keras

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.encoder_dir.resolve()))
    yamnet = importlib.import_module("yamnet")
    params = importlib.import_module("params").Params()
    inputs = tf_keras.layers.Input(shape=(96, 64))
    network = tf_keras.Model(inputs, yamnet.yamnet(inputs, params))
    network.load_weights(str(args.encoder_dir / "yamnet.h5"))
    features = np.fromfile(args.fixtures / "fixture-logmel.bin", dtype="<f4").reshape(
        -1, 96, 64
    )
    expected, embeddings = [x.numpy() for x in network(features, training=False)]
    actual = expit(embeddings.astype(float) @ kernel.astype(float) + bias.astype(float))
    error = float(np.max(np.abs(actual - expected[:, indices])))
    if error > 1e-6:
        raise ValueError("exported semantic scores differ from official graph")
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "feature_source_sha256": sha256(source_path("context_features.py")),
            "maximum_graph_score_error": error,
            "selected_classes": len(indices),
            "files": {n: sha256(args.output_dir / n) for n in ("head.json", "LICENSE")},
        },
    )
    print(json.dumps({"classes": len(indices), "maximum_graph_score_error": error}))


if __name__ == "__main__":
    main()
