#!/usr/bin/env python3
"""Export the pinned pretrained taxonomy head; no corpus fitting occurs here."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from extract_audio_context import WEIGHTS_SHA256, sha256
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path("research/artifacts/yamnet")
    if sha256(source / "yamnet.h5") != WEIGHTS_SHA256:
        raise ValueError("pretrained weights changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {}
    with h5py.File(source / "yamnet.h5") as weights:

        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                arrays[name.rsplit("/", 1)[-1].removesuffix(":0")] = item[...]

        weights["logits"].visititems(collect)
    if arrays["kernel"].shape != (1024, 521) or arrays["bias"].shape != (521,):
        raise ValueError("unexpected YAMNet head")
    np.savez_compressed(args.output_dir / "head.npz", **arrays)
    classes = [
        r["display_name"]
        for r in csv.DictReader((source / "yamnet_class_map.csv").open())
    ]
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "weights_sha256": WEIGHTS_SHA256,
            "head_sha256": sha256(args.output_dir / "head.npz"),
            "class_map_sha256": sha256(source / "yamnet_class_map.csv"),
            "classes": classes,
        },
    )


if __name__ == "__main__":
    main()
