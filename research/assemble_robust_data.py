#!/usr/bin/env python3
"""Assemble validated per-record caches into memory-mapped training matrices."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.data_dir / "manifest.json").read_text())
    encoder = json.loads((args.data_dir / "encoder.json").read_text())
    if encoder["dataset_sha256"] != sha256(args.data_dir / "manifest.json"):
        raise ValueError("encoder belongs to another dataset")
    records, windows = [], []
    for record in source["records"]:
        meta = json.loads(
            (args.data_dir / "cache" / (record["key"] + ".json")).read_text()
        )
        if meta["names"] != source["names"]:
            raise ValueError("feature schema changed")
        records.append(
            record
            | {"first_window": len(windows), "window_count": len(meta["windows"])}
        )
        windows.extend(meta["windows"])
    count = len(windows)
    shape = {
        "physical": ((count, len(source["names"])), np.float64),
        "audio": ((count, 3072), np.float32),
        "targets": ((count,), np.int8),
        "owners": ((count,), np.int32),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            args.output_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=dims
        )
        for name, (dims, dtype) in shape.items()
    }
    for index, record in enumerate(records):
        features = args.data_dir / "cache" / (record["key"] + ".npz")
        audio = args.data_dir / "embeddings" / (record["key"] + ".npz")
        encoder_meta = json.loads(audio.with_suffix(".json").read_text())
        if (
            sha256(features) != record["feature_sha256"]
            or sha256(audio) != encoder_meta["sha256"]
        ):
            raise ValueError("record contents changed")
        lo, hi = record["first_window"], record["first_window"] + record["window_count"]
        with np.load(features, allow_pickle=False) as values:
            arrays["physical"][lo:hi] = values["physical"]
            arrays["targets"][lo:hi] = values["targets"]
        with np.load(audio, allow_pickle=False) as values:
            arrays["audio"][lo:hi] = values["embeddings"]
        arrays["owners"][lo:hi] = index
        if (index + 1) % 1000 == 0:
            print(f"assembled {index + 1}/{len(records)}", flush=True)
    for array in arrays.values():
        array.flush()
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": 1,
            "source_manifest_sha256": sha256(args.data_dir / "manifest.json"),
            "encoder": encoder,
            "names": source["names"],
            "clips": source["clips"],
            "records": records,
            "windows": windows,
            "files": {
                name + ".npy": sha256(args.output_dir / (name + ".npy"))
                for name in arrays
            },
            "source_sha256": sha256(Path(__file__)),
        },
    )
    print(f"assembled {len(records)} recordings, {count} event windows", flush=True)


if __name__ == "__main__":
    main()
