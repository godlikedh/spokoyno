#!/usr/bin/env python3
"""Consume atomic patch files while robust-data generation runs; bound RAM usage."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from event_dataset import atomic_npz
from extract_audio_context import Encoder, ensure_encoder, sha256
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    args = parser.parse_args()
    encoder_info = ensure_encoder(args.encoder_dir)
    encoder = Encoder(args.encoder_dir, threads=3)
    count = 0
    while True:
        pending = sorted((args.data_dir / "pending").glob("*.npz"))
        ready = [
            p
            for p in pending
            if (args.data_dir / "cache" / (p.stem + ".json")).exists()
        ]
        for path in ready:
            meta = json.loads(
                (args.data_dir / "cache" / (path.stem + ".json")).read_text()
            )
            if sha256(path) != meta["patch_sha256"]:
                raise ValueError("patch file checksum mismatch")
            target = args.data_dir / "embeddings" / path.name
            with np.load(path, allow_pickle=False) as source:
                patches = source["patches"]
            values = np.concatenate(
                [
                    encoder(patches[start : start + 96])
                    for start in range(0, len(patches), 96)
                ]
            )
            atomic_npz(target, embeddings=values.reshape(-1, 3072))
            atomic_json(
                target.with_suffix(".json"),
                {"sha256": sha256(target), "patch_sha256": meta["patch_sha256"]},
            )
            # This is a transient file created by this experiment, not corpus audio.
            path.unlink()
            count += 1
            if count % 100 == 0:
                print(f"embedded {count} records", flush=True)
        complete = args.data_dir / "manifest.json"
        if complete.exists() and not list((args.data_dir / "pending").glob("*.npz")):
            manifest = json.loads(complete.read_text())
            for record in manifest["records"]:
                file = args.data_dir / "embeddings" / (record["key"] + ".npz")
                meta = json.loads(file.with_suffix(".json").read_text())
                if sha256(file) != meta["sha256"]:
                    raise ValueError("embedding cache is incomplete")
            atomic_json(
                args.data_dir / "encoder.json",
                {
                    "encoder": encoder_info,
                    "dataset_sha256": sha256(complete),
                    "records": len(manifest["records"]),
                    "source_sha256": sha256(Path(__file__)),
                },
            )
            print("all encoder records verified", flush=True)
            return
        if not ready:
            time.sleep(1)


if __name__ == "__main__":
    main()
