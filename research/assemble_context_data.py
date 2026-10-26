#!/usr/bin/env python3
"""Add label-independent semantic/temporal features without duplicating audio matrices."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from context_features import extract, relative_positions
from extract_audio_context import sha256
from train_models import atomic_json
from train_robust_models import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(args.base)
    head_manifest = json.loads((args.head_dir / "manifest.json").read_text())
    for name, expected in head_manifest["files"].items():
        if sha256(args.head_dir / name) != expected:
            raise ValueError("semantic head changed")
    head = json.loads((args.head_dir / "head.json").read_text())
    positions = relative_positions(data)
    _, extra_names = extract(data["audio"][:1], positions[:1], head)
    names = data["names"] + extra_names
    physical = np.lib.format.open_memmap(
        args.output_dir / "physical.npy",
        mode="w+",
        dtype=np.float64,
        shape=(len(data["owners"]), len(names)),
    )
    for lo in range(0, len(physical), 4096):
        sl = slice(lo, lo + 4096)
        extra, generated = extract(data["audio"][sl], positions[sl], head)
        if generated != extra_names or not np.all(np.isfinite(extra)):
            raise ValueError("semantic feature schema or numerical failure")
        physical[sl] = np.c_[data["physical"][sl], extra]
    physical.flush()
    for name in ("audio.npy", "owners.npy", "targets.npy"):
        os.link(args.base / name, args.output_dir / name)
    original = json.loads((args.base / "manifest.json").read_text())
    manifest = original | {
        "names": names,
        "files": {n: sha256(args.output_dir / n) for n in original["files"]},
        "source_sha256": sha256(Path(__file__)),
        "base_matrix_sha256": sha256(args.base / "manifest.json"),
        "context_features": {
            "head_sha256": sha256(args.head_dir / "head.json"),
            "head_directory": str(args.head_dir),
            "names": extra_names,
            "base_names": data["names"],
            "feature_source_sha256": sha256(
                Path(__file__).with_name("context_features.py")
            ),
            "definition": "Fixed ontology groups and temporal contrasts; no fitting and no class labels used. Opening position is relative to first audible content, invariant to leading digital zeros.",
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(
        f"assembled {len(physical)} windows, {len(extra_names)} new features, {len(names)} total physical/context features"
    )


if __name__ == "__main__":
    main()
