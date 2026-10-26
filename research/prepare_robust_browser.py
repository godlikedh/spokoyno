#!/usr/bin/env python3
"""Prepare browser PCM and excluded-fold model fixtures for the portable runtime."""

import argparse
import json
from pathlib import Path

from extract_audio_context import sha256
from extract_features import read_audio
from improve_events import training_key
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--browser",
        type=Path,
        default=Path("research/artifacts/browser-events-full-20261025-v2"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.run / "plan.json").read_text())
    folds = {f["thread"]: f for f in plan["folds"]}
    source = json.loads((args.browser / "manifest.json").read_text())
    expected = json.loads((args.audit / "browser.json").read_text())
    bundles = {}
    fixtures = []
    for i, (clip, evidence, prediction) in enumerate(
        zip(source["clips"], source["evidence"], expected, strict=True)
    ):
        if clip["path"] != prediction["path"]:
            raise ValueError("browser predictions have another ordering")
        thread = clip["fold_thread"]
        if thread not in bundles:
            fit = args.run / "fits" / training_key(folds[thread]["train"])
            meta = json.loads((fit / "manifest.json").read_text())
            if any(
                sha256(fit / name) != expected
                for name, expected in meta["files"].items()
            ):
                raise ValueError("fold model changed")
            models = json.loads((fit / "models.json").read_text())
            bundle = {
                "transform": json.loads((fit / "transform.json").read_text()),
                "models": {
                    name: {k: v for k, v in m.items() if k != "training_records"}
                    for name, m in models.items()
                },
            }
            bundles[thread] = f"models-{thread}.json"
            atomic_json(args.output_dir / bundles[thread], bundle)
        replay = json.loads(Path(evidence["replay"]).read_text())
        pcm = Path(replay["decoded_audio"])
        if sha256(pcm) != evidence["pcm_sha256"]:
            raise ValueError("browser PCM changed")
        samples = read_audio(pcm)
        target = args.output_dir / f"{i}.bin"
        samples.astype("<f4").tofile(target)
        fixtures.append(
            {
                "file": clip["file"],
                "label": clip["label"],
                "pcm": target.name,
                "channels": samples.shape[1],
                "models": bundles[thread],
                "expected": prediction["predictions"],
                "pcm_sha256": sha256(target),
            }
        )
    atomic_json(args.output_dir / "fixtures.json", fixtures)
    print(f"prepared {len(fixtures)} browser fixtures")


if __name__ == "__main__":
    main()
