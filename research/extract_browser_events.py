#!/usr/bin/env python3
"""Re-extract event inputs from verified, saved original-media browser PCM.

Run in .venv-audio. This tests decoder sensitivity, not a JavaScript ML port.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from event_dataset import atomic_npz, physical_events
from extract_audio_context import Encoder, context_audio, sha256
from extract_features import extract_clip, read_audio
from train_models import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    args = parser.parse_args()
    sources = sorted(Path().glob(args.pattern))
    if not sources:
        raise ValueError("no browser fixtures")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads(
        Path("research/artifacts/event-data-v1/manifest.json").read_text()
    )
    feature_names = json.loads(Path("research/artifacts/features-v1.json").read_text())[
        "feature_names"
    ]
    encoder_manifest = json.loads((args.encoder_dir / "manifest.json").read_text())
    for name, expected in encoder_manifest["sha256"].items():
        if sha256(args.encoder_dir / name) != expected:
            raise ValueError("pinned encoder file changed")
    by_file = {Path(p).name: c for c in original["clips"] for p in c["paths"]}
    encoder = Encoder(args.encoder_dir, threads=2)
    clips, windows, physical, audio, evidence = [], [], [], [], []
    whole_clip = []
    for source in sources:
        replay = json.loads(source.read_text())
        media, pcm = Path(replay["source"]), Path(replay["decoded_audio"])
        if (
            sha256(media) != replay["source_sha256"]
            or sha256(pcm) != replay["decoded_audio_sha256"]
        ):
            raise ValueError("browser media/PCM hash mismatch")
        samples = read_audio(pcm)
        features = extract_clip(pcm)
        whole_clip.append([features.get(name, np.nan) for name in feature_names])
        vectors, points = physical_events(samples, features)
        patches, _ = context_audio(samples, features)
        embeddings = encoder.encode(patches).numpy()
        clip = by_file[media.name] | {
            "first_window": len(windows),
            "window_count": len(points),
        }
        for vector, point in zip(vectors, points, strict=True):
            physical.append([vector[name] for name in original["physical_names"]])
            slot = point["slot"]
            context = (
                embeddings[slot * 3 : slot * 3 + 3]
                if slot < 3
                else np.stack([np.zeros(1024), embeddings[9], np.zeros(1024)])
            )
            audio.append(context.reshape(-1))
            windows.append(point | {"clip_index": len(clips)})
        clips.append(clip)
        evidence.append(
            {
                "replay": str(source),
                "replay_sha256": sha256(source),
                "media_sha256": sha256(media),
                "pcm_sha256": sha256(pcm),
            }
        )
        print(f"browser event features {len(clips)}/{len(sources)}", flush=True)
    atomic_npz(
        args.output_dir / "examples.npz",
        physical=np.asarray(physical),
        whole_clip=np.asarray(whole_clip),
        embeddings=np.asarray(audio),
        owners=np.asarray([w["clip_index"] for w in windows]),
    )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "clips": clips,
            "windows": windows,
            "physical_names": original["physical_names"],
            "feature_names": feature_names,
            "evidence": evidence,
            "matrix_sha256": sha256(args.output_dir / "examples.npz"),
            "encoder_manifest_sha256": sha256(args.encoder_dir / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
