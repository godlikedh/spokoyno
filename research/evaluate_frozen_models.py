#!/usr/bin/env python3
"""Compare every saved model without fitting or changing its decision policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from risk_score import POLICY, risk_tier
from train_audio_context import load_data as load_context
from train_audio_context import predict
from train_events import clip_predictions, load_data, metrics
from train_models import atomic_json, matrix, predict_serialized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite frozen predictions")
    features_path = Path("research/artifacts/features-v1.json")
    labels_path = Path("corpus/labels.json")
    events_path = Path("corpus/events.json")
    logical, _, embeddings, context = load_context(
        features_path, Path("research/artifacts/audio-context-v1"), labels_path
    )
    manifest, event_physical, event_audio, _ = load_data(
        Path("research/artifacts/event-data-v1")
    )
    if (
        manifest["feature_dataset_sha256"] != sha256(features_path)
        or manifest["labels_sha256"] != sha256(labels_path)
        or manifest["events_sha256"] != sha256(events_path)
        or manifest["context_manifest_sha256"]
        != sha256(Path("research/artifacts/audio-context-v1/manifest.json"))
    ):
        raise ValueError("stale event dataset; regenerate features/context/events")
    by_hash = {row["audio_sha256"]: row for row in logical}
    clips = manifest["clips"]
    rows = [
        {
            key: clip[key]
            for key in ("path", "paths", "threads", "label", "audio_sha256")
        }
        | {"models": {}}
        for clip in clips
    ]
    whole_rows = [by_hash[clip["audio_sha256"]] for clip in clips]
    audio = np.stack([embeddings[clip["audio_sha256"]] for clip in clips])
    original_membership = json.loads(
        Path("research/artifacts/audio-models/physical-model.json").read_text()
    )
    encoder_hash = hashlib.sha256(
        json.dumps(context["encoder"], sort_keys=True).encode()
    ).hexdigest()
    if original_membership["encoder_sha256"] != encoder_hash:
        raise ValueError("saved models do not match the extracted audio encoder")
    definitions = [
        ("logistic", Path("research/models/shadow-model.json"), "original"),
        ("forest", Path("research/models/challenger-model.json"), "original"),
        *[
            (
                f"clip-{mode}",
                Path(f"research/artifacts/audio-models/{mode}-model.json"),
                "clip",
            )
            for mode in ("physical", "embedding", "hybrid")
        ],
        *[
            (
                f"event-{mode}",
                Path(f"research/artifacts/event-models-v1/{mode}-model.json"),
                "event",
            )
            for mode in ("physical", "hybrid")
        ],
    ]
    artifacts, summaries = {}, {}
    for name, path, kind in definitions:
        artifact_hash = sha256(path)
        artifact = json.loads(path.read_text())
        threshold = None
        if kind == "original":
            if (
                artifact["feature_dataset_sha256"]
                != original_membership["feature_dataset_sha256"]
            ):
                raise ValueError("cannot establish original-model training membership")
            trained = set(original_membership["training_audio_hashes"])
            model = artifact["model"]
            scores = predict_serialized(
                model, matrix(whole_rows, model["feature_names"])
            )
            threshold = artifact["training_zero_fp_threshold"]
        else:
            trained = set(artifact["training_audio_hashes"])
            if kind == "clip":
                if artifact["encoder_sha256"] != original_membership["encoder_sha256"]:
                    raise ValueError("inconsistent saved audio encoder")
                scores = predict(
                    artifact["model"],
                    matrix(whole_rows, artifact["physical_names"]),
                    audio,
                )
                threshold = artifact["threshold"]
            else:
                if artifact["physical_names"] != manifest["physical_names"]:
                    raise ValueError("event feature schema mismatch")
                window_scores = predict(artifact["model"], event_physical, event_audio)
                predicted = clip_predictions(
                    clips, manifest["windows"], window_scores, list(range(len(clips)))
                )
                scores = np.asarray([row["score"] for row in predicted])
        for row, score in zip(rows, scores, strict=True):
            row["models"][name] = {
                "score": float(score),
                "risk_tier": risk_tier(float(score)),
                "seen_in_training": row["audio_sha256"] in trained,
                "training_threshold": threshold,
                "above_training_threshold": bool(score > threshold)
                if threshold is not None
                else None,
            }
        summaries[name] = {}
        for split in ("all", "training", "new"):
            selected = [
                row | row["models"][name]
                for row in rows
                if split == "all"
                or row["models"][name]["seen_in_training"] == (split == "training")
            ]
            summaries[name][split] = metrics(selected)
            summaries[name][split]["yellow_only_negatives"] = sum(
                row["label"] == "negative" and row["risk_tier"] == "maybe"
                for row in selected
            )
            if threshold is not None:
                summaries[name][split]["historical_threshold"] = {
                    "threshold": threshold,
                    "operator": ">",
                    "detected": sum(
                        row["label"] == "positive" and row["above_training_threshold"]
                        for row in selected
                    ),
                    "false_warnings": sum(
                        row["label"] == "negative" and row["above_training_threshold"]
                        for row in selected
                    ),
                }
        if sha256(path) != artifact_hash:
            raise ValueError(f"model changed while scoring: {path}")
        artifacts[name] = {"path": str(path), "sha256": artifact_hash}
        print(json.dumps({"model": name, "summary": summaries[name]}), flush=True)
    atomic_json(
        args.output,
        {
            "schema": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "risk_policy": POLICY,
            "features_sha256": sha256(features_path),
            "labels_sha256": sha256(labels_path),
            "models": artifacts,
            "warning": "Frozen-model development comparison on retained FFmpeg audio; no retraining. Exact-audio groups counted once. Unknown/visual-only labels excluded from accuracy. New excludes exact training hashes, not all possible related audio; labels were known. Fixed 0.6/0.8 tiers and historical thresholds are different policies.",
            "summary": summaries,
            "rows": rows,
        },
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
