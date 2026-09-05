#!/usr/bin/env python3
"""Apply exported Spokoyno shadow models to a feature dataset without retraining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from risk_score import POLICY, risk_tier
from train_models import predict_serialized


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        dest="models",
        help="exported model JSON; repeat for multiple models",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("research/artifacts/scores-v1.json")
    )
    args = parser.parse_args()
    if not args.models:
        args.models = [
            Path("research/models/shadow-model.json"),
            Path("research/models/challenger-model.json"),
        ]

    features = json.loads(args.features.read_text())
    artifacts = [json.loads(path.read_text()) for path in args.models]
    for artifact in artifacts:
        if artifact["feature_version"] != features["feature_version"]:
            raise ValueError(
                f"feature version mismatch for {artifact['experiment']}: "
                f"model {artifact['feature_version']}, dataset {features['feature_version']}"
            )
    rows = features["rows"]
    result_rows = [
        {
            "path": row["path"],
            "file": row["file"],
            "thread": row["thread"],
            "label_at_scoring": row["label"],
            "audio_sha256": row["audio_sha256"],
            "models": {},
        }
        for row in rows
    ]
    for artifact in artifacts:
        model = artifact["model"]
        names = model["feature_names"]
        matrix = np.asarray(
            [[row["features"].get(name, np.nan) for name in names] for row in rows],
            dtype=np.float64,
        )
        scores = predict_serialized(model, matrix)
        threshold = float(artifact["training_zero_fp_threshold"])
        if artifact.get("decision_operator", ">") != ">":
            raise ValueError(
                f"unsupported decision operator in {artifact['experiment']}"
            )
        for result, score in zip(result_rows, scores, strict=True):
            result["models"][artifact["experiment"]] = {
                "score": float(score),
                "risk_tier": risk_tier(float(score)),
                "threshold": threshold,
                "above_training_threshold": bool(score > threshold),
            }

    payload = {
        "schema": 1,
        "risk_policy": POLICY,
        "feature_version": features["feature_version"],
        "warning": "Shadow scores are uncalibrated and do not change production warnings.",
        "models": [artifact["experiment"] for artifact in artifacts],
        "model_artifacts": {
            artifact["experiment"]: hashlib.sha256(path.read_bytes()).hexdigest()
            for artifact, path in zip(artifacts, args.models, strict=True)
        },
        "rows": result_rows,
    }
    atomic_json(args.output, payload)
    with args.output.with_suffix(".csv").open("w", newline="") as target:
        names = payload["models"]
        fields = ["path", "file", "thread", "label_at_scoring", "audio_sha256"]
        for name in names:
            fields.extend((f"{name}_score", f"{name}_above_training_threshold"))
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result_rows:
            flat = {key: row[key] for key in fields[:5]}
            for name in names:
                flat[f"{name}_score"] = row["models"][name]["score"]
                flat[f"{name}_above_training_threshold"] = row["models"][name][
                    "above_training_threshold"
                ]
            writer.writerow(flat)
    print(f"scored {len(result_rows)} tracks with {len(artifacts)} shadow models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
