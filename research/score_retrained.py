#!/usr/bin/env python3
"""Publish or score trusted, versioned retraining artifacts; never fit a model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from extract_audio_context import sha256
from retrain_models import (
    CANDIDATES,
    load_inputs,
    predict_candidate,
    prediction_rows,
    summary,
)
from train_models import atomic_json


def load_trusted_model(directory: Path) -> tuple[dict, dict]:
    """Checksums detect corruption, NOT malicious pickle; caller must trust origin."""
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "model.joblib"
    if manifest["schema"] != 1 or manifest["model_sha256"] != sha256(path):
        raise ValueError("model checksum/schema mismatch")
    if manifest["sklearn"] != sklearn.__version__:
        raise ValueError(
            "use the recorded scikit-learn version for pickle compatibility"
        )
    return joblib.load(path), manifest


def align_inputs(data: dict, manifest: dict) -> dict:
    if manifest["provenance"]["encoder_sha256"] != data["provenance"]["encoder_sha256"]:
        raise ValueError("audio encoder changed")
    if manifest["event_feature_names"] != data["event_names"]:
        raise ValueError("event feature schema changed")
    expected = manifest["feature_names"]
    if set(expected) - set(data["names"]):
        raise ValueError("missing fitted physical features")
    # Candidate columns index its FITTED schema, not a newly alphabetized schema.
    columns = [data["names"].index(name) for name in expected]
    return data | {"names": expected, "physical": data["physical"][:, columns]}


def publish(source: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refusing to overwrite a published snapshot")
    results = json.loads((source / "results.json").read_text())
    candidates = results["plan"]["candidates"]
    if set(candidates) - set(CANDIDATES) or len(candidates) != len(set(candidates)):
        raise ValueError("unknown or duplicate candidates")
    # Validate every source before creating a partially published snapshot.
    loaded = {name: load_trusted_model(source / name / "final") for name in candidates}
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, (fitted, manifest) in loaded.items():
        directory = output / name / "final"
        directory.mkdir(parents=True)
        path = directory / "model.joblib"
        joblib.dump(fitted, path, compress=3)
        spec = manifest | {
            "source_model_sha256": manifest["model_sha256"],
            "model_sha256": sha256(path),
            "context_seed": 42,
            "seed_note": "seed is the tabular seed; existing context/event fit uses random_state=42",
        }
        atomic_json(directory / "manifest.json", spec)
        artifacts[name] = {
            "manifest": f"{name}/final/manifest.json",
            "model_sha256": spec["model_sha256"],
        }
    experiment_summaries = []
    for result in results["experiments"]:
        # Keep every positive held-out result inspectable in Git. Full per-group
        # predictions and all fold models remain in the immutable local run.
        compact = {k: v for k, v in result.items() if not k.endswith("predictions")}
        compact["held_out_positive_predictions"] = [
            r for r in result["held_out_predictions"] if r["label"] == "positive"
        ]
        compact["latest_thread_positive_predictions"] = [
            r
            for r in result["latest_thread_excluded_predictions"]
            if r["label"] == "positive"
        ]
        compact["final_model_manifest"] = artifacts[result["name"]]["manifest"]
        experiment_summaries.append(compact)
    atomic_json(
        output / "results-summary.json",
        {
            "schema": 1,
            "status": "shadow-only",
            "source_results_sha256": sha256(source / "results.json"),
            "source_run": str(source),
            "source_plan_sha256": sha256(source / "plan.json"),
            "plan": results["plan"]
            | {
                "folds": [
                    {
                        "thread": f["thread"],
                        "train_groups": len(f["train"]),
                        "test_groups": len(f["test"]),
                    }
                    for f in results["plan"]["folds"]
                ]
            },
            "models": artifacts,
            "experiments": experiment_summaries,
            "warning": "Published research snapshots, not browser promotion. Joblib files are executable pickle; only load artifacts from a trusted source. Fold choice/model comparison used this development corpus; future evaluation required.",
        },
    )


def score(run: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refusing to overwrite predictions")
    index = json.loads((run / "results-summary.json").read_text())
    data = load_inputs()
    predictions, summaries = {}, {}
    for name, entry in index["models"].items():
        if name not in CANDIDATES:
            raise ValueError("unknown candidate")
        directory = run / name / "final"
        fitted, manifest = load_trusted_model(directory)
        if manifest["model_sha256"] != entry["model_sha256"]:
            raise ValueError("snapshot/model mismatch")
        scores = predict_candidate(fitted, align_inputs(data, manifest))
        rows = prediction_rows(
            data, scores, list(range(len(data["clips"]))), manifest["threshold"]
        )
        used = set(manifest["training_audio_hashes"])
        eligible = set(manifest["eligible_training_audio_hashes"])
        for row in rows:
            row["seen_in_training"] = row["audio_sha256"] in used
            row["eligible_in_final_fit"] = row["audio_sha256"] in eligible
        predictions[name] = rows
        summaries[name] = summary(rows)
    atomic_json(
        output,
        {
            "schema": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "shadow-only",
            "snapshot_sha256": sha256(run / "results-summary.json"),
            "provenance": data["provenance"],
            "summary": summaries,
            "predictions": predictions,
        },
    )
    print(f"Scored {len(predictions)} frozen models; wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trust-models",
        action="store_true",
        help="acknowledge these joblib files are from a trusted source; pickle can execute code",
    )
    args = parser.parse_args()
    if not args.trust_models:
        raise ValueError(
            "joblib can execute code; verify origin, then explicitly pass --trust-models"
        )
    if args.command == "publish":
        publish(args.run, args.output)
    else:
        score(args.run, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
