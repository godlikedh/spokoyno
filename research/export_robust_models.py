#!/usr/bin/env python3
"""Export the completed, audited experiment as non-executable JSON snapshots."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from robust_tree import window_probabilities
from train_models import atomic_json
from train_robust_models import load_data, pool, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(args.data_dir)
    results = json.loads((args.run / "results.json").read_text())
    fit = args.run / "fits" / results["final_fit_key"]
    meta = json.loads((fit / "manifest.json").read_text())
    for name, expected in meta["files"].items():
        if sha256(fit / name) != expected:
            raise ValueError("final fit was changed")
    transform = json.loads((fit / "transform.json").read_text())
    original_models = json.loads((fit / "models.json").read_text())
    models = {
        name: {k: v for k, v in model.items() if k != "training_records"}
        for name, model in original_models.items()
    }
    atomic_json(args.output_dir / "transform.json", transform)
    atomic_json(args.output_dir / "models.json", models)
    # Re-read the published files, rather than checking in-memory fit objects.
    transform = json.loads((args.output_dir / "transform.json").read_text())
    models = json.loads((args.output_dir / "models.json").read_text())
    scores, errors = {}, {}
    with np.load(fit / "scores.npz", allow_pickle=False) as expected:
        for name, model in models.items():
            probabilities = np.empty(len(data["owners"]))
            for lo in range(0, len(probabilities), 4096):
                sl = slice(lo, lo + 4096)
                x = prepare(
                    transform,
                    data["physical"][sl],
                    data["audio"][sl] if model["audio"] else None,
                )
                probabilities[sl] = window_probabilities(model, x)
            scores[name] = pool(
                probabilities, data["owners"], len(data["records"]), model["power"]
            )[0]
            errors[name] = float(np.max(np.abs(scores[name] - expected[name])))
            if errors[name] > 1e-10:
                raise ValueError(f"published model does not reproduce scores: {name}")
    atomic_json(
        args.output_dir / "serialization-parity.json",
        {"records": len(data["records"]), "maximum_absolute_error": errors},
    )
    atomic_json(
        args.output_dir / "evaluation.json",
        {
            key: results[key]
            for key in (
                "metrics",
                "selection_decisions",
                "final_model",
                "final_fit_key",
            )
        },
    )
    # Keep all positive predictions and every false warning; full arrays remain
    # in the local immutable run and are identified by checksum below.
    atomic_json(
        args.output_dir / "heldout-errors.json",
        {
            name: {
                view: [
                    row
                    for row in rows
                    if row["label"] == "positive" or row["score"] >= 0.6
                ]
                for view, rows in views.items()
            }
            for name, views in results["predictions"].items()
        },
    )
    plan = json.loads((args.run / "plan.json").read_text())
    atomic_json(
        args.output_dir / "experiment-plan.json",
        {
            **{k: v for k, v in plan.items() if k != "folds"},
            "complete_plan_sha256": sha256(args.run / "plan.json"),
            "folds": [
                {
                    "thread": f["thread"],
                    "train_sources": len(f["train"]),
                    "test_sources": len(f["test"]),
                    "inner_folds": len(f["inner"]),
                }
                for f in plan["folds"]
            ],
        },
    )
    for source, target in (
        (args.audit / "summary.json", "audit.json"),
        (args.audit / "browser.json", "browser.json"),
        (args.audit / "stress.json", "stress.json"),
        (args.audit / "delay-invariance.json", "delay-invariance.json"),
    ):
        shutil.copyfile(source, args.output_dir / target)
    for name in ("followup-declaration.json", "resume-provenance.json"):
        if (args.run / name).exists():
            shutil.copyfile(args.run / name, args.output_dir / name)
    manifest = {
        "schema": 1,
        "status": "research-only; within-experiment model selection is not production promotion",
        "default_model": results["final_model"],
        "thresholds": {"maybe": 0.6, "alert": 0.8},
        "encoder": data["encoder"]["encoder"],
        "feature_count": len(data["names"]),
        "original_training_sources": len(meta["training_indices"]),
        "training_hashes": meta["training_hashes"],
        "dataset_sha256": sha256(args.data_dir / "manifest.json"),
        "full_results_sha256": sha256(args.run / "results.json"),
        "human_labels_sha256": sha256(Path("corpus/labels.json")),
        "human_intervals_sha256": sha256(Path("corpus/events.json")),
        "extractor_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "robust_audio.py",
                "acoustic_features.py",
                "extract_features.py",
                "extract_audio_context.py",
                "train_robust_models.py",
                "robust_tree.py",
            )
        },
        "files": {
            p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()
        },
        "limitations": "Scores are not probabilities. The development corpus has been used in prior research. Synthetic and codec copies do not increase the number of independent positives. Nested source/thread exclusions reduce leakage but cannot establish performance on future threads. Published final models have seen current labels; use held-out predictions for corpus accuracy.",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"default_model": manifest["default_model"], "parity": errors}, indent=2
        )
    )


if __name__ == "__main__":
    main()
