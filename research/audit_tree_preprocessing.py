#!/usr/bin/env python3
"""Diagnose tree sensitivity to training/inference PCA roundoff without changing fits."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from robust_tree import window_probabilities
from train_models import atomic_json
from train_robust_models import fitted_transform, load_data, pool, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnostic evidence")
    results = json.loads((args.run / "results.json").read_text())
    directory = args.run / "fits" / results["final_fit_key"]
    meta = json.loads((directory / "manifest.json").read_text())
    for name, expected in meta["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError("saved fit changed")
    data = load_data(args.data_dir)
    transform = json.loads((directory / "transform.json").read_text())
    models = json.loads((directory / "models.json").read_text())
    _, training_x = fitted_transform(data, meta["training_indices"])
    inference_x = prepare(transform, data["physical"], data["audio"])
    evidence = {}

    def tier(values):
        return (values >= 0.6).astype(int) + (values >= 0.8).astype(int)

    with np.load(directory / "scores.npz", allow_pickle=False) as expected:
        for name, model in models.items():
            before = window_probabilities(model, training_x)
            after = window_probabilities(model, inference_x)
            original_scores = pool(
                before, data["owners"], len(data["records"]), model["power"]
            )[0]
            scores = pool(after, data["owners"], len(data["records"]), model["power"])[
                0
            ]
            if np.max(abs(original_scores - expected[name])) > 1e-10:
                raise ValueError(
                    "reconstructed training preprocessing does not reproduce saved scores"
                )
            changed = np.flatnonzero(abs(scores - expected[name]) > 1e-10)
            windows = []
            for row in np.flatnonzero(abs(before - after) > 1e-10):
                rid = int(data["owners"][row])
                nodes = []
                for tree in model["trees"]:
                    node = tree[0]
                    while not node["leaf"]:
                        f, threshold = node["feature"], node["threshold"]
                        if (training_x[row, f] <= threshold) != (
                            inference_x[row, f] <= threshold
                        ):
                            nodes.append(
                                {
                                    "feature": f,
                                    "training": float(training_x[row, f]),
                                    "inference": float(inference_x[row, f]),
                                    "threshold": threshold,
                                }
                            )
                            break
                        node = tree[
                            node["left"]
                            if training_x[row, f] <= threshold
                            else node["right"]
                        ]
                windows.append(
                    {
                        "window": int(row),
                        "record": rid,
                        "probability_before": float(before[row]),
                        "probability_after": float(after[row]),
                        "different_nodes": nodes,
                    }
                )
            evidence[name] = {
                "records": len(scores),
                "changed_records": len(changed),
                "maximum_score_error": float(np.max(abs(scores - expected[name]))),
                "tier_changes": int(
                    np.count_nonzero(tier(scores) != tier(expected[name]))
                ),
                "windows": windows,
                "changed_record_details": [
                    {
                        "record": int(i),
                        "kind": data["records"][i]["kind"],
                        "split": data["records"][i]["split"],
                        "before": float(expected[name][i]),
                        "after": float(scores[i]),
                    }
                    for i in changed
                ],
            }
    atomic_json(
        args.output,
        {
            "source_sha256": sha256(Path(__file__)),
            "data_sha256": sha256(args.data_dir / "manifest.json"),
            "fit_manifest_sha256": sha256(directory / "manifest.json"),
            "maximum_input_difference": float(np.max(abs(training_x - inference_x))),
            "models": evidence,
            "interpretation": "PCA matrix-operation ordering introduces roundoff. A few exact tree-threshold ties change branches. This diagnostic does not relax export checks or change fitted models, labels, thresholds, or held-out results.",
        },
    )
    print(
        json.dumps(
            {
                n: {
                    k: v
                    for k, v in r.items()
                    if k not in ("windows", "changed_record_details")
                }
                for n, r in evidence.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
