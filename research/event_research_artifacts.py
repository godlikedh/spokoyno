#!/usr/bin/env python3
"""Publish and score immutable, trusted event-research snapshots in shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from extract_audio_context import sha256
from improve_event_semantics import ALL_CANDIDATES, NEW_CANDIDATES, predict_semantic
from improve_events import display_scores, predict_fit, rows_for
from retrain_models import event_indices, load_inputs
from train_events import metrics
from train_models import atomic_json


def predict_model(name, model, data, return_windows=False):
    predictor = predict_semantic if name in NEW_CANDIDATES else predict_fit
    return predictor(model, data, return_windows=return_windows)


def load_fit(directory):
    spec = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "models.joblib") != spec["model_sha256"]:
        raise ValueError("model checksum mismatch")
    return joblib.load(directory / "models.joblib"), spec


def compact_results(result):
    return {key: value for key, value in result.items() if key != "predictions"} | {
        "held_out_positives": {
            name: [r for r in rows if r["label"] == "positive"]
            for name, rows in result["predictions"].items()
        },
        "held_out_false_alerts": {
            name: [r for r in rows if r["label"] == "negative" and r["score"] >= 0.8]
            for name, rows in result["predictions"].items()
        },
    }


def publish(round_one, round_two, output):
    if output.exists():
        raise ValueError("refusing to overwrite snapshot")
    data = load_inputs()
    runs, plans, results, models, manifests = [], [], [], {}, []
    for run in (round_one, round_two):
        plan = json.loads((run / "plan.json").read_text())
        result = json.loads((run / "results.json").read_text())
        if plan["provenance"] != data["provenance"] or result["plan_sha256"] != sha256(
            run / "plan.json"
        ):
            raise ValueError("stale run or changed plan")
        if plan["sklearn"] != sklearn.__version__:
            raise ValueError("use the recorded sklearn version")
        loaded, spec = load_fit(run / "fits" / result["final_fit_key"])
        with np.load(
            run / "fits" / result["final_fit_key"] / "scores.npz", allow_pickle=False
        ) as saved:
            for name, model in loaded.items():
                np.testing.assert_allclose(
                    predict_model(name, model, data),
                    saved[name],
                    rtol=1e-12,
                    atol=1e-12,
                )
        models.update(loaded)
        plans.append({k: v for k, v in plan.items() if k != "folds"})
        results.append(compact_results(result))
        manifests.append(spec)
        runs.append(
            {
                "path": str(run),
                "plan_sha256": sha256(run / "plan.json"),
                "results_sha256": sha256(run / "results.json"),
                "final_models_sha256": spec["model_sha256"],
            }
        )
    if set(models) != set(ALL_CANDIDATES):
        raise ValueError("incomplete research snapshot")
    training = manifests[0]["training_indices"]
    if manifests[1]["training_indices"] != training:
        raise ValueError("round training sets differ")
    supervised = set(data["owners"][event_indices(data, training)].tolist())
    used = {}
    for name, model in models.items():
        indices = (
            set(training)
            if name in ("mil-linear-softmax", "clip-gated")
            else supervised
            | {
                int(data["owners"][i])
                for i in model.get("latent_windows_not_annotations", [])
            }
        )
        used[name] = sorted(data["clips"][i]["audio_sha256"] for i in indices)
    output.mkdir(parents=True)
    joblib.dump(models, output / "models.joblib", compress=3)
    reloaded = joblib.load(output / "models.joblib")
    for name in models:
        np.testing.assert_allclose(
            predict_model(name, reloaded[name], data),
            predict_model(name, models[name], data),
            rtol=1e-12,
            atol=1e-12,
        )
    manifest = {
        "schema": 1,
        "status": "shadow-only",
        "sklearn": sklearn.__version__,
        "model_sha256": sha256(output / "models.joblib"),
        "candidates": ALL_CANDIDATES,
        "feature_names": data["names"],
        "event_feature_names": data["event_names"],
        "provenance": data["provenance"],
        "actually_used_training_hashes": used,
        "final_policy": results[1]["final_policy"],
        "sources": runs,
        "score_semantics": "Uncalibrated evidence; final_policy uses a monotone logit shift so its training-selected red threshold displays as .8. Not a probability calibration.",
        "warning": "Joblib executes pickle. Load only trusted snapshots. No production promotion; final fit is not validation. Adaptive research used this development corpus.",
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "results-summary.json",
        {"schema": 1, "status": "shadow-only", "plans": plans, "rounds": results},
    )
    print(f"published {len(models)} frozen research models to {output}")


def load_snapshot(directory, data):
    models, manifest = load_fit(directory)
    if manifest.get("schema") != 1 or manifest["sklearn"] != sklearn.__version__:
        raise ValueError("unsupported snapshot or sklearn version")
    if (
        manifest["feature_names"] != data["names"]
        or manifest["event_feature_names"] != data["event_names"]
    ):
        raise ValueError("feature schema changed")
    if manifest["provenance"]["encoder_sha256"] != data["provenance"]["encoder_sha256"]:
        raise ValueError("encoder changed")
    if set(models) != set(manifest["candidates"]):
        raise ValueError("candidate set mismatch")
    return models, manifest


def score(snapshot, output):
    if output.exists():
        raise ValueError("refusing to overwrite scores")
    data = load_inputs()
    models, manifest = load_snapshot(snapshot, data)
    indices = list(range(len(data["clips"])))
    scores = {name: predict_model(name, model, data) for name, model in models.items()}
    policy = manifest["final_policy"]
    scores["selected-policy"] = display_scores(
        scores[policy["candidate"]], policy["threshold"]
    )
    rows = {name: rows_for(data, values, indices) for name, values in scores.items()}
    for name, predictions in rows.items():
        used = set(
            manifest["actually_used_training_hashes"][
                policy["candidate"] if name == "selected-policy" else name
            ]
        )
        for row in predictions:
            row["seen_in_training"] = row["audio_sha256"] in used
    atomic_json(
        output,
        {
            "snapshot": str(snapshot),
            "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
            "provenance": data["provenance"],
            "warning": "Seen-in-training results are fit diagnostics, not validation.",
            "metrics": {n: metrics(r) for n, r in rows.items()},
            "predictions": rows,
        },
    )
    print(f"scored {len(indices)} audio groups using frozen research models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "score"))
    parser.add_argument("--round-one", type=Path)
    parser.add_argument("--round-two", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-models", action="store_true")
    args = parser.parse_args()
    if not args.trust_models:
        raise ValueError("Joblib executes pickle: verify origin and use --trust-models")
    if args.command == "publish":
        if not args.round_one or not args.round_two:
            parser.error("publish requires --round-one and --round-two")
        publish(args.round_one, args.round_two, args.output)
    else:
        if not args.snapshot:
            parser.error("score requires --snapshot")
        score(args.snapshot, args.output)


if __name__ == "__main__":
    main()
