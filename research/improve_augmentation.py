#!/usr/bin/env python3
"""Declared adaptive follow-up: lower synthetic weight and weak-label training.

This round was motivated by round one's recall/false-alert tradeoff. It remains
development research; inner selection cannot undo adaptive feature-design bias.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from improve_acoustics import CANDIDATES, augmented_weights, load_matrix, transform
from improve_events import (
    aggregate,
    bag_objective,
    display_scores,
    logistic,
    rows_for,
    select_policy,
    training_key,
)
from retrain_models import event_indices, load_inputs
from scipy.optimize import minimize
from scipy.special import expit
from synthetic_events import eligible
from train_events import metrics, training_weights
from train_models import atomic_json

VARIANTS = {
    "synthetic-2pct": 0.02,
    "synthetic-5pct": 0.05,
    "synthetic-10pct": 0.10,
    "synthetic-mil-5pct": 0.05,
    "synthetic-mil-20pct": 0.20,
}


def synthetic_objective(theta, real_args, synthetic_x, synthetic_y, weights, strength):
    loss, gradient = bag_objective(theta, *real_args)
    z = synthetic_x @ theta[:-1] + theta[-1]
    weight = strength * weights / weights.sum()
    loss += np.sum(weight * (np.logaddexp(0, z) - synthetic_y * z))
    residual = weight * (expit(z) - synthetic_y)
    gradient += np.r_[synthetic_x.T @ residual, residual.sum()]
    return loss, gradient


def run_fit(data, features, synthetic, source, target, training):
    key = training_key(training)
    previous = source / "fits" / key
    provenance = json.loads((previous / "manifest.json").read_text())
    if (
        sha256(previous / "models.json") != provenance["model_sha256"]
        or provenance["training_indices"] != training
    ):
        raise ValueError("source fit changed")
    old = json.loads((previous / "models.json").read_text())
    shared = old["all-acoustic"]
    manifest, arrays = features
    real_x = transform(
        shared,
        arrays["physical"][arrays["legacy_indices"] >= 0],
        data["event_audio"],
        manifest["physical_names"],
    )
    sm, sa = synthetic
    allowed = eligible(sm["clips"], training)
    owners = np.asarray([w["clip_index"] for w in sm["windows"]])
    use = np.flatnonzero(allowed[owners] & (sa["targets"] >= 0))
    synthetic_x = transform(
        shared, sa["physical"][use], sa["embeddings"][use], sm["physical_names"]
    )
    synthetic_y = sa["targets"][use]
    known = event_indices(data, training)
    y = data["targets"][known]
    weights = training_weights(data["owners"][known], y)
    paired = augmented_weights(
        weights, synthetic_y, owners[use], [c["parents"] for c in sm["clips"]]
    )
    if paired is None:
        raise ValueError("follow-up fold has no synthetic class support")
    models, predictions = {}, {}
    for name, strength in VARIANTS.items():
        if "mil" in name:
            base = old["all-acoustic-mil"]
            indices = np.flatnonzero(np.isin(data["owners"], training))
            owner_map = {owner: i for i, owner in enumerate(training)}
            mapped = np.asarray([owner_map[int(i)] for i in data["owners"][indices]])
            local_known = np.flatnonzero(data["targets"][indices] >= 0)
            real_args = (
                real_x[indices],
                mapped,
                data["y"][training],
                local_known,
                data["targets"][indices][local_known],
                weights,
            )
            optimized = minimize(
                synthetic_objective,
                np.r_[base["coefficient"], base["intercept"]],
                args=(real_args, synthetic_x, synthetic_y, paired[1], strength),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
            )
            if not optimized.success:
                raise ValueError(optimized.message)
            coefficient, intercept = optimized.x[:-1], float(optimized.x[-1])
        else:
            base = shared
            classifier = logistic(
                np.r_[real_x[known], synthetic_x],
                np.r_[y, synthetic_y],
                np.r_[weights * (1 - strength), paired[1] * strength / 0.2],
            )
            coefficient, intercept = (
                classifier.coef_[0],
                float(classifier.intercept_[0]),
            )
        models[name] = base | {
            "coefficient": coefficient.tolist(),
            "intercept": intercept,
            "synthetic_strength": strength,
            "synthetic_clips": np.flatnonzero(allowed).tolist(),
        }
        predictions[name] = aggregate(
            data,
            expit(real_x @ coefficient + intercept),
            models[name].get("pooling", "max"),
        )
    output = target / "fits" / key
    output.mkdir(parents=True)
    atomic_json(output / "models.json", models)
    np.savez_compressed(output / "scores.npz", **predictions)
    atomic_json(
        output / "manifest.json",
        {
            "training_indices": training,
            "eligible_training_hashes": provenance["eligible_training_hashes"],
            "model_sha256": sha256(output / "models.json"),
            "scores_sha256": sha256(output / "scores.npz"),
        },
    )
    print(f"follow-up {key}", flush=True)
    with np.load(previous / "scores.npz", allow_pickle=False) as saved:
        predictions.update({n: saved[n] for n in CANDIDATES})
    return key, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_inputs()
    features, synthetic = load_matrix(args.features), load_matrix(args.synthetic)
    plan = json.loads((args.source / "plan.json").read_text())
    if (
        sha256(args.features / "manifest.json") != plan["feature_manifest_sha256"]
        or sha256(args.synthetic / "manifest.json") != plan["synthetic_manifest_sha256"]
    ):
        raise ValueError("input snapshot changed")
    names = tuple(CANDIDATES) + tuple(VARIANTS)
    plan.update(
        created_at=datetime.now(UTC).isoformat(),
        source_sha256=sha256(Path(__file__)),
        parent_plan_sha256=sha256(args.source / "plan.json"),
        candidates=names,
        adaptive_variants=VARIANTS,
        motivation="Adaptive second round: first-round synthetic training caught all positives but increased false alerts. Lower weight and MIL objective are declared before this round's fitting.",
    )
    atomic_json(args.output_dir / "plan.json", plan)
    folds = plan["folds"]
    known = sorted(i for f in folds for i in f["test"])
    sets = {
        training_key(f["train"]): f["train"]
        for outer in folds
        for f in [outer, *outer["inner"]]
    }
    sets[training_key(known)] = known
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_fit, data, features, synthetic, args.source, args.output_dir, train
            )
            for train in sets.values()
        ]
        scores = dict(f.result() for f in concurrent.futures.as_completed(futures))
    fixed = {n: np.zeros(len(data["clips"])) for n in names}
    nested = np.zeros(len(data["clips"]))
    decisions = []
    for outer in folds:
        inner = {n: np.zeros(len(data["clips"])) for n in names}
        for fold in outer["inner"]:
            for name in names:
                inner[name][fold["test"]] = scores[training_key(fold["train"])][name][
                    fold["test"]
                ]
        policy = select_policy(
            {n: s[outer["train"]] for n, s in inner.items()},
            data["y"][outer["train"]],
            names,
        )
        current = scores[training_key(outer["train"])]
        for name in names:
            fixed[name][outer["test"]] = current[name][outer["test"]]
        nested[outer["test"]] = display_scores(
            current[policy["candidate"]][outer["test"]], policy["threshold"]
        )
        decisions.append({"thread": outer["thread"], "selected": policy})
    outputs = fixed | {"nested-selection": nested}
    results = {
        "plan_sha256": sha256(args.output_dir / "plan.json"),
        "metrics": {n: metrics(rows_for(data, s, known)) for n, s in outputs.items()},
        "predictions": {n: rows_for(data, s, known) for n, s in outputs.items()},
        "folds": [
            {
                "thread": f["thread"],
                "metrics": {
                    n: metrics(rows_for(data, s, f["test"])) for n, s in outputs.items()
                },
            }
            for f in folds
        ],
        "selection_decisions": decisions,
        "final_policy": select_policy(
            {n: s[known] for n, s in fixed.items()}, data["y"][known], names
        ),
        "final_fit_key": training_key(known),
    }
    atomic_json(args.output_dir / "results.json", results)
    print(
        json.dumps(
            {n: results["metrics"][n] for n in [*VARIANTS, "nested-selection"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
