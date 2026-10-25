#!/usr/bin/env python3
"""Explanatory ablations of the fixed synthetic MIL candidate, not a new selector."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from improve_acoustics import augmented_weights, load_matrix, transform
from improve_augmentation import synthetic_objective
from improve_events import aggregate, rows_for, training_key
from retrain_models import event_indices, load_inputs
from scipy.optimize import minimize
from scipy.special import expit
from synthetic_events import eligible
from train_events import metrics, training_weights
from train_models import atomic_json

ABLATIONS = (
    "without-burst",
    "without-rhythm",
    "without-bands",
    "without-roughness",
    "old-features-only",
)


def fit_ablation(data, features, synthetic, source, output, training):
    key = training_key(training)
    path = source / "fits" / key
    meta = json.loads((path / "manifest.json").read_text())
    if (
        meta["training_indices"] != training
        or sha256(path / "models.json") != meta["model_sha256"]
    ):
        raise ValueError("fixed candidate changed")
    base = json.loads((path / "models.json").read_text())["synthetic-mil-20pct"]
    fm, fa = features
    x = transform(
        base,
        fa["physical"][fa["legacy_indices"] >= 0],
        data["event_audio"],
        fm["physical_names"],
    )
    sm, sa = synthetic
    allowed = eligible(sm["clips"], training)
    owners = np.asarray([w["clip_index"] for w in sm["windows"]])
    use = np.flatnonzero(allowed[owners] & (sa["targets"] >= 0))
    sx = transform(
        base, sa["physical"][use], sa["embeddings"][use], sm["physical_names"]
    )
    sy = sa["targets"][use]
    known = event_indices(data, training)
    weights = training_weights(data["owners"][known], data["targets"][known])
    _, sw = augmented_weights(
        weights, sy, owners[use], [c["parents"] for c in sm["clips"]]
    )
    indices = np.flatnonzero(np.isin(data["owners"], training))
    owner_map = {owner: i for i, owner in enumerate(training)}
    mapped = np.asarray([owner_map[int(i)] for i in data["owners"][indices]])
    local_known = np.flatnonzero(data["targets"][indices] >= 0)
    names = (
        base["old_names"] + base["extra_names"] + [f"audio_pc_{i}" for i in range(8)]
    )
    models, predictions = {}, {}
    for name in ABLATIONS:
        columns = [
            i
            for i, feature in enumerate(names)
            if (
                feature not in base["extra_names"]
                if name == "old-features-only"
                else not feature.startswith(name.removeprefix("without-") + "_")
            )
        ]
        real_args = (
            x[indices][:, columns],
            mapped,
            data["y"][training],
            local_known,
            data["targets"][indices][local_known],
            weights,
        )
        initial = np.r_[np.asarray(base["coefficient"])[columns], base["intercept"]]
        optimized = minimize(
            synthetic_objective,
            initial,
            args=(real_args, sx[:, columns], sy, sw, 0.2),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
        )
        if not optimized.success:
            raise ValueError(optimized.message)
        models[name] = base | {
            "columns": columns,
            "coefficient": optimized.x[:-1].tolist(),
            "intercept": float(optimized.x[-1]),
        }
        predictions[name] = aggregate(
            data,
            expit(x[:, columns] @ optimized.x[:-1] + optimized.x[-1]),
            "linear-softmax",
        )
    target = output / "fits" / key
    target.mkdir(parents=True)
    atomic_json(target / "models.json", models)
    np.savez_compressed(target / "scores.npz", **predictions)
    atomic_json(
        target / "manifest.json",
        meta
        | {
            "model_sha256": sha256(target / "models.json"),
            "scores_sha256": sha256(target / "scores.npz"),
        },
    )
    print(f"ablated {key}", flush=True)
    return key, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_inputs()
    features, synthetic = load_matrix(args.features), load_matrix(args.synthetic)
    source_plan = json.loads((args.source / "plan.json").read_text())
    if (
        sha256(args.features / "manifest.json")
        != source_plan["feature_manifest_sha256"]
        or sha256(args.synthetic / "manifest.json")
        != source_plan["synthetic_manifest_sha256"]
    ):
        raise ValueError("ablation input changed")
    folds = source_plan["folds"]
    known = sorted(i for f in folds for i in f["test"])
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "parent_plan_sha256": sha256(args.source / "plan.json"),
        "ablations": ABLATIONS,
        "purpose": "Explanatory feature removal after candidate discovery. No new threshold or candidate selection from this experiment.",
        "folds": [{k: v for k, v in f.items() if k != "inner"} for f in folds],
    }
    atomic_json(args.output_dir / "plan.json", plan)
    sets = [f["train"] for f in folds] + [known]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                fit_ablation,
                data,
                features,
                synthetic,
                args.source,
                args.output_dir,
                training,
            )
            for training in sets
        ]
        scores = dict(f.result() for f in concurrent.futures.as_completed(futures))
    fixed = {n: np.zeros(len(data["clips"])) for n in ABLATIONS}
    for fold in folds:
        for name in ABLATIONS:
            fixed[name][fold["test"]] = scores[training_key(fold["train"])][name][
                fold["test"]
            ]
    atomic_json(
        args.output_dir / "results.json",
        {
            "metrics": {n: metrics(rows_for(data, s, known)) for n, s in fixed.items()},
            "predictions": {n: rows_for(data, s, known) for n, s in fixed.items()},
            "plan_sha256": sha256(args.output_dir / "plan.json"),
        },
    )
    print(
        json.dumps(
            {n: metrics(rows_for(data, s, known)) for n, s in fixed.items()}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
