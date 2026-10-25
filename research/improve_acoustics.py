#!/usr/bin/env python3
"""Ablate new waveform features, proposals and synthetic training on real folds.

Reuses the previous 67 split definitions and training-only baseline transforms.
No final-fit scores are used as validation; all preprocessing is fold-specific.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from extract_audio_context import sha256
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from synthetic_events import eligible
from train_audio_context import prepare
from train_events import metrics, training_weights
from train_models import atomic_json

CANDIDATES = (
    "event-hybrid",
    "add-burst",
    "duration-burst",
    "add-rhythm",
    "add-bands",
    "add-roughness",
    "all-acoustic",
    "all-acoustic-mil",
    "physical-old",
    "physical-new",
    "physical-expanded",
    "synthetic-acoustic",
)
DROP_DURATION = ("event_0_6s_db", "event_1_0s_db", "persistence_1s", "loudness_area_db")


def load_matrix(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "examples.npz") != manifest["matrix_sha256"]:
        raise ValueError("acoustic matrix checksum mismatch")
    with np.load(directory / "examples.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return manifest, arrays


def transform(model, physical, audio, names):
    old = [names.index(n) for n in model["old_names"]]
    new = [names.index(n) for n in model["extra_names"]]
    original = prepare(model["preprocessor"], physical[:, old], audio)
    spec = model["extra_scale"]
    extras = (
        np.where(np.isnan(physical[:, new]), spec["median"], physical[:, new])
        - spec["mean"]
    ) / spec["scale"]
    return np.c_[original[:, : len(old)], extras, original[:, len(old) :]]


def predict_model(model, physical, audio, names, owners, size):
    x = transform(model, physical, audio, names)[:, model["columns"]]
    windows = expit(x @ model["coefficient"] + model["intercept"])
    if model.get("pooling") == "linear-softmax":
        from improve_events import pooled

        return pooled(windows, owners, size)[0], windows
    scores = np.zeros(size)
    np.maximum.at(scores, owners, windows)
    return scores, windows


def augmented_weights(real, targets, source, parents):
    """Fixed 20% synthetic weight, balanced by class and original donor.

    Adding copies of a derivative cannot increase its donor's total influence.
    Returns None if a fold has no positive or negative synthetic observations.
    """
    if len(np.unique(targets)) < 2:
        return None
    weights = np.zeros(len(targets))
    for label in (0, 1):
        mask = targets == label
        donors = np.asarray([parents[int(i)][0] for i in source])
        identities = np.unique(donors[mask])
        for donor in identities:
            selected = mask & (donors == donor)
            weights[selected] = real.sum() * 0.1 / len(identities) / sum(selected)
    return real * 0.8, weights


def run_fit(data, manifest, arrays, synthetic, baseline, training, output):
    key = training_key(training)
    target = output / "fits" / key
    target.mkdir(parents=True)
    source = baseline / "fits" / key
    metadata = json.loads((source / "manifest.json").read_text())
    if (
        metadata["training_indices"] != training
        or sha256(source / "models.joblib") != metadata["model_sha256"]
    ):
        raise ValueError("baseline training provenance changed")
    base = joblib.load(source / "models.joblib")["event-hybrid"]["preprocessor"]
    legacy = arrays["legacy_indices"] >= 0
    old = [manifest["physical_names"].index(n) for n in data["event_names"]]
    extras = [n for n in manifest["physical_names"] if n not in data["event_names"]]
    new = [manifest["physical_names"].index(n) for n in extras]
    known = event_indices(data, training)
    y = data["targets"][known]
    weights = training_weights(data["owners"][known], y)
    median = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        arrays["physical"][legacy][known][:, new]
    )
    scaler = StandardScaler().fit(
        median.transform(arrays["physical"][legacy][known][:, new]),
        sample_weight=weights,
    )
    shared = {
        "preprocessor": base,
        "old_names": data["event_names"],
        "extra_names": extras,
        "extra_scale": {
            "median": median.statistics_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }
    x = transform(
        shared,
        arrays["physical"][legacy],
        data["event_audio"],
        manifest["physical_names"],
    )
    names = data["event_names"] + extras + [f"audio_pc_{i}" for i in range(8)]
    old_columns = [
        i
        for i, n in enumerate(names)
        if n in data["event_names"] or n.startswith("audio_pc_")
    ]
    all_columns = list(range(len(names)))
    models, predictions = {}, {}
    with np.load(source / "scores.npz", allow_pickle=False) as stored:
        predictions["event-hybrid"] = stored["event-hybrid"]
    baseline_windows = expit(
        x[:, old_columns] @ base["coefficient"] + base["intercept"]
    )
    np.testing.assert_allclose(
        aggregate(data, baseline_windows),
        predictions["event-hybrid"],
        atol=1e-12,
        rtol=1e-12,
    )
    models["event-hybrid"] = shared | {
        "columns": old_columns,
        "coefficient": base["coefficient"],
        "intercept": base["intercept"],
    }
    for name in CANDIDATES[1:]:
        if name.startswith("add-"):
            prefix = name.removeprefix("add-") + "_"
            columns = [
                i
                for i, n in enumerate(names)
                if i in old_columns or n.startswith(prefix)
            ]
        elif name == "duration-burst":
            columns = [
                i
                for i, n in enumerate(names)
                if (i in old_columns and n not in DROP_DURATION)
                or n.startswith("burst_")
            ]
        elif name == "physical-old":
            columns = list(range(len(old)))
        elif name.startswith("physical-"):
            columns = list(range(len(old) + len(new)))
        else:
            columns = all_columns
        model = shared | {"columns": columns}
        current = x[:, columns]
        fit_x, fit_y, fit_weights = current[known], y, weights
        if name == "physical-expanded":
            spec = base["physical"]
            filled = np.where(
                np.isnan(arrays["physical"][:, old]),
                spec["median"],
                arrays["physical"][:, old],
            )
            expanded = np.c_[
                (filled - spec["mean"]) / spec["scale"],
                scaler.transform(median.transform(arrays["physical"][:, new])),
            ]
            owners = np.asarray([w["clip_index"] for w in manifest["windows"]])
            eligible_windows = np.flatnonzero(
                np.isin(owners, training) & (arrays["targets"] >= 0)
            )
            fit_x, fit_y = (
                expanded[eligible_windows],
                arrays["targets"][eligible_windows],
            )
            fit_weights = training_weights(owners[eligible_windows], fit_y)
            current = expanded
        elif name == "synthetic-acoustic":
            sm, sa = synthetic
            allowed = eligible(sm["clips"], training)
            owners = np.asarray([w["clip_index"] for w in sm["windows"]])
            use = np.flatnonzero(allowed[owners] & (sa["targets"] >= 0))
            model["synthetic_clips"] = np.flatnonzero(allowed).tolist()
            paired = augmented_weights(
                weights,
                sa["targets"][use],
                owners[use],
                [c["parents"] for c in sm["clips"]],
            )
            if paired is not None:
                sx = transform(
                    shared,
                    sa["physical"][use],
                    sa["embeddings"][use],
                    sm["physical_names"],
                )
                fit_x, fit_y, fit_weights = (
                    np.r_[fit_x, sx[:, columns]],
                    np.r_[y, sa["targets"][use]],
                    np.r_[paired[0], paired[1]],
                )
        classifier = logistic(fit_x, fit_y, fit_weights)
        coefficient, intercept = classifier.coef_[0], float(classifier.intercept_[0])
        if name == "all-acoustic-mil":
            indices = np.flatnonzero(np.isin(data["owners"], training))
            owner_map = {owner: i for i, owner in enumerate(training)}
            owners = np.asarray([owner_map[int(i)] for i in data["owners"][indices]])
            local_known = np.flatnonzero(data["targets"][indices] >= 0)
            optimized = minimize(
                bag_objective,
                np.r_[coefficient, intercept],
                args=(
                    current[indices],
                    owners,
                    data["y"][training],
                    local_known,
                    data["targets"][indices][local_known],
                    weights,
                ),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
            )
            if not optimized.success:
                raise ValueError(f"MIL failed: {optimized.message}")
            coefficient, intercept = optimized.x[:-1], float(optimized.x[-1])
            model["pooling"] = "linear-softmax"
        model.update(coefficient=coefficient.tolist(), intercept=intercept)
        windows = expit(current @ coefficient + intercept)
        if name == "physical-expanded":
            scores = np.zeros(len(data["clips"]))
            np.maximum.at(
                scores,
                np.asarray([w["clip_index"] for w in manifest["windows"]]),
                windows,
            )
            model["expanded_proposals"] = True
        else:
            scores = aggregate(data, windows, model.get("pooling", "max"))
            np.testing.assert_allclose(
                predict_model(
                    model,
                    arrays["physical"][legacy],
                    data["event_audio"],
                    manifest["physical_names"],
                    data["owners"],
                    len(data["clips"]),
                )[0],
                scores,
                atol=1e-12,
                rtol=1e-12,
            )
        models[name], predictions[name] = model, scores
    atomic_json(target / "models.json", models)
    np.savez_compressed(target / "scores.npz", **predictions)
    atomic_json(
        target / "manifest.json",
        {
            "training_indices": training,
            "eligible_training_hashes": [
                data["clips"][i]["audio_sha256"] for i in training
            ],
            "model_sha256": sha256(target / "models.json"),
            "scores_sha256": sha256(target / "scores.npz"),
        },
    )
    print(f"finished {key}: {len(training)} real clips", flush=True)
    return key, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("research/artifacts/improve-events-20261025"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--trust-models", action="store_true")
    args = parser.parse_args()
    if not args.trust_models:
        raise ValueError("Baseline pickle requires verified origin and --trust-models")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_inputs()
    manifest, arrays = load_matrix(args.features)
    synthetic = load_matrix(args.synthetic)
    if manifest["base_manifest_sha256"] != sha256(
        Path("research/artifacts/event-data-v1/manifest.json")
    ):
        raise ValueError("base event dataset changed")
    if [c["audio_sha256"] for c in manifest["clips"]] != [
        c["audio_sha256"] for c in data["clips"]
    ]:
        raise ValueError("clip order changed")
    legacy = arrays["legacy_indices"] >= 0
    np.testing.assert_array_equal(
        arrays["legacy_indices"][legacy], np.arange(len(data["windows"]))
    )
    np.testing.assert_array_equal(arrays["targets"][legacy], data["targets"])
    np.testing.assert_allclose(
        arrays["physical"][legacy][
            :, [manifest["physical_names"].index(n) for n in data["event_names"]]
        ],
        data["event_physical"],
        atol=1e-10,
        rtol=1e-10,
    )
    prior = json.loads((args.baseline / "plan.json").read_text())
    folds = prior["folds"]
    known = sorted(i for f in folds for i in f["test"])
    sets = {
        training_key(f["train"]): f["train"]
        for outer in folds
        for f in [outer, *outer["inner"]]
    }
    sets[training_key(known)] = known
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "synthetic_manifest_sha256": sha256(args.synthetic / "manifest.json"),
        "baseline_plan_sha256": sha256(args.baseline / "plan.json"),
        "candidates": CANDIDATES,
        "folds": folds,
        "drop_duration": DROP_DURATION,
        "synthetic_weight": 0.2,
        "regularization_C": 0.05,
        "selection": prior["selection"],
        "limitations": prior["limitations"],
        "status": "research, no production promotion",
    }
    atomic_json(args.output_dir / "plan.json", plan)
    print(
        f"predeclared {len(CANDIDATES)} candidates on {len(sets)} training partitions",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                run_fit,
                data,
                manifest,
                arrays,
                synthetic,
                args.baseline,
                train,
                args.output_dir,
            )
            for train in sets.values()
        ]
        scores = dict(f.result() for f in concurrent.futures.as_completed(futures))
    fixed = {n: np.zeros(len(data["clips"])) for n in CANDIDATES}
    nested = np.zeros(len(data["clips"]))
    decisions = []
    for outer in folds:
        inner = {n: np.zeros(len(data["clips"])) for n in CANDIDATES}
        for fold in outer["inner"]:
            for name in CANDIDATES:
                inner[name][fold["test"]] = scores[training_key(fold["train"])][name][
                    fold["test"]
                ]
        policy = select_policy(
            {n: s[outer["train"]] for n, s in inner.items()},
            data["y"][outer["train"]],
            CANDIDATES,
        )
        current = scores[training_key(outer["train"])]
        for name in CANDIDATES:
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
            {n: s[known] for n, s in fixed.items()}, data["y"][known], CANDIDATES
        ),
        "final_fit_key": training_key(known),
    }
    atomic_json(args.output_dir / "results.json", results)
    print(json.dumps(results["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
