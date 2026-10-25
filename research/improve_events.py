#!/usr/bin/env python3
"""Predeclared event-model research with nested thread/family-held-out selection.

All candidates keep the frozen encoder and label-independent event proposals.
Raw scores are evidence scores, not calibrated screamer probabilities.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from extract_audio_context import sha256
from retrain_models import event_indices, folds_for, load_inputs, weak_choices
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from train_audio_context import fit, predict, prepare
from train_events import metrics, training_weights
from train_models import atomic_json

CANDIDATES = (
    "event-hybrid",
    "strong-ridge",
    "semantic-contrast",
    "hard-negatives",
    "mil-hybrid",
    "mil-linear-softmax",
    "additive-spline",
    "rbf-event",
)
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
SEED = 20261025


def logistic(x, y, weights, c=0.05):
    return LogisticRegression(
        C=c, solver="liblinear", max_iter=5000, random_state=42
    ).fit(x, y, sample_weight=weights)


def contrast(audio):
    before, event, after = np.split(audio, 3, axis=1)
    return np.concatenate([event, event - before, event - after], axis=1)


def pooled(probabilities, owners, size):
    total = np.bincount(owners, weights=probabilities, minlength=size)
    square = np.bincount(owners, weights=probabilities**2, minlength=size)
    return square / np.maximum(total, 1e-12), total


def bag_objective(theta, x, owners, labels, known, targets, window_weights):
    """Linear-softmax bag BCE + 1/4 annotated-window BCE + L2.

    Bag classes each have half of the total weight. Only training bags and
    genuine known windows enter this objective. Unknown windows stay unknown.
    """
    z = x @ theta[:-1] + theta[-1]
    p = expit(z)
    bag, total = pooled(p, owners, len(labels))
    bag = np.clip(bag, 1e-12, 1 - 1e-12)
    weights = np.where(labels == 1, 0.5 / sum(labels), 0.5 / sum(labels == 0))
    loss = -np.sum(weights * (labels * np.log(bag) + (1 - labels) * np.log1p(-bag)))
    dbag = weights * (bag - labels) / (bag * (1 - bag))
    dz = dbag[owners] * (2 * p - bag[owners]) / np.maximum(total[owners], 1e-12)
    dz *= p * (1 - p)
    # Auxiliary human labels prevent treating other positive-bag windows as
    # known negatives, while retaining the available timing supervision.
    auxiliary = 0.25 * window_weights / window_weights.sum()
    loss += np.sum(auxiliary * (np.logaddexp(0, z[known]) - targets * z[known]))
    dz[known] += auxiliary * (p[known] - targets)
    penalty = 1 / (0.05 * len(labels))
    loss += 0.5 * penalty * np.dot(theta, theta)
    gradient = np.r_[x.T @ dz, dz.sum()] + penalty * theta
    return float(loss), gradient


def aggregate(data, window_scores, pooling="max"):
    if pooling == "linear-softmax":
        return pooled(window_scores, data["owners"], len(data["clips"]))[0]
    return np.asarray(
        [
            max(
                window_scores[c["first_window"] : c["first_window"] + c["window_count"]]
            )
            for c in data["clips"]
        ]
    )


def predict_fit(model, data, return_windows=False):
    audio = data["event_audio"]
    if model.get("contrast"):
        audio = contrast(audio)
    base = model["preprocessor"]
    x = prepare(base, data["event_physical"], audio)
    if "spline" in model:
        curved = model["spline"].transform(x[:, : len(data["event_names"])])
        x = model["scale"].transform(np.c_[curved, x[:, len(data["event_names"]) :]])
    if "classifier" in model:
        classifier = model["classifier"]
        windows = expit(classifier.decision_function(x))
    elif "theta" in model:
        windows = expit(x @ model["theta"][:-1] + model["theta"][-1])
    else:
        windows = predict(base, data["event_physical"], audio)
    scores = aggregate(data, windows, model.get("pooling", "max"))
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid model predictions")
    return (scores, windows) if return_windows else scores


def fit_all(data, training):
    known = event_indices(data, training)
    targets = data["targets"][known]
    weights = training_weights(data["owners"][known], targets)
    base = fit(
        data["event_physical"][known],
        data["event_audio"][known],
        targets,
        "hybrid",
        8,
        weights,
    )
    x = prepare(base, data["event_physical"], data["event_audio"])
    models = {"event-hybrid": {"preprocessor": base}}
    models["strong-ridge"] = {
        "preprocessor": base,
        "classifier": logistic(x[known], targets, weights, 0.01),
    }
    models["semantic-contrast"] = {
        "contrast": True,
        "preprocessor": fit(
            data["event_physical"][known],
            contrast(data["event_audio"][known]),
            targets,
            "hybrid",
            8,
            weights,
        ),
    }
    current = predict(base, data["event_physical"], data["event_audio"])
    # Three fixed rounds. Class totals remain unchanged; hard negatives receive
    # up to eight times a same-class easy window's original weight.
    for _ in range(3):
        hard = weights.copy()
        mask = targets == 0
        hard[mask] *= 1 + 7 * current[known][mask] ** 2
        hard[mask] *= weights[mask].sum() / hard[mask].sum()
        classifier = logistic(x[known], targets, hard)
        current = classifier.predict_proba(x)[:, 1]
    models["hard-negatives"] = {"preprocessor": base, "classifier": classifier}
    current = predict(base, data["event_physical"], data["event_audio"])
    for _ in range(3):
        latent = weak_choices(data, training, current)
        indices = np.r_[known, np.asarray(latent, dtype=int)]
        y = np.r_[targets, np.ones(len(latent), dtype=np.int8)]
        classifier = logistic(
            x[indices], y, training_weights(data["owners"][indices], y)
        )
        current = classifier.predict_proba(x)[:, 1]
    models["mil-hybrid"] = {
        "preprocessor": base,
        "classifier": classifier,
        "latent_windows_not_annotations": latent,
    }
    # Smooth MIL uses every TRAINING bag, including untimed positives.
    indices = np.flatnonzero(np.isin(data["owners"], training))
    owner_map = {owner: i for i, owner in enumerate(training)}
    owners = np.asarray([owner_map[int(i)] for i in data["owners"][indices]])
    local_known = np.flatnonzero(data["targets"][indices] >= 0)
    optimized = minimize(
        bag_objective,
        np.r_[base["coefficient"], base["intercept"]],
        args=(
            x[indices],
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
        raise ValueError(f"MIL optimization failed: {optimized.message}")
    models["mil-linear-softmax"] = {
        "preprocessor": base,
        "theta": optimized.x,
        "pooling": "linear-softmax",
        "optimizer_iterations": optimized.nit,
    }
    dimensions = len(data["event_names"])
    spline = SplineTransformer(
        n_knots=3,
        degree=2,
        knots="uniform",
        extrapolation="constant",
        include_bias=False,
    )
    curved = spline.fit_transform(x[known, :dimensions])
    expanded = np.c_[curved, x[known, dimensions:]]
    scale = StandardScaler().fit(expanded, sample_weight=weights)
    models["additive-spline"] = {
        "preprocessor": base,
        "spline": spline,
        "scale": scale,
        "classifier": logistic(scale.transform(expanded), targets, weights),
    }
    models["rbf-event"] = {
        "preprocessor": base,
        "classifier": SVC(
            C=1, gamma=1 / x.shape[1], kernel="rbf", probability=False, cache_size=256
        ).fit(x[known], targets, sample_weight=weights),
    }
    return models


def display_scores(scores, threshold):
    """Keep the UI red boundary at .8; this monotone shift is NOT calibration."""
    return expit(
        logit(np.clip(scores, 1e-12, 1 - 1e-12)) + logit(0.8) - logit(threshold)
    )


def select_policy(scores, labels, names=CANDIDATES):
    baseline = scores["event-hybrid"] >= 0.8
    preserve = baseline & (labels == 1)
    base_fp = int(sum(baseline & (labels == 0)))
    # Without positive validation evidence, never attempt threshold selection.
    fallback = {
        "candidate": "event-hybrid",
        "threshold": 0.8,
        "false_positives": base_fp,
        "detected": int(sum(preserve)),
    }
    if not np.any(preserve):
        return fallback
    choices = []
    for order, name in enumerate(names):
        for threshold in THRESHOLDS:
            alerts = scores[name] >= threshold
            if np.any(preserve & ~alerts):
                continue
            fp, tp = int(sum(alerts & (labels == 0))), int(sum(alerts & (labels == 1)))
            # Require >=20% fewer inner false alerts, otherwise retain baseline.
            if fp < base_fp and fp <= 0.8 * base_fp:
                choices.append((fp, -tp, order, abs(threshold - 0.8), threshold, name))
    if not choices:
        return fallback
    fp, negative_tp, _, _, threshold, name = min(choices)
    return {
        "candidate": name,
        "threshold": threshold,
        "false_positives": fp,
        "detected": -negative_tp,
    }


def training_key(training):
    return hashlib.sha256(json.dumps(training).encode()).hexdigest()[:16]


def nested_splits(folds):
    result = []
    for outer in folds:
        allowed = set(outer["train"])
        inner = []
        for fold in folds:
            testing = sorted(allowed & set(fold["test"]))
            if testing:
                inner.append(
                    {
                        "thread": fold["thread"],
                        "train": sorted(allowed & set(fold["train"])),
                        "test": testing,
                    }
                )
        if sorted(i for f in inner for i in f["test"]) != sorted(allowed):
            raise ValueError(
                "inner validation does not cover outer training exactly once"
            )
        result.append(outer | {"inner": inner})
    return result


def rows_for(data, scores, indices):
    return [
        {
            k: data["clips"][i][k]
            for k in ("path", "paths", "audio_sha256", "label", "fold_thread")
        }
        | {"score": float(scores[i])}
        for i in indices
    ]


def run_training(data, training, output):
    key = training_key(training)
    target = output / "fits" / key
    target.mkdir(parents=True)
    models = fit_all(data, training)
    predictions = {name: predict_fit(model, data) for name, model in models.items()}
    joblib.dump(models, target / "models.joblib", compress=3)
    reloaded = joblib.load(target / "models.joblib")
    for name in CANDIDATES:
        np.testing.assert_allclose(
            predict_fit(reloaded[name], data), predictions[name], rtol=1e-12, atol=1e-12
        )
    np.savez_compressed(target / "scores.npz", **predictions)
    atomic_json(
        target / "manifest.json",
        {
            "training_indices": training,
            "eligible_training_hashes": [
                data["clips"][i]["audio_sha256"] for i in training
            ],
            "model_sha256": sha256(target / "models.joblib"),
            "scores_sha256": sha256(target / "scores.npz"),
        },
    )
    print(f"finished training set {key}: {len(training)} clips", flush=True)
    return key, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_inputs()
    folds = nested_splits(folds_for(data["clips"], data["links"]))
    known = [
        i
        for i, clip in enumerate(data["clips"])
        if clip["label"] in ("positive", "negative")
    ]
    training_sets = {
        training_key(f["train"]): f["train"]
        for outer in folds
        for f in [outer, *outer["inner"]]
    }
    training_sets[training_key(known)] = known
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "shadow-only",
        "source_sha256": sha256(Path(__file__)),
        "provenance": data["provenance"],
        "candidates": CANDIDATES,
        "threshold_grid": THRESHOLDS,
        "folds": folds,
        "selection": "Preserve EVERY inner-validation positive caught by event-hybrid at .8; require at least 20% fewer inner false alerts; minimize FP, maximize TP, prefer declaration order then threshold nearest .8. Otherwise baseline .8. Separate baseline-only threshold control. No outer labels used for selection.",
        "limitations": "Development corpus already informed feature design. Nested selection is not a new prospective test. Known families only; new related audio can remain. Final fits are not validation. Scores are uncalibrated.",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
    }
    atomic_json(args.output_dir / "plan.json", plan)
    print(
        f"predeclared {len(CANDIDATES)} candidates, {len(training_sets)} training sets",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_training, data, train, args.output_dir)
            for train in training_sets.values()
        ]
        scores = dict(
            future.result() for future in concurrent.futures.as_completed(futures)
        )
    fixed = {name: np.zeros(len(data["clips"])) for name in CANDIDATES}
    nested = np.zeros(len(data["clips"]))
    threshold_control = np.zeros(len(data["clips"]))
    decisions = []
    for outer in folds:
        inner_scores = {name: np.zeros(len(data["clips"])) for name in CANDIDATES}
        for inner in outer["inner"]:
            for name in CANDIDATES:
                inner_scores[name][inner["test"]] = scores[
                    training_key(inner["train"])
                ][name][inner["test"]]
        policies = [
            select_policy(
                {n: s[outer["train"]] for n, s in inner_scores.items()},
                data["y"][outer["train"]],
                names,
            )
            for names in (CANDIDATES, ("event-hybrid",))
        ]
        current = scores[training_key(outer["train"])]
        for name in CANDIDATES:
            fixed[name][outer["test"]] = current[name][outer["test"]]
        for output, policy in zip((nested, threshold_control), policies, strict=True):
            output[outer["test"]] = display_scores(
                current[policy["candidate"]][outer["test"]], policy["threshold"]
            )
        decisions.append(
            {
                "thread": outer["thread"],
                "selected": policies[0],
                "threshold_control": policies[1],
            }
        )
    policy = select_policy({n: s[known] for n, s in fixed.items()}, data["y"][known])
    outputs = fixed | {
        "nested-selection": nested,
        "nested-baseline-threshold": threshold_control,
    }
    results = {
        "plan_sha256": sha256(args.output_dir / "plan.json"),
        "metrics": {
            name: metrics(rows_for(data, s, known)) for name, s in outputs.items()
        },
        "predictions": {name: rows_for(data, s, known) for name, s in outputs.items()},
        "folds": [
            {
                "thread": fold["thread"],
                "metrics": {
                    name: metrics(rows_for(data, s, fold["test"]))
                    for name, s in outputs.items()
                },
            }
            for fold in folds
        ],
        "selection_decisions": decisions,
        "final_policy": policy,
        "final_fit_key": training_key(known),
    }
    atomic_json(args.output_dir / "results.json", results)
    print(json.dumps(results["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
