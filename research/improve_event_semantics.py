#!/usr/bin/env python3
"""Second, explicitly adaptive research round: sound taxonomy and clip context.

Reuses round one's immutable fits/splits. This round was designed after seeing
round-one outer results; nested validation cannot erase that development reuse.
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
    CANDIDATES,
    aggregate,
    display_scores,
    logistic,
    predict_fit,
    rows_for,
    select_policy,
    training_key,
)
from retrain_models import event_indices, load_inputs, tabular_pipeline, weak_choices
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from train_audio_context import prepare
from train_events import metrics, training_weights
from train_models import atomic_json

NEW_CANDIDATES = (
    "semantic-taxonomy",
    "semantic-mil",
    "semantic-boost",
    "clip-gated",
    "event-boost",
)
ALL_CANDIDATES = CANDIDATES + NEW_CANDIDATES
# Predeclared from the audio-only task and taxonomy, not selected by correlations
# or by listening to errors. Context includes pre-event, event, and post-event.
SOUND_CLASSES = (
    "Speech",
    "Shout",
    "Screaming",
    "Whispering",
    "Laughter",
    "Crying, sobbing",
    "Wail, moan",
    "Singing",
    "Groan",
    "Grunt",
    "Breathing",
    "Gasp",
    "Pant",
    "Music",
    "Musical instrument",
    "Animal",
    "Sound effect",
    "Silence",
    "Noise",
    "Static",
    "Distortion",
    "White noise",
)


def semantic_features(audio, head):
    windows = audio.reshape(-1, 3, 1024)
    probabilities = expit(windows @ head["kernel"] + head["bias"])
    # Missing context (e.g. before an opening) is not a real sound prediction.
    probabilities[np.all(windows == 0, axis=2)] = 0
    before, event, after = (probabilities[:, i] for i in range(3))
    return np.c_[event, event - before, event - after]


def load_head(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "head.npz") != manifest["head_sha256"]:
        raise ValueError("head changed")
    columns = [manifest["classes"].index(name) for name in SOUND_CLASSES]
    with np.load(directory / "head.npz", allow_pickle=False) as source:
        return {"kernel": source["kernel"][:, columns], "bias": source["bias"][columns]}


def predict_semantic(model, data, return_windows=False):
    if model["kind"] == "clip-gated":
        scores, windows = predict_fit(model["base"], data, return_windows=True)
        clip = model["clip"].predict_proba(data["physical"])[:, 1]
        scores *= clip**0.25
        windows *= clip[data["owners"]] ** 0.25
    else:
        x = prepare(
            model["base"]["preprocessor"], data["event_physical"], data["event_audio"]
        )
        if "head" in model:
            semantic = semantic_features(data["event_audio"], model["head"])
            x = np.c_[x, model["semantic_scale"].transform(semantic)]
        windows = model["classifier"].predict_proba(x)[:, 1]
        scores = aggregate(data, windows)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid scores")
    return (scores, windows) if return_windows else scores


def boost(x, y, weights):
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=2,
        max_leaf_nodes=4,
        min_samples_leaf=20,
        l2_regularization=10,
        early_stopping=False,
        random_state=20261025,
    ).fit(x, y, sample_weight=weights)


def fit_semantics(data, training, base, head):
    known = event_indices(data, training)
    targets = data["targets"][known]
    weights = training_weights(data["owners"][known], targets)
    x = prepare(base["preprocessor"], data["event_physical"], data["event_audio"])
    semantic = data["semantic"]
    scale = StandardScaler().fit(semantic[known], sample_weight=weights)
    expanded = np.c_[x, scale.transform(semantic)]
    common = {"kind": "semantic", "base": base, "head": head, "semantic_scale": scale}
    classifier = logistic(expanded[known], targets, weights, 0.01)
    models = {"semantic-taxonomy": common | {"classifier": classifier}}
    current = classifier.predict_proba(expanded)[:, 1]
    for _ in range(3):
        latent = weak_choices(data, training, current)
        indices = np.r_[known, np.asarray(latent, dtype=int)]
        y = np.r_[targets, np.ones(len(latent), dtype=np.int8)]
        classifier = logistic(
            expanded[indices], y, training_weights(data["owners"][indices], y), 0.01
        )
        current = classifier.predict_proba(expanded)[:, 1]
    models["semantic-mil"] = common | {
        "classifier": classifier,
        "latent_windows_not_annotations": latent,
    }
    models["semantic-boost"] = common | {
        "classifier": boost(expanded[known], targets, weights)
    }
    models["event-boost"] = {
        "kind": "event-boost",
        "base": base,
        "classifier": boost(x[known], targets, weights),
    }
    clip = tabular_pipeline("logistic-rich").fit(
        data["physical"][training], data["y"][training]
    )
    models["clip-gated"] = {"kind": "clip-gated", "base": base, "clip": clip}
    return models


def run_training(data, training, source, output, head):
    key = training_key(training)
    origin, target = source / "fits" / key, output / "fits" / key
    spec = json.loads((origin / "manifest.json").read_text())
    if (
        spec["training_indices"] != training
        or sha256(origin / "models.joblib") != spec["model_sha256"]
        or sha256(origin / "scores.npz") != spec["scores_sha256"]
    ):
        raise ValueError("round-one fit changed")
    # Source is explicitly trusted by the CLI; checksum is not a trust decision.
    base = joblib.load(origin / "models.joblib")["event-hybrid"]
    models = fit_semantics(data, training, base, head)
    predictions = {
        name: predict_semantic(model, data) for name, model in models.items()
    }
    target.mkdir(parents=True)
    joblib.dump(models, target / "models.joblib", compress=3)
    reloaded = joblib.load(target / "models.joblib")
    for name in NEW_CANDIDATES:
        np.testing.assert_allclose(
            predict_semantic(reloaded[name], data),
            predictions[name],
            rtol=1e-12,
            atol=1e-12,
        )
    np.savez_compressed(target / "scores.npz", **predictions)
    atomic_json(
        target / "manifest.json",
        spec
        | {
            "source_model_sha256": spec["model_sha256"],
            "model_sha256": sha256(target / "models.joblib"),
            "scores_sha256": sha256(target / "scores.npz"),
        },
    )
    with np.load(origin / "scores.npz", allow_pickle=False) as previous:
        predictions.update({name: previous[name] for name in CANDIDATES})
    print(f"finished semantic training set {key}", flush=True)
    return key, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--trust-models", action="store_true")
    args = parser.parse_args()
    if not args.trust_models:
        raise ValueError("Joblib executes pickle: verify origin and use --trust-models")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_plan = json.loads((args.source / "plan.json").read_text())
    data = load_inputs()
    if source_plan["provenance"] != data["provenance"]:
        raise ValueError("corpus changed between research rounds")
    folds = source_plan["folds"]
    known = sorted(i for f in folds for i in f["test"])
    training_sets = {
        training_key(f["train"]): f["train"]
        for outer in folds
        for f in [outer, *outer["inner"]]
    }
    training_sets[training_key(known)] = known
    head = load_head(args.head)
    data["semantic"] = semantic_features(data["event_audio"], head)
    plan = source_plan | {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "source_run": str(args.source),
        "source_plan_sha256": sha256(args.source / "plan.json"),
        "source_results_sha256": sha256(args.source / "results.json"),
        "head_manifest_sha256": sha256(args.head / "manifest.json"),
        "candidates": ALL_CANDIDATES,
        "new_candidates": NEW_CANDIDATES,
        "sound_classes": SOUND_CLASSES,
        "adaptive_round": "Round two was designed after observing round-one outer results. Report all candidates; this is still development research, not a fresh blind test.",
    }
    atomic_json(args.output_dir / "plan.json", plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_training, data, train, args.source, args.output_dir, head)
            for train in training_sets.values()
        ]
        scores = dict(
            future.result() for future in concurrent.futures.as_completed(futures)
        )
    fixed = {name: np.zeros(len(data["clips"])) for name in ALL_CANDIDATES}
    selected = np.zeros(len(data["clips"]))
    decisions = []
    for outer in folds:
        inner_scores = {name: np.zeros(len(data["clips"])) for name in ALL_CANDIDATES}
        for inner in outer["inner"]:
            for name in ALL_CANDIDATES:
                inner_scores[name][inner["test"]] = scores[
                    training_key(inner["train"])
                ][name][inner["test"]]
        policy = select_policy(
            {n: s[outer["train"]] for n, s in inner_scores.items()},
            data["y"][outer["train"]],
            ALL_CANDIDATES,
        )
        current = scores[training_key(outer["train"])]
        for name in ALL_CANDIDATES:
            fixed[name][outer["test"]] = current[name][outer["test"]]
        selected[outer["test"]] = display_scores(
            current[policy["candidate"]][outer["test"]], policy["threshold"]
        )
        decisions.append({"thread": outer["thread"], "selected": policy})
    outputs = fixed | {"nested-selection-round2": selected}
    result = {
        "plan_sha256": sha256(args.output_dir / "plan.json"),
        "metrics": {
            name: metrics(rows_for(data, s, known)) for name, s in outputs.items()
        },
        "predictions": {name: rows_for(data, s, known) for name, s in outputs.items()},
        "folds": [
            {
                "thread": f["thread"],
                "metrics": {
                    name: metrics(rows_for(data, s, f["test"]))
                    for name, s in outputs.items()
                },
            }
            for f in folds
        ],
        "selection_decisions": decisions,
        "final_policy": select_policy(
            {n: s[known] for n, s in fixed.items()}, data["y"][known], ALL_CANDIDATES
        ),
        "final_fit_key": training_key(known),
    }
    atomic_json(args.output_dir / "results.json", result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
