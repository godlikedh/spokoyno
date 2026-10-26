#!/usr/bin/env python3
"""Match boosted-tree supervision to recording-level false-alarm decisions.

Source-balanced normal recordings devote a fixed fraction of their weight to
their highest-scoring events. Three fixed MIL rounds update positive assignments
and negative weights. A same-round all-window control isolates the mining change.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import train_robust_models as trainer
from extract_audio_context import sha256
from improve_events import training_key
from robust_tree import export, window_probabilities
from sklearn.ensemble import HistGradientBoostingClassifier
from train_models import atomic_json

PRIMARY = "hard-negative-mixed"
COMMON = {
    "audio": False,
    "augment": True,
    "power": 8,
    "consistency": 0,
    "leaves": 7,
    "depth": 3,
    "l2": 10.0,
    "min_leaf": 30,
    "rounds": 3,
}
CONFIGS = {
    PRIMARY: COMMON | {"hard_mass": 0.75, "hard_count": 3},
    "hard-negative-max": COMMON | {"hard_mass": 1.0, "hard_count": 1},
    "all-window-control": COMMON | {"hard_mass": 0.0, "hard_count": 1},
}


def instance_supervision(
    probabilities, targets, starts, labels, bag_weights, hard_mass, hard_count
):
    y = np.full(len(probabilities), -1, dtype=np.int8)
    weights = np.zeros(len(probabilities))
    for owner, positive in enumerate(labels):
        lo, hi = starts[owner : owner + 2]
        if not positive:
            y[lo:hi] = 0
            weights[lo:hi] = (1 - hard_mass) * bag_weights[owner] / (hi - lo)
            hardest = np.argsort(-probabilities[lo:hi], kind="stable")[:hard_count] + lo
            weights[hardest] += hard_mass * bag_weights[owner] / len(hardest)
            continue
        candidates = np.flatnonzero(targets[lo:hi] != 0) + lo
        if not len(candidates):
            raise ValueError("positive recording has only known negative events")
        chosen = candidates[np.argmax(probabilities[candidates])]
        positives = np.unique(np.r_[chosen, np.flatnonzero(targets[lo:hi] == 1) + lo])
        y[positives] = 1
        weights[positives] = bag_weights[owner] / len(positives)
        negatives = np.flatnonzero(targets[lo:hi] == 0) + lo
        if len(negatives):
            y[negatives] = 0
            weights[negatives] = 0.1 * bag_weights[owner] / len(negatives)
    return y, weights


def fit(data, x, training, config, seed_run):
    directory = seed_run / "fits" / training_key(training)
    provenance = json.loads((directory / "manifest.json").read_text())
    seed_file = directory / "models.json"
    if (
        provenance["training_indices"] != training
        or sha256(seed_file) != provenance["files"]["models.json"]
    ):
        raise ValueError("seed model belongs to a different training partition")
    seed = json.loads(seed_file.read_text())["codec-stable-hybrid"]
    eligible = trainer.eligible_records(data["records"], training, True)
    selected = np.flatnonzero(eligible)
    records = [data["records"][i] for i in selected]
    ids = np.flatnonzero(eligible[data["owners"]])
    mapping = np.full(len(data["records"]), -1, dtype=int)
    mapping[selected] = np.arange(len(selected))
    owners = mapping[data["owners"][ids]]
    counts = np.bincount(owners, minlength=len(records))
    starts = np.r_[0, np.cumsum(counts)]
    labels = np.asarray([r["label"] == "positive" for r in records])
    targets = data["targets"][ids]
    bag_weights = trainer.group_weights(records)
    probabilities = window_probabilities(seed, x[ids])
    values = x[ids, : len(data["names"])].copy()
    clipping = data["names"].index("near_clip_fraction")
    values[:, clipping] = 0
    diagnostics = []
    for iteration in range(config["rounds"]):
        y, weights = instance_supervision(
            probabilities,
            targets,
            starts,
            labels,
            bag_weights,
            config["hard_mass"],
            config["hard_count"],
        )
        use = (y >= 0) & (weights > 0)
        estimator = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=config["leaves"],
            max_depth=config["depth"],
            l2_regularization=config["l2"],
            min_samples_leaf=config["min_leaf"],
            learning_rate=0.08,
            max_bins=63,
            early_stopping=False,
            random_state=42,
        )
        estimator.fit(values[use], y[use], sample_weight=weights[use] * len(training))
        probabilities = estimator.predict_proba(values)[:, 1]
        bag = trainer.pool(probabilities, owners, len(records), config["power"])[0]
        diagnostics.append(
            {
                "iteration": iteration + 1,
                "supervised_events": int(use.sum()),
                "training_red_positives": int(np.sum(bag[labels] >= 0.8)),
                "training_red_negatives": int(np.sum(bag[~labels] >= 0.8)),
            }
        )
    model = (
        config
        | export(estimator)
        | {
            "training_records": selected.tolist(),
            "training_windows": len(ids),
            "disabled_features": ["near_clip_fraction"],
            "seed_model_sha256": sha256(seed_file),
            "mil_rounds": config["rounds"],
            "training_diagnostics": diagnostics,
        }
    )
    if np.max(np.abs(window_probabilities(model, values) - probabilities)) > 1e-12:
        raise ValueError("portable tree differs from fitted estimator")
    if any(
        not n["leaf"] and n["feature"] == clipping
        for tree in model["trees"]
        for n in tree
    ):
        raise ValueError("tree used the disabled clipping feature")
    scores = trainer.pool(
        window_probabilities(model, x[:, : len(data["names"])]),
        data["owners"],
        len(data["records"]),
        config["power"],
    )[0]
    return model, scores


def main():
    seed_index = sys.argv.index("--seed-run")
    seed_run = Path(sys.argv[seed_index + 1])
    del sys.argv[seed_index : seed_index + 2]
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    declaration = {
        "created_at": datetime.now(UTC).isoformat(),
        "entry_sha256": sha256(Path(__file__)),
        "trainer_sha256": sha256(Path(trainer.__file__)),
        "seed_run": str(seed_run),
        "primary": PRIMARY,
        "candidates": CONFIGS,
        "protocol": "Same bounded input, proposals, parent exclusions, source weights, no clipping feature, and 120-iteration small boosting heads as the previous tree run. All configurations use exactly three MIL rounds. Negative source weight is either uniform, 75% on top three events plus 25% uniform, or entirely on the top event. Mining uses training events only; unknown evaluation labels and evaluation codecs are excluded.",
        "limitations": "Adaptive research after the first robust-model study; not new independent validation. Candidate budget and hyperparameters fixed before these fits.",
    }
    file = output / "followup-declaration.json"
    if file.exists():
        old = json.loads(file.read_text())
        if any(
            old[k] != declaration[k]
            for k in ("entry_sha256", "trainer_sha256", "candidates")
        ):
            raise ValueError("cannot resume changed experiment")
    else:
        atomic_json(file, declaration)
    trainer.fit_one = lambda data, x, training, config: fit(
        data, x, training, config, seed_run
    )
    trainer.CONFIGS = CONFIGS
    trainer.PRIMARY = PRIMARY
    trainer.main()


if __name__ == "__main__":
    main()
