#!/usr/bin/env python3
"""Small nonlinear MIL heads, with codec augmentation and no clipping feature.

Each weak positive contributes its strongest non-negative-labeled event. Two
fixed rounds update those assignments; normal recordings supervise every event.
The seed model, all source parents and PCA/scaling are training-fold-only.
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import train_robust_models as trainer
from extract_audio_context import sha256
from improve_events import training_key
from robust_tree import export, window_probabilities
from sklearn.ensemble import HistGradientBoostingClassifier
from train_models import atomic_json

PRIMARY = "mil-boost-7"
CONFIGS = {
    PRIMARY: {
        "audio": False,
        "augment": True,
        "power": 8,
        "consistency": 0,
        "leaves": 7,
        "depth": 3,
        "l2": 10.0,
        "min_leaf": 30,
    },
    "mil-boost-15": {
        "audio": False,
        "augment": True,
        "power": 8,
        "consistency": 0,
        "leaves": 15,
        "depth": 4,
        "l2": 20.0,
        "min_leaf": 50,
    },
}


def fit(data, x, training, config, seed_run):
    file = seed_run / "fits" / training_key(training) / "manifest.json"
    # This process may follow the independently running linear experiment.
    while not file.exists():
        time.sleep(2)
    provenance = json.loads(file.read_text())
    seed_file = file.with_name("models.json")
    if (
        provenance["training_indices"] != training
        or sha256(seed_file) != provenance["files"]["models.json"]
    ):
        raise ValueError("seed model does not belong to this exact training partition")
    seed = json.loads(seed_file.read_text())["codec-stable-hybrid"]
    eligible = trainer.eligible_records(data["records"], training, True)
    selected = np.flatnonzero(eligible)
    records = [data["records"][i] for i in selected]
    ids = np.flatnonzero(eligible[data["owners"]])
    mapping = np.full(len(data["records"]), -1, dtype=int)
    mapping[selected] = np.arange(len(selected))
    owners = mapping[data["owners"][ids]]
    targets = data["targets"][ids]
    labels = np.asarray([r["label"] == "positive" for r in records])
    bag_weights = trainer.group_weights(records)
    counts = np.bincount(owners, minlength=len(records))
    starts = np.r_[0, np.cumsum(counts)]
    probabilities = window_probabilities(seed, x[ids])
    values = x[ids, : len(data["names"])].copy()
    clipping = data["names"].index("near_clip_fraction")
    values[:, clipping] = 0
    for _ in range(2):
        y = np.full(len(ids), -1, dtype=np.int8)
        weights = np.zeros(len(ids))
        for owner, positive in enumerate(labels):
            lo, hi = starts[owner : owner + 2]
            if not positive:
                y[lo:hi] = 0
                weights[lo:hi] = bag_weights[owner] / (hi - lo)
                continue
            candidates = np.flatnonzero(targets[lo:hi] != 0) + lo
            if not len(candidates):
                raise ValueError("positive recording has only known negative windows")
            chosen = candidates[np.argmax(probabilities[candidates])]
            positives = np.unique(
                np.r_[chosen, np.flatnonzero(targets[lo:hi] == 1) + lo]
            )
            y[positives] = 1
            weights[positives] = bag_weights[owner] / len(positives)
            negatives = np.flatnonzero(targets[lo:hi] == 0) + lo
            if len(negatives):
                y[negatives] = 0
                weights[negatives] = 0.1 * bag_weights[owner] / len(negatives)
        use = y >= 0
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
    model = (
        config
        | export(estimator)
        | {
            "training_records": selected.tolist(),
            "training_windows": len(ids),
            "disabled_features": ["near_clip_fraction"],
            "seed_model_sha256": sha256(seed_file),
            "mil_rounds": 2,
        }
    )
    portable = window_probabilities(model, values)
    if np.max(np.abs(portable - probabilities)) > 1e-12:
        raise ValueError("tree export differs from fitted estimator")
    # No trained split can use a constant feature, so extraction need not zero it.
    if any(
        not node["leaf"] and node["feature"] == clipping
        for tree in model["trees"]
        for node in tree
    ):
        raise ValueError("tree used disabled clipping feature")
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
        "tree_runtime_sha256": sha256(Path(__file__).with_name("robust_tree.py")),
        "trainer_sha256": sha256(Path(trainer.__file__)),
        "primary": PRIMARY,
        "candidates": CONFIGS,
        "seed_run": str(seed_run),
        "protocol": "Exactly two MIL pseudo-label rounds, 120 iterations, learning rate .08, 63 bins, early stopping disabled. Seed classifier is codec-stable-hybrid from the same source-excluded training partition. Every derivative parent must be eligible. Evaluation codecs are never used for fitting or selection.",
        "limitation": "Adaptive development research on the existing corpus; declared before inspecting these nonlinear fits or the codec-stable linear follow-up results.",
    }
    file = output / "followup-declaration.json"
    if file.exists():
        old = json.loads(file.read_text())
        if any(
            old[k] != declaration[k]
            for k in (
                "entry_sha256",
                "tree_runtime_sha256",
                "trainer_sha256",
                "candidates",
            )
        ):
            raise ValueError("cannot resume changed tree experiment")
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
