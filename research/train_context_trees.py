#!/usr/bin/env python3
"""Fixed small-tree test of interactions between acoustic and sound-context features."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import train_hard_negative_mil as mining
import train_robust_models as trainer
from extract_audio_context import sha256
from improve_events import training_key
from robust_tree import export, window_probabilities
from sklearn.ensemble import HistGradientBoostingClassifier
from train_models import atomic_json

PRIMARY = "context-boost"
COMMON = mining.COMMON | {"audio": True, "hard_mass": 0.75, "hard_count": 3}
CONFIGS = {
    PRIMARY: COMMON | {"context": True},
    "context-boost-control": COMMON | {"context": False},
}


def fit(data, x, training, config, seed_run):
    directory = seed_run / "fits" / training_key(training)
    provenance = json.loads((directory / "manifest.json").read_text())
    seed_file = directory / "models.json"
    if (
        provenance["training_indices"] != training
        or sha256(seed_file) != provenance["files"]["models.json"]
    ):
        raise ValueError("seed does not match training partition")
    seed = json.loads(seed_file.read_text())["context-hybrid"]
    eligible = trainer.eligible_records(data["records"], training, True)
    selected = np.flatnonzero(eligible)
    records = [data["records"][i] for i in selected]
    ids = np.flatnonzero(eligible[data["owners"]])
    mapping = np.full(len(data["records"]), -1, dtype=int)
    mapping[selected] = np.arange(len(selected))
    owners = mapping[data["owners"][ids]]
    starts = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(records)))]
    labels = np.asarray([r["label"] == "positive" for r in records])
    targets = data["targets"][ids]
    bag_weights = trainer.group_weights(records)
    probabilities = window_probabilities(seed, x[ids])
    values = x[ids].copy()
    disabled = [
        i
        for i, name in enumerate(data["names"])
        if name == "near_clip_fraction"
        or (not config["context"] and name.startswith("ctx_"))
    ]
    values[:, disabled] = 0
    diagnostics = []
    for iteration in range(config["rounds"]):
        y, weights = mining.instance_supervision(
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
            "disabled_features": [data["names"][i] for i in disabled],
            "seed_model_sha256": sha256(seed_file),
            "seed_name": "context-hybrid",
            "mil_rounds": config["rounds"],
            "training_diagnostics": diagnostics,
        }
    )
    if np.max(np.abs(window_probabilities(model, values) - probabilities)) > 1e-12:
        raise ValueError("portable tree differs from estimator")
    if any(
        not node["leaf"] and node["feature"] in disabled
        for tree in model["trees"]
        for node in tree
    ):
        raise ValueError("tree used disabled input")
    scores = trainer.pool(
        window_probabilities(model, x),
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
    data_directory = Path(sys.argv[sys.argv.index("--data-dir") + 1])
    context = json.loads((data_directory / "manifest.json").read_text())[
        "context_features"
    ]
    head_file = Path(context["head_directory"]) / "head.json"
    if sha256(head_file) != context["head_sha256"]:
        raise ValueError("frozen semantic head changed")
    head = json.loads(head_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    declaration = {
        "created_at": datetime.now(UTC).isoformat(),
        "entry_sha256": sha256(Path(__file__)),
        "trainer_sha256": sha256(Path(trainer.__file__)),
        "mining_sha256": sha256(Path(mining.__file__)),
        "seed_run": str(seed_run),
        "primary": PRIMARY,
        "candidates": CONFIGS,
        "context_feature_source_sha256": context["feature_source_sha256"],
        "semantic_head_sha256": sha256(head_file),
        "protocol": "Two fixed 7-leaf, depth-3, 120-tree heads with three MIL rounds, 75% normal weight on the three hardest events. Both use identical same-fold linear context seeds and training-only PCA8. Compare all context measurements with all 32 context inputs disabled. Clipping disabled in both. Same source weights, parent exclusions and .6/.8 thresholds. No evaluation codec enters fitting or selection.",
        "limitations": "Adaptive development after inspecting linear context results; no new independent validation. Configurations fixed before tree results or hard-negative study results are inspected.",
    }
    file = output / "followup-declaration.json"
    if file.exists():
        old = json.loads(file.read_text())
        if any(old[k] != declaration[k] for k in declaration if k != "created_at"):
            raise ValueError("cannot resume changed study")
    else:
        atomic_json(file, declaration)
    original_transform = trainer.fitted_transform

    def fitted_transform(data, training):
        transform, x = original_transform(data, training)
        transform["context_features"] = context | {"head": head}
        return transform, x

    trainer.fitted_transform = fitted_transform
    trainer.fit_one = lambda data, x, training, config: fit(
        data, x, training, config, seed_run
    )
    trainer.CONFIGS = CONFIGS
    trainer.PRIMARY = PRIMARY
    trainer.main()


if __name__ == "__main__":
    main()
