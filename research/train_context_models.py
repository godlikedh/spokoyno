#!/usr/bin/env python3
"""Fixed-budget context-feature study using the existing source-excluded MIL folds."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import train_robust_models as trainer
from extract_audio_context import sha256
from train_models import atomic_json

PRIMARY = "context-hybrid"
COMMON = {"audio": True, "augment": True, "consistency": 1.0}
CONFIGS = {
    PRIMARY: COMMON | {"power": 8, "feature_group": "full", "use_synthetic": True},
    "context-softmax": COMMON
    | {"power": 2, "feature_group": "full", "use_synthetic": True},
    "context-real-augmentation": COMMON
    | {"power": 8, "feature_group": "full", "use_synthetic": False},
    "boundary-control": COMMON
    | {"power": 8, "feature_group": "boundary", "use_synthetic": True},
}


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    data_directory = Path(sys.argv[sys.argv.index("--data-dir") + 1])
    data_manifest = json.loads((data_directory / "manifest.json").read_text())
    context = data_manifest["context_features"]
    head_file = Path(context["head_directory"]) / "head.json"
    if sha256(head_file) != context["head_sha256"]:
        raise ValueError("frozen semantic head changed")
    head = json.loads(head_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    declaration = {
        "created_at": datetime.now(UTC).isoformat(),
        "entry_sha256": sha256(Path(__file__)),
        "trainer_sha256": sha256(Path(trainer.__file__)),
        "primary": PRIMARY,
        "candidates": CONFIGS,
        "context_feature_source_sha256": context["feature_source_sha256"],
        "semantic_head_sha256": sha256(head_file),
        "protocol": "Same source weights, parent exclusions, fixed .6/.8 thresholds, regularization and source-variant consistency as the first linear study. Near-clipping input disabled in all heads. Compare full fixed context features, lower-order pooling, real codec augmentation without synthetic mixtures, and relative-content-boundary-only control. All ontology groups and feature formulas fixed before fitting.",
        "limitations": "Adaptive development research following PR12; existing corpus, not prospective evaluation. Frozen AudioSet categories are features, not screamer labels.",
    }
    file = output / "followup-declaration.json"
    if file.exists():
        old = json.loads(file.read_text())
        if any(
            old[k] != declaration[k]
            for k in (
                "entry_sha256",
                "trainer_sha256",
                "candidates",
                "context_feature_source_sha256",
                "semantic_head_sha256",
            )
        ):
            raise ValueError("cannot resume changed context experiment")
    else:
        atomic_json(file, declaration)
    original_fit = trainer.fit_one
    original_transform = trainer.fitted_transform

    def fitted_transform(data, training):
        transform, x = original_transform(data, training)
        transform["context_features"] = context | {"head": head}
        return transform, x

    def fit(data, x, training, config):
        disabled = [
            i
            for i, name in enumerate(data["names"])
            if name == "near_clip_fraction"
            or (
                config["feature_group"] == "boundary"
                and name.startswith("ctx_")
                and name != "ctx_boundary_opening"
            )
        ]
        values = x.copy()
        values[:, disabled] = 0
        selected_data = data
        if not config["use_synthetic"]:
            # A private training view changes eligibility only. Corpus labels,
            # held-out recording labels and the published dataset stay intact.
            selected_data = data | {
                "records": [
                    r | {"label": "unlabeled"} if r["kind"] == "synthetic" else r
                    for r in data["records"]
                ]
            }
        model, scores = original_fit(selected_data, values, training, config)
        if any(abs(model["coefficient"][i]) > 1e-12 for i in disabled):
            raise ValueError("disabled context feature acquired a coefficient")
        model["disabled_features"] = [data["names"][i] for i in disabled]
        return model, scores

    trainer.fitted_transform = fitted_transform
    trainer.fit_one = fit
    trainer.CONFIGS = CONFIGS
    trainer.PRIMARY = PRIMARY
    trainer.main()


if __name__ == "__main__":
    main()
