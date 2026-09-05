#!/usr/bin/env python3
"""Compare physical features, frozen audio embeddings, and their combination."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dataset import label_for
from extract_audio_context import sha256
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from train_models import (
    atomic_json,
    collapse_labeled,
    compact_features,
    conservative_threshold,
    matrix,
)

EXPERIMENTS = (("physical", 0), ("embedding", 8), ("hybrid", 8))


def standardize(
    values: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler().fit(values, sample_weight=sample_weight)
    return scaler.transform(values), {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def prepare(model: dict, physical: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    parts = []
    if "physical" in model:
        spec = model["physical"]
        filled = np.where(np.isnan(physical), np.asarray(spec["median"]), physical)
        parts.append((filled - spec["mean"]) / spec["scale"])
    if "audio" in model:
        spec = model["audio"]
        projected = (embeddings - spec["pca_mean"]) @ np.asarray(spec["components"]).T
        parts.append((projected - spec["mean"]) / spec["scale"])
    return np.concatenate(parts, axis=1)


def predict(model: dict, physical: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    features = prepare(model, physical, embeddings)
    logits = features @ np.asarray(model["coefficient"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def fit(
    physical: np.ndarray,
    embeddings: np.ndarray,
    y: np.ndarray,
    mode: str,
    components: int,
    sample_weight: np.ndarray | None = None,
) -> dict:
    model = {"type": "audio-context-logistic", "mode": mode}
    parts = []
    if mode in ("physical", "hybrid"):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
            physical
        )
        transformed, spec = standardize(imputer.transform(physical), sample_weight)
        model["physical"] = spec | {"median": imputer.statistics_.tolist()}
        parts.append(transformed)
    if mode in ("embedding", "hybrid"):
        # Both dimensionality reduction and scaling see training rows only.
        pca = PCA(
            n_components=min(components, len(y) - 1),
            svd_solver="randomized",
            random_state=42,
        )
        pca.fit(embeddings.astype(np.float64))
        projected = pca.transform(embeddings.astype(np.float64))
        transformed, spec = standardize(projected, sample_weight)
        model["audio"] = spec | {
            "pca_mean": pca.mean_.tolist(),
            "components": pca.components_.tolist(),
        }
        parts.append(transformed)
    features = np.concatenate(parts, axis=1)
    classifier = LogisticRegression(
        C=0.05,
        class_weight="balanced" if sample_weight is None else None,
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )
    classifier.fit(features, y, sample_weight=sample_weight)
    model.update(
        {
            "coefficient": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
        }
    )
    actual = predict(model, physical, embeddings)
    np.testing.assert_allclose(
        actual,
        classifier.predict_proba(prepare(model, physical, embeddings))[:, 1],
        rtol=1e-9,
        atol=1e-10,
    )
    return model


def split_indices(rows: list[dict], held_out: str) -> tuple[np.ndarray, np.ndarray]:
    # Every test group is counted once, while all copies belonging to its
    # source thread are excluded from training even if assigned an earlier fold.
    train = np.asarray(
        [i for i, row in enumerate(rows) if held_out not in row["threads"]],
        dtype=np.int64,
    )
    test = np.asarray(
        [i for i, row in enumerate(rows) if row["fold_thread"] == held_out],
        dtype=np.int64,
    )
    return train, test


def evaluate(
    rows: list[dict],
    physical: np.ndarray,
    embeddings: np.ndarray,
    mode: str,
    components: int,
) -> dict:
    y = np.asarray([row["label"] == "positive" for row in rows], dtype=np.int8)
    predictions, folds = [], []
    for thread in sorted({row["fold_thread"] for row in rows}):
        train, test = split_indices(rows, thread)
        if not len(test) or len(np.unique(y[train])) != 2:
            raise ValueError(f"cannot evaluate source fold {thread}")
        fitted = fit(physical[train], embeddings[train], y[train], mode, components)
        threshold = conservative_threshold(
            predict(fitted, physical[train], embeddings[train]), y[train]
        )
        scores = predict(fitted, physical[test], embeddings[test])
        folds.append(
            {
                "thread": thread,
                "train_positive": int(sum(y[train])),
                "train_negative": int(sum(y[train] == 0)),
                "threshold": threshold,
            }
        )
        for index, score in zip(test, scores, strict=True):
            row = rows[index]
            predictions.append(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "score": float(score),
                    "threshold": threshold,
                    "warned": bool(score > threshold),
                }
            )
    labels = np.asarray([r["label"] == "positive" for r in predictions])
    margins = np.asarray([r["score"] - r["threshold"] for r in predictions])
    warnings = np.asarray([r["warned"] for r in predictions])
    return {
        "mode": mode,
        "pca_components": components,
        "positive": int(sum(labels)),
        "negative": int(sum(~labels)),
        "true_positive": int(sum(labels & warnings)),
        "false_positive": int(sum(~labels & warnings)),
        "average_precision": float(average_precision_score(labels, margins)),
        "folds": folds,
        "predictions": predictions,
    }


def load_data(features_path: Path, context_dir: Path, labels_path: Path) -> tuple:
    payload = json.loads(features_path.read_text())
    manifest = json.loads((context_dir / "manifest.json").read_text())
    if (
        manifest.get("schema") != 1
        or manifest.get("feature_version") != 1
        or payload.get("feature_version") != 1
    ):
        raise ValueError("unsupported context/physical feature schema")
    if manifest["feature_dataset_sha256"] != sha256(features_path):
        raise ValueError(
            "context is stale for this feature dataset; rerun extraction (cached)"
        )
    if sha256(context_dir / "embeddings.npz") != manifest["matrix_sha256"]:
        raise ValueError("embedding matrix hash mismatch")
    with np.load(context_dir / "embeddings.npz", allow_pickle=False) as source:
        embedding = source["embeddings"]
    if len(embedding) != len(manifest["rows"]) or not np.all(np.isfinite(embedding)):
        raise ValueError("invalid context matrix")
    mapping = {r["audio_sha256"]: embedding[i] for i, r in enumerate(manifest["rows"])}
    labels = json.loads(labels_path.read_text())
    rows = [
        dict(r, label=label_for(r["path"], labels))
        for r in payload["rows"]
        if label_for(r["path"], labels) != "visual-only"
    ]
    missing = {r["audio_sha256"] for r in rows} - mapping.keys()
    if missing:
        raise ValueError(
            f"missing embeddings for {len(missing)} groups; run full extraction"
        )
    return rows, compact_features(payload["feature_names"]), mapping, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument(
        "--context-dir", type=Path, default=Path("research/artifacts/audio-context-v1")
    )
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/artifacts/audio-models")
    )
    parser.add_argument(
        "--score", type=Path, help="score a saved model without retraining"
    )
    args = parser.parse_args()
    logical, names, mapping, manifest = load_data(
        args.features, args.context_dir, args.labels
    )
    encoder_id = hashlib.sha256(
        json.dumps(manifest["encoder"], sort_keys=True).encode()
    ).hexdigest()
    if args.score:
        artifact = json.loads(args.score.read_text())
        if (
            artifact["encoder_sha256"] != encoder_id
            or artifact["context_schema"] != manifest["schema"]
        ):
            raise ValueError("encoder/context schema does not match model")
        scores = predict(
            artifact["model"],
            matrix(logical, artifact["physical_names"]),
            np.stack([mapping[r["audio_sha256"]] for r in logical]),
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = args.output_dir / f"scores-{stamp}.json"
        atomic_json(
            target,
            {
                "schema": 1,
                "created_at": stamp,
                "model_sha256": sha256(args.score),
                "context_matrix_sha256": manifest["matrix_sha256"],
                "status": "shadow-only",
                "rows": [
                    {
                        "path": row["path"],
                        "audio_sha256": row["audio_sha256"],
                        "label_at_scoring": row["label"],
                        "seen_in_training": row["audio_sha256"]
                        in artifact["training_audio_hashes"],
                        "score": float(score),
                        "above_training_threshold": bool(score > artifact["threshold"]),
                    }
                    for row, score in zip(logical, scores, strict=True)
                ],
            },
        )
        print(f"wrote {target}")
        return 0
    rows, stats = collapse_labeled(logical)
    physical = matrix(rows, names)
    embeddings = np.stack([mapping[r["audio_sha256"]] for r in rows])
    results = []
    for mode, components in EXPERIMENTS:
        result = evaluate(rows, physical, embeddings, mode, components)
        results.append(result)
        print(
            f"{mode}: {result['true_positive']}/{result['positive']} TP, {result['false_positive']}/{result['negative']} FP",
            flush=True,
        )
    selected = min(
        results,
        key=lambda r: (
            r["false_positive"],
            -r["true_positive"],
            -r["average_precision"],
        ),
    )
    y = np.asarray([r["label"] == "positive" for r in rows], dtype=np.int8)
    final_fits = {}
    for result in results:
        fitted = fit(physical, embeddings, y, result["mode"], result["pca_components"])
        fitted_scores = predict(fitted, physical, embeddings)
        threshold = conservative_threshold(fitted_scores, y)
        artifact = {
            "schema": 1,
            "status": "shadow-only",
            "score_semantics": "uncalibrated ranking score",
            "context_schema": manifest["schema"],
            "encoder_sha256": encoder_id,
            "physical_names": names,
            "model": fitted,
            "threshold": threshold,
            "decision_operator": ">",
            "training_audio_hashes": [r["audio_sha256"] for r in rows],
            "feature_dataset_sha256": sha256(args.features),
            "context_matrix_sha256": manifest["matrix_sha256"],
            "labels_sha256": sha256(args.labels),
        }
        atomic_json(args.output_dir / f"{result['mode']}-model.json", artifact)
        if result["mode"] == selected["mode"]:
            atomic_json(args.output_dir / "model.json", artifact)
        final_fits[result["mode"]] = {
            "training_true_positive": int(sum((fitted_scores > threshold) & (y == 1))),
            "training_false_positive": int(sum((fitted_scores > threshold) & (y == 0))),
            "threshold": threshold,
        }
    atomic_json(
        args.output_dir / "results.json",
        {
            "schema": 1,
            "dataset": stats,
            "selected": selected["mode"],
            "experiments": results,
            "final_fits": final_fits,
        },
    )
    lines = [
        "# Frozen audio-context experiment",
        "",
        "All results are development diagnostics, not prospective accuracy. Production remains unchanged.",
        "",
        "| Features | Held-out TP | Held-out FP | AP |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['mode']} | {r['true_positive']}/{r['positive']} | {r['false_positive']}/{r['negative']} | {r['average_precision']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Selected `{selected['mode']}` by fewest FP, then TP and AP. Selection uses the same folds, so its estimate is optimistic.",
            "",
            "PCA and scaling are fitted inside source-thread folds. All groups with membership in a test thread are excluded from training. Thresholds are above the largest fitted-training negative, not validated production thresholds. Event candidates never use annotation timings. Embeddings remain frozen.",
            "",
            "See results.json for all predictions and fold thresholds. Future scoring records model hashes and training membership.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
