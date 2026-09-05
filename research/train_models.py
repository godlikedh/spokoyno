#!/usr/bin/env python3
"""Train and compare compact shadow models on grouped Spokoyno features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    complexity: int
    factory: Callable[[], Pipeline]


def compact_features(names: list[str]) -> list[str]:
    global_names = {
        "duration_s",
        "global_raw_peak_dbfs",
        "global_raw_clip_fraction",
        "global_median_db",
        "global_dynamic_range_db",
        "global_derivative_db",
        "global_loud_fraction",
        "global_quiet_fraction",
    }
    start_names = {
        "start_position_s",
        "start_event_db",
        "start_duration_s",
        "start_near_clip_fraction",
        "start_spectral_flatness",
        "start_spectral_centroid_hz",
    }
    event_suffixes = {
        "position_fraction",
        "event_level_db",
        "jump_db",
        "baseline_mad_db",
        "robust_z",
        "attack_ms",
        "event_0_1s_db",
        "event_0_3s_db",
        "event_0_6s_db",
        "event_1_0s_db",
        "jump_vs_0_5s_db",
        "jump_vs_1_0s_db",
        "jump_vs_3_0s_db",
        "jump_vs_6_0s_db",
        "duration_s",
        "duration_plus6_s",
        "duration_plus12_s",
        "persistence_1s",
        "loudness_area_db",
        "baseline_iqr_db",
        "baseline_derivative_db",
        "prior_abrupt_changes",
        "near_clip_fraction",
        "spectral_shape_distance",
        "spectral_flatness_delta",
        "spectral_centroid_delta_hz",
        "spectral_flux",
        "nearby_spectral_flux",
    }
    return [
        name
        for name in names
        if name in global_names
        or name in start_names
        or (
            name.startswith("event_1_")
            and name.removeprefix("event_1_") in event_suffixes
        )
        or (name.startswith("event_1_band_") and name.endswith("_jump_db"))
    ]


def experiments() -> list[Experiment]:
    def logistic(c: float) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=20260905,
                    ),
                ),
            ]
        )

    def forest(depth: int, features: float | str) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=depth,
                        min_samples_leaf=3,
                        max_features=features,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=20260905,
                    ),
                ),
            ]
        )

    return [
        Experiment("logistic-compact-c005", "compact", 1, lambda: logistic(0.05)),
        Experiment("logistic-compact-c02", "compact", 2, lambda: logistic(0.2)),
        Experiment("forest-compact-depth2", "compact", 3, lambda: forest(2, "sqrt")),
        Experiment("forest-compact-depth3", "compact", 4, lambda: forest(3, "sqrt")),
        Experiment("forest-rich-depth3", "rich", 5, lambda: forest(3, 0.35)),
    ]


def collapse_labeled(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["label"] in ("positive", "negative"):
            grouped.setdefault(row["audio_sha256"], []).append(row)
    collapsed = []
    cross_label = []
    for digest, members in grouped.items():
        labels = {row["label"] for row in members}
        if len(labels) != 1:
            cross_label.append(
                {"audio_sha256": digest, "paths": [row["path"] for row in members]}
            )
            continue
        threads = sorted({row["thread"] for row in members})
        representative = min(members, key=lambda row: row["path"])
        collapsed.append(
            {
                **representative,
                "paths": sorted(row["path"] for row in members),
                "threads": threads,
                "fold_thread": threads[0],
            }
        )
    if cross_label:
        raise ValueError(f"conflicting labels for identical audio: {cross_label}")
    collapsed.sort(key=lambda row: row["path"])
    return collapsed, {
        "logical_labeled": sum(len(members) for members in grouped.values()),
        "content_groups": len(collapsed),
        "duplicate_groups": sum(len(members) > 1 for members in grouped.values()),
    }


def matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[row["features"].get(name, np.nan) for name in names] for row in rows],
        dtype=np.float64,
    )


def conservative_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    negatives = scores[labels == 0]
    return min(1.0, float(np.max(negatives)) + 1e-9) if len(negatives) else 1.0


def evaluate_experiment(
    experiment: Experiment, rows: list[dict], all_features: list[str]
) -> dict:
    names = (
        compact_features(all_features)
        if experiment.feature_set == "compact"
        else all_features
    )
    positive_threads = sorted(
        {row["fold_thread"] for row in rows if row["label"] == "positive"}
    )
    predictions = []
    folds = []
    for held_out in positive_threads:
        test = [row for row in rows if row["fold_thread"] == held_out]
        test_hashes = {row["audio_sha256"] for row in test}
        train = [
            row
            for row in rows
            if row["fold_thread"] != held_out and row["audio_sha256"] not in test_hashes
        ]
        train_y = np.asarray(
            [row["label"] == "positive" for row in train], dtype=np.int8
        )
        test_y = np.asarray([row["label"] == "positive" for row in test], dtype=np.int8)
        if len(np.unique(train_y)) < 2 or not len(test):
            continue
        model = experiment.factory()
        model.fit(matrix(train, names), train_y)
        train_score = model.predict_proba(matrix(train, names))[:, 1]
        threshold = conservative_threshold(train_score, train_y)
        test_score = model.predict_proba(matrix(test, names))[:, 1]
        warned = test_score > threshold
        folds.append(
            {
                "thread": held_out,
                "train_positive": int(np.sum(train_y)),
                "train_negative": int(np.sum(train_y == 0)),
                "test_positive": int(np.sum(test_y)),
                "test_negative": int(np.sum(test_y == 0)),
                "threshold": threshold,
                "true_positive": int(np.sum(warned & (test_y == 1))),
                "false_positive": int(np.sum(warned & (test_y == 0))),
            }
        )
        for row, score, decision in zip(test, test_score, warned, strict=True):
            predictions.append(
                {
                    "path": row["path"],
                    "thread": held_out,
                    "label": row["label"],
                    "score": float(score),
                    "threshold": threshold,
                    "margin": float(score - threshold),
                    "warned": bool(decision),
                }
            )
    labels = np.asarray(
        [row["label"] == "positive" for row in predictions], dtype=np.int8
    )
    scores = np.asarray([row["margin"] for row in predictions], dtype=np.float64)
    warned = np.asarray([row["warned"] for row in predictions], dtype=bool)
    false_positives = [
        row for row in predictions if row["label"] == "negative" and row["warned"]
    ]
    false_negatives = [
        row for row in predictions if row["label"] == "positive" and not row["warned"]
    ]
    positive_predictions = [row for row in predictions if row["label"] == "positive"]
    return {
        "name": experiment.name,
        "feature_set": experiment.feature_set,
        "feature_count": len(names),
        "complexity": experiment.complexity,
        "positive": int(np.sum(labels)),
        "negative": int(np.sum(labels == 0)),
        "true_positive": int(np.sum(warned & (labels == 1))),
        "false_positive": int(np.sum(warned & (labels == 0))),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "folds": folds,
        "positive_predictions": positive_predictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def serialize_model(pipeline: Pipeline, feature_names: list[str]) -> dict:
    imputer = pipeline.named_steps["imputer"]
    base = {
        "feature_names": feature_names,
        "imputer_median": [float(value) for value in imputer.statistics_],
    }
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        scale = pipeline.named_steps["scale"]
        return base | {
            "type": "logistic-regression",
            "mean": [float(value) for value in scale.mean_],
            "scale": [float(value) for value in scale.scale_],
            "coefficient": [float(value) for value in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
        }
    if isinstance(model, RandomForestClassifier):
        trees = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            values = tree.value.reshape(tree.node_count, -1)
            positive = []
            for row in values:
                total = float(np.sum(row))
                positive.append(float(row[-1] / total) if total else 0.0)
            trees.append(
                {
                    "left": tree.children_left.tolist(),
                    "right": tree.children_right.tolist(),
                    "feature": tree.feature.tolist(),
                    "threshold": [float(value) for value in tree.threshold],
                    "positive": positive,
                }
            )
        return base | {
            "type": "random-forest",
            "input_precision": "float32",
            "trees": trees,
        }
    raise TypeError(f"cannot serialize {type(model).__name__}")


def predict_serialized(model: dict, raw: np.ndarray) -> np.ndarray:
    values = raw.copy()
    median = np.asarray(model["imputer_median"])
    missing = np.isnan(values)
    values[missing] = np.take(median, np.where(missing)[1])
    if model["type"] == "logistic-regression":
        scaled = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
        linear = scaled @ np.asarray(model["coefficient"]) + model["intercept"]
        return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))
    if model.get("input_precision") == "float32":
        # scikit-learn converts forest inputs to float32 before comparing them
        # with double-precision tree thresholds. Browser inference must mirror
        # this with Math.fround() to make boundary decisions identical.
        values = values.astype(np.float32).astype(np.float64)
    scores = np.zeros(len(values), dtype=np.float64)
    for tree in model["trees"]:
        for row_index, row in enumerate(values):
            node = 0
            while tree["left"][node] != -1:
                feature = tree["feature"][node]
                node = (
                    tree["left"][node]
                    if row[feature] <= tree["threshold"][node]
                    else tree["right"][node]
                )
            scores[row_index] += tree["positive"][node]
    return scores / len(model["trees"])


def feature_ranking(
    pipeline: Pipeline, names: list[str], limit: int = 15
) -> list[dict]:
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        values = model.coef_[0]
        rows = [
            {"feature": name, "weight": float(value), "importance": abs(float(value))}
            for name, value in zip(names, values, strict=True)
        ]
    elif isinstance(model, RandomForestClassifier):
        rows = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(names, model.feature_importances_, strict=True)
        ]
    else:
        return []
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:limit]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def model_card(
    results: list[dict],
    selected: dict,
    challenger: dict,
    dataset: dict,
    max_oof_fp: int,
    selected_features: list[dict],
    challenger_features: list[dict],
    dataset_hash: str,
    selected_fit: dict,
    challenger_fit: dict,
) -> str:
    lines = [
        "# Spokoyno shadow-model card",
        "",
        "This model is an offline research/shadow scorer. It does not replace the v5.7 production warning rule and its output is not a calibrated probability.",
        "",
        "## Dataset",
        "",
        f"- {dataset['logical_labeled']} logical labeled audio tracks",
        f"- {dataset['content_groups']} exact-content groups after deduplication",
        f"- {dataset['positive']} positive and {dataset['negative']} negative groups",
        f"- feature dataset SHA-256: `{dataset_hash}`",
        "- visual-only and any unlabeled clips excluded from training",
        "- folds hold out one positive-bearing thread at a time",
        "- ranking metrics pool score-minus-training-threshold margins so separately fitted fold scores share a conservative reference",
        "",
        "## Grouped experiment",
        "",
        "| model | features | TP | FP | average precision | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['feature_count']} | "
            f"{result['true_positive']}/{result['positive']} | "
            f"{result['false_positive']}/{result['negative']} | "
            f"{result['average_precision']:.3f} | {result['roc_auc']:.3f} |"
        )
    lines += ["", "## Held-out errors", ""]
    for result in results:
        false_positive = (
            ", ".join(Path(row["path"]).name for row in result["false_positives"])
            or "none"
        )
        false_negative = (
            ", ".join(Path(row["path"]).name for row in result["false_negatives"])
            or "none"
        )
        lines.append(
            f"- `{result['name']}` — false positives: {false_positive}; "
            f"missed positives: {false_negative}."
        )
    lines += [
        "",
        "## Exported candidate",
        "",
        f"`{selected['name']}` was selected from models with at most {max_oof_fp} out-of-fold false positives, then by held-out true positives, average precision, and simplicity.",
        "The exported threshold is the largest fitted-training negative score plus a 1e-9 serialization margin, and the decision operator is strictly greater-than. That guarantees zero training false positives only; it is not a validated production threshold.",
        f"`{challenger['name']}` is also exported as the higher-recall challenger; its grouped errors prevent it from satisfying the conservative selection constraint.",
        f"On the final all-data fit, the conservative model marks {selected_fit['training_true_positive']}/{dataset['positive']} positives and {selected_fit['training_false_positive']}/{dataset['negative']} negatives; the challenger marks {challenger_fit['training_true_positive']}/{dataset['positive']} positives and {challenger_fit['training_false_positive']}/{dataset['negative']} negatives. These are training results, not validation.",
        "",
        "## Strongest fitted features",
        "",
        f"- `{selected['name']}`: "
        + ", ".join(row["feature"] for row in selected_features),
        f"- `{challenger['name']}`: "
        + ", ".join(row["feature"] for row in challenger_features),
        "",
        "## Limitations",
        "",
        "All feature design has seen the current ten positives, several source threads contain positives without fully labeled negatives, and the selected model is chosen using these same grouped diagnostics. Future prospectively labeled threads are required before promotion.",
        "Synthetic gain/codec variants may be used for invariance tests but must never be counted as independent examples.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("research/models"))
    parser.add_argument(
        "--select", help="force a named experiment instead of automatic selection"
    )
    parser.add_argument(
        "--max-oof-fp",
        type=int,
        default=0,
        help="maximum grouped out-of-fold false positives for automatic selection",
    )
    args = parser.parse_args()
    payload = json.loads(args.features.read_text())
    dataset_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    rows, dataset_stats = collapse_labeled(payload["rows"])
    all_features = payload["feature_names"]
    experiment_defs = experiments()
    results = [
        evaluate_experiment(experiment, rows, all_features)
        for experiment in experiment_defs
    ]
    by_name = {result["name"]: result for result in results}
    if args.select:
        if args.select not in by_name:
            raise SystemExit(f"unknown experiment: {args.select}")
        selected_result = by_name[args.select]
    else:
        eligible = [
            result for result in results if result["false_positive"] <= args.max_oof_fp
        ]
        if not eligible:
            fewest = min(result["false_positive"] for result in results)
            eligible = [
                result for result in results if result["false_positive"] == fewest
            ]
        selected_result = max(
            eligible,
            key=lambda result: (
                result["true_positive"],
                result["average_precision"],
                -result["complexity"],
            ),
        )
    challenger_result = max(
        (result for result in results if result["name"] != selected_result["name"]),
        key=lambda result: (
            result["true_positive"],
            -result["false_positive"],
            result["average_precision"],
        ),
    )
    selected_def = next(
        item for item in experiment_defs if item.name == selected_result["name"]
    )
    selected_features = (
        compact_features(all_features)
        if selected_def.feature_set == "compact"
        else all_features
    )
    y = np.asarray([row["label"] == "positive" for row in rows], dtype=np.int8)
    x = matrix(rows, selected_features)
    fitted = selected_def.factory()
    fitted.fit(x, y)
    fitted_scores = fitted.predict_proba(x)[:, 1]
    threshold = conservative_threshold(fitted_scores, y)
    fitted_warned = fitted_scores > threshold
    serialized = serialize_model(fitted, selected_features)
    parity = predict_serialized(serialized, x)
    if not np.allclose(fitted_scores, parity, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            f"export parity failure: max error {np.max(np.abs(fitted_scores - parity))}"
        )

    challenger_def = next(
        item for item in experiment_defs if item.name == challenger_result["name"]
    )
    challenger_features = (
        compact_features(all_features)
        if challenger_def.feature_set == "compact"
        else all_features
    )
    challenger_x = matrix(rows, challenger_features)
    challenger_fitted = challenger_def.factory()
    challenger_fitted.fit(challenger_x, y)
    challenger_scores = challenger_fitted.predict_proba(challenger_x)[:, 1]
    challenger_threshold = conservative_threshold(challenger_scores, y)
    challenger_serialized = serialize_model(challenger_fitted, challenger_features)
    challenger_parity = predict_serialized(challenger_serialized, challenger_x)
    if not np.allclose(challenger_scores, challenger_parity, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            f"challenger export parity failure: max error "
            f"{np.max(np.abs(challenger_scores - challenger_parity))}"
        )

    dataset_stats.update(
        {
            "positive": int(np.sum(y)),
            "negative": int(np.sum(y == 0)),
            "threads": sorted({row["fold_thread"] for row in rows}),
        }
    )
    output = {
        "schema": 1,
        "feature_version": payload["feature_version"],
        "feature_dataset_sha256": dataset_hash,
        "dataset": dataset_stats,
        "selection_policy": (
            f"out-of-fold FP <= {args.max_oof_fp}, then held-out TP, "
            "average precision, and simplicity"
        ),
        "selected": selected_result["name"],
        "challenger": challenger_result["name"],
        "experiments": results,
        "final_fit": {
            selected_result["name"]: {
                "threshold": threshold,
                "training_true_positive": int(np.sum(fitted_warned & (y == 1))),
                "training_false_positive": int(np.sum(fitted_warned & (y == 0))),
                "strongest_features": feature_ranking(fitted, selected_features),
            },
            challenger_result["name"]: {
                "threshold": challenger_threshold,
                "training_true_positive": int(
                    np.sum((challenger_scores > challenger_threshold) & (y == 1))
                ),
                "training_false_positive": int(
                    np.sum((challenger_scores > challenger_threshold) & (y == 0))
                ),
                "strongest_features": feature_ranking(
                    challenger_fitted, challenger_features
                ),
            },
        },
    }
    exported = {
        "schema": 1,
        "status": "shadow-only",
        "score_semantics": "uncalibrated ranking score",
        "feature_version": payload["feature_version"],
        "feature_dataset_sha256": dataset_hash,
        "experiment": selected_result["name"],
        "training_content_groups": len(rows),
        "training_positive": int(np.sum(y)),
        "training_negative": int(np.sum(y == 0)),
        "training_zero_fp_threshold": threshold,
        "decision_operator": ">",
        "model": serialized,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "model-results.json", output)
    atomic_json(args.output_dir / "shadow-model.json", exported)
    atomic_json(
        args.output_dir / "challenger-model.json",
        {
            "schema": 1,
            "status": "shadow-only",
            "score_semantics": "uncalibrated ranking score",
            "feature_version": payload["feature_version"],
            "feature_dataset_sha256": dataset_hash,
            "experiment": challenger_result["name"],
            "training_content_groups": len(rows),
            "training_positive": int(np.sum(y)),
            "training_negative": int(np.sum(y == 0)),
            "training_zero_fp_threshold": challenger_threshold,
            "decision_operator": ">",
            "model": challenger_serialized,
        },
    )
    (args.output_dir / "MODEL_CARD.md").write_text(
        model_card(
            results,
            selected_result,
            challenger_result,
            dataset_stats,
            args.max_oof_fp,
            output["final_fit"][selected_result["name"]]["strongest_features"],
            output["final_fit"][challenger_result["name"]]["strongest_features"],
            dataset_hash,
            output["final_fit"][selected_result["name"]],
            output["final_fit"][challenger_result["name"]],
        )
    )
    print(
        f"selected {selected_result['name']}; trained on {int(np.sum(y))} positive and "
        f"{int(np.sum(y == 0))} negative content groups; threshold {threshold:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
