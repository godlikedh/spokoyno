#!/usr/bin/env python3
"""Train localized shadow scorers; evaluate max event score on unseen whole clips."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from risk_score import POLICY, risk_tier
from sklearn.metrics import average_precision_score
from train_audio_context import fit, predict
from train_models import atomic_json


def family_links(payload: dict) -> dict[str, set[str]]:
    links = {}
    for row in payload.get("rows", []):
        for match in row.get("matches", []):
            a, b = row["audio_sha256"], match["reference_audio_sha256"]
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
    return links


def split_clips(
    clips: list[dict], held_out: str, links: dict
) -> tuple[list[int], list[int]]:
    excluded = {r["audio_sha256"] for r in clips if held_out in r["threads"]}
    pending = list(excluded)
    while pending:
        for neighbor in links.get(pending.pop(), set()) - excluded:
            excluded.add(neighbor)
            pending.append(neighbor)
    train = [
        i
        for i, r in enumerate(clips)
        if r["audio_sha256"] not in excluded and r["label"] in ("positive", "negative")
    ]
    test = [
        i
        for i, r in enumerate(clips)
        if r["fold_thread"] == held_out and r["label"] in ("positive", "negative")
    ]
    return train, test


def training_weights(owners: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Equal total weight per clip before balancing the two event classes."""
    _, inverse, counts = np.unique(owners, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].astype(float)
    for value in (0, 1):
        mask = targets == value
        if not np.any(mask):
            raise ValueError("training fold needs both event classes")
        weights[mask] *= len(counts) / (2 * weights[mask].sum())
    return weights


def clip_predictions(
    clips: list[dict], windows: list[dict], scores: np.ndarray, indices: list[int]
) -> list[dict]:
    predictions = []
    for index in indices:
        clip = clips[index]
        lo = clip["first_window"]
        chosen = lo + int(np.argmax(scores[lo : lo + clip["window_count"]]))
        score = float(scores[chosen])
        predictions.append(
            {
                "path": clip["path"],
                "audio_sha256": clip["audio_sha256"],
                "label": clip["label"],
                "score": score,
                "risk_tier": risk_tier(score),
                "event_at_s": windows[chosen]["start_s"],
            }
        )
    return predictions


def metrics(predictions: list[dict]) -> dict:
    known = [r for r in predictions if r["label"] in ("positive", "negative")]
    positive = sum(r["label"] == "positive" for r in known)
    negative = len(known) - positive
    result = {"positive": positive, "negative": negative}
    for tier, threshold in (("alert", 0.8), ("maybe_or_alert", 0.6)):
        tp = sum(r["label"] == "positive" and r["score"] >= threshold for r in known)
        fp = sum(r["label"] == "negative" and r["score"] >= threshold for r in known)
        result[tier] = {
            "detected": tp,
            "missed": positive - tp,
            "false_warnings": fp,
            "false_warnings_per_1000": 1000 * fp / negative if negative else None,
        }
    result["average_precision"] = (
        float(
            average_precision_score(
                [r["label"] == "positive" for r in known], [r["score"] for r in known]
            )
        )
        if positive
        else None
    )
    return result


def load_data(directory: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["matrix_sha256"] != sha256(
        directory / "examples.npz"
    ):
        raise ValueError("unsupported or modified event dataset")
    with np.load(directory / "examples.npz", allow_pickle=False) as source:
        physical, audio, targets = (
            source["physical"],
            source["embeddings"],
            source["targets"],
        )
    if (
        len(physical) != len(manifest["windows"])
        or len(physical) != len(audio)
        or len(targets) != len(audio)
    ):
        raise ValueError("event matrix/metadata shape mismatch")
    if np.any(np.isinf(physical)) or not np.all(np.isfinite(audio)):
        raise ValueError("invalid feature values")
    return manifest, physical, audio, targets


def train(
    manifest: dict,
    physical: np.ndarray,
    audio: np.ndarray,
    targets: np.ndarray,
    mode: str,
    directory: Path,
    links: dict,
) -> dict:
    clips, windows = manifest["clips"], manifest["windows"]
    owners = np.asarray([w["clip_index"] for w in windows])
    predictions, folds = [], []
    directory.mkdir(parents=True, exist_ok=True)
    for thread in sorted(
        {r["fold_thread"] for r in clips if r["label"] in ("positive", "negative")}
    ):
        training, testing = split_clips(clips, thread, links)
        indices = np.flatnonzero(np.isin(owners, training) & (targets >= 0))
        weights = training_weights(owners[indices], targets[indices])
        model = fit(
            physical[indices], audio[indices], targets[indices], mode, 8, weights
        )
        scores = predict(model, physical, audio)
        fold_predictions = clip_predictions(clips, windows, scores, testing)
        predictions.extend(fold_predictions)
        train_predictions = clip_predictions(clips, windows, scores, training)
        max_negative = max(
            r["score"] for r in train_predictions if r["label"] == "negative"
        )
        for row in fold_predictions:
            row["above_training_negative_max"] = row["score"] > max_negative
        fitted = {
            "model": model,
            "training_audio_hashes": [clips[i]["audio_sha256"] for i in training],
        }
        atomic_json(directory / f"{mode}-fold-{thread}.json", fitted)
        folds.append(
            {
                "thread": thread,
                "train_clips": len(training),
                "test_clips": len(testing),
                "train_windows": len(indices),
                "training_negative_max": max_negative,
            }
        )
        print(f"{mode}: held out {thread}: {metrics(fold_predictions)}", flush=True)
    indices = np.flatnonzero(targets >= 0)
    model = fit(
        physical[indices],
        audio[indices],
        targets[indices],
        mode,
        8,
        training_weights(owners[indices], targets[indices]),
    )
    scores = predict(model, physical, audio)
    artifact = {
        "schema": 1,
        "status": "shadow-only",
        "mode": mode,
        "risk_policy": POLICY,
        "model": model,
        "physical_names": manifest["physical_names"],
        "training_audio_hashes": sorted(
            {clips[i]["audio_sha256"] for i in owners[indices]}
        ),
        "training_matrix_sha256": manifest["matrix_sha256"],
        "labels_sha256": manifest["labels_sha256"],
        "events_sha256": manifest["events_sha256"],
        "proposal_policy": manifest["proposal_policy"],
        "warning": "Max localized event score, uncalibrated. Fixed 0.6/0.8 tiers do not guarantee a production false-alarm rate.",
    }
    atomic_json(directory / f"{mode}-model.json", artifact)
    result = metrics(predictions) | {
        "mode": mode,
        "predictions": predictions,
        "folds": folds,
        "training": metrics(
            clip_predictions(clips, windows, scores, list(range(len(clips))))
        ),
        "above_training_negative_max": {
            "detected": sum(
                r["label"] == "positive" and r["above_training_negative_max"]
                for r in predictions
            ),
            "false_warnings": sum(
                r["label"] == "negative" and r["above_training_negative_max"]
                for r in predictions
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/artifacts/event-models-v1")
    )
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--events", type=Path, default=Path("corpus/events.json"))
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("research/artifacts/fingerprints/evaluation.json"),
        help="existing identity matches conservatively join related audio families",
    )
    parser.add_argument(
        "--score", type=Path, help="score an existing model without fitting anything"
    )
    args = parser.parse_args()
    manifest, physical, audio, targets = load_data(args.data_dir)
    if manifest["labels_sha256"] != sha256(args.labels) or manifest[
        "events_sha256"
    ] != sha256(args.events):
        raise ValueError(
            "labels/timings changed; rebuild event dataset (features are cached)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.score:
        artifact = json.loads(args.score.read_text())
        if (
            artifact["schema"] != 1
            or artifact["physical_names"] != manifest["physical_names"]
        ):
            raise ValueError("model/event feature schema mismatch")
        scores = predict(artifact["model"], physical, audio)
        rows = clip_predictions(
            manifest["clips"],
            manifest["windows"],
            scores,
            list(range(len(manifest["clips"]))),
        )
        for row in rows:
            row["label_at_scoring"] = row.pop("label")
            row["seen_in_training"] = (
                row["audio_sha256"] in artifact["training_audio_hashes"]
            )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = args.output_dir / f"scores-{stamp}.json"
        atomic_json(
            target,
            {
                "schema": 1,
                "status": "shadow-only",
                "risk_policy": POLICY,
                "model_sha256": sha256(args.score),
                "dataset_manifest_sha256": sha256(args.data_dir / "manifest.json"),
                "rows": rows,
            },
        )
        print(f"wrote {target}")
        return 0
    links = (
        family_links(json.loads(args.fingerprints.read_text()))
        if args.fingerprints.exists()
        else {}
    )
    results = [
        train(manifest, physical, audio, targets, mode, args.output_dir, links)
        for mode in ("physical", "hybrid")
    ]
    selected = min(
        results,
        key=lambda r: (
            r["alert"]["false_warnings"],
            -r["alert"]["detected"],
            r["mode"] != "physical",
        ),
    )["mode"]
    atomic_json(
        args.output_dir / "model.json",
        json.loads((args.output_dir / f"{selected}-model.json").read_text()),
    )
    atomic_json(
        args.output_dir / "results.json",
        {
            "schema": 1,
            "risk_policy": POLICY,
            "selected_shadow": selected,
            "dataset_manifest_sha256": sha256(args.data_dir / "manifest.json"),
            "related_audio_source_sha256": sha256(args.fingerprints)
            if args.fingerprints.exists()
            else None,
            "related_audio_links": {key: sorted(value) for key, value in links.items()},
            "warning": "Development diagnostics, not independent validation. Whole source threads and known shared-audio families are excluded from training. Timings only supervise training windows, never test proposal selection. Twelve intervals remain ten recordings.",
            "experiments": results,
        },
    )
    print(f"Selected shadow model: {selected}; no browser model promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
