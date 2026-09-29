#!/usr/bin/env python3
"""Versioned shadow retraining with shared source-thread/family-held-out folds."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from dataset import label_for
from extract_audio_context import sha256
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_audio_context import fit as fit_context
from train_audio_context import load_data as load_context
from train_audio_context import predict as predict_context
from train_events import family_links, load_data, metrics, split_clips, training_weights
from train_models import atomic_json, compact_features, conservative_threshold, matrix

SEED = 20260930
CANDIDATES = (
    "logistic-compact",
    "forest-rich-depth3",
    "clip-embedding",
    "clip-hybrid",
    "event-physical",
    "event-hybrid",
    "extra-trees-rich-depth3",
    "hist-gradient-rich-depth2",
    "logistic-rich",
    "event-mil-physical",
)
FEATURES = Path("research/artifacts/features-v1.json")
LABELS = Path("corpus/labels.json")
EVENTS = Path("corpus/events.json")
CONTEXT = Path("research/artifacts/audio-context-v1")
EVENT_DATA = Path("research/artifacts/event-data-v1")
FAMILIES = Path("research/artifacts/fingerprints/evaluation.json")


def load_inputs() -> dict:
    logical, _, embeddings, context = load_context(FEATURES, CONTEXT, LABELS)
    manifest, event_physical, event_audio, targets = load_data(EVENT_DATA)
    expected = {
        "feature_dataset_sha256": sha256(FEATURES),
        "labels_sha256": sha256(LABELS),
        "events_sha256": sha256(EVENTS),
        "context_manifest_sha256": sha256(CONTEXT / "manifest.json"),
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("stale event dataset; regenerate features/context/events")
    labels = json.loads(LABELS.read_text())
    features = json.loads(FEATURES.read_text())
    if any(row["label"] != label_for(row["path"], labels) for row in features["rows"]):
        raise ValueError("stale feature labels")
    by_hash = {row["audio_sha256"]: row for row in logical}
    clips = manifest["clips"]
    # The event manifest retains ALL duplicate paths, including unlabeled copies.
    # Do not rebuild groups from labeled paths only and lose source membership.
    names = features["feature_names"]
    compact = compact_features(names)
    links = family_links(json.loads(FAMILIES.read_text()))
    provenance = expected | {
        "event_manifest_sha256": sha256(EVENT_DATA / "manifest.json"),
        "event_matrix_sha256": manifest["matrix_sha256"],
        "context_matrix_sha256": context["matrix_sha256"],
        "family_source_sha256": sha256(FAMILIES),
        "encoder_sha256": hashlib.sha256(
            json.dumps(context["encoder"], sort_keys=True).encode()
        ).hexdigest(),
    }
    return {
        "clips": clips,
        "windows": manifest["windows"],
        "names": names,
        "compact_names": compact,
        "compact_columns": [names.index(name) for name in compact],
        "physical": matrix([by_hash[c["audio_sha256"]] for c in clips], names),
        "audio": np.stack([embeddings[c["audio_sha256"]] for c in clips]),
        "event_physical": event_physical,
        "event_audio": event_audio,
        "event_names": manifest["physical_names"],
        "targets": targets,
        "owners": np.asarray([w["clip_index"] for w in manifest["windows"]]),
        "y": np.asarray([c["label"] == "positive" for c in clips], dtype=np.int8),
        "links": links,
        "provenance": provenance,
    }


def folds_for(clips: list[dict], links: dict) -> list[dict]:
    folds, tested = [], []
    for thread in sorted(
        {c["fold_thread"] for c in clips if c["label"] in ("positive", "negative")}
    ):
        train, test = split_clips(clips, thread, links)
        if {clips[i]["label"] for i in train} != {"positive", "negative"}:
            raise ValueError(f"fold {thread} does not have both training classes")
        if set(train) & set(test) or any(thread in clips[i]["threads"] for i in train):
            raise ValueError("source-thread leakage")
        folds.append({"thread": thread, "train": train, "test": test})
        tested.extend(test)
    known = [i for i, c in enumerate(clips) if c["label"] in ("positive", "negative")]
    if sorted(tested) != known:
        raise ValueError("each labeled exact-audio group must be tested exactly once")
    return folds


def tabular_pipeline(name: str) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name.startswith("logistic"):
        steps.append(("scale", StandardScaler()))
        model = LogisticRegression(
            C=0.05,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
    elif name == "forest-rich-depth3":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=3,
            max_features=0.35,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=SEED,
        )
    elif name == "extra-trees-rich-depth3":
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=3,
            max_features=0.35,
            class_weight="balanced",
            n_jobs=2,
            random_state=SEED,
        )
    elif name == "hist-gradient-rich-depth2":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=10,
            l2_regularization=10,
            class_weight="balanced",
            early_stopping=False,
            random_state=SEED,
        )
    else:
        raise ValueError(f"unknown tabular candidate: {name}")
    return Pipeline([*steps, ("model", model)])


def event_indices(data: dict, training: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(data["owners"], training) & (data["targets"] >= 0))


def weak_choices(data: dict, training: list[int], scores: np.ndarray) -> list[int]:
    """Latent positives from TRAINING bags only; never edit human annotations."""
    chosen = []
    for index in training:
        clip = data["clips"][index]
        if clip["label"] == "positive" and clip["annotation"] is None:
            lo = clip["first_window"]
            chosen.append(lo + int(np.argmax(scores[lo : lo + clip["window_count"]])))
    return chosen


def fit_candidate(name: str, data: dict, training: list[int]) -> dict:
    train = np.asarray(training, dtype=np.int64)
    if name.startswith("event-"):
        mode = "hybrid" if name == "event-hybrid" else "physical"
        known = event_indices(data, training)
        indices, targets = known, data["targets"][known]
        chosen = []
        # Existing annotated-event models do not use untimed positives. MIL is
        # separate: initialize on real event targets, then three fixed latent
        # updates, never inventing negatives for the other positive-bag windows.
        for _ in range(4 if name == "event-mil-physical" else 1):
            fitted = fit_context(
                data["event_physical"][indices],
                data["event_audio"][indices],
                targets,
                mode,
                8,
                training_weights(data["owners"][indices], targets),
            )
            if name == "event-mil-physical":
                used = chosen
                chosen = weak_choices(
                    data,
                    training,
                    predict_context(
                        fitted, data["event_physical"], data["event_audio"]
                    ),
                )
                indices = np.concatenate([known, np.asarray(chosen, dtype=np.int64)])
                targets = np.concatenate(
                    [data["targets"][known], np.ones(len(chosen), np.int8)]
                )
        supervised = set(data["owners"][known].tolist())
        latent = used if name == "event-mil-physical" else []
        return {
            "kind": "event",
            "model": fitted,
            "event_names": data["event_names"],
            "used_clips": sorted(supervised | {int(data["owners"][i]) for i in latent}),
            "annotated_training_windows": len(known),
            "latent_training_windows": latent,
        }
    columns = (
        data["compact_columns"]
        if name in ("logistic-compact", "clip-hybrid", "clip-embedding")
        else list(range(len(data["names"])))
    )
    if name.startswith("clip-"):
        mode = name.removeprefix("clip-")
        fitted = fit_context(
            data["physical"][train][:, columns],
            data["audio"][train],
            data["y"][train],
            mode,
            8,
        )
        return {
            "kind": "context",
            "model": fitted,
            "columns": columns,
            "used_clips": training,
        }
    pipeline = tabular_pipeline(name)
    pipeline.fit(data["physical"][train][:, columns], data["y"][train])
    return {
        "kind": "tabular",
        "pipeline": pipeline,
        "columns": columns,
        "used_clips": training,
    }


def predict_candidate(fitted: dict, data: dict) -> np.ndarray:
    if fitted["kind"] == "event":
        scores = predict_context(
            fitted["model"], data["event_physical"], data["event_audio"]
        )
        result = np.asarray(
            [
                np.max(
                    scores[c["first_window"] : c["first_window"] + c["window_count"]]
                )
                for c in data["clips"]
            ]
        )
    elif fitted["kind"] == "context":
        result = predict_context(
            fitted["model"], data["physical"][:, fitted["columns"]], data["audio"]
        )
    else:
        result = fitted["pipeline"].predict_proba(
            data["physical"][:, fitted["columns"]]
        )[:, 1]
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("invalid scores")
    return result


def prediction_rows(
    data: dict, scores: np.ndarray, indices: list[int], threshold: float
) -> list[dict]:
    return [
        {
            key: data["clips"][i][key]
            for key in ("path", "paths", "audio_sha256", "label", "fold_thread")
        }
        | {
            "score": float(scores[i]),
            "threshold": threshold,
            "above_training_threshold": bool(scores[i] > threshold),
        }
        for i in indices
    ]


def summary(rows: list[dict]) -> dict:
    known = [r for r in rows if r["label"] in ("positive", "negative")]
    return metrics(known) | {
        "yellow_only_positives": sum(
            r["label"] == "positive" and 0.6 <= r["score"] < 0.8 for r in known
        ),
        "yellow_only_negatives": sum(
            r["label"] == "negative" and 0.6 <= r["score"] < 0.8 for r in known
        ),
        "training_negative_max_policy": {
            "operator": ">",
            "detected": sum(
                r["label"] == "positive" and r["above_training_threshold"]
                for r in known
            ),
            "false_warnings": sum(
                r["label"] == "negative" and r["above_training_threshold"]
                for r in known
            ),
        },
    }


def save_fit(
    directory: Path,
    fitted: dict,
    data: dict,
    training: list[int],
    scores: np.ndarray,
    name: str,
) -> dict:
    directory.mkdir(parents=True)
    model_path = directory / "model.joblib"
    joblib.dump(fitted, model_path, compress=3)
    # Only deserialize the artifact this process just created, never user media.
    reloaded = joblib.load(model_path)
    np.testing.assert_allclose(
        predict_candidate(reloaded, data), scores, rtol=1e-12, atol=1e-12
    )
    threshold = conservative_threshold(scores[training], data["y"][training])
    spec = {
        "schema": 1,
        "status": "shadow-only",
        "candidate": name,
        "score_semantics": "uncalibrated evidence, not probability",
        "model_file": "model.joblib",
        "model_sha256": sha256(model_path),
        "feature_names": data["names"],
        "event_feature_names": data["event_names"],
        "training_audio_hashes": sorted(
            data["clips"][i]["audio_sha256"] for i in fitted["used_clips"]
        ),
        "eligible_training_audio_hashes": sorted(
            data["clips"][i]["audio_sha256"] for i in training
        ),
        "threshold": threshold,
        "decision_operator": ">",
        "provenance": data["provenance"],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "seed": SEED,
        "warning": "Joblib is executable pickle. Load only your own trusted artifacts. Final-fit scores are not validation. No production promotion.",
    }
    if fitted["kind"] == "event":
        spec["annotated_training_windows"] = fitted["annotated_training_windows"]
        spec["latent_positive_windows_not_annotations"] = [
            {
                "audio_sha256": data["clips"][int(data["owners"][i])]["audio_sha256"],
                **data["windows"][i],
            }
            for i in fitted["latent_training_windows"]
        ]
    atomic_json(directory / "manifest.json", spec)
    return spec


def run_candidate(name: str, data: dict, folds: list[dict], output: Path) -> dict:
    oof, fold_rows, latest_rows = [], [], []
    for fold in folds:
        training, testing, thread = fold["train"], fold["test"], fold["thread"]
        fitted = fit_candidate(name, data, training)
        scores = predict_candidate(fitted, data)
        spec = save_fit(
            output / name / f"fold-{thread}", fitted, data, training, scores, name
        )
        rows = prediction_rows(data, scores, testing, spec["threshold"])
        oof.extend(rows)
        fold_rows.append(
            {
                "thread": thread,
                "training": len(training),
                "testing": len(testing),
                "training_positive": int(data["y"][training].sum()),
                "threshold": spec["threshold"],
                "metrics": summary(rows),
            }
        )
        if thread == "336991612":
            # Full thread includes groups owned by earlier folds. This extra
            # view overlaps OOF rows and must never be pooled into OOF totals.
            indices = [i for i, c in enumerate(data["clips"]) if thread in c["threads"]]
            latest_rows = prediction_rows(data, scores, indices, spec["threshold"])
        print(f"{name} held out {thread}: {summary(rows)['alert']}", flush=True)
    training = [
        i for i, c in enumerate(data["clips"]) if c["label"] in ("positive", "negative")
    ]
    fitted = fit_candidate(name, data, training)
    scores = predict_candidate(fitted, data)
    spec = save_fit(output / name / "final", fitted, data, training, scores, name)
    all_rows = prediction_rows(
        data, scores, list(range(len(data["clips"]))), spec["threshold"]
    )
    result = {
        "name": name,
        "held_out": summary(oof),
        "folds": fold_rows,
        "held_out_predictions": oof,
        "latest_thread_excluded": summary(latest_rows),
        "latest_thread_excluded_predictions": latest_rows,
        "final_fit_all_labeled_diagnostic": summary(all_rows),
        "final_predictions": all_rows,
        "actually_used_training_clips": len(fitted["used_clips"]),
        "final_model_manifest": str(output / name / "final/manifest.json"),
    }
    atomic_json(output / name / "results.json", result)
    print(f"DONE {name}: {result['held_out']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES)
    )
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4 or len(set(args.candidates)) != len(args.candidates):
        raise ValueError("use 1..4 jobs and distinct candidates")
    if args.output_dir.exists():
        raise ValueError(
            "refusing to overwrite a retraining run; choose a fresh directory"
        )
    data = load_inputs()
    folds = folds_for(data["clips"], data["links"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "shadow-only",
        "candidates": args.candidates,
        "seed": SEED,
        "data": dict(Counter(c["label"] for c in data["clips"])),
        "source_sha256": sha256(Path(__file__)),
        "provenance": data["provenance"],
        "folds": folds,
        "related_audio_links": {k: sorted(v) for k, v in data["links"].items()},
        "policy": "Fixed configurations declared before fitting. All source threads held out, including negative-only threads; all duplicate source memberships and known transitive families excluded. Each labeled audio group tested once. Preprocessing and latent event selection use training rows only. 0.6/0.8 fixed tiers plus training-negative-max diagnostic; no held-out threshold tuning. Model comparisons are development diagnostics, not nested model-selection validation.",
        "limitations": "Only 14 distinct positives. Feature design has seen this corpus; incomplete related-audio detection remains possible. Ten timed positive recordings; four untimed positives can supervise clip models and experimental MIL, not original annotated-event models. Future threads are needed before promotion.",
    }
    atomic_json(args.output_dir / "plan.json", plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_candidate, name, data, folds, args.output_dir): name
            for name in args.candidates
        }
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    results.sort(key=lambda r: args.candidates.index(r["name"]))
    atomic_json(
        args.output_dir / "results.json", {"plan": plan, "experiments": results}
    )
    print(f"Completed {len(results)} shadow candidates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
