#!/usr/bin/env python3
"""Localized examples at label-independent v1 proposals, with frozen audio context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dataset import label_for, thread_group
from event_annotations import load_annotations, validate_intervals
from extract_audio_context import sha256
from extract_features import (
    RATE,
    WINDOW,
    WINDOW_S,
    Spectra,
    event_features,
    percentile,
    read_audio,
    timelines,
)
from train_models import atomic_json

VERSION = 1
CORE_SECONDS = 0.3


def proposal_points(features: dict, sample_count: int) -> list[tuple[int, int]]:
    """Slots 0..2 are existing transition candidates; slot 3 is the opening."""
    last = max(1, sample_count // WINDOW - 1)
    points = []
    for slot in range(3):
        fraction = features.get(f"event_{slot + 1}_position_fraction")
        if fraction is not None:
            points.append((slot, round(fraction * last)))
    points.append((3, round(features["start_position_s"] / WINDOW_S)))
    return points


def physical_events(
    samples: np.ndarray, features: dict
) -> tuple[list[dict], list[dict]]:
    levels, fine, stats = timelines(samples)
    spectra = Spectra(samples)
    vectors, windows = [], []
    for slot, index in proposal_points(features, len(samples)):
        guard = max(0, index - 2)
        history = levels[max(0, index - 60) : guard]
        baseline = percentile(history, 50)
        event = percentile(levels[index : index + 6], 25)
        candidate = {
            "index": index,
            "baseline": baseline,
            "event": event,
            "jump": event - baseline,
            "mad": percentile(np.abs(history - baseline), 50) if len(history) else 0,
        }
        vector = event_features(candidate, levels, fine, stats["raw_clip"], spectra)
        vector.pop("position_fraction")
        vector["is_opening"] = float(slot == 3)
        if not len(history):
            for name in vector:
                if (
                    "baseline" in name
                    or "jump" in name
                    or name in ("robust_z", "spectral_shape_distance")
                ):
                    vector[name] = np.nan
        vectors.append(vector)
        windows.append(
            {
                "slot": slot,
                "start_s": index * WINDOW_S,
                "end_s": min(len(samples) / RATE, index * WINDOW_S + CORE_SECONDS),
            }
        )
    return vectors, windows


def window_label(
    clip_label: str, annotation: dict | None, start: float, end: float, duration: float
) -> int:
    """-1 is unlabeled/uncertain, never a fabricated negative."""
    if clip_label == "negative":
        return 0
    if clip_label != "positive" or annotation is None:
        return -1
    intervals = validate_intervals(annotation["intervals"], duration)
    error = annotation.get("estimated_boundary_error_s", 0.05)
    for interval in intervals:
        # An explicit EOF has no visual measurement uncertainty.
        end_error = 0 if abs(interval["end_s"] - duration) < 1 / RATE else error
        if (
            start >= interval["start_s"] + error - 1e-9
            and end <= interval["end_s"] - end_error + 1e-9
        ):
            return 1
    # Physical features examine up to six seconds before/after the proposal.
    # Do not label a setup window negative while its context contains a screamer.
    if all(
        end + 6 <= r["start_s"] - error or start - 6 >= r["end_s"] + error
        for r in intervals
    ):
        return 0
    return -1


def grouped_rows(rows: list[dict], labels: dict, annotations: dict) -> list[dict]:
    groups = {}
    for row in rows:
        if label_for(row["path"], labels) != "visual-only":
            groups.setdefault(row["audio_sha256"], []).append(row)
    result = []
    for digest, members in groups.items():
        known = {label_for(r["path"], labels) for r in members} - {"unlabeled"}
        if len(known) > 1:
            raise ValueError(f"conflicting labels for identical audio: {digest}")
        annotated = [r for r in members if r["path"] in annotations["events"]]
        if annotated:
            first = annotations["events"][annotated[0]["path"]]["intervals"]
            if any(
                annotations["events"][r["path"]]["intervals"] != first
                for r in annotated
            ):
                raise ValueError(f"conflicting timings for identical audio: {digest}")
        representative = min(annotated or members, key=lambda r: r["path"])
        threads = sorted({thread_group(r["path"]) for r in members})
        result.append(
            representative
            | {
                "label": next(iter(known), "unlabeled"),
                "threads": threads,
                "fold_thread": threads[0],
                "paths": sorted(r["path"] for r in members),
            }
        )
    return sorted(result, key=lambda r: r["path"])


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    parser.add_argument(
        "--context-dir", type=Path, default=Path("research/artifacts/audio-context-v1")
    )
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--events", type=Path, default=Path("corpus/events.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    args = parser.parse_args()
    features = json.loads(args.features.read_text())
    context = json.loads((args.context_dir / "manifest.json").read_text())
    if (
        features["feature_version"] != 1
        or context["schema"] != 1
        or context["feature_dataset_sha256"] != sha256(args.features)
    ):
        raise ValueError("missing/stale v1 features or context; extract them first")
    if context["matrix_sha256"] != sha256(args.context_dir / "embeddings.npz"):
        raise ValueError("context matrix hash mismatch")
    with np.load(args.context_dir / "embeddings.npz", allow_pickle=False) as saved:
        context_vectors = saved["embeddings"].reshape(-1, 10, 1024)
    context_by_hash = {
        row["audio_sha256"]: context_vectors[i] for i, row in enumerate(context["rows"])
    }
    labels = json.loads(args.labels.read_text())
    annotations = load_annotations(args.events, labels)
    rows = grouped_rows(features["rows"], labels, annotations)
    cache = args.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    extractor_hash = sha256(Path(__file__)) + sha256(
        Path(__file__).with_name("extract_features.py")
    )
    all_physical, all_audio, targets, window_rows, clips = [], [], [], [], []
    names = None
    for number, row in enumerate(rows):
        source = args.audio_dir / f"{row['file']}.audio.wav"
        # Validate even a cached entry against the actual retained audio.
        if sha256(source) != row["audio_sha256"]:
            raise ValueError(f"audio changed: {source}")
        key = hashlib.sha256(
            json.dumps(
                [extractor_hash, row["audio_sha256"], row["features"]], sort_keys=True
            ).encode()
        ).hexdigest()
        target = cache / f"{key}.json"
        if target.exists():
            data = json.loads(target.read_text())
            vectors, windows = data["vectors"], data["windows"]
        else:
            vectors, windows = physical_events(read_audio(source), row["features"])
            # JSON null represents missing physical features, not nonstandard NaN.
            vectors = [
                {k: float(v) if np.isfinite(v) else None for k, v in r.items()}
                for r in vectors
            ]
            atomic_json(target, {"vectors": vectors, "windows": windows})
        names = names or sorted(vectors[0])
        annotation = annotations["events"].get(row["path"])
        if annotation and annotation["audio_sha256"] != row["audio_sha256"]:
            raise ValueError(f"stale timing annotation: {row['path']}")
        first = len(window_rows)
        for vector, window in zip(vectors, windows, strict=True):
            if sorted(vector) != names:
                raise ValueError("inconsistent feature schema")
            all_physical.append(
                [np.nan if vector[name] is None else vector[name] for name in names]
            )
            slots = context_by_hash[row["audio_sha256"]]
            audio = (
                slots[window["slot"] * 3 : window["slot"] * 3 + 3]
                if window["slot"] < 3
                else np.stack([np.zeros(1024), slots[9], np.zeros(1024)])
            )
            all_audio.append(audio.reshape(-1))
            targets.append(
                window_label(
                    row["label"],
                    annotation,
                    window["start_s"],
                    window["end_s"],
                    row["features"]["duration_s"],
                )
            )
            window_rows.append(window | {"clip_index": number})
        clips.append(
            {
                k: row[k]
                for k in (
                    "path",
                    "file",
                    "audio_sha256",
                    "paths",
                    "label",
                    "threads",
                    "fold_thread",
                )
            }
            | {
                "first_window": first,
                "window_count": len(windows),
                "annotation": annotation,
                "duration_s": row["features"]["duration_s"],
            }
        )
        if (number + 1) % 100 == 0 or number + 1 == len(rows):
            print(f"event examples {number + 1}/{len(rows)}", flush=True)
    matrix = args.output_dir / "examples.npz"
    atomic_npz(
        matrix,
        physical=np.asarray(all_physical, np.float64),
        embeddings=np.asarray(all_audio, np.float32),
        targets=np.asarray(targets, np.int8),
    )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": VERSION,
            "physical_names": names,
            "clips": clips,
            "windows": window_rows,
            "matrix_sha256": sha256(matrix),
            "labels_sha256": sha256(args.labels),
            "events_sha256": sha256(args.events),
            "feature_dataset_sha256": sha256(args.features),
            "context_manifest_sha256": sha256(args.context_dir / "manifest.json"),
            "annotation_usage": "Targets only. All proposals and features are label-independent; missing timings never create negative windows.",
            "proposal_policy": "Existing top three whole-recording transition candidates plus the opening; not an exhaustive sliding-window classifier.",
        },
    )
    print(
        f"{len(clips)} clips; {len(targets)} windows; positive={targets.count(1)}, negative={targets.count(0)}, uncertain/unlabeled={targets.count(-1)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
