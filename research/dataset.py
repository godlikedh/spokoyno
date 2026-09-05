"""Corpus labels and grouping shared by feature extraction and model training."""

from __future__ import annotations

import json
from pathlib import Path


def load_labels(path: Path) -> dict:
    return json.loads(path.read_text())


def label_for(media_path: str, labels: dict) -> str:
    if media_path in labels.get("confirmed_positives", {}):
        return "positive"
    if media_path in labels.get("confirmed_visual_only_screamers", {}):
        return "visual-only"
    if media_path in labels.get("reviewed_negatives", {}):
        return "negative"

    original = labels.get("provisional_negative_set", {})
    thread = str(original.get("thread", ""))
    if thread and media_path.startswith(f"/b/src/{thread}/"):
        excluded = set(original.get("exclude_confirmed_positive_files", []))
        return "unlabeled" if Path(media_path).name in excluded else "negative"

    for prefix, spec in labels.get("additional_provisional_negative_sets", {}).items():
        if media_path.startswith(prefix):
            excluded = set(spec.get("exclude_confirmed_screamer_files", []))
            return "unlabeled" if Path(media_path).name in excluded else "negative"

    for prefix, spec in labels.get("unlabeled_batches", {}).items():
        if media_path.startswith(prefix):
            excluded = set(spec.get("exclude_confirmed_positive_files", []))
            return "unlabeled" if Path(media_path).name not in excluded else "positive"
    return "unlabeled"


def thread_group(media_path: str) -> str:
    parts = media_path.strip("/").split("/")
    return parts[2] if len(parts) >= 4 and parts[1] == "src" else "unknown"
