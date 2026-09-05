"""Apply an explicit whole-thread review to exactly the supplied attachment snapshot."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

from build_corpus import canonical_path, thread_parts


def reviewed_labels(
    labels: dict, thread_url: str, rows: list[dict], screamers: list[str]
) -> dict:
    board, thread, _ = thread_parts(thread_url)
    prefix = f"/{board}/src/{thread}/"
    if not rows or any(not row["path"].startswith(prefix) for row in rows):
        raise ValueError("review requires a nonempty snapshot belonging to this thread")
    positives = set()
    for value in screamers:
        matches = {
            row["path"]
            for row in rows
            if value == row["file"] or canonical_path(value) == row["path"]
        }
        if len(matches) != 1:
            raise ValueError(
                f"screamer must identify one attachment in this snapshot: {value}"
            )
        positives.update(matches)
    result = deepcopy(labels)
    for path in positives:
        if path in result.get("confirmed_visual_only_screamers", {}):
            raise ValueError(
                f"explicitly revise the existing visual-only label first: {path}"
            )
        result.setdefault("confirmed_positives", {}).setdefault(
            path, "User-confirmed audio screamer; timing may follow later"
        )
        result.get("reviewed_negatives", {}).pop(path, None)
    reviews = result.setdefault("reviewed_thread_sets", {})
    previous = reviews.get(prefix, {})
    paths = sorted(
        set(previous.get("reviewed_paths", [])) | {row["path"] for row in rows}
    )
    reviews[prefix] = {
        "source_thread": thread_url,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "reviewed_paths": paths,
        "note": "User reviewed these attachments. Every listed path without an explicit screamer label is negative. Later unseen attachments are not covered by this review. Existing explicit positives and visual-only exclusions are preserved.",
    }
    return result
