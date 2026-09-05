#!/usr/bin/env python3
"""List or record user-confirmed screamer intervals without changing clip labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from extract_features import RATE, read_audio
from train_models import atomic_json


def validate_intervals(intervals: list, duration: float | None = None) -> list[dict]:
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("provide at least one start/end interval")
    result = []
    for interval in intervals:
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"invalid interval: {interval}")
        if duration is not None and end > duration + 0.001:
            raise ValueError(
                f"interval ends at {end}s, beyond audio end {duration:.3f}s"
            )
        result.append({"start_s": start, "end_s": end})
    result.sort(key=lambda item: item["start_s"])
    if any(a["end_s"] > b["start_s"] for a, b in zip(result, result[1:], strict=False)):
        raise ValueError("intervals must not overlap")
    return result


def load_annotations(path: Path, labels: dict) -> dict:
    if not path.exists():
        return {"schema": 1, "events": {}}
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError("unsupported event annotation schema")
    for media_path, entry in payload["events"].items():
        if media_path not in labels.get("confirmed_positives", {}):
            raise ValueError(
                f"timing annotation is not an audio positive: {media_path}"
            )
        if entry.get("source") != "user":
            raise ValueError(f"timings must be user-confirmed: {media_path}")
        validate_intervals(entry["intervals"])
        uncertainty = entry.get("estimated_boundary_error_s", 0)
        if (
            not isinstance(uncertainty, (int, float))
            or not math.isfinite(uncertainty)
            or uncertainty < 0
        ):
            raise ValueError(f"invalid estimated boundary error: {media_path}")
    return payload


def candidate_coverage(annotations: dict, rows: list[dict]) -> list[dict]:
    """Report proposal localization; never use annotations to select proposals."""
    by_path = {row["path"]: row for row in rows}
    results = []
    for media_path, entry in annotations["events"].items():
        row = by_path.get(media_path)
        if row is None or row["audio_sha256"] != entry["audio_sha256"]:
            raise ValueError(f"features missing or stale for annotation: {media_path}")
        features = row["features"]
        last_window = max(1, int(features["duration_s"] * RATE) // 800 - 1)
        onsets = [float(features["start_position_s"])]
        for number in range(1, 4):
            position = features.get(f"event_{number}_position_fraction")
            if position is not None:
                onsets.append(position * last_window * 0.05)
        for interval in entry["intervals"]:
            nearest = min(onsets, key=lambda at: abs(at - interval["start_s"]))
            error = abs(nearest - interval["start_s"])
            results.append(
                {
                    "path": media_path,
                    **interval,
                    "nearest_candidate_s": nearest,
                    "onset_error_s": error,
                    "within_250ms": error <= 0.25
                    or math.isclose(error, 0.25, rel_tol=0, abs_tol=1e-9),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("corpus/index.json"))
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--events", type=Path, default=Path("corpus/events.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument("--set", dest="media", help="canonical path or unique filename")
    parser.add_argument(
        "--interval",
        action="append",
        help="START:END seconds; END can be 'end'; repeat",
    )
    parser.add_argument(
        "--uncertainty",
        type=float,
        help="estimated plus/minus boundary error in seconds",
    )
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="compare confirmed onsets with existing feature candidates",
    )
    parser.add_argument(
        "--features", type=Path, default=Path("research/artifacts/features-v1.json")
    )
    args = parser.parse_args()
    if args.uncertainty is not None and (
        not args.media or not math.isfinite(args.uncertainty) or args.uncertainty < 0
    ):
        parser.error("--uncertainty requires --set and a finite nonnegative value")
    labels = json.loads(args.labels.read_text())
    payload = load_annotations(args.events, labels)
    items = json.loads(args.index.read_text())["items"]
    if args.media:
        matches = [row for row in items if args.media in (row["path"], row["file"])]
        if len(matches) != 1 or not args.interval:
            parser.error("provide a unique media path and at least one --interval")
        row = matches[0]
        if row["path"] not in labels["confirmed_positives"]:
            parser.error("only confirmed audio positives can receive screamer timings")
        duration = len(read_audio(args.audio_dir / row["audio_file"])) / RATE
        intervals = []
        for value in args.interval:
            start, end = value.split(":")
            intervals.append(
                {
                    "start_s": float(start),
                    "end_s": duration if end.lower() == "end" else float(end),
                }
            )
        previous = payload["events"].get(row["path"], {})
        payload["events"][row["path"]] = {
            "source": "user",
            "audio_sha256": row["audio_sha256"],
            "intervals": validate_intervals(intervals, duration),
            "note": args.note,
        }
        uncertainty = (
            args.uncertainty
            if args.uncertainty is not None
            else previous.get("estimated_boundary_error_s")
        )
        if uncertainty is not None:
            payload["events"][row["path"]]["estimated_boundary_error_s"] = uncertainty
        atomic_json(args.events, payload)
    elif args.interval:
        parser.error("--interval requires --set")
    for row in items:
        if row["path"] not in labels["confirmed_positives"]:
            continue
        entry = payload["events"].get(row["path"])
        times = (
            ", ".join(f"{r['start_s']:g}–{r['end_s']:g}s" for r in entry["intervals"])
            if entry
            else "timing pending"
        )
        print(f"{row['file']}: {times}")
    if args.coverage:
        coverage = candidate_coverage(
            payload, json.loads(args.features.read_text())["rows"]
        )
        print(
            json.dumps(
                {
                    "annotated_events": len(coverage),
                    "onsets_within_250ms": sum(row["within_250ms"] for row in coverage),
                    "rows": coverage,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
