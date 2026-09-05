#!/usr/bin/env python3
"""Local landmark matching for known positive clips and annotated screamer events.

Inspired by Wang's ISMIR 2003 landmark fingerprinting approach. This is a small
research implementation, not Shazam, and its matches never change the userscript.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dataset import label_for
from event_annotations import load_annotations, validate_intervals
from extract_audio_context import sha256
from extract_features import RATE, read_audio
from scipy.ndimage import maximum_filter
from scipy.signal import stft
from train_models import atomic_json

FINGERPRINT_VERSION = 1
HOP = 160
PARAMETERS = {
    "sample_rate": RATE,
    "hop_samples": HOP,
    "window_samples": 512,
    "fft_size": 1024,
    "frequency_quantization_bins": 3,
    "delta_quantization_frames": 2,
    "fanout": 8,
    "target_min_frames": 3,
    "target_max_frames": 60,
    "minimum_anchors": 6,
    "minimum_hashes": 8,
    "minimum_frequency_bins": 3,
    "minimum_anchor_fraction": 0.25,
    "offset_tolerance_frames": 3,
}


def mono_channel(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    energy = np.mean(np.square(samples[::16], dtype=np.float64), axis=0)
    return samples[:, int(np.argmax(energy))]


def landmarks(samples: np.ndarray) -> np.ndarray:
    """Return (hash, anchor frame) pairs with bounded spectrogram memory."""
    mono = mono_channel(samples)
    pairs = set()
    frames = max(0, (len(mono) - 512) // HOP + 1)
    for core in range(0, frames, 2000):
        first = max(0, core - 8)
        last = min(frames, core + 2000 + 70)
        chunk = mono[first * HOP : (last - 1) * HOP + 512]
        if len(chunk) < 512:
            continue
        _, _, spectrum = stft(
            chunk,
            fs=RATE,
            nperseg=512,
            noverlap=512 - HOP,
            nfft=1024,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(spectrum)
        levels = 20 * np.log10(np.maximum(magnitude, 1e-10))
        local = maximum_filter(levels, size=(15, 9), mode="constant", cval=-200)
        selected = (levels == local) & (
            levels >= np.maximum(levels.max(axis=0) - 30, -100)
        )
        selected[:7] = False
        selected[481:] = False
        frequencies, times = np.nonzero(selected)
        peaks = sorted(
            zip(
                times.tolist(),
                frequencies.tolist(),
                levels[frequencies, times].tolist(),
                strict=True,
            )
        )
        peak_times = np.asarray([peak[0] for peak in peaks])
        for t1, f1, _ in peaks:
            absolute = t1 + first
            if not core <= absolute < core + 2000:
                continue
            lo, hi = np.searchsorted(peak_times, [t1 + 3, t1 + 61])
            targets = sorted(peaks[lo:hi], key=lambda peak: peak[2], reverse=True)[:8]
            for t2, f2, _ in targets:
                token = ((f1 // 3) << 18) | ((f2 // 3) << 9) | ((t2 - t1) // 2)
                pairs.add((token, absolute))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


class Matcher:
    def __init__(self, references: list[dict]):
        self.references = references
        self.postings = defaultdict(list)
        self.anchor_counts = []
        for number, reference in enumerate(references):
            pairs = reference["pairs"]
            self.anchor_counts.append(len({t for _, t in pairs}))
            for token, frame in pairs:
                self.postings[token].append((number, frame))
        # Highly repetitive landmarks are weak identity evidence.
        self.postings = {
            token: values
            for token, values in self.postings.items()
            if len(values) <= 100
        }

    def match(
        self,
        pairs: np.ndarray,
        exclude_audio_hash: str | None = None,
        query_audio_hash: str | None = None,
    ) -> list[dict]:
        if query_audio_hash and query_audio_hash != exclude_audio_hash:
            exact = [
                r for r in self.references if r["audio_sha256"] == query_audio_hash
            ]
            if exact:
                return [
                    {
                        "reference_path": r["path"],
                        "reference_audio_sha256": query_audio_hash,
                        "reference_scope": r["scope"],
                        "reference_start_s": r["start_s"],
                        "query_match_start_s": r["start_s"],
                        "query_match_end_s": r["start_s"] + r["duration_s"],
                        "match_type": "exact-audio",
                        "anchor_fraction": 1.0,
                        "confirmed_event_match": r["scope"] == "user-annotated-event",
                    }
                    for r in exact
                ]
        matches = defaultdict(set)
        for token, query_frame in pairs:
            token, query_frame = int(token), int(query_frame)
            for delta in (-1, 0, 1):
                if not 0 <= (token & 511) + delta < 512:
                    continue
                for number, reference_frame in self.postings.get(token + delta, []):
                    reference = self.references[number]
                    if reference["audio_sha256"] != exclude_audio_hash:
                        matches[number].add(
                            (reference_frame, query_frame, token + delta)
                        )
        found = []
        for number, aligned in matches.items():
            reference = self.references[number]
            histogram = Counter(round((q - r) / 3) for r, q, _ in aligned)
            best = None
            for bucket, _ in histogram.most_common(3):
                cluster = [
                    (r, q, token)
                    for r, q, token in aligned
                    if abs(q - r - bucket * 3) <= 3
                ]
                anchor_times = {r for r, _, _ in cluster}
                hashes = {token for _, _, token in cluster}
                frequencies = {token >> 18 for token in hashes}
                count = len(anchor_times)
                fraction = count / max(1, self.anchor_counts[number])
                span = (
                    (max(anchor_times) - min(anchor_times)) * HOP / RATE if count else 0
                )
                required_span = min(0.15, reference["duration_s"] * 0.3)
                accepted = (
                    count >= PARAMETERS["minimum_anchors"]
                    and len(hashes) >= PARAMETERS["minimum_hashes"]
                    and len(frequencies) >= PARAMETERS["minimum_frequency_bins"]
                    and fraction >= PARAMETERS["minimum_anchor_fraction"]
                    and span >= required_span
                )
                if not accepted:
                    continue
                offset = float(np.median([q - r for r, q, _ in cluster])) * HOP / RATE
                result = {
                    "match_type": "landmark",
                    "reference_path": reference["path"],
                    "reference_audio_sha256": reference["audio_sha256"],
                    "reference_scope": reference["scope"],
                    "reference_start_s": reference["start_s"],
                    "query_match_start_s": offset,
                    "query_match_end_s": offset + reference["duration_s"],
                    "matched_anchors": count,
                    "anchor_fraction": fraction,
                    "matched_span_s": span,
                    "confirmed_event_match": reference["scope"]
                    == "user-annotated-event",
                }
                if best is None or count > best["matched_anchors"]:
                    best = result
            if best:
                found.append(best)
        return sorted(
            found,
            key=lambda row: (
                row["confirmed_event_match"],
                row["anchor_fraction"],
                row["matched_anchors"],
            ),
            reverse=True,
        )


def cached_pairs(row: dict, audio_dir: Path, cache_dir: Path) -> np.ndarray:
    target = cache_dir / f"v{FINGERPRINT_VERSION}-{row['audio_sha256']}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as source:
            return source["pairs"]
    source = audio_dir / row["audio_file"]
    if sha256(source) != row["audio_sha256"]:
        raise ValueError(f"audio hash mismatch: {source}")
    pairs = landmarks(read_audio(source))
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as out:
        np.savez_compressed(out, pairs=pairs)
    os.replace(temporary, target)
    return pairs


def build_references(
    items: list[dict], labels: dict, annotations: dict, audio_dir: Path, cache_dir: Path
) -> list[dict]:
    references, seen = [], set()
    for row in items:
        if label_for(row["path"], labels) != "positive" or row["audio_sha256"] in seen:
            continue
        seen.add(row["audio_sha256"])
        entry = annotations["events"].get(row["path"])
        if sha256(audio_dir / row["audio_file"]) != row["audio_sha256"]:
            raise ValueError(f"reference audio has changed: {row['path']}")
        samples = read_audio(audio_dir / row["audio_file"])
        duration = len(samples) / RATE
        if entry:
            if entry["audio_sha256"] != row["audio_sha256"]:
                raise ValueError(f"annotated audio has changed: {row['path']}")
            intervals = validate_intervals(entry["intervals"], duration)
        else:
            intervals = [{"start_s": 0.0, "end_s": duration}]
        for interval in intervals:
            start, end = interval["start_s"], interval["end_s"]
            pairs = (
                landmarks(samples[round(start * RATE) : round(end * RATE)])
                if entry
                else cached_pairs(row, audio_dir, cache_dir)
            )
            references.append(
                {
                    "path": row["path"],
                    "audio_sha256": row["audio_sha256"],
                    "scope": "user-annotated-event" if entry else "whole-positive-clip",
                    "start_s": start,
                    "duration_s": end - start,
                    "pairs": pairs.tolist(),
                }
            )
    return references


def aac_roundtrip(mono: np.ndarray) -> np.ndarray:
    samples = np.asarray(mono, dtype=np.float32)
    if samples.ndim not in (1, 2):
        raise ValueError("AAC roundtrip expects mono/stereo audio")
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    if channels not in (1, 2):
        raise ValueError("AAC roundtrip expects mono/stereo audio")
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-f",
            "adts",
            "pipe:1",
        ],
        input=samples.tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "aac",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "pipe:1",
        ],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    result = np.frombuffer(decoded, dtype=np.float32)
    return result if samples.ndim == 1 else result.reshape(-1, channels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "score", "evaluate"))
    parser.add_argument("--index", type=Path, default=Path("corpus/index.json"))
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument("--events", type=Path, default=Path("corpus/events.json"))
    parser.add_argument(
        "--file",
        action="append",
        dest="query_files",
        help="score only this filename/path; repeat for more",
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/artifacts/fingerprints")
    )
    parser.add_argument("--cache-dir", type=Path, help="reuse existing landmark cache")
    args = parser.parse_args()
    if args.query_files and args.command != "score":
        parser.error("--file is only available for score")
    items = [
        row
        for row in json.loads(args.index.read_text())["items"]
        if row["status"] == "ok"
    ]
    labels = json.loads(args.labels.read_text())
    cache = args.cache_dir or args.output_dir / "cache"
    reference_path = args.output_dir / "references.json"
    if args.command in ("build", "evaluate"):
        annotations = load_annotations(args.events, labels)
        references = build_references(items, labels, annotations, args.audio_dir, cache)
        atomic_json(
            reference_path,
            {
                "schema": FINGERPRINT_VERSION,
                "parameters": PARAMETERS,
                "labels_sha256": sha256(args.labels),
                "events_sha256": sha256(args.events),
                "references": references,
            },
        )
        print(
            f"built {len(references)} references ({sum(r['scope'] == 'user-annotated-event' for r in references)} user-annotated intervals)",
            flush=True,
        )
        if args.command == "build":
            return 0
    artifact = json.loads(reference_path.read_text())
    if (
        artifact["schema"] != FINGERPRINT_VERSION
        or artifact["parameters"] != PARAMETERS
    ):
        raise ValueError("fingerprint configuration changed; rebuild references")
    matcher = Matcher(artifact["references"])
    if args.query_files:
        missing = set(args.query_files) - {
            value for row in items for value in (row["file"], row["path"])
        }
        if missing:
            parser.error(f"unknown audio files: {sorted(missing)}")
        items = [
            row
            for row in items
            if row["file"] in args.query_files or row["path"] in args.query_files
        ]
    results, seen, scored = [], set(), {}
    for row in items:
        label = label_for(row["path"], labels)
        if label == "visual-only" or (
            args.command == "evaluate" and row["audio_sha256"] in seen
        ):
            continue
        seen.add(row["audio_sha256"])
        if row["audio_sha256"] not in scored:
            pairs = cached_pairs(row, args.audio_dir, cache)
            scored[row["audio_sha256"]] = matcher.match(
                pairs,
                exclude_audio_hash=row["audio_sha256"]
                if args.command == "evaluate"
                else None,
                query_audio_hash=row["audio_sha256"]
                if args.command == "score"
                else None,
            )
        matches = scored[row["audio_sha256"]]
        results.append(
            {
                "path": row["path"],
                "audio_sha256": row["audio_sha256"],
                "label_at_scoring": label,
                "matches": matches,
            }
        )
        if len(results) % 100 == 0:
            print(f"fingerprints {len(results)} clips", flush=True)
    payload = {
        "schema": 1,
        "status": "shadow-only",
        "reference_index_sha256": sha256(reference_path),
        "warning": "Partial whole-positive-clip matches are identity hints, not confirmed screamer events.",
        "exact_sources_excluded": args.command == "evaluate",
        "rows": results,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = args.output_dir / f"scores-{stamp}.json"
    atomic_json(snapshot, payload)
    print(f"wrote {snapshot}", flush=True)
    if args.command == "evaluate":
        controlled = []
        variant_names = ("same-recording", "prepend-25ms", "gain-minus-3db", "aac-96k")
        for row in items:
            if label_for(row["path"], labels) != "positive":
                continue
            mono = mono_channel(read_audio(args.audio_dir / row["audio_file"]))
            variants = {
                "same-recording": mono,
                "prepend-25ms": np.pad(mono, (400, 0)),
                "gain-minus-3db": mono * 10 ** (-3 / 20),
                "aac-96k": aac_roundtrip(mono),
            }
            for name, samples in variants.items():
                matches = matcher.match(landmarks(samples))
                controlled.append(
                    {
                        "path": row["path"],
                        "variant": name,
                        "recovered_reference": any(
                            m["reference_audio_sha256"] == row["audio_sha256"]
                            for m in matches
                        ),
                        "matches": matches,
                    }
                )
        negatives = [r for r in results if r["label_at_scoring"] == "negative"]
        positives = [r for r in results if r["label_at_scoring"] == "positive"]
        summary = {
            "negative": len(negatives),
            "negative_with_identity_match": sum(bool(r["matches"]) for r in negatives),
            "negative_with_confirmed_event_match": sum(
                any(m["confirmed_event_match"] for m in r["matches"]) for r in negatives
            ),
            "positive": len(positives),
            "positive_matching_other_reference": sum(
                bool(r["matches"]) for r in positives
            ),
            "annotated_references": sum(
                r["scope"] == "user-annotated-event" for r in artifact["references"]
            ),
            "known_reference_robustness": {
                name: {
                    "recovered": sum(
                        r["recovered_reference"]
                        for r in controlled
                        if r["variant"] == name
                    ),
                    "total": sum(r["variant"] == name for r in controlled),
                }
                for name in variant_names
            },
        }
        atomic_json(
            args.output_dir / "evaluation.json",
            payload | {"summary": summary, "controlled_variants": controlled},
        )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
