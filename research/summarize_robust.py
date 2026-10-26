#!/usr/bin/env python3
"""Paired source-level codec changes and comparisons with the previous best ML."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_event_research import paired_cluster_interval
from extract_audio_context import sha256
from train_events import metrics
from train_models import atomic_json


def tier(score):
    return 2 if score >= 0.8 else 1 if score >= 0.6 else 0


def paired(original, transformed):
    by_hash = {r["audio_sha256"]: r for r in transformed}
    result = {}
    for label in ("positive", "negative"):
        pairs = [
            (r, by_hash[r["audio_sha256"]]) for r in original if r["label"] == label
        ]
        delta = np.asarray([abs(a["score"] - b["score"]) for a, b in pairs])
        result[label] = {
            "sources": len(pairs),
            "absolute_score_change_median": float(np.median(delta)),
            "absolute_score_change_p95": float(np.quantile(delta, 0.95)),
            "absolute_score_change_max": float(max(delta)),
            "changed_tier": sum(tier(a["score"]) != tier(b["score"]) for a, b in pairs),
            "red_to_lower": sum(
                a["score"] >= 0.8 and b["score"] < 0.8 for a, b in pairs
            ),
            "warning_to_low": sum(
                a["score"] >= 0.6 and b["score"] < 0.6 for a, b in pairs
            ),
            "low_to_warning": sum(
                a["score"] < 0.6 and b["score"] >= 0.6 for a, b in pairs
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("research/artifacts/improve-augmentation-20261025/results.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a comparison")
    old = json.loads(args.baseline.read_text())["predictions"]["synthetic-mil-20pct"]
    ordered_old = sorted(old, key=lambda r: r["audio_sha256"])
    runs = {}
    for run in args.run:
        results = json.loads((run / "results.json").read_text())
        details = {}
        for name, views in results["predictions"].items():
            original = views["original"]
            encoded = {
                v: {r["audio_sha256"]: r for r in views[v]}
                for v in ("aac80", "opus48", "mp3_128")
            }
            worst = [
                r
                | {
                    "score": float(
                        (min if r["label"] == "positive" else max)(
                            [
                                r["score"],
                                *[
                                    encoded[v][r["audio_sha256"]]["score"]
                                    for v in encoded
                                ],
                            ]
                        )
                    )
                }
                for r in original
            ]
            details[name] = {
                "metrics": results["metrics"][name],
                "worst_original_or_evaluation_codec": metrics(worst),
                "paired_codec_changes": {
                    v: paired(original, views[v]) for v in encoded
                },
                "paired_thread_bootstrap_vs_previous_ml": paired_cluster_interval(
                    ordered_old, sorted(original, key=lambda r: r["audio_sha256"])
                ),
                "original_per_thread": {
                    thread: metrics([r for r in original if r["fold_thread"] == thread])
                    for thread in sorted({r["fold_thread"] for r in original})
                },
            }
        runs[str(run)] = {
            "results_sha256": sha256(run / "results.json"),
            "default_model": results["final_model"],
            "models": details,
        }
    atomic_json(
        args.output,
        {
            "baseline": {
                "name": "synthetic-mil-20pct",
                "metrics": metrics(old),
                "results_sha256": sha256(args.baseline),
            },
            "runs": runs,
            "limitations": "All codec copies are paired measurements of the same recordings. Cluster intervals describe existing development predictions and do not correct adaptive research or establish future-thread performance.",
        },
    )


if __name__ == "__main__":
    main()
