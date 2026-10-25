#!/usr/bin/env python3
"""Frozen rule/MIL union and a separately labeled exploratory replay.

No thresholds are fitted. Any malformed/missing component aborts scoring instead
of silently producing a safe result. Neither operation changes browser alerts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from improve_event_semantics import ALL_CANDIDATES
from risk_score import risk_tier
from train_events import metrics
from train_models import atomic_json

SCRIPT_SHA256 = "1c51b0ac4cd552932a0e6af90c59faac79072492c9d97e4459dde2597305a47b"


def valid_score(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("invalid component score; cannot infer safe")
    return float(value)


def heuristic_scores(payload, provenance):
    if payload["current_sha256"] != SCRIPT_SHA256:
        raise ValueError("frozen production reference changed")
    if (
        payload["features_sha256"] != provenance["feature_dataset_sha256"]
        or payload["labels_sha256"] != provenance["labels_sha256"]
    ):
        raise ValueError("component corpus snapshots differ")
    result = {}
    for row in payload["rows"]:
        if row["after"]["status"] != "ok":
            raise ValueError("failed heuristic analysis; cannot infer safe")
        digest = row["audio_sha256"]
        if digest in result:
            raise ValueError("duplicate heuristic audio group")
        result[digest] = valid_score(row["after"]["score"])
    return result


def fuse(rows, heuristic):
    result = []
    for row in rows:
        if row["audio_sha256"] not in heuristic:
            raise ValueError("missing heuristic result; cannot infer safe")
        ml, rules = (
            valid_score(row["score"]),
            valid_score(heuristic[row["audio_sha256"]]),
        )
        score = max(ml, rules)
        result.append(
            row
            | {
                "score": score,
                "ml_score": ml,
                "rule_score": rules,
                "risk_tier": risk_tier(score),
            }
        )
    return result


def score(model_scores, heuristic_path, policy_path, output):
    policy = json.loads(policy_path.read_text())
    ml = json.loads(model_scores.read_text())
    heuristic = json.loads(heuristic_path.read_text())
    if (
        policy["schema"] != 1
        or policy["operation"] != "maximum"
        or policy["userscript_sha256"] != SCRIPT_SHA256
        or policy["yellow_threshold"] != 0.6
        or policy["red_threshold"] != 0.8
        or policy["candidate"] not in ALL_CANDIDATES
    ):
        raise ValueError("unsupported fusion policy")
    if ml["snapshot_manifest_sha256"] != policy["model_snapshot_manifest_sha256"]:
        raise ValueError("model snapshot changed")
    rules = heuristic_scores(heuristic, ml["provenance"])
    rows = fuse(ml["predictions"][policy["candidate"]], rules)
    atomic_json(
        output,
        {
            "policy_sha256": sha256(policy_path),
            "model_scores_sha256": sha256(model_scores),
            "heuristic_scores_sha256": sha256(heuristic_path),
            "warning": "Frozen shadow fusion. Seen-in-training rows are fit diagnostics, not validation. Score is uncalibrated.",
            "metrics": metrics(rows),
            "rows": rows,
        },
    )


def evaluate(run, heuristic_path, audit_path, browser_directory, output):
    plan = json.loads((run / "plan.json").read_text())
    result = json.loads((run / "results.json").read_text())
    heuristic = json.loads(heuristic_path.read_text())
    rules = heuristic_scores(heuristic, plan["provenance"])
    predictions = {
        name: fuse(result["predictions"][name], rules) for name in ALL_CANDIDATES
    }
    audit = json.loads(audit_path.read_text())
    by_file = {
        Path(p).name: row["audio_sha256"]
        for row in heuristic["rows"]
        for p in row["paths"]
    }
    browser_rules = {}
    # Only the previously saved, verified original-media fixtures are used.
    browser_manifest = browser_directory / "manifest.json"
    manifest = json.loads(browser_manifest.read_text())
    if sha256(browser_manifest) != audit["browser_manifest_sha256"]:
        raise ValueError("browser audit changed")
    for evidence in manifest["evidence"]:
        replay_path = Path(evidence["replay"])
        if sha256(replay_path) != evidence["replay_sha256"]:
            raise ValueError("browser replay changed")
        replay = json.loads(replay_path.read_text())
        if (
            replay["current_sha256"] != SCRIPT_SHA256
            or replay["after"]["status"] != "ok"
        ):
            raise ValueError("invalid original-media heuristic fixture")
        browser_rules[by_file[Path(replay["source"]).name]] = valid_score(
            replay["after"]["score"]
        )
    browser = {
        name: fuse(audit["browser_predictions"][name], browser_rules)
        for name in ALL_CANDIDATES
    }
    candidate = "mil-linear-softmax"
    rows = predictions[candidate]
    baseline = {
        row["audio_sha256"]: row for row in result["predictions"]["event-hybrid"]
    }
    changes = [
        row | {"baseline_score": baseline[row["audio_sha256"]]["score"]}
        for row in rows
        if row["label"] == "positive"
        and risk_tier(row["score"]) != risk_tier(baseline[row["audio_sha256"]]["score"])
    ]
    artifact = {
        "status": "exploratory-shadow-only",
        "source_sha256": sha256(Path(__file__)),
        "design": "Post hoc fixed maximum of each ML candidate and unchanged v5.8.2 rules, at .6/.8. No new fitting or threshold optimization. All 13 combinations are reported.",
        "limitation": "ML scores exclude each source thread; handcrafted rules were developed using part of this corpus. These are NOT fully held-out ensemble results. Fusion was investigated after the ML experiments; prospective confirmation required.",
        "source_results_sha256": sha256(run / "results.json"),
        "heuristic_scores_sha256": sha256(heuristic_path),
        "audit_sha256": sha256(audit_path),
        "metrics": {name: metrics(r) for name, r in predictions.items()},
        "browser_metrics": {name: metrics(r) for name, r in browser.items()},
        "candidate": candidate,
        "positive_tier_changes_from_event_hybrid": changes,
        "candidate_held_out_ml_plus_fixed_rules_predictions": rows,
        "candidate_browser_predictions": browser[candidate],
        "folds": [
            {
                "thread": thread,
                "metrics": metrics([r for r in rows if r["fold_thread"] == thread]),
            }
            for thread in sorted({r["fold_thread"] for r in rows})
        ],
    }
    atomic_json(output, artifact)
    print(
        json.dumps(
            {
                "candidate": candidate,
                "metrics": artifact["metrics"][candidate],
                "browser": artifact["browser_metrics"][candidate],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "evaluate"))
    parser.add_argument("--model-scores", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--heuristic-scores", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("research/models/rules-mil-20261025/policy.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite fusion scores")
    if args.command == "score":
        if not args.model_scores:
            parser.error("score requires --model-scores")
        score(args.model_scores, args.heuristic_scores, args.policy, args.output)
    else:
        if not args.run or not args.audit or not args.browser:
            parser.error("evaluate requires --run, --audit and --browser")
        evaluate(args.run, args.heuristic_scores, args.audit, args.browser, args.output)


if __name__ == "__main__":
    main()
