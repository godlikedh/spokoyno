#!/usr/bin/env python3
"""Audit nested research, baseline parity, browser PCM, and uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from event_research_artifacts import load_fit, predict_model
from extract_audio_context import sha256
from improve_events import display_scores, rows_for, training_key
from retrain_models import load_inputs
from train_events import metrics
from train_models import atomic_json


def paired_cluster_interval(baseline, candidate, seed=20261025):
    """Descriptive thread bootstrap of fixed OOF predictions, not new validation."""
    threads = sorted({row["fold_thread"] for row in baseline})
    assert [r["audio_sha256"] for r in baseline] == [
        r["audio_sha256"] for r in candidate
    ]
    counts = []
    for thread in threads:
        rows = [
            (a, b)
            for a, b in zip(baseline, candidate, strict=True)
            if a["fold_thread"] == thread
        ]
        positive = sum(a["label"] == "positive" for a, _ in rows)
        tp_delta = sum(
            int(b["score"] >= 0.8) - int(a["score"] >= 0.8)
            for a, b in rows
            if a["label"] == "positive"
        )
        fp_delta = sum(
            int(b["score"] >= 0.8) - int(a["score"] >= 0.8)
            for a, b in rows
            if a["label"] == "negative"
        )
        counts.append([positive, len(rows) - positive, tp_delta, fp_delta])
    counts = np.asarray(counts)
    draws = np.random.default_rng(seed).integers(
        len(threads), size=(10000, len(threads))
    )
    sums = counts[draws].sum(axis=1)
    sums = sums[(sums[:, 0] > 0) & (sums[:, 1] > 0)]
    return {
        "recall_delta_95_percentile": np.quantile(
            sums[:, 2] / sums[:, 0], [0.025, 0.975]
        ).tolist(),
        "fp_per_1000_delta_95_percentile": np.quantile(
            1000 * sums[:, 3] / sums[:, 1], [0.025, 0.975]
        ).tolist(),
        "warning": "Descriptive clustered resampling of this development corpus. Does not account for adaptive model design, selection, or unseen sound families.",
    }


def browser_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "examples.npz") != manifest["matrix_sha256"]:
        raise ValueError("browser feature matrix changed")
    for evidence in manifest["evidence"]:
        source = Path(evidence["replay"])
        if sha256(source) != evidence["replay_sha256"]:
            raise ValueError("browser replay changed")
        replay = json.loads(source.read_text())
        if (
            sha256(Path(replay["source"])) != evidence["media_sha256"]
            or sha256(Path(replay["decoded_audio"])) != evidence["pcm_sha256"]
        ):
            raise ValueError("original browser media or PCM changed")
    with np.load(directory / "examples.npz", allow_pickle=False) as saved:
        return {
            "clips": manifest["clips"],
            "windows": manifest["windows"],
            "event_names": manifest["physical_names"],
            "names": manifest["feature_names"],
            "event_physical": saved["physical"],
            "event_audio": saved["embeddings"],
            "owners": saved["owners"],
            "physical": saved["whole_clip"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-one", type=Path, required=True)
    parser.add_argument("--round-two", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-models", action="store_true")
    args = parser.parse_args()
    if not args.trust_models:
        raise ValueError("Joblib executes pickle: verify origin and use --trust-models")
    if args.output.exists():
        raise ValueError("refusing to overwrite audit")
    data = load_inputs()
    one, two = [
        json.loads((run / "results.json").read_text())
        for run in (args.round_one, args.round_two)
    ]
    plan = json.loads((args.round_one / "plan.json").read_text())
    known = {
        i for i, c in enumerate(data["clips"]) if c["label"] in ("positive", "negative")
    }
    assertions = 0
    for outer in plan["folds"]:
        assert set(outer["train"]) <= known
        assert not set(outer["train"]) & set(outer["test"])
        for index in outer["train"]:
            assert outer["thread"] not in data["clips"][index]["threads"]
        for inner in outer["inner"]:
            assert set(inner["train"]) <= set(outer["train"])
            assert set(inner["test"]) <= set(outer["train"])
            assert not set(inner["train"]) & set(inner["test"])
            for index in inner["train"]:
                assert inner["thread"] not in data["clips"][index]["threads"]
            assertions += 1
    baseline = one["predictions"]["event-hybrid"]
    historical_path = Path(
        "research/artifacts/retrain-20261025-known-families/event-hybrid/results.json"
    )
    historical = json.loads(historical_path.read_text())
    expected = {
        row["audio_sha256"]: row["score"] for row in historical["held_out_predictions"]
    }
    assert len(expected) == len(baseline) == len(known)
    parity = max(abs(expected[row["audio_sha256"]] - row["score"]) for row in baseline)
    if parity > 1e-12:
        raise ValueError("baseline was not reproduced")
    paired = {
        name: paired_cluster_interval(baseline, rows)
        for name, rows in two["predictions"].items()
    }
    # Audit very close pretrained context neighbors as an additional limited
    # identity screen. It is not a substitute for complete family discovery.
    audio = data["event_audio"]
    normalized = audio / np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-12)
    positive_windows = np.flatnonzero(data["y"][data["owners"]] == 1)
    similarities = normalized[positive_windows] @ normalized.T
    neighbors = {}
    for row, window in enumerate(positive_windows):
        owner = int(data["owners"][window])
        similarities[row, data["owners"] == owner] = -1
        for other in np.flatnonzero(similarities[row] >= 0.995):
            neighbor = int(data["owners"][other])
            key = tuple(sorted((owner, neighbor)))
            neighbors[key] = max(neighbors.get(key, 0), float(similarities[row, other]))
    identity = [
        {"paths": [data["clips"][i]["path"] for i in pair], "cosine": value}
        for pair, value in neighbors.items()
    ]
    # All browser fixtures are from this one held-out thread. Never use final
    # corpus-trained fits for this check.
    outer = next(f for f in plan["folds"] if f["thread"] == "337690329")
    key = training_key(outer["train"])
    models = {}
    for run in (args.round_one, args.round_two):
        loaded, spec = load_fit(run / "fits" / key)
        assert spec["training_indices"] == outer["train"]
        models.update(loaded)
    browser = browser_data(args.browser)
    if any("337690329" not in c["threads"] for c in browser["clips"]):
        raise ValueError("browser fixtures must belong to held-out thread")
    if (
        browser["names"] != data["names"]
        or browser["event_names"] != data["event_names"]
    ):
        raise ValueError("browser feature schema mismatch")
    full_index = {clip["audio_sha256"]: i for i, clip in enumerate(data["clips"])}
    wav_indices = [full_index[c["audio_sha256"]] for c in browser["clips"]]
    browser_predictions, decoder_comparison = {}, {}
    for name, model in models.items():
        scores, windows = predict_model(name, model, browser, return_windows=True)
        wav_scores = predict_model(name, model, data)[wav_indices]
        rows = rows_for(browser, scores, list(range(len(browser["clips"]))))
        for index, row in enumerate(rows):
            c = browser["clips"][index]
            lo, count = c["first_window"], c["window_count"]
            row["peak_proposal_s"] = browser["windows"][
                lo + int(np.argmax(windows[lo : lo + count]))
            ]["start_s"]
            row["wav_score"] = float(wav_scores[index])
        browser_predictions[name] = rows
        decoder_comparison[name] = {
            "browser": metrics(rows),
            "same_subset_wav": metrics([r | {"score": r["wav_score"]} for r in rows]),
            "red_boundary_changes": int(sum((scores >= 0.8) != (wav_scores >= 0.8))),
        }
    selected = next(
        r["selected"] for r in two["selection_decisions"] if r["thread"] == "337690329"
    )
    selected_rows = browser_predictions[selected["candidate"]]
    transformed = display_scores(
        np.asarray([r["score"] for r in selected_rows]), selected["threshold"]
    )
    wav_transformed = display_scores(
        np.asarray([r["wav_score"] for r in selected_rows]), selected["threshold"]
    )
    browser_predictions["nested-selection-round2"] = [
        r
        | {
            "raw_score": r["score"],
            "raw_wav_score": r["wav_score"],
            "score": float(s),
            "wav_score": float(w),
        }
        for r, s, w in zip(selected_rows, transformed, wav_transformed, strict=True)
    ]
    selected_predictions = browser_predictions["nested-selection-round2"]
    decoder_comparison["nested-selection-round2"] = {
        "browser": metrics(selected_predictions),
        "same_subset_wav": metrics(
            [r | {"score": r["wav_score"]} for r in selected_predictions]
        ),
        "red_boundary_changes": int(
            sum((transformed >= 0.8) != (wav_transformed >= 0.8))
        ),
    }
    artifact = {
        "round_one_results_sha256": sha256(args.round_one / "results.json"),
        "round_two_results_sha256": sha256(args.round_two / "results.json"),
        "baseline_reference_sha256": sha256(historical_path),
        "baseline_max_absolute_error": parity,
        "audited_inner_folds": assertions,
        "source_folds": len(plan["folds"]),
        "paired_thread_bootstrap": paired,
        "near_identical_context_audit": {
            "threshold": 0.995,
            "pairs": identity,
            "warning": "Limited embedding-neighbor screen only. No new complete transformed-audio family scan.",
        },
        "browser_manifest_sha256": sha256(args.browser / "manifest.json"),
        "browser_warning": "Selected 14 positives, nine named hard negatives, four unknown cases. Not a representative negative sample. Python inference on verified browser PCM; not a JavaScript model port.",
        "browser_comparison": decoder_comparison,
        "browser_selected_policy": selected,
        "browser_predictions": browser_predictions,
        "protected_data_sha256": {
            str(p): sha256(p)
            for p in (
                Path("spokoyno.user.js"),
                Path("corpus/labels.json"),
                Path("corpus/events.json"),
            )
        },
    }
    atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "baseline_error": parity,
                "inner_folds": assertions,
                "browser_comparison": decoder_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
