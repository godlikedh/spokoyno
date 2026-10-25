#!/usr/bin/env python3
"""Publish portable JSON acoustic models or score a media file with a snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from audit_acoustics import encode, score
from event_dataset import physical_events
from extract_audio_context import Encoder, ensure_encoder, sha256
from extract_features import extract_clip, read_audio
from improve_acoustics import load_matrix, predict_model
from retrain_models import load_inputs
from risk_score import risk_tier
from train_models import atomic_json

SHARED = ("preprocessor", "old_names", "extra_names", "extra_scale")


def load_snapshot(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"snapshot checksum mismatch: {name}")
    shared = json.loads((directory / "transform.json").read_text())
    models = json.loads((directory / "models.json").read_text())
    return manifest, {name: shared | model for name, model in models.items()}


def publish(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = json.loads((args.followup / "results.json").read_text())
    models, provenance, expected = {}, {}, {}
    for directory in (args.run, args.followup):
        target = directory / "fits" / results["final_fit_key"]
        meta = json.loads((target / "manifest.json").read_text())
        if (
            sha256(target / "models.json") != meta["model_sha256"]
            or sha256(target / "scores.npz") != meta["scores_sha256"]
        ):
            raise ValueError("final fit changed")
        models.update(json.loads((target / "models.json").read_text()))
        with np.load(target / "scores.npz", allow_pickle=False) as stored:
            expected.update({n: stored[n] for n in stored.files})
        provenance[str(directory)] = meta
    shared = {k: models["synthetic-mil-20pct"][k] for k in SHARED}
    for model in models.values():
        if {k: model[k] for k in SHARED} != shared:
            raise ValueError("models do not share the same fitted transform")
    atomic_json(args.output_dir / "transform.json", shared)
    atomic_json(
        args.output_dir / "models.json",
        {
            name: {k: v for k, v in model.items() if k not in SHARED}
            for name, model in models.items()
        },
    )
    summary = {k: v for k, v in results.items() if k != "predictions"}
    summary["held_out_positives_and_false_warnings"] = {
        name: [r for r in rows if r["label"] == "positive" or r["score"] >= 0.6]
        for name, rows in results["predictions"].items()
    }
    summary["round_one_metrics"] = json.loads((args.run / "results.json").read_text())[
        "metrics"
    ]
    ablation = json.loads((args.ablation / "results.json").read_text())
    summary["ablation_metrics"] = ablation["metrics"]
    summary["ablation_positive_predictions"] = {
        n: [r for r in rows if r["label"] == "positive"]
        for n, rows in ablation["predictions"].items()
    }
    atomic_json(args.output_dir / "results-summary.json", summary)
    for filename in ("summary.json", "browser.json", "stress.json"):
        atomic_json(
            args.output_dir / ("audit-" + filename),
            json.loads((args.audit / filename).read_text()),
        )
    for label, directory in (
        ("round-one", args.run),
        ("round-two", args.followup),
        ("ablation", args.ablation),
    ):
        plan = json.loads((directory / "plan.json").read_text())
        # Split identities are hashed; complete full plans remain local with fits.
        compact = {k: v for k, v in plan.items() if k != "folds"}
        compact["complete_plan_sha256"] = sha256(directory / "plan.json")
        atomic_json(args.output_dir / f"{label}-plan.json", compact)
    data = load_inputs()
    fm, fa = load_matrix(args.features)
    legacy = fa["legacy_indices"] >= 0
    parity = {}
    for name, model in models.items():
        if model.get("expanded_proposals"):
            predictions = []
            for clip in fm["clips"]:
                lo, hi = (
                    clip["first_window"],
                    clip["first_window"] + clip["window_count"],
                )
                s, _ = predict_model(
                    model,
                    fa["physical"][lo:hi],
                    np.zeros((hi - lo, 3072)),
                    fm["physical_names"],
                    np.zeros(hi - lo, dtype=int),
                    1,
                )
                predictions.append(s[0])
        else:
            predictions, _ = predict_model(
                model,
                fa["physical"][legacy],
                data["event_audio"],
                fm["physical_names"],
                data["owners"],
                len(data["clips"]),
            )
        error = float(np.max(np.abs(np.asarray(predictions) - expected[name])))
        if error > 1e-12:
            raise ValueError(f"published model score mismatch: {name}: {error}")
        parity[name] = error
    files = {
        file.name: sha256(file) for file in args.output_dir.iterdir() if file.is_file()
    }
    manifest = {
        "schema": 1,
        "status": "frozen research candidate; no production promotion",
        "recommended_candidate": "synthetic-mil-20pct",
        "score_kind": "uncalibrated evidence, not probability",
        "thresholds": {"yellow": 0.6, "red": 0.8},
        "training_provenance": provenance,
        "source_provenance": data["provenance"],
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "extractor_sha256": {
            name: sha256(Path("research") / name)
            for name in (
                "acoustic_features.py",
                "extract_features.py",
                "event_dataset.py",
                "extract_audio_context.py",
            )
        },
        "files": files,
        "reload_maximum_score_error": parity,
        "validation": "30 positive and 3466 negative real audio groups, thread/known-family excluded; adaptive development research, not prospective validation. Nested selector regressed and is rejected. See report for browser and stress limits.",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    reloaded_manifest, reloaded = load_snapshot(args.output_dir)
    if reloaded != models or reloaded_manifest != manifest:
        raise ValueError("portable JSON snapshot failed roundtrip")
    print(
        json.dumps(
            {
                "snapshot": str(args.output_dir),
                "models": len(models),
                "maximum_score_error": max(parity.values()),
            }
        ),
        flush=True,
    )


def score_media(args):
    manifest, models = load_snapshot(args.snapshot)
    for file, expected in manifest["extractor_sha256"].items():
        if sha256(Path("research") / file) != expected:
            raise ValueError(f"extractor differs from frozen model: {file}")
    if args.output.exists():
        raise ValueError("refusing to overwrite predictions")
    names = list(models) if args.all_models else [args.model]
    chosen = {n: models[n] for n in names}
    ensure_encoder(args.encoder_dir)
    encoder = Encoder(args.encoder_dir, threads=2)
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "decoded.wav"
        if args.media.suffix.lower() == ".wav":
            wav = args.media
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(args.media),
                    "-vn",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_f32le",
                    str(wav),
                ],
                check=True,
            )
        samples = read_audio(wav)
        features = extract_clip(wav)
        _, windows = physical_events(samples, features)
        embeddings = encode(samples, features, windows, encoder)
        predictions = score(chosen, samples, windows, embeddings)
        digest = sha256(wav)
    training = next(iter(manifest["training_provenance"].values()))[
        "eligible_training_hashes"
    ]
    for row in predictions.values():
        row["risk_tier"] = risk_tier(row["score"])
    result = {
        "source": str(args.media),
        "source_sha256": sha256(args.media),
        "decoded_sha256": digest,
        "snapshot_sha256": sha256(args.snapshot / "manifest.json"),
        "exact_wav_hash_seen_in_training": digest in training,
        "score_kind": manifest["score_kind"],
        "timing_kind": "machine proposals, not verified scream boundaries",
        "predictions": predictions,
    }
    atomic_json(args.output, result)
    print(json.dumps(predictions, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("publish")
    for name in ("run", "followup", "ablation", "audit", "features", "output-dir"):
        export.add_argument("--" + name, type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--snapshot", type=Path, required=True)
    score_parser.add_argument("--media", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    score_parser.add_argument("--model", default="synthetic-mil-20pct")
    score_parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()
    publish(args) if args.command == "publish" else score_media(args)


if __name__ == "__main__":
    main()
