#!/usr/bin/env python3
"""Evaluate frozen acoustic candidates on browser PCM and waveform perturbations."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from acoustic_features import extract
from audit_event_research import paired_cluster_interval
from event_dataset import physical_events
from extract_audio_context import Encoder, context_audio, ensure_encoder, sha256
from extract_features import RATE, extract_clip, read_audio
from improve_acoustics import predict_model
from improve_events import training_key
from scipy.io import wavfile
from stress_events import variants
from synthetic_events import eligible
from train_events import metrics
from train_models import atomic_json


def encode(samples, features, windows, encoder):
    patches, _ = context_audio(samples, features)
    embedded = encoder(patches)
    return np.asarray(
        [
            np.concatenate([np.zeros(1024), embedded[9], np.zeros(1024)])
            if point["slot"] == 3
            else embedded[point["slot"] * 3 : point["slot"] * 3 + 3].ravel()
            for point in windows
        ]
    )


def score(models, samples, original, embeddings):
    vectors, points = extract(samples, original)
    names = sorted(vectors[0])
    physical = np.asarray([[v[n] for n in names] for v in vectors])
    result = {}
    for name, model in models.items():
        size = len(points) if model.get("expanded_proposals") else len(original)
        audio = np.pad(embeddings, ((0, size - len(original)), (0, 0)))
        scores, windows = predict_model(
            model, physical[:size], audio, names, np.zeros(size, dtype=int), 1
        )
        chosen = int(np.argmax(windows))
        result[name] = {
            "score": float(scores[0]),
            "event_at_s": points[chosen]["start_s"],
            "top_windows": [
                {"start_s": points[i]["start_s"], "score": float(windows[i])}
                for i in np.argsort(-windows)[:3]
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--followup", type=Path)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument(
        "--browser",
        type=Path,
        default=Path("research/artifacts/browser-events-full-20261025-v2"),
    )
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    latest = args.followup or args.run
    plan = json.loads((latest / "plan.json").read_text())
    results = json.loads((latest / "results.json").read_text())
    names = plan["candidates"]
    if args.followup and plan["parent_plan_sha256"] != sha256(args.run / "plan.json"):
        raise ValueError("follow-up belongs to a different initial experiment")
    real = json.loads(
        Path("research/artifacts/event-data-v1/manifest.json").read_text()
    )
    synthetic = json.loads((args.synthetic / "manifest.json").read_text())
    # Verify the exact plan, all real parent identities, and every derivative's
    # eligibility in all saved outer, inner, and final training partitions.
    if sha256(args.synthetic / "manifest.json") != plan["synthetic_manifest_sha256"]:
        raise ValueError("synthetic snapshot changed")
    for clip in synthetic["clips"]:
        if clip["parent_hashes"] != [
            real["clips"][i]["audio_sha256"] for i in clip["parents"]
        ]:
            raise ValueError("synthetic parent identity mismatch")
        if sha256(Path(clip["file"])) != clip["audio_sha256"]:
            raise ValueError("synthetic waveform changed")
    assertions = 0
    saved_manifests = sorted((args.run / "fits").glob("*/manifest.json"))
    if args.followup:
        saved_manifests += sorted((args.followup / "fits").glob("*/manifest.json"))
    for saved in saved_manifests:
        meta = json.loads(saved.read_text())
        file = saved.with_name("models.json")
        if sha256(file) != meta["model_sha256"]:
            raise ValueError("model checksum mismatch")
        saved_models = json.loads(file.read_text())
        expected = np.flatnonzero(
            eligible(synthetic["clips"], meta["training_indices"])
        ).tolist()
        for model in saved_models.values():
            if "synthetic_clips" in model and model["synthetic_clips"] != expected:
                raise ValueError("synthetic source crossed a split")
        assertions += 1
    models = {}
    for fold in plan["folds"]:
        models[fold["thread"]] = json.loads(
            (
                args.run / "fits" / training_key(fold["train"]) / "models.json"
            ).read_text()
        )
        if args.followup:
            models[fold["thread"]].update(
                json.loads(
                    (
                        args.followup
                        / "fits"
                        / training_key(fold["train"])
                        / "models.json"
                    ).read_text()
                )
            )
    browser = json.loads((args.browser / "manifest.json").read_text())
    if sha256(args.browser / "examples.npz") != browser["matrix_sha256"]:
        raise ValueError("browser matrix changed")
    with np.load(args.browser / "examples.npz", allow_pickle=False) as saved:
        browser_audio = saved["embeddings"]
    browser_rows = []
    for clip, evidence in zip(browser["clips"], browser["evidence"], strict=True):
        replay_file = Path(evidence["replay"])
        if sha256(replay_file) != evidence["replay_sha256"]:
            raise ValueError("browser replay changed")
        replay = json.loads(replay_file.read_text())
        if (
            sha256(Path(replay["decoded_audio"])) != evidence["pcm_sha256"]
            or sha256(Path(replay["source"])) != evidence["media_sha256"]
        ):
            raise ValueError("browser media changed")
        lo, hi = clip["first_window"], clip["first_window"] + clip["window_count"]
        scored = score(
            models[clip["fold_thread"]],
            read_audio(Path(replay["decoded_audio"])),
            browser["windows"][lo:hi],
            browser_audio[lo:hi],
        )
        browser_rows.append(clip | {"predictions": scored, "evidence": evidence})
    atomic_json(args.output_dir / "browser.json", browser_rows)
    print("browser audit: 27 original-media fixtures scored", flush=True)
    encoder_manifest = ensure_encoder(args.encoder_dir)
    encoder = Encoder(args.encoder_dir, threads=2)
    hard = {c["audio_sha256"] for c in browser["clips"] if c["label"] == "negative"}
    selected = [
        c
        for c in real["clips"]
        if c["label"] == "positive" or c["audio_sha256"] in hard
    ]
    stress = []
    for clip in selected:
        source = Path("corpus/audio") / f"{clip['file']}.audio.wav"
        if sha256(source) != clip["audio_sha256"]:
            raise ValueError("stress source changed")
        samples = read_audio(source)
        for name, transformed in variants(samples):
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / "variant.wav"
                wavfile.write(file, RATE, transformed)
                features = extract_clip(file)
            _, original = physical_events(transformed, features)
            embeddings = encode(transformed, features, original, encoder)
            scores = score(
                models[clip["fold_thread"]], transformed, original, embeddings
            )
            stress.append(
                {
                    "path": clip["path"],
                    "audio_sha256": clip["audio_sha256"],
                    "label": clip["label"],
                    "variant": name,
                    "predictions": scores,
                }
            )
        print(f"stress {len(stress) // 5}/{len(selected)}", flush=True)

    def summarize(rows):
        return {
            name: metrics(
                [
                    {"label": r["label"], "score": r["predictions"][name]["score"]}
                    for r in rows
                    if r["label"] in ("positive", "negative")
                ]
            )
            for name in names
        }

    summary = {
        "source_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(latest / "plan.json"),
        "synthetic_parent_audits": assertions,
        "encoder": encoder_manifest,
        "browser": summarize(browser_rows),
        "stress": {
            v: summarize([r for r in stress if r["variant"] == v])
            for v in sorted({r["variant"] for r in stress})
        },
        "clustered_intervals": {
            n: paired_cluster_interval(results["predictions"]["event-hybrid"], rows)
            for n, rows in results["predictions"].items()
        },
        "limitations": "Controlled variants and selected browser fixtures, not new independent examples. Gain changes can change perceived severity. Python inference on browser PCM is not a JS model port. No tuning on these scores.",
    }
    atomic_json(args.output_dir / "stress.json", stress)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["browser"], indent=2), flush=True)


if __name__ == "__main__":
    main()
