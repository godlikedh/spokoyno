#!/usr/bin/env python3
"""Verify every augmentation exclusion, evaluate browser PCM, and stress delays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from context_runtime import physical_features
from extract_audio_context import Encoder, ensure_encoder, sha256
from extract_features import RATE, read_audio
from improve_events import training_key
from robust_audio import augment, extract
from robust_tree import window_probabilities
from train_events import family_links, metrics, split_clips
from train_models import atomic_json
from train_robust_models import eligible_records, load_data, pool, prepare


def infer(samples, transform, models, encoder):
    extracted = extract(samples)
    patches = extracted["patches"]
    embedded = np.concatenate(
        [encoder(patches[lo : lo + 96]) for lo in range(0, len(patches), 96)]
    ).reshape(-1, 3072)
    physical, names = physical_features(extracted, embedded, transform)
    if names != transform["names"]:
        raise ValueError("live feature schema differs from model")
    x = prepare(transform, physical, embedded)
    extracted["embeddings"] = embedded
    extracted["scoring_physical"] = physical
    result = {}
    for name, model in models.items():
        dimensions = x.shape[1] if model["audio"] else len(names)
        p = window_probabilities(model, x[:, :dimensions])
        score = float(pool(p, np.zeros(len(p), int), 1, model["power"])[0][0])
        result[name] = {
            "score": score,
            "event_at_s": extracted["windows"][int(np.argmax(p))]["start_s"],
        }
    return result, extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("research/artifacts/fingerprints/evaluation.json"),
    )
    parser.add_argument(
        "--browser",
        type=Path,
        default=Path("research/artifacts/browser-events-full-20261025-v2"),
    )
    parser.add_argument(
        "--encoder-dir", type=Path, default=Path("research/artifacts/yamnet")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(args.data_dir)
    plan = json.loads((args.run / "plan.json").read_text())
    results = json.loads((args.run / "results.json").read_text())
    choices = {r["thread"]: r["chosen"] for r in results["selection_decisions"]}
    expected_original = {
        name: {r["path"]: r["score"] for r in views["original"]}
        for name, views in results["predictions"].items()
    }
    declaration_file = args.run / "followup-declaration.json"
    declaration = (
        json.loads(declaration_file.read_text()) if declaration_file.exists() else {}
    )
    if plan["dataset_sha256"] != sha256(args.data_dir / "manifest.json"):
        raise ValueError("run belongs to another dataset")
    audits = []
    for file in sorted((args.run / "fits").glob("*/manifest.json")):
        meta = json.loads(file.read_text())
        for name, expected in meta["files"].items():
            if sha256(file.parent / name) != expected:
                raise ValueError("saved fit changed")
        models = json.loads((file.parent / "models.json").read_text())
        for model in models.values():
            expected = np.flatnonzero(
                eligible_records(
                    data["records"], meta["training_indices"], model["augment"]
                )
            ).tolist()
            if not model.get("use_synthetic", True):
                expected = [i for i in expected if data["records"][i]["kind"] == "real"]
            if model["training_records"] != expected:
                raise ValueError(
                    "a synthetic parent or evaluation codec crossed a fold"
                )
            if model.get("kind") == "histogram-boosting":
                seed = Path(declaration["seed_run"]) / "fits" / file.parent.name
                seed_meta = json.loads((seed / "manifest.json").read_text())
                if (
                    seed_meta["training_indices"] != meta["training_indices"]
                    or sha256(seed / "models.json") != model["seed_model_sha256"]
                ):
                    raise ValueError("tree seed belongs to another training partition")
        audits.append(
            {
                "key": file.parent.name,
                "models": len(models),
                "original_sources": len(meta["training_indices"]),
            }
        )
    # Recompute complete transitive known-family exclusions as well as threads.
    links = family_links(json.loads(args.families.read_text()))
    for outer in plan["folds"]:
        train, test = split_clips(data["clips"], outer["thread"], links)
        if train != outer["train"] or test != outer["test"]:
            raise ValueError("outer source/family split differs")
        for fold in [outer, *outer["inner"]]:
            if set(fold["train"]) & set(fold["test"]):
                raise ValueError("train/test source overlap")
            if any(
                fold["thread"] in data["clips"][i]["threads"] for i in fold["train"]
            ):
                raise ValueError("source thread crossed a fold")
            if not set(fold["train"]) <= set(outer["train"]):
                raise ValueError("inner split reintroduced outer exclusions")
        for inner in outer["inner"]:
            train, test = split_clips(data["clips"], inner["thread"], links)
            if (
                sorted(set(train) & set(outer["train"])) != inner["train"]
                or sorted(set(test) & set(outer["train"])) != inner["test"]
            ):
                raise ValueError("inner source/family split differs")
    encoder_info = ensure_encoder(args.encoder_dir)
    encoder = Encoder(args.encoder_dir, threads=3)
    fits = {}
    for fold in plan["folds"]:
        directory = args.run / "fits" / training_key(fold["train"])
        fits[fold["thread"]] = (
            json.loads((directory / "transform.json").read_text()),
            json.loads((directory / "models.json").read_text()),
        )
    browser = json.loads((args.browser / "manifest.json").read_text())
    browser_rows = []
    for clip, evidence in zip(browser["clips"], browser["evidence"], strict=True):
        replay_path = Path(evidence["replay"])
        if sha256(replay_path) != evidence["replay_sha256"]:
            raise ValueError("browser replay changed")
        replay = json.loads(replay_path.read_text())
        if (
            sha256(Path(replay["decoded_audio"])) != evidence["pcm_sha256"]
            or sha256(Path(replay["source"])) != evidence["media_sha256"]
        ):
            raise ValueError("browser source changed")
        scores, _ = infer(
            read_audio(Path(replay["decoded_audio"])),
            *fits[clip["fold_thread"]],
            encoder,
        )
        scores["nested-selection"] = scores[choices[clip["fold_thread"]]]
        browser_rows.append(
            {
                "path": clip["path"],
                "label": clip["label"],
                "predictions": scores,
                "evidence": evidence,
            }
        )
    atomic_json(args.output_dir / "browser.json", browser_rows)
    print("browser fixtures scored", flush=True)
    hard = {c["audio_sha256"] for c in browser["clips"] if c["label"] == "negative"}
    selected = [
        c
        for c in data["clips"]
        if c["label"] == "positive" or c["audio_sha256"] in hard
    ]
    stress, invariance = [], []
    maximum_original_score_error = 0.0
    for clip in selected:
        path = Path("corpus/audio") / f"{clip['file']}.audio.wav"
        if sha256(path) != clip["audio_sha256"]:
            raise ValueError("source audio changed")
        samples = read_audio(path)
        original, features = infer(samples, *fits[clip["fold_thread"]], encoder)
        original["nested-selection"] = original[choices[clip["fold_thread"]]]
        for name, row in original.items():
            error = abs(row["score"] - expected_original[name][clip["path"]])
            maximum_original_score_error = max(maximum_original_score_error, error)
            if error > 1e-5:
                raise ValueError(
                    "live extraction differs from held-out matrix prediction"
                )
        worst_feature_difference = 0.0
        for delay in (0.013, 0.125, 1.75, 11.0, 30.0):
            shifted = extract(np.pad(samples, ((round(delay * RATE), 0), (0, 0))))
            np.testing.assert_array_equal(shifted["physical"], features["physical"])
            np.testing.assert_array_equal(shifted["patches"], features["patches"])
            shifted_physical, _ = physical_features(
                shifted, features["embeddings"], fits[clip["fold_thread"]][0]
            )
            np.testing.assert_array_equal(
                shifted_physical, features["scoring_physical"]
            )
            np.testing.assert_allclose(
                [w["start_s"] + delay for w in features["windows"]],
                [w["start_s"] for w in shifted["windows"]],
                atol=1e-12,
                rtol=0,
            )
        invariance.append(
            {
                "path": clip["path"],
                "delays_s": [0.013, 0.125, 1.75, 11.0, 30.0],
                "maximum_feature_or_encoder_input_difference": worst_feature_difference,
            }
        )
        stress.append(
            {
                "path": clip["path"],
                "label": clip["label"],
                "variant": "original",
                "predictions": original,
            }
        )
        for name, recipe in {
            "aac-aligned": {"codec": "aac80", "align": True},
            "aac16k-aligned": {"codec": "aac16k96", "align": True},
            "gain-minus-3db": {"gain_db": -3},
            "gain-plus-3db": {"gain_db": 3},
            "codec-plus-11s": {"codec": "opus48", "delay_s": 11},
        }.items():
            audio, metadata = augment(samples, recipe)
            scores, _ = infer(audio, *fits[clip["fold_thread"]], encoder)
            scores["nested-selection"] = scores[choices[clip["fold_thread"]]]
            stress.append(
                {
                    "path": clip["path"],
                    "label": clip["label"],
                    "variant": name,
                    "transform": metadata,
                    "predictions": scores,
                }
            )
        print(f"robust stress {len(invariance)}/{len(selected)}", flush=True)

    def summarize(rows):
        return {
            name: metrics(
                [
                    {"label": r["label"], "score": r["predictions"][name]["score"]}
                    for r in rows
                    if r["label"] in ("positive", "negative")
                ]
            )
            for name in [*plan["candidates"], "nested-selection"]
        }

    summary = {
        "source_sha256": sha256(Path(__file__)),
        "encoder": encoder_info,
        "fit_audits": audits,
        "browser": summarize(browser_rows),
        "stress": {
            variant: summarize([r for r in stress if r["variant"] == variant])
            for variant in sorted({r["variant"] for r in stress})
        },
        "exact_delay_checks": len(invariance) * 5,
        "maximum_delay_feature_difference": 0.0,
        "maximum_live_extraction_score_error": maximum_original_score_error,
        "family_source_sha256": sha256(args.families),
        "limitations": "Codec and delay derivatives are not new independent recordings. Gain can change perceived severity. Python inference on browser PCM is not a JavaScript runtime.",
    }
    atomic_json(args.output_dir / "stress.json", stress)
    atomic_json(args.output_dir / "delay-invariance.json", invariance)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["browser"], indent=2), flush=True)


if __name__ == "__main__":
    main()
