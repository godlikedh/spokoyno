#!/usr/bin/env python3
"""Large, source-tracked codec/delay augmentation and independently mixed events.

Per-record extraction is restartable. Synthetic parent eligibility is enforced
later for every training fold; evaluation profiles never enter training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from event_annotations import validate_intervals
from event_dataset import atomic_npz, window_label
from extract_audio_context import sha256
from extract_features import RATE, read_audio
from robust_audio import augment, bounded, extract, recipe_key, waveform_hash
from train_models import atomic_json

SEED = 2026102601
TRAIN_CODECS = ("aac64", "aac96", "opus32", "opus64")
EVALUATION = {
    "aac80": {"codec": "aac80", "delay_s": 0.013},
    "opus48": {"codec": "opus48", "delay_s": 0.125},
    "mp3_128": {"codec": "mp3_128", "delay_s": 1.75},
}


def plans(clips, mixes_per_donor):
    rng = np.random.default_rng(SEED)
    result = []
    for i, clip in enumerate(clips):
        result.append(
            {
                "kind": "real",
                "source": i,
                "parents": [i],
                "split": "base",
                "variant": "original",
                "recipe": {},
                "label": clip["label"],
                "pair": f"real-{i}",
            }
        )
        if clip["label"] not in ("positive", "negative"):
            continue
        count = 12 if clip["label"] == "positive" else 2
        for v in range(count):
            result.append(
                {
                    "kind": "real",
                    "source": i,
                    "parents": [i],
                    "split": "train",
                    "variant": f"train-{v}",
                    "recipe": {
                        "codec": TRAIN_CODECS[(i + v) % 4],
                        "delay_s": float(
                            rng.choice([1 / RATE, 0.007, 0.025, 0.064, 0.333, 2.0, 7.0])
                        ),
                        "gain_db": float(rng.choice([0.0, 0.0, 0.0, -1.5, 1.5])),
                    },
                    "label": clip["label"],
                    "pair": f"real-{i}",
                }
            )
        for name, recipe in EVALUATION.items():
            result.append(
                {
                    "kind": "real",
                    "source": i,
                    "parents": [i],
                    "split": "evaluation",
                    "variant": name,
                    "recipe": recipe,
                    "label": clip["label"],
                    "pair": f"real-{i}",
                }
            )
    donors = [
        i for i, c in enumerate(clips) if c["label"] == "positive" and c["annotation"]
    ]
    backgrounds = [
        i
        for i, c in enumerate(clips)
        if c["label"] == "negative" and c["duration_s"] >= 12
    ]
    for donor in donors:
        intervals = validate_intervals(
            clips[donor]["annotation"]["intervals"], clips[donor]["duration_s"]
        )
        error = clips[donor]["annotation"].get("estimated_boundary_error_s", 0.05)
        for v in range(mixes_per_donor):
            background = int(rng.choice(backgrounds))
            interval = intervals[v % len(intervals)]
            lower, upper = interval["start_s"] + error, interval["end_s"] - error
            duration = min(
                float(np.exp(rng.uniform(np.log(0.2), np.log(3)))), upper - lower
            )
            settings = {
                "donor_start_s": float(
                    rng.uniform(lower, max(lower, upper - duration))
                ),
                "duration_s": duration,
                "background_start_s": float(
                    rng.uniform(0, max(0, clips[background]["duration_s"] - 12))
                ),
                "onset_s": float(rng.uniform(0.5, 8)),
                "foreground_db": float(rng.uniform(-14, -3)),
                "contrast_db": float(rng.uniform(8, 30)),
                "additive": bool(rng.integers(0, 2)),
                "control_gain_db": float(rng.choice([0, 0, 6, 12])),
            }
            recipe = {
                "codec": [None, *TRAIN_CODECS][v % 5],
                "delay_s": float(rng.uniform(0, 20)),
            }
            for label in ("positive", "negative"):
                result.append(
                    {
                        "kind": "synthetic",
                        "source": donor,
                        "parents": [donor, background],
                        "split": "train",
                        "variant": f"mix-{v}-{label}",
                        "recipe": recipe,
                        "mix": settings,
                        "label": label,
                        "pair": f"mix-{donor}-{v}-{label}",
                    }
                )
    # Generate base and train examples before evaluation; encoder can run in parallel.
    return sorted(
        result, key=lambda p: {"base": 0, "train": 1, "evaluation": 2}[p["split"]]
    )


def source_audio(index, clips, audio_dir):
    clip = clips[index]
    path = audio_dir / f"{clip['file']}.audio.wav"
    if sha256(path) != clip["audio_sha256"]:
        raise ValueError(f"source changed: {path}")
    return bounded(read_audio(path))


def strong_channel(audio):
    return audio[:, int(np.argmax(np.mean(audio[::16].astype(float) ** 2, axis=0)))]


def synthesize(plan, clips, audio_dir):
    spec = plan["mix"]
    donor = strong_channel(source_audio(plan["parents"][0], clips, audio_dir))
    background = strong_channel(source_audio(plan["parents"][1], clips, audio_dir))
    lo = round(spec["background_start_s"] * RATE)
    background = background[lo : lo + 12 * RATE].astype(float)
    lo = round(spec["donor_start_s"] * RATE)
    foreground = donor[lo : lo + round(spec["duration_s"] * RATE)].astype(float)
    gain = min(
        10 ** (spec["foreground_db"] / 20) / max(1e-9, np.sqrt(np.mean(foreground**2))),
        0.99 / max(1e-9, max(abs(foreground))),
    )
    foreground *= gain
    fg_rms = np.sqrt(np.mean(foreground**2))
    onset = round(spec["onset_s"] * RATE)
    context = background[max(0, onset - 3 * RATE) : onset]
    bg_gain = (
        fg_rms
        / 10 ** (spec["contrast_db"] / 20)
        / max(np.sqrt(np.mean(context**2)), 1e-9)
    )
    background *= min(bg_gain, 0.99 / max(1e-9, max(abs(background))))
    if plan["label"] == "negative":
        # Uniform level change preserves the background's temporal structure.
        # No arbitrarily inserted loud sound is mislabeled as a negative.
        return bounded(background[:, None] * 10 ** (spec["control_gain_db"] / 20)), None
    envelope = np.ones(len(foreground))
    fade = min(80, len(foreground) // 2)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    stop = onset + len(foreground)
    if not spec["additive"]:
        background[onset:stop] *= 1 - envelope
    background[onset:stop] += foreground * envelope
    annotation = {
        "intervals": [{"start_s": onset / RATE, "end_s": stop / RATE}],
        "estimated_boundary_error_s": 0.0,
    }
    return bounded(background[:, None]), annotation


def generate_one(plan, clips, audio_dir, output, source_hash):
    key = recipe_key(
        {
            "plan": plan,
            "parents": [clips[i]["audio_sha256"] for i in plan["parents"]],
            "source_hash": source_hash,
        }
    )
    meta_file = output / "cache" / f"{key}.json"
    if meta_file.exists():
        saved = json.loads(meta_file.read_text())
        if sha256(output / "cache" / f"{key}.npz") != saved["feature_sha256"]:
            raise ValueError("cached feature checksum mismatch")
        return saved
    if plan["kind"] == "real":
        samples = source_audio(plan["source"], clips, audio_dir)
        annotation = clips[plan["source"]]["annotation"]
    else:
        samples, annotation = synthesize(plan, clips, audio_dir)
    intervals = (
        validate_intervals(annotation["intervals"], len(samples) / RATE)
        if annotation
        else []
    )
    samples, transform = augment(samples, plan["recipe"])
    if annotation:
        offset = transform["annotation_delay_samples"] / RATE
        annotation = {
            "intervals": [
                {
                    "start_s": r["start_s"] + offset,
                    "end_s": min(len(samples) / RATE, r["end_s"] + offset),
                }
                for r in intervals
            ],
            "estimated_boundary_error_s": annotation.get(
                "estimated_boundary_error_s", 0.05
            )
            + (0.005 if plan["recipe"].get("codec") else 0),
        }
    extracted = extract(samples)
    targets = [
        window_label(
            plan["label"], annotation, w["start_s"], w["end_s"], len(samples) / RATE
        )
        for w in extracted["windows"]
    ]
    feature_file = output / "cache" / f"{key}.npz"
    atomic_npz(
        feature_file,
        physical=extracted["physical"],
        targets=np.asarray(targets, np.int8),
    )
    patch_file = output / "pending" / f"{key}.npz"
    atomic_npz(patch_file, patches=extracted["patches"])
    saved = plan | {
        "key": key,
        "parent_hashes": [clips[i]["audio_sha256"] for i in plan["parents"]],
        "feature_sha256": sha256(feature_file),
        "patch_sha256": sha256(patch_file),
        "waveform_sha256": waveform_hash(samples),
        "transform": transform,
        "canonical": extracted["metadata"],
        "windows": extracted["windows"],
        "names": extracted["names"],
        "duration_s": len(samples) / RATE,
    }
    atomic_json(meta_file, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("research/artifacts/event-data-v1")
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("corpus/audio"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixes-per-donor", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise ValueError("dataset already complete")
    for name in ("cache", "pending", "embeddings"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base / "manifest.json").read_text())
    source_hash = sha256(Path(__file__)) + sha256(Path("research/robust_audio.py"))
    recipes = plans(base["clips"], args.mixes_per_donor)
    plan = {
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_hash,
        "base_manifest_sha256": sha256(args.base / "manifest.json"),
        "train_codecs": TRAIN_CODECS,
        "evaluation_profiles": EVALUATION,
        "mixes_per_donor": args.mixes_per_donor,
        "total_records": len(recipes),
        "policy": "Real sources and every synthetic parent remain on the same side of each exclusion. Evaluation profiles never used for fitting. Derivatives do not increase independent sample counts.",
        "recipes": recipes,
    }
    if (args.output_dir / "plan.json").exists():
        prior = json.loads((args.output_dir / "plan.json").read_text())
        if prior["source_sha256"] != source_hash or prior["recipes"] != recipes:
            raise ValueError("cannot resume a changed generation plan")
    else:
        atomic_json(args.output_dir / "plan.json", plan)
    print(f"generating {len(recipes)} records", flush=True)
    from functools import partial

    work = partial(
        generate_one,
        clips=base["clips"],
        audio_dir=args.audio_dir,
        output=args.output_dir,
        source_hash=source_hash,
    )
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(
            pool.map(work, recipes, buffersize=args.workers * 2)
        ):
            records.append(
                {k: v for k, v in record.items() if k not in ("names", "windows")}
            )
            if (i + 1) % 100 == 0 or i + 1 == len(recipes):
                print(
                    f"features {i + 1}/{len(recipes)} ({record['split']})", flush=True
                )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": 1,
            "plan_sha256": sha256(args.output_dir / "plan.json"),
            "names": record["names"],
            "clips": base["clips"],
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
