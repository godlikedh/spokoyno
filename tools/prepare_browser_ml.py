#!/usr/bin/env python3
"""Prepare immutable feature/patch/fold-score fixtures for the production ML port."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from training.export.acoustic_artifacts import load_snapshot
from training.features.acoustic_features import extract
from training.features.extract_audio_context import context_audio
from training.features.extract_features import extract_clip, read_audio
from training.models.improve_acoustics import predict_model
from training.models.improve_events import training_key
from training.paths import ROOT, artifact_path, corpus_path
from training.storage import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="event dataset, or explicitly supplied browser-replay dataset",
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="synthetic MIL training run containing fold fits",
    )
    parser.add_argument(
        "--acoustic",
        type=Path,
        help="acoustic feature matrix; required for corpus fixtures",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="optional new model snapshot; otherwise use the shipped model",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.browser and args.data_dir is None:
        parser.error("--browser requires --data-dir with saved browser replays")
    if not args.browser and args.acoustic is None:
        parser.error("corpus fixtures require --acoustic")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = args.data_dir or artifact_path("event-data-v1")
    data = json.loads((base / "manifest.json").read_text())
    if sha256(base / "examples.npz") != data["matrix_sha256"]:
        raise ValueError("base matrix changed")
    with np.load(base / "examples.npz", allow_pickle=False) as saved:
        embeddings = saved["embeddings"].astype("<f4")
    embeddings.tofile(args.output_dir / "embeddings.bin")
    if args.snapshot:
        _, exported = load_snapshot(args.snapshot)
        final = exported["synthetic-mil-20pct"]
    else:
        final = json.loads((ROOT / "ml/assets/model.json").read_text())["model"]
    (args.output_dir / "model-final.json").write_text(json.dumps(final))
    fits = args.run
    plan = json.loads((fits / "plan.json").read_text())
    folds = {f["thread"]: f for f in plan["folds"]}
    models = {}
    if not args.browser:
        acoustic = args.acoustic
        fm = json.loads((acoustic / "manifest.json").read_text())
        if sha256(acoustic / "examples.npz") != fm["matrix_sha256"]:
            raise ValueError("acoustic matrix changed")
        with np.load(acoustic / "examples.npz", allow_pickle=False) as arrays:
            physical = arrays["physical"][arrays["legacy_indices"] >= 0]
        names = fm["physical_names"]
        source_features = json.loads(artifact_path("features-v1.json").read_text())
        features_by_hash = {
            r["audio_sha256"]: r["features"] for r in source_features["rows"]
        }
    fixtures = args.output_dir / "fixtures.jsonl"
    with fixtures.open("w") as stream:
        for index, clip in enumerate(data["clips"]):
            if args.limit and index >= args.limit:
                break
            thread = clip["fold_thread"]
            if thread not in models:
                fit = fits / "fits" / training_key(folds[thread]["train"])
                meta = json.loads((fit / "manifest.json").read_text())
                if sha256(fit / "models.json") != meta["model_sha256"]:
                    raise ValueError("excluded model changed")
                models[thread] = json.loads((fit / "models.json").read_text())[
                    "synthetic-mil-20pct"
                ]
                (args.output_dir / f"model-{thread}.json").write_text(
                    json.dumps(models[thread])
                )
            if args.browser:
                replay = json.loads(Path(data["evidence"][index]["replay"]).read_text())
                source = Path(replay["decoded_audio"])
                if sha256(source) != data["evidence"][index]["pcm_sha256"]:
                    raise ValueError("browser PCM changed")
                features = extract_clip(source)
            else:
                source = corpus_path("audio") / f"{clip['file']}.audio.wav"
                if sha256(source) != clip["audio_sha256"]:
                    raise ValueError("source PCM changed")
                features = features_by_hash[clip["audio_sha256"]]
            samples = read_audio(source)
            lo, hi = clip["first_window"], clip["first_window"] + clip["window_count"]
            windows = data["windows"][lo:hi]
            if args.browser:
                vectors, _ = extract(samples, windows)
                names = sorted(vectors[0])
                values = np.asarray(
                    [[row[n] for n in names] for row in vectors[: len(windows)]]
                )
            else:
                values = physical[lo:hi]
            patches, _ = context_audio(samples, features)
            selected = np.concatenate(
                [
                    patches[9:10]
                    if w["slot"] == 3
                    else patches[w["slot"] * 3 : w["slot"] * 3 + 3]
                    for w in windows
                ]
            )
            expected = {}
            for label, model in (("heldout", models[thread]), ("final", final)):
                score, _ = predict_model(
                    model,
                    values,
                    embeddings[lo:hi],
                    names,
                    np.zeros(len(windows), int),
                    1,
                )
                expected[label] = float(score[0])
            row = {
                "index": index,
                "file": clip["file"],
                "label": clip["label"],
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "channels": samples.shape[1],
                "names": names,
                "physical": [
                    [float(v) if np.isfinite(v) else None for v in row]
                    for row in values
                ],
                "windows": [
                    {k: w[k] for k in ("slot", "start_s", "end_s")} for w in windows
                ],
                "patch_sha256": hashlib.sha256(
                    selected.astype("<f4").tobytes()
                ).hexdigest(),
                "embedding_offset": lo * 3072 * 4,
                "model": f"model-{thread}.json",
                "expected": expected,
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if index % 100 == 0:
                print(f"prepared {index + 1}", flush=True)
    print(f"fixtures: {fixtures}", flush=True)


if __name__ == "__main__":
    main()
