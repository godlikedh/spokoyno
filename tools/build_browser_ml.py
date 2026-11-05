#!/usr/bin/env python3
"""Rebuild the browser bundle from committed assets, or package a new acoustic model."""

import argparse
import base64
import json
import shutil
from pathlib import Path

from training.paths import ROOT, artifact_path, source_path
from training.storage import sha256


def package_model(snapshot, encoder, output, name):
    from scipy.signal import butter, firwin

    from training.export.acoustic_artifacts import load_snapshot

    manifest, models = load_snapshot(snapshot)
    for source, expected in manifest["extractor_sha256"].items():
        if sha256(source_path(source)) != expected:
            raise ValueError("snapshot extractor changed")
    encoder_manifest = json.loads((encoder / "model.json").read_text())
    if sha256(encoder / "weights.bin") != encoder_manifest["weights_sha256"]:
        raise ValueError("encoder weights changed")
    payload = {
        "schema": 1,
        "name": name,
        "snapshot_sha256": sha256(snapshot / "manifest.json"),
        "thresholds": {"maybe": 0.6, "alert": 0.8},
        "model": models[name],
        "encoder": encoder_manifest,
        "dsp": {
            "sos": encoder_manifest["dsp_sos"],
            "roughness_sos": butter(
                2, 250, btype="highpass", fs=16000, output="sos"
            ).tolist(),
            "envelope_fir": firwin(641, 1 / 32, window=("kaiser", 5.0)).tolist(),
        },
    }
    (output / "model.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n"
    )
    (output / "weights.bin.b64").write_text(
        base64.b64encode((encoder / "weights.bin").read_bytes()).decode("ascii")
    )
    shutil.copyfile(artifact_path("yamnet/LICENSE"), output / "LICENSE-yamnet")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="explicitly package a newly trained acoustic snapshot",
    )
    parser.add_argument(
        "--encoder", type=Path, default=artifact_path("portable-encoder")
    )
    parser.add_argument("--model", default="synthetic-mil-20pct")
    args = parser.parse_args()
    output = ROOT / "ml/assets"
    if args.snapshot:
        payload = package_model(args.snapshot, args.encoder, output, args.model)
    else:
        previous = json.loads((output / "manifest.json").read_text())
        for name in ("model.json", "weights.bin.b64"):
            if sha256(output / name) != previous["resources"][name]:
                raise ValueError(
                    f"committed model data changed: {name}; use --snapshot for a new model"
                )
        payload = json.loads((output / "model.json").read_text())
    deps = ROOT / "node_modules/@tensorflow"
    for package, source, target in (
        ("tfjs", "dist/tf.min.js", "tf.min.js"),
        ("tfjs-backend-wasm", "dist/tf-backend-wasm.min.js", "tf-wasm.min.js"),
    ):
        metadata = json.loads((deps / package / "package.json").read_text())
        if metadata["version"] != "4.22.0":
            raise ValueError(
                "unexpected TensorFlow.js version; install the lockfile with npm ci"
            )
        shutil.copyfile(deps / package / source, output / target)
    for license_name in ("LICENSE-tfjs", "LICENSE-yamnet"):
        if not (output / license_name).is_file():
            raise ValueError(f"missing release license: {license_name}")
    for source, name in (
        ("tfjs-backend-wasm.wasm", "tf.wasm"),
        ("tfjs-backend-wasm-simd.wasm", "tf-simd.wasm"),
    ):
        encoded = base64.b64encode(
            (deps / "tfjs-backend-wasm/dist" / source).read_bytes()
        ).decode("ascii")
        (output / (name + ".b64")).write_text(encoded)
    runtime = [output / "tf.min.js", output / "tf-wasm.min.js"] + [
        ROOT / "ml" / name
        for name in ("encoder.cjs", "roughness.cjs", "features.cjs", "model.cjs")
    ]
    (output / "runtime.js").write_text("\n;\n".join(p.read_text() for p in runtime))
    manifest = {
        "model": payload["name"],
        "snapshot_sha256": payload["snapshot_sha256"],
        "tfjs_version": "4.22.0",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in runtime},
        "resources": {
            name: sha256(output / name)
            for name in (
                "model.json",
                "weights.bin.b64",
                "tf.wasm.b64",
                "tf-simd.wasm.b64",
                "runtime.js",
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": payload["name"],
                "asset_bytes": sum(p.stat().st_size for p in output.iterdir()),
            }
        )
    )


if __name__ == "__main__":
    main()
