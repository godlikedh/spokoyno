#!/usr/bin/env python3
"""Package the unchanged synthetic-mil-20pct snapshot and pinned browser dependencies."""

import json
import shutil
from pathlib import Path

from acoustic_artifacts import load_snapshot
from extract_audio_context import sha256
from scipy.signal import butter, firwin

ROOT = Path(__file__).resolve().parent.parent


def main():
    snapshot = ROOT / "research/models/acoustic-mil-20261025"
    manifest, models = load_snapshot(snapshot)
    for name, expected in manifest["extractor_sha256"].items():
        if sha256(ROOT / "research" / name) != expected:
            raise ValueError("frozen extractor changed")
    encoder = ROOT / "research/artifacts/robust-portable-encoder-20261026-v3"
    encoder_manifest = json.loads((encoder / "model.json").read_text())
    if sha256(encoder / "weights.bin") != encoder_manifest["weights_sha256"]:
        raise ValueError("frozen encoder changed")
    out = ROOT / "ml/assets"
    out.mkdir(parents=True, exist_ok=True)
    model = models["synthetic-mil-20pct"]
    payload = {
        "schema": 1,
        "name": "synthetic-mil-20pct",
        "snapshot_sha256": sha256(snapshot / "manifest.json"),
        "thresholds": {"maybe": 0.6, "alert": 0.8},
        "model": model,
        "encoder": encoder_manifest,
        "dsp": {
            "sos": encoder_manifest["dsp_sos"],
            "roughness_sos": butter(
                2, 250, btype="highpass", fs=16000, output="sos"
            ).tolist(),
            "envelope_fir": firwin(641, 1 / 32, window=("kaiser", 5.0)).tolist(),
        },
    }
    (out / "model.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    shutil.copyfile(ROOT / "research/artifacts/yamnet/LICENSE", out / "LICENSE-yamnet")
    deps = ROOT / "node_modules/@tensorflow"
    if not deps.exists():
        deps = (
            ROOT
            / "research/artifacts/robust-js-runtime-20261026/node_modules/@tensorflow"
        )
    for package, source, target in (
        ("tfjs", "dist/tf.min.js", "tf.min.js"),
        ("tfjs-backend-wasm", "dist/tf-backend-wasm.min.js", "tf-wasm.min.js"),
    ):
        metadata = json.loads((deps / package / "package.json").read_text())
        if metadata["version"] != "4.22.0":
            raise ValueError("unexpected TensorFlow.js version")
        shutil.copyfile(deps / package / source, out / target)
    if not (out / "LICENSE-tfjs").exists():
        raise ValueError(
            "include the TensorFlow.js 4.22.0 Apache license before building"
        )
    # A text base64 resource avoids depending on page CSP for binary resource fetches.
    import base64

    for source, name in (
        (encoder / "weights.bin", "weights.bin"),
        (deps / "tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm", "tf.wasm"),
        (deps / "tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm", "tf-simd.wasm"),
    ):
        (out / (name + ".b64")).write_text(
            base64.b64encode(source.read_bytes()).decode("ascii")
        )
    # Tampermonkey executes @require in its sandbox; the same cached text is
    # embedded in a local worker Blob. No runtime CDN requests are necessary.
    runtime = [out / "tf.min.js", out / "tf-wasm.min.js"] + [
        ROOT / "ml" / name
        for name in ("encoder.cjs", "roughness.cjs", "features.cjs", "model.cjs")
    ]
    (out / "runtime.js").write_text("\n;\n".join(p.read_text() for p in runtime))
    evidence = {
        "model": payload["name"],
        "snapshot_sha256": payload["snapshot_sha256"],
        "tfjs_version": "4.22.0",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in runtime},
        "resources": {
            n: sha256(out / n)
            for n in (
                "model.json",
                "weights.bin.b64",
                "tf.wasm.b64",
                "tf-simd.wasm.b64",
                "runtime.js",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    # Only cache the single selected model; research alternatives are not browser dependencies.
    print(
        json.dumps(
            {
                "model": payload["name"],
                "asset_bytes": sum(p.stat().st_size for p in out.iterdir()),
            }
        )
    )


if __name__ == "__main__":
    main()
