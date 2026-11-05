"""Independent scipy/numpy numerical checks for the browser DSP implementation."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from training.features.acoustic_features import modulation
from training.models.improve_acoustics import predict_model

ROOT = Path(__file__).resolve().parent.parent


def javascript(script, payload):
    result = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
        timeout=60,
    )
    return json.loads(result.stdout)


class BrowserMLTests(unittest.TestCase):
    def test_shipped_model_matches_python_inference_without_local_dataset(self):
        model = json.loads((ROOT / "ml/assets/model.json").read_text())["model"]
        names = model["old_names"] + model["extra_names"]
        rng = np.random.default_rng(31)
        physical = rng.normal(size=(7, len(names)))
        audio = rng.normal(0, 0.05, (7, 3072)).astype(np.float32)
        expected, events = predict_model(
            model, physical, audio, names, np.zeros(7, dtype=int), 1
        )
        actual = javascript(
            "const fs=require('fs'),{predict}=require('./ml/model.cjs');"
            "const d=JSON.parse(fs.readFileSync(0,'utf8'));"
            "console.log(JSON.stringify(predict(d.model,d.physical,"
            "Float32Array.from(d.audio.flat()),d.names)));",
            {
                "model": model,
                "physical": physical.tolist(),
                "audio": audio.tolist(),
                "names": names,
            },
        )
        self.assertAlmostEqual(actual["score"], expected[0], places=10)
        np.testing.assert_allclose(actual["events"], events, atol=1e-10, rtol=0)

    def test_release_resources_match_sources_and_manifest(self):
        assets = ROOT / "ml/assets"
        manifest = json.loads((assets / "manifest.json").read_text())
        for paths, base in (
            (manifest["source_sha256"], ROOT),
            (manifest["resources"], assets),
        ):
            for name, expected in paths.items():
                self.assertEqual(
                    hashlib.sha256((base / name).read_bytes()).hexdigest(),
                    expected,
                    name,
                )
        sources = [
            ROOT / name
            for name in (
                "ml/assets/tf.min.js",
                "ml/assets/tf-wasm.min.js",
                "ml/encoder.cjs",
                "ml/roughness.cjs",
                "ml/features.cjs",
                "ml/model.cjs",
            )
        ]
        self.assertEqual(
            (assets / "runtime.js").read_text(),
            "\n;\n".join(p.read_text() for p in sources),
        )
        script = (ROOT / "spokoyno.user.js").read_text()
        embedded = script.split("  function spokoynoMLRuntime() {\n", 1)[1].split(
            "\n    return { tf: globalThis.tf, ML: globalThis.SpokoynoML };", 1
        )[0]
        self.assertEqual(embedded, (assets / "runtime.js").read_text())
        self.assertNotIn("// @resource", script)
        self.assertNotIn("// @require", script)

    def test_arbitrary_length_fft_matches_numpy(self):
        rng = np.random.default_rng(13)
        arrays = [rng.normal(0, 1, n) for n in (32, 127, 155, 295, 9600)]
        actual = javascript(
            "const fs=require('fs'),{transform}=require('./ml/roughness.cjs');"
            "const d=JSON.parse(fs.readFileSync(0,'utf8'));"
            "console.log(JSON.stringify(d.map(a=>transform(Float64Array.from(a)).map(v=>Array.from(v)))));",
            [a.tolist() for a in arrays],
        )
        for values, (real, imaginary) in zip(arrays, actual, strict=True):
            np.testing.assert_allclose(
                np.asarray(real) + 1j * np.asarray(imaginary),
                np.fft.fft(values),
                atol=2e-9,
                rtol=0,
            )

    def test_hilbert_resampling_and_frequency_boundaries_match_scipy(self):
        # Truncated bursts exercise exact 200 Hz boundary rounding as well as
        # ordinary full-length windows; these caught a real port discrepancy.
        rng = np.random.default_rng(17)
        arrays = [
            rng.normal(0, 0.2, (n, 2)).astype(np.float32) for n in (4942, 9433, 9600)
        ]
        arrays += [np.zeros((9600, 1), np.float32)]
        actual = javascript(
            "const fs=require('fs'),{modulation}=require('./ml/roughness.cjs');"
            "const dsp=JSON.parse(fs.readFileSync('./ml/assets/model.json','utf8')).dsp;"
            "const d=JSON.parse(fs.readFileSync(0,'utf8'));"
            "console.log(JSON.stringify(d.map(r=>modulation(Float32Array.from(r.values),r.channels,0,r.values.length/r.channels,dsp))));",
            [{"values": a.ravel().tolist(), "channels": a.shape[1]} for a in arrays],
        )
        for samples, result in zip(arrays, actual, strict=True):
            np.testing.assert_allclose(result, modulation(samples), atol=1e-9, rtol=0)


if __name__ == "__main__":
    unittest.main()
