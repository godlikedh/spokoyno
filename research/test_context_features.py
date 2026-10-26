"""Temporal semantic measurements remain independent of leading audio delays."""

import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
from context_features import GROUPS, cosine_distance, extract, relative_positions


class ContextFeatureTests(unittest.TestCase):
    def head(self):
        indices = sorted({i for group in GROUPS.values() for i in group})
        return {
            "indices": indices,
            "kernel": np.zeros((1024, len(indices))),
            "bias": np.zeros(len(indices)),
        }

    def test_repeated_sound_has_zero_temporal_changes(self):
        embedding = np.random.default_rng(7).uniform(0, 1, (2, 1024))
        values, names = extract(np.tile(embedding, (1, 3)), [0, 5], self.head())
        self.assertEqual(len(names), 32)
        for name in names:
            if name.endswith(("_rise", "_fall", "_distance", "_log_norm")):
                np.testing.assert_allclose(values[:, names.index(name)], 0, atol=1e-12)
        np.testing.assert_array_equal(
            values[:, names.index("ctx_boundary_opening")], [1, 0]
        )
        np.testing.assert_array_equal(
            cosine_distance(np.zeros((1, 2)), np.zeros((1, 2))), [0]
        )

    def test_leading_delay_does_not_change_content_boundary_or_context_features(self):
        original = {
            "windows": [{"start_s": 0.51}, {"start_s": 1.51}, {"start_s": 7.51}],
            "records": [{"canonical": {"trimmed_samples": 8160}}],
            "owners": np.zeros(3, int),
        }
        embeddings = np.random.default_rng(7).uniform(0, 1, (3, 3072))
        expected, names = extract(embeddings, relative_positions(original), self.head())
        for delay in (0.013, 11, 30):
            shifted = {
                "windows": [
                    {"start_s": w["start_s"] + delay} for w in original["windows"]
                ],
                "records": [
                    {"canonical": {"trimmed_samples": 8160 + round(delay * 16000)}}
                ],
                "owners": original["owners"],
            }
            actual, actual_names = extract(
                embeddings, relative_positions(shifted), self.head()
            )
            self.assertEqual(actual_names, names)
            np.testing.assert_array_equal(actual, expected)

    def test_javascript_matches_python_for_temporal_and_semantic_features(self):
        rng = np.random.default_rng(47)
        head = self.head()
        head["groups"] = GROUPS
        head["kernel"] = rng.normal(0, 0.03, head["kernel"].shape).tolist()
        head["bias"] = rng.normal(0, 1, len(head["bias"])).tolist()
        embeddings = rng.uniform(0, 2, (7, 3072)).astype(np.float32)
        embeddings[0] = 0
        embeddings[1, :1024] = 0
        embeddings[2, :1024] = embeddings[2, 2048:]
        positions = [0, 1, 1.025, 1.05, 1.075, 2, 30]
        expected, names = extract(embeddings, positions, head)
        process = subprocess.run(
            [
                "node",
                "-e",
                "const fs = require('fs'); "
                "const {extract} = require('./robust_context.cjs'); "
                "const d = JSON.parse(fs.readFileSync(0, 'utf8')); "
                "process.stdout.write(JSON.stringify(extract("
                "Float32Array.from(d.embeddings), d.positions, d.head)));",
            ],
            input=json.dumps(
                {
                    "embeddings": embeddings.ravel().tolist(),
                    "positions": positions,
                    "head": head,
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        actual = json.loads(process.stdout)
        self.assertEqual(actual["names"], names)
        np.testing.assert_allclose(actual["rows"], expected, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
