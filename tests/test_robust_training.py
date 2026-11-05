"""Source-level weighting, evaluation isolation, and analytic objective checks."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

from training.models.robust_detector import load_snapshot
from training.models.train_robust_models import (
    eligible_records,
    group_weights,
    objective,
    pool,
)
from training.paths import source_path
from training.storage import sha256


class RobustTrainingTests(unittest.TestCase):
    def test_portable_snapshot_checks_payload_and_extractor_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in {
                "transform.json": {"names": ["level"]},
                "models.json": {"candidate": {"coefficient": [1]}},
            }.items():
                (root / name).write_text(json.dumps(data))
            manifest = {
                "files": {
                    name: sha256(root / name)
                    for name in ("transform.json", "models.json")
                },
                "extractor_sha256": {
                    "robust_audio.py": sha256(source_path("robust_audio.py"))
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            _, transform, models = load_snapshot(root)
            self.assertEqual(transform["names"], ["level"])
            self.assertIn("candidate", models)
            manifest["extractor_sha256"]["robust_audio.py"] = "changed"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "extractor"):
                load_snapshot(root)
            (root / "models.json").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                load_snapshot(root)

    def test_evaluation_profiles_and_every_parent_are_excluded(self):
        rows = [
            {"split": "base", "label": "positive", "parents": [1]},
            {"split": "train", "label": "positive", "parents": [1, 2]},
            {"split": "train", "label": "positive", "parents": [1, 3]},
            {"split": "evaluation", "label": "positive", "parents": [1]},
            {"split": "base", "label": "unlabeled", "parents": [2]},
        ]
        np.testing.assert_array_equal(
            eligible_records(rows, [1, 2]), [True, True, False, False, False]
        )
        np.testing.assert_array_equal(
            eligible_records(rows, [1, 2], False), [True, False, False, False, False]
        )

    def test_copy_counts_do_not_change_source_or_class_mass(self):
        rows = [
            {"kind": "real", "label": "positive", "source": 1},
            {"kind": "real", "label": "positive", "source": 1},
            {"kind": "real", "label": "positive", "source": 2},
            {"kind": "real", "label": "negative", "source": 3},
            {"kind": "synthetic", "label": "positive", "source": 1},
            {"kind": "synthetic", "label": "negative", "source": 1},
        ]
        w = group_weights(rows)
        np.testing.assert_allclose(w, [0.125, 0.125, 0.25, 0.5, 0.1, 0.1])

    def test_power_pool_preserves_strong_event_better_than_linear_softmax(self):
        p = np.array([0.92, 0.43, 0.24, 0.05])
        owners = np.zeros(4, dtype=int)
        self.assertLess(pool(p, owners, 1, 2)[0][0], 0.8)
        self.assertGreater(pool(p, owners, 1, 8)[0][0], 0.9)
        for power in (2, 8):
            value = pool(p, owners, 1, power)[0][0]
            self.assertGreaterEqual(value, min(p))
            self.assertLessEqual(value, max(p))

    def test_complete_objective_gradient_with_shared_source_consistency(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(18, 5))
        owners = np.repeat(np.arange(6), 3)
        labels = np.array([1, 1, 0, 0, 1, 0])
        weights = np.array([0.15, 0.15, 0.2, 0.2, 0.2, 0.1])
        pairs = np.array([0, 0, 1, 1, 2, 3])
        strong = np.array([0, 3, 6, 7, 9, 12, 15])
        targets = np.array([1, 1, 0, 0, 0, 1, 0])
        for power in (2, 8):
            args = (
                x,
                owners,
                labels,
                weights,
                pairs,
                strong,
                targets,
                np.ones(7) / 70,
                power,
                1.0,
                0.03,
            )
            theta = rng.normal(0, 0.3, 6)
            error = check_grad(
                lambda t, a=args: objective(t, *a)[0],
                lambda t, a=args: objective(t, *a)[1],
                theta,
            )
            self.assertLess(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
