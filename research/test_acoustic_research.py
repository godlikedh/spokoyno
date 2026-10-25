"""Guard derivative eligibility and training influence across nested folds."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from acoustic_artifacts import load_snapshot
from extract_audio_context import sha256
from improve_acoustics import augmented_weights
from improve_augmentation import synthetic_objective
from scipy.optimize import check_grad
from synthetic_events import eligible, mix


class AcousticResearchTests(unittest.TestCase):
    def test_portable_snapshot_rejects_changed_model_before_loading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transform = {"old_names": ["level"]}
            (root / "transform.json").write_text(json.dumps(transform))
            (root / "models.json").write_text(
                json.dumps({"candidate": {"coefficient": [1.0]}})
            )
            files = {
                name: sha256(root / name) for name in ("transform.json", "models.json")
            }
            (root / "manifest.json").write_text(json.dumps({"files": files}))
            _, models = load_snapshot(root)
            self.assertEqual(models["candidate"]["old_names"], ["level"])
            (root / "models.json").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_snapshot(root)

    def test_synthetic_mil_gradient(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(8, 3))
        owners = np.repeat(np.arange(4), 2)
        y = np.array([0, 1, 0, 1])
        known = np.array([0, 1, 2, 4, 5, 6])
        targets = np.array([0, 0, 1, 0, 0, 1])
        args = (x, owners, y, known, targets, np.ones(len(known)))
        sx, sy = rng.normal(size=(5, 3)), np.array([0, 0, 1, 1, 1])
        theta = rng.normal(0, 0.2, 4)
        self.assertLess(
            check_grad(
                lambda t: synthetic_objective(t, args, sx, sy, np.ones(5), 0.2)[0],
                lambda t: synthetic_objective(t, args, sx, sy, np.ones(5), 0.2)[1],
                theta,
            ),
            1e-5,
        )

    def test_every_parent_must_be_in_training(self):
        clips = [{"parents": [1, 2]}, {"parents": [2, 3]}, {"parents": [1, 4]}]
        np.testing.assert_array_equal(eligible(clips, [1, 2, 3]), [True, True, False])
        np.testing.assert_array_equal(eligible(clips, [1, 2]), [True, False, False])
        np.testing.assert_array_equal(eligible(clips, []), [False, False, False])

    def test_synthetic_class_and_donor_mass_does_not_grow_with_copies(self):
        real = np.ones(10)
        targets = np.asarray([0, 0, 1, 1, 1, 1])
        source = np.arange(6)
        parents = [[1, 5], [2, 6], [1, 5], [2, 6], [2, 7], [2, 8]]
        rw, sw = augmented_weights(real, targets, source, parents)
        self.assertAlmostEqual(rw.sum(), 8)
        self.assertAlmostEqual(sw[targets == 0].sum(), 1)
        self.assertAlmostEqual(sw[targets == 1].sum(), 1)
        self.assertAlmostEqual(sw[2], sw[3:].sum())
        self.assertIsNone(augmented_weights(real, np.zeros(6), source, parents))

    def test_waveform_mixing_is_bounded_and_control_has_no_inserted_sound(self):
        rng = np.random.default_rng(2)
        background = rng.normal(0, 0.1, 12 * 16000)
        foreground = rng.normal(0, 0.5, 16000)
        positive, control = mix(background, foreground, 4, -3, 20)
        self.assertLessEqual(max(abs(positive)), 0.981)
        np.testing.assert_allclose(positive[: 4 * 16000], control[: 4 * 16000])
        np.testing.assert_allclose(positive[5 * 16000 :], control[5 * 16000 :])
        ratio = control / background
        self.assertLess(np.std(ratio), 1e-7)
        self.assertGreater(
            np.sqrt(np.mean(positive[4 * 16000 : 5 * 16000] ** 2)),
            9 * np.sqrt(np.mean(control**2)),
        )


if __name__ == "__main__":
    unittest.main()
