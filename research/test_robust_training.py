"""Source-level weighting, evaluation isolation, and analytic objective checks."""

import unittest

import numpy as np
from scipy.optimize import check_grad
from train_robust_models import eligible_records, group_weights, objective, pool


class RobustTrainingTests(unittest.TestCase):
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
