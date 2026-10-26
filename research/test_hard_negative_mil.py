"""Hard-event mining preserves source weight and honors human event labels."""

import unittest

import numpy as np
from train_hard_negative_mil import instance_supervision


class HardNegativeTests(unittest.TestCase):
    def test_negative_budget_is_concentrated_without_changing_source_mass(self):
        p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        for fraction, count in ((0, 1), (0.75, 3), (1, 1)):
            y, weights = instance_supervision(
                p, np.zeros(6), np.array([0, 6]), [False], [0.5], fraction, count
            )
            np.testing.assert_array_equal(y, np.zeros(6))
            self.assertAlmostEqual(sum(weights), 0.5)
            self.assertGreaterEqual(weights[1], weights[0])
        np.testing.assert_allclose(weights, [0, 0.5, 0, 0, 0, 0])

    def test_weak_positive_cannot_select_a_known_negative_and_keeps_strong_events(self):
        p = np.array([0.99, 0.7, 0.2, 0.1])
        targets = np.array([0, -1, 1, -1])
        y, weights = instance_supervision(
            p, targets, np.array([0, 4]), [True], [0.4], 0.75, 3
        )
        np.testing.assert_array_equal(y, [0, 1, 1, -1])
        np.testing.assert_allclose(weights, [0.04, 0.2, 0.2, 0])


if __name__ == "__main__":
    unittest.main()
