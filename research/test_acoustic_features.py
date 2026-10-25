"""Physical invariants for new audio measurements and proposal generation."""

import unittest

import numpy as np
from acoustic_features import (
    band_features,
    burst_features,
    modulation,
    relative_attack,
    short_proposals,
)


class AcousticFeatureTests(unittest.TestCase):
    def test_relative_attack_is_gain_invariant(self):
        fine = np.full(200, -50.0)
        fine[100:111] = np.linspace(-50, -5, 11)
        fine[111:130] = -5
        self.assertEqual(relative_attack(fine, 100), relative_attack(fine + 6, 100))
        self.assertGreater(relative_attack(fine, 100), 0)

    def test_quiet_tail_does_not_erase_active_burst_level(self):
        short = np.full(300, -50.0)
        short[100:120] = -3
        long = short.copy()
        long[120:200] = -3
        a, b = burst_features(short, 100), burst_features(long, 100)
        self.assertEqual(a["burst_active_median_db"], b["burst_active_median_db"])
        self.assertLess(a["burst_active_duration_s"], b["burst_active_duration_s"])
        self.assertGreater(
            a["burst_peak_to_1s_median_db"], b["burst_peak_to_1s_median_db"]
        )

    def test_band_absolute_levels_shift_with_gain_but_jumps_do_not(self):
        time = np.arange(32000) / 16000
        x = (0.05 * np.sin(2 * np.pi * 1500 * time))[:, None]
        x[16000:] *= 4
        a, b = band_features(x, 1), band_features(x * 2, 1)
        self.assertAlmostEqual(
            b["bands_event_db_1000_2000"] - a["bands_event_db_1000_2000"],
            20 * np.log10(2),
            places=5,
        )
        self.assertAlmostEqual(
            b["bands_jump_db_1000_2000"], a["bands_jump_db_1000_2000"], places=5
        )

    def test_roughness_proxy_distinguishes_fast_and_slow_modulation(self):
        time = np.arange(16000) / 16000

        def sound(rate):
            return (
                (1 + 0.8 * np.sin(2 * np.pi * rate * time))
                * np.sin(2 * np.pi * 1000 * time)
            )[:, None]

        fast, slow = modulation(sound(80)), modulation(sound(5))
        self.assertGreater(fast[0], slow[0] + 0.5)
        np.testing.assert_allclose(modulation(sound(80) * 0.1), fast, atol=1e-8)

    def test_short_proposals_find_brief_burst_without_using_labels(self):
        fine = np.full(1000, -50.0)
        fine[433:443] = -4
        points = short_proposals(fine, [1.0])
        self.assertTrue(any(abs(t - 4.33) <= 0.1 for t in points))
        self.assertTrue(all(abs(t - 1.0) >= 0.15 for t in points))
        self.assertEqual(points, short_proposals(fine, [1.0]))


if __name__ == "__main__":
    unittest.main()
