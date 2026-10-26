"""Structural invariance, explicit codec alignment and channel-preservation tests."""

import unittest

import numpy as np
from robust_audio import RATE, alignment, bounded, canonical, extract, transcode


class RobustAudioTests(unittest.TestCase):
    def test_every_sample_of_leading_silence_is_removed_without_changing_content(self):
        rng = np.random.default_rng(91)
        samples = np.pad(
            rng.normal(0, 0.3, (RATE, 2)).astype(np.float32), ((97, 0), (0, 0))
        )
        base, meta = canonical(samples)
        for delay in (1, 207, 1024, 28000, 11 * RATE):
            other, offset = canonical(np.pad(samples, ((delay, 0), (0, 0))))
            np.testing.assert_array_equal(base, other)
            self.assertEqual(offset["trimmed_samples"] - meta["trimmed_samples"], delay)

    def test_features_and_embedding_inputs_are_exactly_delay_invariant(self):
        rng = np.random.default_rng(7)
        audio = rng.normal(0, 0.01, (3 * RATE, 1)).astype(np.float32)
        audio[2 * RATE : 2 * RATE + 4000] *= 25
        base = extract(audio)
        delay = 207
        shifted = extract(np.pad(audio, ((delay, 0), (0, 0))))
        np.testing.assert_array_equal(base["physical"], shifted["physical"])
        np.testing.assert_array_equal(base["patches"], shifted["patches"])
        np.testing.assert_allclose(
            [w["start_s"] + delay / RATE for w in base["windows"]],
            [w["start_s"] for w in shifted["windows"]],
            atol=1e-14,
        )

    def test_bounding_preserves_antiphase_channels_and_rejects_invalid_audio(self):
        samples = np.array([[16.0, -16.0], [0.25, -0.25]], dtype=np.float32)
        np.testing.assert_array_equal(bounded(samples), [[1, -1], [0.25, -0.25]])
        for audio in (np.zeros((0, 1)), np.zeros((3, 3)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                bounded(audio)

    def test_alignment_measures_positive_and_negative_offset(self):
        source = (
            np.random.default_rng(5).normal(0, 0.1, (2 * RATE, 1)).astype(np.float32)
        )
        for delay in (341, 1024):
            measured, corr = alignment(source, np.pad(source, ((delay, 0), (0, 0))))
            self.assertEqual(measured, delay)
            self.assertGreater(corr, 0.99999)
        measured, corr = alignment(source, source[120:])
        self.assertEqual(measured, -120)
        self.assertGreater(corr, 0.99999)

    def test_aac_control_is_bounded_before_encoding_and_keeps_stereo(self):
        t = np.arange(RATE) / RATE
        source = np.c_[
            0.7 * np.sin(2 * np.pi * 400 * t), 0.5 * np.sin(2 * np.pi * 700 * t)
        ].astype(np.float32)
        decoded, meta = transcode(source, "aac96", align=True)
        self.assertEqual(decoded.shape, source.shape)
        self.assertTrue(meta["aligned"])
        self.assertGreater(meta["alignment_correlation"], 0.95)
        self.assertLess(abs(10 * np.log10(np.mean(decoded**2) / np.mean(source**2))), 1)


if __name__ == "__main__":
    unittest.main()
