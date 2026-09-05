"""Regression checks for leakage, localized matching, and annotation integrity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from event_annotations import candidate_coverage, load_annotations, validate_intervals
from extract_audio_context import PATCH_SAMPLES, context_audio, crop
from fingerprint_audio import Matcher, landmarks
from train_audio_context import fit, predict, split_indices


def sound(seed: int, duration: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(duration * 16000)
    time = np.arange(count) / 16000
    waveform = rng.normal(0, 0.001, count)
    for _ in range(12):
        frequency = rng.uniform(150, 6000)
        slope = rng.uniform(-50, 50)
        envelope = 0.02 * (1 + np.sin(time * rng.uniform(1, 15) + rng.uniform(0, 6)))
        waveform += envelope * np.sin(2 * np.pi * (frequency * time + slope * time**2))
    return waveform.astype(np.float32)


class TimingTests(unittest.TestCase):
    def test_coverage_uses_existing_candidates(self):
        annotations = {
            "events": {
                "clip": {
                    "audio_sha256": "hash",
                    "intervals": [
                        {"start_s": 2.0, "end_s": 2.4},
                        {"start_s": 3.5, "end_s": 3.8},
                    ],
                }
            }
        }
        rows = [
            {
                "path": "clip",
                "audio_sha256": "hash",
                "features": {
                    "duration_s": 4.0,
                    "start_position_s": 0.1,
                    "event_1_position_fraction": 40 / 79,
                },
            }
        ]
        result = candidate_coverage(annotations, rows)
        self.assertTrue(result[0]["within_250ms"])
        self.assertFalse(result[1]["within_250ms"])
        self.assertEqual(result[1]["nearest_candidate_s"], 2.0)

    def test_coverage_includes_exact_threshold_despite_float_roundoff(self):
        annotations = {
            "events": {
                "clip": {
                    "audio_sha256": "hash",
                    "estimated_boundary_error_s": 0.05,
                    "intervals": [
                        {"start_s": 5.3, "end_s": 5.4},
                        {"start_s": 5.800001, "end_s": 6.0},
                    ],
                }
            }
        }
        rows = [
            {
                "path": "clip",
                "audio_sha256": "hash",
                "features": {
                    "duration_s": 10.0,
                    "start_position_s": 0.1,
                    "event_1_position_fraction": 111 / 199,
                },
            }
        ]
        result = candidate_coverage(annotations, rows)
        self.assertAlmostEqual(result[0]["onset_error_s"], 0.25)
        self.assertTrue(result[0]["within_250ms"])
        # Measurement uncertainty must not silently widen the nominal metric.
        self.assertFalse(result[1]["within_250ms"])

    def test_invalid_or_overlapping_intervals_are_rejected(self):
        for intervals in (
            [],
            [{"start_s": -1, "end_s": 2}],
            [{"start_s": 1, "end_s": 1}],
            [{"start_s": 0, "end_s": float("nan")}],
            [{"start_s": 0, "end_s": 2}, {"start_s": 1, "end_s": 3}],
        ):
            with self.assertRaises(ValueError):
                validate_intervals(intervals)
        with self.assertRaises(ValueError):
            validate_intervals([{"start_s": 1, "end_s": 3}], duration=2)

    def test_no_machine_proposals_or_negative_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            entry = {"source": "machine", "intervals": [{"start_s": 1, "end_s": 2}]}
            path.write_text(
                json.dumps({"schema": 1, "events": {"/b/src/1/a.mp4": entry}})
            )
            with self.assertRaises(ValueError):
                load_annotations(
                    path, {"confirmed_positives": {"/b/src/1/a.mp4": "positive"}}
                )
            entry["source"] = "user"
            path.write_text(
                json.dumps({"schema": 1, "events": {"/b/src/1/a.mp4": entry}})
            )
            with self.assertRaises(ValueError):
                load_annotations(path, {"confirmed_positives": {}})


class ContextTests(unittest.TestCase):
    def test_crop_padding_keeps_time_alignment(self):
        mono = np.ones(1000, dtype=np.float32) * 0.25
        result = crop(mono, -0.01)
        self.assertEqual(len(result), PATCH_SAMPLES)
        np.testing.assert_array_equal(result[:160], 0)
        np.testing.assert_array_equal(result[160:1160], mono)
        np.testing.assert_array_equal(result[1160:], 0)

    def test_antiphase_channels_do_not_cancel_context(self):
        mono = sound(1)
        stereo = np.column_stack((mono, -mono))
        patches, starts = context_audio(stereo, {"start_position_s": 0})
        self.assertEqual(patches.shape, (10, PATCH_SAMPLES))
        self.assertIsNone(starts[0])
        self.assertGreater(float(np.std(patches[-1])), 0.01)

    def test_cross_thread_duplicates_are_excluded_from_training(self):
        rows = [
            {"threads": ["1", "2"], "fold_thread": "1"},
            {"threads": ["2"], "fold_thread": "2"},
            {"threads": ["3"], "fold_thread": "3"},
        ]
        train, test = split_indices(rows, "2")
        self.assertEqual(train.tolist(), [2])
        self.assertEqual(test.tolist(), [1])

    def test_export_roundtrip_and_missing_physical_feature(self):
        rng = np.random.default_rng(10)
        physical, embedding = rng.normal(size=(40, 4)), rng.normal(size=(40, 24))
        physical[0, 1] = np.nan
        y = np.asarray([0] * 30 + [1] * 10)
        for mode, components in (("physical", 0), ("embedding", 3), ("hybrid", 3)):
            fitted = fit(physical, embedding, y, mode, components)
            exported = json.loads(json.dumps(fitted))
            actual = predict(exported, physical, embedding)
            np.testing.assert_allclose(
                actual, predict(fitted, physical, embedding), rtol=1e-10, atol=1e-10
            )
            self.assertTrue(np.all(np.isfinite(actual)))


class FingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.waveform = sound(5)
        cls.pairs = landmarks(cls.waveform)
        cls.reference = {
            "path": "/b/src/1/positive.mp4",
            "audio_sha256": "source",
            "scope": "user-annotated-event",
            "start_s": 10.0,
            "duration_s": 2.0,
            "pairs": cls.pairs.tolist(),
        }
        cls.matcher = Matcher([cls.reference])

    def test_timing_gain_and_localization(self):
        query = np.concatenate(
            (
                np.zeros(48400, np.float32),
                self.waveform * 0.5,
                np.zeros(8000, np.float32),
            )
        )
        matches = self.matcher.match(landmarks(query))
        self.assertTrue(matches)
        self.assertTrue(matches[0]["confirmed_event_match"])
        self.assertAlmostEqual(matches[0]["query_match_start_s"], 3.025, delta=0.04)

    def test_self_source_is_excluded(self):
        self.assertEqual(
            self.matcher.match(self.pairs, exclude_audio_hash="source"), []
        )

    def test_silence_and_unrelated_recording_do_not_match(self):
        self.assertEqual(len(landmarks(np.zeros(16000, np.float32))), 0)
        self.assertEqual(self.matcher.match(landmarks(sound(99))), [])

    def test_exact_reupload_is_retained_when_scoring(self):
        matches = self.matcher.match(self.pairs, query_audio_hash="source")
        self.assertEqual(matches[0]["match_type"], "exact-audio")
        self.assertEqual(matches[0]["query_match_start_s"], 10.0)
        self.assertEqual(
            self.matcher.match(
                self.pairs, query_audio_hash="source", exclude_audio_hash="source"
            ),
            [],
        )

    def test_unannotated_clip_is_only_an_identity_hint(self):
        matcher = Matcher([self.reference | {"scope": "whole-positive-clip"}])
        matches = matcher.match(self.pairs)
        self.assertTrue(matches)
        self.assertFalse(matches[0]["confirmed_event_match"])


if __name__ == "__main__":
    unittest.main()
