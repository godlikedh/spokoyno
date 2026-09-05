#!/usr/bin/env python3
"""Fast smoke tests for the corpus, feature, and model utilities."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import build_corpus
import extract_features
import numpy as np
import train_models
from dataset import label_for
from scipy.io import wavfile


class CorpusTests(unittest.TestCase):
    def test_raw_api_normalization_and_hardlink(self) -> None:
        payload = {
            "threads": [
                {
                    "posts": [
                        {
                            "files": [
                                {"path": "/b/src/1/a.mp4", "md5": "same", "size": 2},
                                {"path": "/b/src/1/a.jpg", "md5": "image"},
                                {"path": "/b/src/1/b.webm", "md5": "same", "size": 2},
                            ]
                        }
                    ]
                }
            ]
        }
        rows = build_corpus.rows_from_payload(payload, "fixture")
        self.assertEqual([row["file"] for row in rows], ["a.mp4", "b.webm"])
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)
            first = audio / "a.mp4.audio.wav"
            first.write_bytes(b"fixture")
            self.assertEqual(build_corpus.seed_media_duplicates(rows, audio), 1)
            self.assertTrue(os.path.samefile(first, audio / "b.webm.audio.wav"))

    def test_decoded_audio_deduplication(self) -> None:
        rows = [
            {"path": "/b/src/1/a.mp4", "file": "a.mp4"},
            {"path": "/b/src/2/b.webm", "file": "b.webm"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)
            first = audio / "a.mp4.audio.wav"
            second = audio / "b.webm.audio.wav"
            first.write_bytes(b"same decoded audio")
            second.write_bytes(b"same decoded audio")
            hashes = {}
            for path in (first, second):
                stat = path.stat()
                hashes[(stat.st_dev, stat.st_ino)] = build_corpus.file_sha256(path)
            self.assertEqual(build_corpus.deduplicate_audio(rows, audio, hashes), 1)
            self.assertTrue(os.path.samefile(first, second))


class LabelTests(unittest.TestCase):
    def test_repository_labels(self) -> None:
        labels = json.loads(Path("corpus/labels.json").read_text())
        self.assertEqual(
            label_for("/b/src/336185346/17883557324650588814.webm", labels),
            "positive",
        )
        self.assertEqual(
            label_for("/b/src/336291305/17885181380740506509.mp4", labels),
            "visual-only",
        )
        self.assertEqual(
            label_for("/b/src/336272252/17884170000000000000.mp4", labels),
            "negative",
        )


class FeatureTests(unittest.TestCase):
    def test_synthetic_quiet_to_loud_event(self) -> None:
        rng = np.random.default_rng(7)
        samples = np.zeros((RATE := extract_features.RATE) * 4, dtype=np.float32)
        samples[: RATE * 2] = rng.normal(0, 0.001, RATE * 2)
        samples[RATE * 2 :] = rng.normal(0, 0.5, RATE * 2)
        stereo = np.column_stack((samples, samples))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.wav"
            wavfile.write(path, RATE, stereo)
            features = extract_features.extract_clip(path)
        self.assertGreater(features["event_1_jump_db"], 25)
        self.assertGreaterEqual(features["event_1_duration_s"], 1)
        self.assertGreater(len(features), 100)


class ModelTests(unittest.TestCase):
    def test_logistic_export_parity(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(40, 6))
        x[0, 2] = np.nan
        y = np.asarray([0] * 30 + [1] * 10)
        pipeline = train_models.experiments()[0].factory()
        pipeline.fit(x, y)
        exported = train_models.serialize_model(
            pipeline, [f"feature_{index}" for index in range(x.shape[1])]
        )
        expected = pipeline.predict_proba(x)[:, 1]
        actual = train_models.predict_serialized(exported, x)
        np.testing.assert_allclose(expected, actual, rtol=1e-8, atol=1e-10)

    def test_forest_export_parity_at_float32_boundaries(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.normal(size=(120, 8))
        x[0, 1] = np.nan
        y = np.asarray([0] * 90 + [1] * 30)
        pipeline = train_models.experiments()[-1].factory()
        pipeline.fit(x, y)
        exported = train_models.serialize_model(
            pipeline, [f"feature_{index}" for index in range(x.shape[1])]
        )
        self.assertEqual(exported["input_precision"], "float32")
        expected = pipeline.predict_proba(x)[:, 1]
        actual = train_models.predict_serialized(exported, x)
        np.testing.assert_allclose(expected, actual, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
