"""Regression tests for staged reviews, uncertain event labels, and score policies."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_corpus
import numpy as np
from dataset import label_for
from event_annotations import main as annotate
from event_dataset import grouped_rows, physical_events, proposal_points, window_label
from extract_features import RATE
from fingerprint_audio import aac_roundtrip
from review_labels import reviewed_labels
from risk_score import risk_tier
from scipy.io import wavfile
from train_audio_context import fit, predict
from train_events import clip_predictions, family_links, split_clips, training_weights


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"file": "a.mp4", "path": "/b/src/1/a.mp4"},
            {"file": "b.webm", "path": "/b/src/1/b.webm"},
        ]

    def test_review_labels_other_videos_negative_but_not_future_posts(self):
        original = {"schema": 1, "confirmed_positives": {}}
        labels = reviewed_labels(
            original, "https://2ch.life/b/res/1.html", self.rows, ["a.mp4"]
        )
        self.assertEqual(original["confirmed_positives"], {})
        self.assertEqual(label_for("/b/src/1/a.mp4", labels), "positive")
        self.assertEqual(label_for("/b/src/1/b.webm", labels), "negative")
        self.assertEqual(label_for("/b/src/1/later.mp4", labels), "unlabeled")

    def test_zero_screamers_is_a_valid_review_and_typo_fails(self):
        labels = reviewed_labels({}, "/b/res/1.html", self.rows, [])
        self.assertEqual(label_for("/b/src/1/a.mp4", labels), "negative")
        with self.assertRaises(ValueError):
            reviewed_labels({}, "/b/res/1.html", self.rows, ["typo.mp4"])
        with self.assertRaises(ValueError):
            reviewed_labels({}, "/b/res/2.html", self.rows, [])

    def test_incremental_reviews_preserve_earlier_positive_and_reviewed_paths(self):
        labels = reviewed_labels({}, "/b/res/1.html", self.rows, ["a.mp4"])
        labels = reviewed_labels(
            labels, "/b/res/1.html", [{"path": "/b/src/1/c.mp4", "file": "c.mp4"}], []
        )
        self.assertEqual(label_for("/b/src/1/a.mp4", labels), "positive")
        self.assertEqual(label_for("/b/src/1/b.webm", labels), "negative")
        self.assertEqual(label_for("/b/src/1/c.mp4", labels), "negative")

    def test_later_annotation_accepts_end_and_preserves_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wavfile.write(root / "a.wav", RATE, np.zeros((RATE * 2, 2), np.float32))
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "path": "/b/src/1/a.mp4",
                                "file": "a.mp4",
                                "audio_file": "a.wav",
                                "audio_sha256": "hash",
                            }
                        ]
                    }
                )
            )
            (root / "labels.json").write_text(
                json.dumps({"confirmed_positives": {"/b/src/1/a.mp4": "user"}})
            )
            command = [
                "event_annotations.py",
                "--index",
                str(root / "index.json"),
                "--labels",
                str(root / "labels.json"),
                "--events",
                str(root / "events.json"),
                "--audio-dir",
                str(root),
                "--set",
                "a.mp4",
                "--interval",
                "1.2:end",
            ]
            with patch("sys.argv", command + ["--uncertainty", "0.05"]):
                self.assertEqual(annotate(), 0)
            with patch("sys.argv", command):
                self.assertEqual(annotate(), 0)
            entry = json.loads((root / "events.json").read_text())["events"][
                "/b/src/1/a.mp4"
            ]
            self.assertEqual(entry["estimated_boundary_error_s"], 0.05)
            self.assertEqual(entry["intervals"], [{"start_s": 1.2, "end_s": 2.0}])

    def test_builder_records_review_but_keeps_failed_download_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.json"
            labels_path.write_text(json.dumps({"schema": 1, "confirmed_positives": {}}))
            args = [
                "build_corpus.py",
                "--thread",
                "https://2ch.life/b/res/1.html",
                "--reviewed",
                "--screamer",
                "a.mp4",
                "--labels",
                str(labels_path),
                "--index",
                str(root / "index.json"),
                "--audio-dir",
                str(root / "audio"),
            ]
            outcome = {
                "items": [
                    {
                        "path": self.rows[0]["path"],
                        "status": "ok",
                        "audio_sha256": "hash",
                    },
                    {"path": self.rows[1]["path"], "status": "failed"},
                ]
            }
            with (
                patch("sys.argv", args),
                patch("build_corpus.fetch_thread", return_value=self.rows),
                patch("build_corpus.build_index", return_value=(outcome, 1)),
            ):
                self.assertEqual(build_corpus.main(), 1)
            labels = json.loads(labels_path.read_text())
            self.assertEqual(label_for(self.rows[0]["path"], labels), "positive")
            self.assertEqual(label_for(self.rows[1]["path"], labels), "negative")
            self.assertEqual(outcome["items"][1]["status"], "failed")

    def test_builder_rejects_bad_screamer_before_download_or_label_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.json"
            original = '{"schema": 1, "confirmed_positives": {}}'
            labels_path.write_text(original)
            args = [
                "build_corpus.py",
                "--thread",
                "/b/res/1.html",
                "--reviewed",
                "--screamer",
                "missing.mp4",
                "--labels",
                str(labels_path),
                "--index",
                str(root / "index.json"),
                "--audio-dir",
                str(root / "audio"),
            ]
            with (
                patch("sys.argv", args),
                patch("build_corpus.fetch_thread", return_value=self.rows),
                patch("build_corpus.build_index") as build,
                self.assertRaises(ValueError),
            ):
                build_corpus.main()
            build.assert_not_called()
            self.assertEqual(labels_path.read_text(), original)


class EventDatasetTests(unittest.TestCase):
    def test_aac_roundtrip_preserves_stereo_duration_and_channel_order(self):
        time = np.arange(RATE) / RATE
        samples = np.column_stack(
            (0.1 * np.sin(2 * np.pi * 440 * time), 0.1 * np.sin(2 * np.pi * 880 * time))
        ).astype(np.float32)
        decoded = aac_roundtrip(samples)
        self.assertEqual(decoded.shape[1], 2)
        self.assertLess(abs(len(decoded) - len(samples)), RATE // 4)
        for channel, expected in ((0, 440), (1, 880)):
            spectrum = np.abs(np.fft.rfft(decoded[:, channel]))
            frequency = np.argmax(spectrum) * RATE / len(decoded)
            self.assertAlmostEqual(frequency, expected, delta=5)

    def test_uncertain_edges_missing_timings_and_context_are_not_negative(self):
        annotation = {
            "estimated_boundary_error_s": 0.05,
            "intervals": [{"start_s": 10.0, "end_s": 12.0}],
        }
        self.assertEqual(window_label("positive", None, 10.2, 10.5, 20), -1)
        self.assertEqual(window_label("positive", annotation, 10.02, 10.32, 20), -1)
        self.assertEqual(window_label("positive", annotation, 10.1, 10.4, 20), 1)
        self.assertEqual(window_label("positive", annotation, 9, 9.3, 20), -1)
        self.assertEqual(window_label("positive", annotation, 0, 0.3, 20), 0)
        self.assertEqual(window_label("negative", None, 10.2, 10.5, 20), 0)
        self.assertEqual(window_label("unlabeled", None, 10.2, 10.5, 20), -1)
        self.assertEqual(window_label("positive", annotation, 11.7, 12, 12), 1)

    def test_proposals_have_no_annotation_input_and_opening_has_no_future_baseline(
        self,
    ):
        features = {"start_position_s": 0.0, "event_1_position_fraction": 40 / 79}
        self.assertEqual(proposal_points(features, RATE * 4), [(0, 40), (3, 0)])
        samples = np.zeros((RATE * 4, 2), np.float32)
        samples[RATE * 2 :] = 0.25
        vectors, windows = physical_events(samples, features)
        self.assertEqual(windows[-1]["start_s"], 0.0)
        self.assertTrue(np.isnan(vectors[-1]["baseline_3_0s_db"]))
        self.assertNotIn("position_fraction", vectors[0])

    def test_exact_audio_conflicts_are_rejected(self):
        rows = [
            {"path": "/b/src/1/a.mp4", "audio_sha256": "same"},
            {"path": "/b/src/2/b.mp4", "audio_sha256": "same"},
        ]
        labels = {
            "confirmed_positives": {rows[0]["path"]: "user"},
            "reviewed_negatives": {rows[1]["path"]: "user"},
        }
        with self.assertRaises(ValueError):
            grouped_rows(rows, labels, {"events": {}})


class EventTrainingTests(unittest.TestCase):
    def test_family_exclusion_includes_quiet_edits_and_transitive_matches(self):
        clips = [
            {
                "audio_sha256": "a",
                "threads": ["1"],
                "fold_thread": "1",
                "label": "positive",
            },
            {
                "audio_sha256": "b",
                "threads": ["2"],
                "fold_thread": "2",
                "label": "negative",
            },
            {
                "audio_sha256": "c",
                "threads": ["3"],
                "fold_thread": "3",
                "label": "negative",
            },
            {
                "audio_sha256": "d",
                "threads": ["3"],
                "fold_thread": "3",
                "label": "positive",
            },
        ]
        links = family_links(
            {
                "rows": [
                    {"audio_sha256": "a", "matches": [{"reference_audio_sha256": "b"}]},
                    {"audio_sha256": "b", "matches": [{"reference_audio_sha256": "c"}]},
                ]
            }
        )
        train, test = split_clips(clips, "1", links)
        self.assertEqual(train, [3])
        self.assertEqual(test, [0])

    def test_repeated_windows_do_not_increase_clip_weight(self):
        owners = np.asarray([0, 0, 0, 1, 2, 3])
        y = np.asarray([1, 1, 1, 1, 0, 0])
        weights = training_weights(owners, y)
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights[y == 0].sum(), weights[y == 1].sum())

    def test_weighted_export_roundtrip(self):
        rng = np.random.default_rng(44)
        x, audio = rng.normal(size=(24, 4)), rng.normal(size=(24, 12))
        y = np.asarray([0] * 16 + [1] * 8)
        weights = training_weights(np.arange(24), y)
        for mode in ("physical", "hybrid"):
            model = fit(x, audio, y, mode, 3, weights)
            np.testing.assert_allclose(
                predict(model, x, audio),
                predict(json.loads(json.dumps(model)), x, audio),
            )

    def test_clip_score_uses_all_automatic_windows_not_manual_timing(self):
        clip = {
            "path": "clip",
            "audio_sha256": "hash",
            "label": "positive",
            "first_window": 0,
            "window_count": 2,
            "annotation": {"intervals": [{"start_s": 1, "end_s": 2}]},
        }
        result = clip_predictions(
            [clip], [{"start_s": 1}, {"start_s": 9}], np.asarray([0.2, 0.9]), [0]
        )[0]
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["event_at_s"], 9)
        self.assertEqual(result["risk_tier"], "alert")

    def test_tier_boundaries_and_invalid_scores(self):
        for score, tier in (
            (0.0, "low"),
            (0.59999, "low"),
            (0.6, "maybe"),
            (0.79999, "maybe"),
            (0.8, "alert"),
            (1.0, "alert"),
            (None, "unknown"),
            (float("nan"), "unknown"),
            (-1.0, "unknown"),
        ):
            self.assertEqual(risk_tier(score), tier)


if __name__ == "__main__":
    unittest.main()
