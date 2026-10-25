"""Checks for the research objective, nested selection, and score semantics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import event_research_artifacts as artifacts
import numpy as np
from improve_event_semantics import semantic_features
from improve_events import (
    bag_objective,
    display_scores,
    nested_splits,
    pooled,
    select_policy,
)
from scipy.optimize import check_grad
from score_event_fusion import SCRIPT_SHA256, fuse, heuristic_scores


class EventResearchTests(unittest.TestCase):
    def test_fusion_preserves_continuous_scores_and_rejects_unknown_components(self):
        row = {"audio_sha256": "a", "score": 0.73}
        fused = fuse([row], {"a": 0.2})[0]
        self.assertEqual(fused["score"], 0.73)
        self.assertEqual(fused["risk_tier"], "maybe")
        self.assertEqual(fuse([row], {"a": 0.8})[0]["risk_tier"], "alert")
        for missing in ({}, {"a": None}, {"a": float("nan")}, {"a": True}):
            with self.assertRaises(ValueError):
                fuse([row], missing)

    def test_fusion_rejects_stale_or_failed_heuristic_runs(self):
        provenance = {"feature_dataset_sha256": "f", "labels_sha256": "l"}
        payload = {
            "current_sha256": SCRIPT_SHA256,
            "features_sha256": "f",
            "labels_sha256": "l",
            "rows": [{"audio_sha256": "a", "after": {"status": "ok", "score": 0.7}}],
        }
        self.assertEqual(heuristic_scores(payload, provenance), {"a": 0.7})
        for changed in (
            payload | {"current_sha256": "changed"},
            payload | {"labels_sha256": "stale"},
            payload | {"rows": [{"after": {"status": "error"}}]},
        ):
            with self.assertRaises(ValueError):
                heuristic_scores(changed, provenance)

    def test_pretrained_taxonomy_zero_padding_is_not_an_audio_prediction(self):
        audio = np.zeros((1, 3072))
        audio[0, 1024:2048] = 1
        head = {"kernel": np.zeros((1024, 2)), "bias": np.zeros(2)}
        # Real event produces sigmoid(0)=.5; missing before/after produces 0.
        np.testing.assert_allclose(semantic_features(audio, head), [[0.5] * 6])

    def test_snapshot_hash_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"model_sha256": "bad"}))
            (path / "models.joblib").write_bytes(b"untrusted fixture")
            with patch.object(artifacts.joblib, "load") as load:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    artifacts.load_fit(path)
                load.assert_not_called()

    def test_snapshot_cli_requires_explicit_pickle_trust(self):
        with (
            patch(
                "sys.argv",
                ["tool", "score", "--snapshot", "unused", "--output", "unused"],
            ),
            patch.object(artifacts, "score") as score,
        ):
            with self.assertRaisesRegex(ValueError, "verify origin"):
                artifacts.main()
            score.assert_not_called()

    def test_linear_softmax_objective_gradient(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(12, 5))
        owners = np.repeat(np.arange(4), 3)
        labels = np.asarray([0, 1, 0, 1])
        known = np.asarray([0, 1, 2, 3, 6, 7, 8])
        targets = np.asarray([0, 0, 0, 1, 0, 0, 0])
        weights = np.asarray([0.5, 0.5, 0.5, 3, 0.5, 0.5, 0.5])
        args = (x, owners, labels, known, targets, weights)
        theta = rng.normal(0, 0.3, 6)
        error = check_grad(
            lambda t: bag_objective(t, *args)[0],
            lambda t: bag_objective(t, *args)[1],
            theta,
        )
        self.assertLess(error, 1e-5)

    def test_pooling_is_bounded_and_independent_between_bags(self):
        result, _ = pooled(np.asarray([0.2, 0.8, 0.6]), np.asarray([0, 0, 1]), 2)
        np.testing.assert_allclose(result, [0.68, 0.6])

    def test_selection_preserves_each_detected_positive_not_just_total_recall(self):
        labels = np.asarray([1, 1, 0, 0, 0, 0])
        scores = {
            "event-hybrid": np.asarray([0.81, 0.1, 0.85, 0.83, 0.82, 0.1]),
            "swaps-miss": np.asarray([0.2, 0.99, 0.1, 0.1, 0.1, 0.1]),
            "better": np.asarray([0.9, 0.1, 0.2, 0.2, 0.2, 0.1]),
        }
        policy = select_policy(scores, labels, tuple(scores))
        self.assertEqual(policy["candidate"], "better")
        self.assertEqual(policy["false_positives"], 0)
        self.assertGreaterEqual(scores["better"][0], policy["threshold"])

    def test_no_positive_evidence_or_no_false_alerts_keeps_baseline(self):
        for labels, scores in (([0, 0], [0.9, 0.1]), ([1, 0], [0.9, 0.1])):
            policy = select_policy(
                {"event-hybrid": np.asarray(scores)},
                np.asarray(labels),
                ("event-hybrid",),
            )
            self.assertEqual(policy["threshold"], 0.8)

    def test_display_transform_keeps_threshold_at_red_and_is_monotonic(self):
        values = np.asarray([0, 0.2, 0.6, 0.85, 0.99, 1])
        scores = display_scores(values, 0.85)
        self.assertAlmostEqual(scores[3], 0.8)
        self.assertTrue(np.all(np.diff(scores) > 0))
        np.testing.assert_allclose(display_scores(values[1:-1], 0.8), values[1:-1])

    def test_inner_folds_cannot_reintroduce_outer_excluded_copies_or_families(self):
        folds = [
            {"thread": "a", "train": [2, 3, 4, 5], "test": [0, 1]},
            {"thread": "b", "train": [0, 1, 5], "test": [2, 3]},
            {"thread": "c", "train": [0, 1, 2, 3], "test": [4, 5]},
        ]
        for outer in nested_splits(folds):
            for inner in outer["inner"]:
                self.assertLessEqual(set(inner["train"]), set(outer["train"]))
                self.assertLessEqual(set(inner["test"]), set(outer["train"]))
                self.assertFalse(set(inner["train"]) & set(inner["test"]))
        inner_b = nested_splits(folds)[0]["inner"][0]
        self.assertEqual(inner_b["train"], [5])


if __name__ == "__main__":
    unittest.main()
