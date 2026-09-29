"""Safeguards for versioned retraining, grouped splits, and weak event labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import retrain_models as retrain
import score_retrained as score


class RetrainingTests(unittest.TestCase):
    def test_new_feature_columns_cannot_shift_saved_model_inputs(self):
        data = {
            "names": ["new", "a", "b"],
            "physical": np.asarray([[99.0, 1.0, 2.0]]),
            "event_names": ["event"],
            "provenance": {"encoder_sha256": "encoder"},
        }
        manifest = {
            "feature_names": ["b", "a"],
            "event_feature_names": ["event"],
            "provenance": {"encoder_sha256": "encoder"},
        }
        aligned = score.align_inputs(data, manifest)
        np.testing.assert_array_equal(aligned["physical"], [[2.0, 1.0]])
        np.testing.assert_array_equal(data["physical"], [[99.0, 1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "missing fitted"):
            score.align_inputs(data, manifest | {"feature_names": ["missing"]})
        with self.assertRaisesRegex(ValueError, "encoder changed"):
            score.align_inputs(
                data, manifest | {"provenance": {"encoder_sha256": "other"}}
            )

    def test_pickle_loading_requires_explicit_trust(self):
        with (
            patch(
                "sys.argv",
                [
                    "score_retrained.py",
                    "score",
                    "--run",
                    "unused",
                    "--output",
                    "unused",
                ],
            ),
            patch.object(score, "score") as run,
            self.assertRaisesRegex(ValueError, "verify origin"),
        ):
            score.main()
        run.assert_not_called()

    def test_folds_include_negative_only_threads_and_all_copy_memberships(self):
        clips = [
            {
                "audio_sha256": "p1",
                "threads": ["a", "c"],
                "fold_thread": "a",
                "label": "positive",
            },
            {
                "audio_sha256": "p2",
                "threads": ["b"],
                "fold_thread": "b",
                "label": "positive",
            },
            {
                "audio_sha256": "p3",
                "threads": ["d"],
                "fold_thread": "d",
                "label": "positive",
            },
            {
                "audio_sha256": "n1",
                "threads": ["a"],
                "fold_thread": "a",
                "label": "negative",
            },
            {
                "audio_sha256": "n2",
                "threads": ["c"],
                "fold_thread": "c",
                "label": "negative",
            },
            {
                "audio_sha256": "n3",
                "threads": ["b"],
                "fold_thread": "b",
                "label": "negative",
            },
            {
                "audio_sha256": "u",
                "threads": ["a"],
                "fold_thread": "a",
                "label": "unlabeled",
            },
        ]
        links = {"p1": {"bridge"}, "bridge": {"p1", "p3"}, "p3": {"bridge"}}
        folds = retrain.folds_for(clips, links)
        c = next(f for f in folds if f["thread"] == "c")
        self.assertEqual(c["test"], [4])
        self.assertNotIn(0, c["train"])
        self.assertNotIn(2, c["train"])
        self.assertEqual(sorted(i for f in folds for i in f["test"]), list(range(6)))
        self.assertTrue(all(6 not in f["train"] for f in folds))

    def test_event_mil_selects_only_training_bags_and_never_mutates_targets(self):
        targets = np.asarray([0, 0, 1, -1, -1, -1, -1, -1], dtype=np.int8)
        data = {
            "clips": [
                {
                    "label": label,
                    "annotation": annotation,
                    "first_window": 2 * i,
                    "window_count": 2,
                }
                for i, (label, annotation) in enumerate(
                    [
                        ("negative", None),
                        ("positive", {}),
                        ("positive", None),
                        ("positive", None),
                    ]
                )
            ],
            "owners": np.repeat(np.arange(4), 2),
            "targets": targets,
            "event_physical": np.arange(8, dtype=float).reshape(-1, 1),
            "event_audio": np.zeros((8, 3)),
            "event_names": ["fixture"],
        }
        calls = []

        def fit(physical, audio, y, mode, components, weights):
            calls.append((physical[:, 0].tolist(), y.tolist()))
            return {}

        with (
            patch.object(retrain, "fit_context", side_effect=fit),
            patch.object(
                retrain,
                "predict_context",
                return_value=np.asarray([0.1, 0.1, 0.9, 0.3, 0.2, 0.8, 0.99, 0.99]),
            ),
        ):
            fitted = retrain.fit_candidate("event-mil-physical", data, [0, 1, 2])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0], ([0, 1, 2], [0, 0, 1]))
        self.assertTrue(all(call == ([0, 1, 2, 5], [0, 0, 1, 1]) for call in calls[1:]))
        self.assertEqual(fitted["latent_training_windows"], [5])
        self.assertEqual(fitted["used_clips"], [0, 1, 2])
        np.testing.assert_array_equal(targets, [0, 0, 1, -1, -1, -1, -1, -1])

    def test_annotated_event_model_does_not_claim_untimed_positives_as_training(self):
        data = {
            "clips": [],
            "owners": np.asarray([0, 0, 1, 1, 2, 2]),
            "targets": np.asarray([0, 0, 1, -1, -1, -1]),
            "event_physical": np.arange(6).reshape(-1, 1),
            "event_audio": np.zeros((6, 3)),
            "event_names": ["fixture"],
        }
        with patch.object(retrain, "fit_context", return_value={}):
            model = retrain.fit_candidate("event-physical", data, [0, 1, 2])
        self.assertEqual(model["used_clips"], [0, 1])
        self.assertEqual(model["latent_training_windows"], [])

    def test_threshold_diagnostic_is_distinct_from_fixed_red_and_yellow(self):
        rows = [
            {"label": label, "score": score, "above_training_threshold": score > 0.9}
            for label, score in [
                ("positive", 0.95),
                ("positive", 0.7),
                ("negative", 0.85),
                ("unlabeled", 0.99),
            ]
        ]
        result = retrain.summary(rows)
        self.assertEqual(result["positive"], 2)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["alert"]["detected"], 1)
        self.assertEqual(result["alert"]["false_warnings"], 1)
        self.assertEqual(result["yellow_only_positives"], 1)
        self.assertEqual(result["training_negative_max_policy"]["false_warnings"], 0)

    def test_all_tabular_families_handle_empty_feature_columns(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 4))
        x[:, -1] = np.nan
        y = np.asarray([0] * 40 + [1] * 20)
        for name in [
            "logistic-compact",
            "forest-rich-depth3",
            "extra-trees-rich-depth3",
            "hist-gradient-rich-depth2",
        ]:
            with self.subTest(name=name):
                model = retrain.tabular_pipeline(name).fit(x, y)
                scores = model.predict_proba(x)[:, 1]
                self.assertTrue(np.all(np.isfinite(scores)))
                self.assertEqual(
                    model.named_steps["imputer"].transform(x).shape, x.shape
                )
                if name == "hist-gradient-rich-depth2":
                    self.assertFalse(model.named_steps["model"].early_stopping)

    def test_existing_output_directory_is_rejected_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sentinel"
            target.write_text("keep")
            with (
                patch("sys.argv", ["retrain_models.py", "--output-dir", directory]),
                patch.object(retrain, "load_inputs") as load,
                self.assertRaisesRegex(ValueError, "refusing to overwrite"),
            ):
                retrain.main()
            load.assert_not_called()
            self.assertEqual(target.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
