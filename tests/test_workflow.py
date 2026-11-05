"""The public workflow works with a new local dataset and stops failed pipelines."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from training.__main__ import main
from training.data import build_corpus
from training.data.dataset import label_for
from training.workflow import plan, run_commands


class WorkflowTests(unittest.TestCase):
    def test_initialize_is_local_and_preserves_existing_labels_and_timings(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--corpus", directory, "init"])
            labels = root / "labels.json"
            data = json.loads(labels.read_text())
            data["confirmed_positives"]["/b/src/1/a.mp4"] = "human label"
            labels.write_text(json.dumps(data))
            events = root / "events.json"
            before = {p: p.read_bytes() for p in (labels, events)}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--corpus", directory, "init"])
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--corpus", directory, "labels"])
            self.assertIn(
                "a.mp4\tpositive\ttiming pending\taudio unavailable", output.getvalue()
            )

    def test_first_reviewed_import_creates_labels_without_a_seed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            for name in ("screamer.mp4", "safe.mp4"):
                (audio / f"{name}.audio.wav").write_bytes(name.encode())
            snapshot = [
                {"path": "/b/src/1/screamer.mp4"},
                {"path": "/b/src/1/safe.mp4"},
            ]
            argv = [
                "build",
                "--thread",
                "https://2ch.life/b/res/1.html",
                "--reviewed",
                "--screamer",
                "screamer.mp4",
                "--audio-dir",
                str(audio),
                "--index",
                str(root / "index.json"),
                "--labels",
                str(root / "labels.json"),
            ]
            with (
                patch("sys.argv", argv),
                patch.object(build_corpus, "fetch_thread", return_value=snapshot),
                patch.object(build_corpus.subprocess, "run") as download,
            ):
                self.assertEqual(build_corpus.main(), 0)
                download.assert_not_called()
            labels = json.loads((root / "labels.json").read_text())
            self.assertEqual(label_for("/b/src/1/screamer.mp4", labels), "positive")
            self.assertEqual(label_for("/b/src/1/safe.mp4", labels), "negative")
            self.assertEqual(label_for("/b/src/1/later.mp4", labels), "unlabeled")
            self.assertEqual(
                json.loads((root / "manifests/b-1.json").read_text()), snapshot
            )

    def test_old_failed_downloads_do_not_trigger_network_retries(self):
        row = {
            "path": "/b/src/1/expired.mp4",
            "file": "expired.mp4",
            "_retained_failure": True,
            "error": "expired",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(build_corpus.subprocess, "run") as download,
        ):
            self.assertEqual(
                build_corpus.extract_one(row, Path(directory), build_corpus.MIRRORS),
                (row["path"], "failed", "expired"),
            )
            download.assert_not_called()

    def test_dry_run_works_without_personal_data_and_uses_new_run_dependencies(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(
                    [
                        "--corpus",
                        str(root / "corpus"),
                        "--cache",
                        str(root / "cache"),
                        "train",
                        "--experiment",
                        "all",
                        "--output",
                        str(root / "run"),
                        "--dry-run",
                    ]
                )
            self.assertFalse((root / "corpus").exists())
            self.assertFalse((root / "run").exists())
            self.assertNotIn("research/", output.getvalue())
            self.assertIn(
                "training.augmentation.generate_robust_data", output.getvalue()
            )
            self.assertIn(str(root / "run/events"), output.getvalue())
        untimed = plan(Path("new-run"), "models", timed=False)
        fitting = next(
            s.commands[0] for s in untimed if s.name == "Compare model families"
        )
        self.assertIn("event-mil-physical", fitting)
        self.assertNotIn("event-physical", fitting)
        self.assertNotIn("event-hybrid", fitting)

    def test_failed_encoder_stops_its_companion_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "producer"
            producer = [
                sys.executable,
                "-c",
                "import os,pathlib,time,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
                str(ready),
            ]
            consumer = [
                sys.executable,
                "-c",
                "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.01)\nraise SystemExit(7)",
                str(ready),
            ]
            start = time.monotonic()
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                run_commands((producer, consumer))
            self.assertEqual(failure.exception.returncode, 7)
            self.assertLess(time.monotonic() - start, 10)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(ready.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
