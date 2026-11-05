"""Import reviewed threads, record exact timings, and train local audio models."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from training.data.dataset import label_for, load_labels
from training.paths import ROOT, corpus_path
from training.storage import atomic_json
from training.workflow import run_commands, train


def initialize():
    labels = corpus_path("labels.json")
    if not labels.exists():
        atomic_json(labels, load_labels(labels))
    events = corpus_path("events.json")
    if not events.exists():
        atomic_json(
            events,
            {
                "schema": 1,
                "time_reference": "Seconds from the beginning of retained audio; user-confirmed intervals only.",
                "events": {},
            },
        )
    corpus_path("audio").mkdir(parents=True, exist_ok=True)


def show_labels():
    labels = load_labels(corpus_path("labels.json"))
    index = corpus_path("index.json")
    rows = json.loads(index.read_text())["items"] if index.exists() else []
    by_path = {r["path"]: r for r in rows}
    events_path = corpus_path("events.json")
    events = (
        json.loads(events_path.read_text())["events"] if events_path.exists() else {}
    )
    positives = labels.get("confirmed_positives", {})
    print(
        f"{len(positives)} confirmed screamer paths; {len(events)} with exact timings; "
        f"{sum(r.get('status') == 'ok' for r in rows)} retained audio paths."
    )
    for path in sorted(positives):
        row = by_path.get(path, {})
        intervals = events.get(path, {}).get("intervals", [])
        times = (
            ", ".join(f"{v['start_s']:g}–{v['end_s']:g}s" for v in intervals)
            or "timing pending"
        )
        audio = corpus_path("audio") / row.get("audio_file", "missing")
        state = (
            "available"
            if row.get("status") == "ok" and audio.is_file()
            else "audio unavailable"
        )
        label = label_for(path, labels)
        print(f"{Path(path).name}\t{label}\t{times}\t{state}")


def training_options(parser):
    parser.add_argument(
        "--experiment",
        choices=("models", "features", "augmentation", "all"),
        default="models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--audio-python", default=str(ROOT / ".venv-audio/bin/python"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the complete training sequence without executing it",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, help="local dataset directory; defaults to corpus/"
    )
    parser.add_argument(
        "--cache", type=Path, help="local generated feature/encoder cache"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize an empty local label/timing store")
    commands.add_parser(
        "labels", help="list screamers, exact timings, and retained-audio availability"
    )
    add = commands.add_parser(
        "add",
        help="import a reviewed thread; unlisted attachments in its snapshot are negative",
    )
    add.add_argument("thread")
    add.add_argument("--screamer", action="append", default=[])
    add.add_argument(
        "--train",
        action="store_true",
        help="retrain after importing the reviewed snapshot",
    )
    training_options(add)
    timing = commands.add_parser(
        "timings", help="record user-provided intervals on an existing screamer"
    )
    timing.add_argument("file")
    timing.add_argument(
        "--interval",
        action="append",
        required=True,
        help="START:END seconds; END may be end",
    )
    timing.add_argument("--uncertainty", type=float)
    timing.add_argument("--note", default="")
    fitting = commands.add_parser(
        "train", help="rebuild inputs and train/evaluate the selected experiments"
    )
    training_options(fitting)
    args = parser.parse_args(argv)
    if args.corpus:
        os.environ["SPOKOYNO_CORPUS"] = str(args.corpus.resolve())
    if args.cache:
        os.environ["SPOKOYNO_CACHE"] = str(args.cache.resolve())
    try:
        if args.command == "init":
            initialize()
            print(f"Local dataset: {corpus_path()}")
        elif args.command == "labels":
            show_labels()
        elif args.command == "add":
            if args.dry_run:
                raise ValueError(
                    "Use train --dry-run to preview training; add downloads and records a real reviewed snapshot."
                )
            initialize()
            command = [
                sys.executable,
                "-m",
                "training.data.build_corpus",
                "--thread",
                args.thread,
                "--reviewed",
                "--allow-download-failures",
            ]
            for name in args.screamer:
                command += ["--screamer", name]
            run_commands((command,))
            if args.train:
                train(args)
        elif args.command == "timings":
            command = [
                sys.executable,
                "-m",
                "training.data.event_annotations",
                "--set",
                args.file,
                "--note",
                args.note,
            ]
            for interval in args.interval:
                command += ["--interval", interval]
            if args.uncertainty is not None:
                command += ["--uncertainty", str(args.uncertainty)]
            run_commands((command,))
        else:
            train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(
            error.returncode,
            "Training/import stopped after the failed command above.\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
