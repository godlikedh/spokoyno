"""Build the current dataset and run reproducible model/feature/augmentation studies."""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from training.paths import ROOT, artifact_path, corpus_path
from training.storage import atomic_json, sha256


@dataclass(frozen=True)
class Step:
    name: str
    commands: tuple


def plan(run: Path, experiment: str, jobs=2, audio_python=None, timed=True):
    audio_python = audio_python or str(ROOT / ".venv-audio/bin/python")
    steps = []

    def command(module, *args, audio=False):
        return [
            audio_python if audio else sys.executable,
            "-m",
            module,
            *map(str, args),
        ]

    def add(name, module, *args, audio=False):
        steps.append(Step(name, (command(module, *args, audio=audio),)))

    add(
        "Extract physical features",
        "training.features.extract_features",
        "--workers",
        jobs,
    )
    add("Encode audio context", "training.features.extract_audio_context", audio=True)
    add(
        "Build timed and weakly labeled events",
        "training.data.event_dataset",
        audio=True,
    )
    add(
        "Build duplicate/family fingerprints",
        "training.data.fingerprint_audio",
        "build",
    )
    add("Group related sources", "training.data.fingerprint_audio", "evaluate")

    if experiment in {"models", "all"}:
        args = ["--output-dir", run / "models", "--jobs", jobs]
        if not timed:
            args += [
                "--candidates",
                "logistic-compact",
                "logistic-rich",
                "forest-rich-depth3",
                "extra-trees-rich-depth3",
                "hist-gradient-rich-depth2",
                "clip-embedding",
                "clip-hybrid",
                "event-mil-physical",
            ]
        add("Compare model families", "training.models.retrain_models", *args)
        add(
            "Save final baseline models",
            "training.export.score_retrained",
            "publish",
            "--run",
            run / "models",
            "--output",
            run / "final/models",
            "--trust-models",
        )

    if experiment in {"features", "augmentation", "all"}:
        add(
            "Fit event models and grouped splits",
            "training.models.improve_events",
            "--output-dir",
            run / "events",
            "--jobs",
            jobs,
        )

    if experiment in {"features", "all"}:
        add(
            "Extract additional acoustic features",
            "training.features.acoustic_features",
            "--output-dir",
            run / "acoustic",
            "--workers",
            jobs,
        )
        add(
            "Generate timed synthetic examples",
            "training.augmentation.synthetic_events",
            "--output-dir",
            run / "synthetic",
            audio=True,
        )
        add(
            "Compare feature families",
            "training.models.improve_acoustics",
            "--features",
            run / "acoustic",
            "--synthetic",
            run / "synthetic",
            "--baseline",
            run / "events",
            "--output-dir",
            run / "features",
            "--jobs",
            jobs,
            "--trust-models",
        )
        add(
            "Fit synthetic MIL variants",
            "training.models.improve_augmentation",
            "--features",
            run / "acoustic",
            "--synthetic",
            run / "synthetic",
            "--source",
            run / "features",
            "--output-dir",
            run / "synthetic-mil",
        )
        add(
            "Measure feature ablations",
            "training.evaluation.ablate_acoustics",
            "--features",
            run / "acoustic",
            "--synthetic",
            run / "synthetic",
            "--source",
            run / "synthetic-mil",
            "--output-dir",
            run / "ablation",
        )
        add(
            "Save final acoustic models",
            "training.export.acoustic_artifacts",
            "publish",
            "--run",
            run / "features",
            "--followup",
            run / "synthetic-mil",
            "--ablation",
            run / "ablation",
            "--features",
            run / "acoustic",
            "--output-dir",
            run / "final/acoustic",
        )
        add(
            "Export pretrained sound categories",
            "training.export.export_yamnet_head",
            "--output-dir",
            run / "taxonomy",
            audio=True,
        )
        add(
            "Compare semantic event models",
            "training.models.improve_event_semantics",
            "--source",
            run / "events",
            "--head",
            run / "taxonomy",
            "--output-dir",
            run / "semantic",
            "--jobs",
            jobs,
            "--trust-models",
        )
        add(
            "Save final event models",
            "training.export.event_research_artifacts",
            "publish",
            "--round-one",
            run / "events",
            "--round-two",
            run / "semantic",
            "--output",
            run / "final/events",
            "--trust-models",
        )

    if experiment in {"augmentation", "all"}:
        steps.append(
            Step(
                "Generate and encode codec/synthetic variants",
                (
                    command(
                        "training.augmentation.generate_robust_data",
                        "--output-dir",
                        run / "variants",
                        "--workers",
                        jobs,
                    ),
                    command(
                        "training.augmentation.encode_robust_data",
                        "--data-dir",
                        run / "variants",
                        audio=True,
                    ),
                ),
            )
        )
        add(
            "Assemble augmentation matrix",
            "training.augmentation.assemble_robust_data",
            "--data-dir",
            run / "variants",
            "--output-dir",
            run / "matrix",
        )
        common = [
            "--data-dir",
            run / "matrix",
            "--splits",
            run / "events/plan.json",
            "--jobs",
            jobs,
        ]
        add(
            "Train robust MIL",
            "training.models.train_robust_models",
            *common,
            "--output-dir",
            run / "robust",
        )
        add(
            "Compare codec-stable features",
            "training.models.train_codec_stable",
            *common,
            "--output-dir",
            run / "codec",
        )
        add(
            "Train robust trees",
            "training.models.train_robust_trees",
            *common,
            "--seed-run",
            run / "codec",
            "--output-dir",
            run / "trees",
        )
        add(
            "Train hard-negative models",
            "training.models.train_hard_negative_mil",
            *common,
            "--seed-run",
            run / "codec",
            "--output-dir",
            run / "hard-negative",
        )
        add(
            "Export encoder for context checks",
            "training.export.export_robust_encoder",
            "--output-dir",
            run / "encoder",
            audio=True,
        )
        add(
            "Export semantic context head",
            "training.export.export_context_head",
            "--fixtures",
            run / "encoder",
            "--output-dir",
            run / "context-head",
            audio=True,
        )
        add(
            "Assemble context features",
            "training.augmentation.assemble_context_data",
            "--base",
            run / "matrix",
            "--head-dir",
            run / "context-head",
            "--output-dir",
            run / "context-matrix",
        )
        context = [
            "--data-dir",
            run / "context-matrix",
            "--splits",
            run / "events/plan.json",
            "--jobs",
            jobs,
        ]
        add(
            "Compare sound-context features",
            "training.models.train_context_models",
            *context,
            "--output-dir",
            run / "context",
        )
        add(
            "Compare context trees",
            "training.models.train_context_trees",
            *context,
            "--seed-run",
            run / "context",
            "--output-dir",
            run / "context-trees",
        )
        for name in ("robust", "codec", "trees", "hard-negative", "context"):
            matrix = "context-matrix" if name == "context" else "matrix"
            add(
                f"Save final {name} models",
                "training.export.export_robust_models",
                "--data-dir",
                run / matrix,
                "--run",
                run / name,
                "--output-dir",
                run / "final" / name,
            )
        # Context-tree full-matrix serialization failed in the historical study.
        # Keep their fit/evaluation local; do not claim they are portable exports.
    return steps


def run_commands(commands, env=None):
    """Stop companion generation/encoding workers if either side fails."""
    processes = []
    try:
        for command in commands:
            print("+ " + shlex.join(command), flush=True)
            processes.append(
                (
                    command,
                    subprocess.Popen(
                        command, cwd=ROOT, env=env, start_new_session=True
                    ),
                )
            )
        while True:
            statuses = [p.poll() for _, p in processes]
            for (command, _), status in zip(processes, statuses, strict=True):
                if status not in (None, 0):
                    raise subprocess.CalledProcessError(status, command)
            if all(status is not None for status in statuses):
                return
            time.sleep(0.1)
    finally:
        for _, process in processes:
            if process.poll() is None:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()


def train(args):
    run = args.output.resolve()
    events = corpus_path("events.json")
    timed = events.exists() and bool(json.loads(events.read_text())["events"])
    steps = plan(run, args.experiment, args.jobs, args.audio_python, timed)
    if args.dry_run:
        for step in steps:
            print(step.name)
            for command in step.commands:
                print("  " + shlex.join(command))
        return
    required = [
        corpus_path(name) for name in ("index.json", "labels.json", "events.json")
    ]
    if any(not p.exists() for p in required):
        raise ValueError(
            "No complete local dataset. Import a reviewed thread or restore the corpus backup first."
        )
    if args.experiment != "models" and not timed:
        raise ValueError(
            "Feature/synthetic studies need exact event timings. Use --experiment models until timings are available."
        )
    if run.exists():
        raise ValueError(
            "Choose a new --output directory; existing training runs are preserved."
        )
    identity = {p.name: sha256(p) for p in required}
    identity["source"] = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / "training").rglob("*.py"))
    }
    identity["experiment"] = args.experiment
    identity["corpus"] = str(corpus_path())
    identity["cache"] = str(artifact_path())
    identity["commands"] = [step.commands for step in steps]
    metadata = run / "inputs.json"
    run.mkdir(parents=True, exist_ok=False)
    atomic_json(metadata, identity)
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("OMP_NUM_THREADS", "2")
    for number, step in enumerate(steps):
        print(f"[{number + 1}/{len(steps)}] {step.name}", flush=True)
        run_commands(step.commands, env)
    summaries = {}
    for p in sorted(run.rglob("results.json")):
        data = json.loads(p.read_text())
        if "metrics" in data:
            summaries[str(p.relative_to(run))] = data["metrics"]
        elif "experiments" in data:
            summaries[str(p.relative_to(run))] = {
                r["name"]: r["held_out"] for r in data["experiments"]
            }
    atomic_json(run / "summary.json", summaries)
    print(
        f"Final models: {run / 'final'}\nHeld-out comparisons: {run / 'summary.json'}"
    )
