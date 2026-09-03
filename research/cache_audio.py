#!/usr/bin/env python3
"""Cache analysis-ready float32 WAV audio for rows produced by analyze_audio.py."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

MIRRORS = ("2ch.org", "2ch.su", "2ch.life")


def cache_one(row: dict, output: Path) -> tuple[str, str]:
    name = row["file"]
    target = output / f"{name}.audio.wav"
    if target.exists() and target.stat().st_size:
        return name, "cached"
    path = row["path"]
    errors = []
    for mirror in MIRRORS:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-user_agent",
            "Mozilla/5.0",
            "-headers",
            f"Referer: https://{mirror}/b/\r\n",
            "-i",
            f"https://{mirror}{path}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "-y",
            str(temporary),
        ]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and temporary.exists() and temporary.stat().st_size:
            temporary.replace(target)
            return name, mirror
        temporary.unlink(missing_ok=True)
        errors.append(result.stderr.decode("utf-8", "replace").strip())
    return name, "FAILED: " + " | ".join(error for error in errors if error)[-500:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="analyze_audio.py JSON outputs"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.inputs:
        rows.extend(
            row for row in json.loads(source.read_text()) if row.get("status") == "ok"
        )
    unique = {row["file"]: row for row in rows}
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(cache_one, row, args.output) for row in unique.values()
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, result = future.result()
            failures += result.startswith("FAILED")
            print(
                f"[{index}/{len(futures)}] {name}: {result}",
                file=sys.stderr,
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
