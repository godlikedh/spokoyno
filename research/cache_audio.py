#!/usr/bin/env python3
"""Cache analysis-ready float32 WAV audio from analysis rows or a 2ch thread API response."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MIRRORS = ("2ch.org", "2ch.su", "2ch.life")


def video_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if row.get("status", "ok") == "ok"]
    if not isinstance(payload, dict):
        return []
    rows = []
    for thread in payload.get("threads", []):
        for post in thread.get("posts", []):
            for media in post.get("files") or []:
                path = media.get("path", "")
                if Path(path).suffix.lower() not in {".mp4", ".webm", ".m4v", ".mov", ".ogv"}:
                    continue
                rows.append(
                    {
                        "file": Path(path).name,
                        "path": path,
                        "md5": media.get("md5", ""),
                        "status": "ok",
                    }
                )
    return rows


def hardlink_replace(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


def seed_media_duplicates(rows: list[dict], output: Path) -> int:
    existing_by_md5 = {}
    for row in rows:
        target = output / f"{row['file']}.audio.wav"
        digest = row.get("md5")
        if digest and target.exists() and target.stat().st_size:
            existing_by_md5.setdefault(digest, target)
    linked = 0
    for row in rows:
        target = output / f"{row['file']}.audio.wav"
        source = existing_by_md5.get(row.get("md5"))
        if source is None or target.exists():
            continue
        hardlink_replace(source, target)
        linked += 1
    return linked


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def deduplicate_audio(output: Path) -> tuple[int, int]:
    by_size: dict[int, list[Path]] = {}
    for path in output.glob("*.audio.wav"):
        if path.stat().st_size:
            by_size.setdefault(path.stat().st_size, []).append(path)
    canonical: dict[tuple[int, str], Path] = {}
    linked = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in sorted(paths):
            key = (size, file_sha256(path))
            source = canonical.setdefault(key, path)
            if source == path or os.path.samefile(source, path):
                continue
            hardlink_replace(source, path)
            linked += 1
    physical = len({(path.stat().st_dev, path.stat().st_ino) for path in output.glob("*.audio.wav")})
    return linked, physical


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
        error = result.stderr.decode("utf-8", "replace").strip()
        if "matches no streams" in error or "does not contain any stream" in error:
            return name, "no-audio"
        errors.append(error)
    return name, "FAILED: " + " | ".join(error for error in errors if error)[-500:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="analyze_audio.py JSON outputs"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.inputs:
        rows.extend(video_rows(json.loads(source.read_text())))
    unique = {row["file"]: row for row in rows}
    seeded = seed_media_duplicates(list(unique.values()), args.output)
    if seeded:
        print(f"seeded {seeded} known media duplicates as hard links", file=sys.stderr)
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
    if args.deduplicate:
        linked, physical = deduplicate_audio(args.output)
        print(
            f"deduplicated {linked} audio files; {physical} physical payloads remain",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
