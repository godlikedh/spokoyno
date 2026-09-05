#!/usr/bin/env python3
"""Build Spokoyno's local, audio-only corpus and a deterministic content index."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MIRRORS = ("2ch.org", "2ch.su", "2ch.life")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".mov", ".ogv"}
AUDIO_SUFFIX = ".audio.wav"
THREAD_RE = re.compile(r"^/([^/]+)/res/(\d+)(?:\.(?:html|json))?/?$")


def canonical_path(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value.split("?", 1)[0]
    return path if path.startswith("/") else f"/{path}"


def thread_parts(value: str) -> tuple[str, str, str | None]:
    parsed = urlparse(value)
    match = THREAD_RE.match(parsed.path if parsed.scheme else canonical_path(value))
    if not match:
        raise ValueError(f"not a 2ch thread URL/path: {value}")
    board, thread = match.groups()
    return board, thread, parsed.hostname


def fetch_thread(value: str, mirrors: tuple[str, ...]) -> dict:
    board, thread, requested_host = thread_parts(value)
    order = [requested_host, *mirrors] if requested_host in mirrors else list(mirrors)
    errors = []
    for host in dict.fromkeys(order):
        url = f"https://{host}/{board}/res/{thread}.json"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": f"https://{host}/{board}/",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.URLError) as error:
            errors.append(f"{host}: {error}")
    raise RuntimeError("thread fetch failed: " + " | ".join(errors))


def normalized_row(media: dict, source: str) -> dict | None:
    path = canonical_path(str(media.get("path", "")))
    if Path(path).suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    parts = path.strip("/").split("/")
    thread = parts[2] if len(parts) >= 4 and parts[1] == "src" else ""
    return {
        "path": path,
        "file": Path(path).name,
        "board": parts[0] if parts else "",
        "thread": thread,
        "media_md5": media.get("md5") or media.get("media_md5") or "",
        "source_bytes": media.get("size_bytes")
        or media.get("source_bytes")
        or (int(media["size"]) * 1024 if media.get("size") is not None else None)
        or (
            int(media["api_size_kib"]) * 1024
            if media.get("api_size_kib") is not None
            else None
        ),
        "source_duration_s": media.get("duration_s")
        or media.get("duration_secs")
        or media.get("api_duration_s"),
        "status": media.get("status", "ok"),
        "source": source,
    }


def rows_from_payload(payload: object, source: str) -> list[dict]:
    if isinstance(payload, list):
        return [row for item in payload if (row := normalized_row(item, source))]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported JSON payload in {source}")
    rows = []
    for thread in payload.get("threads", []):
        for post in thread.get("posts", []):
            for media in post.get("files") or []:
                row = normalized_row(media, source)
                if row:
                    rows.append(row)
    return rows


def merge_rows(rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    filenames: dict[str, str] = {}
    for row in rows:
        previous_path = filenames.setdefault(row["file"], row["path"])
        if previous_path != row["path"]:
            raise ValueError(
                f"audio filename collision: {row['file']} identifies both "
                f"{previous_path} and {row['path']}"
            )
        current = merged.setdefault(row["path"], row.copy())
        for key, value in row.items():
            if value not in (None, "", "unlabeled") and current.get(key) in (None, ""):
                current[key] = value
        if row.get("status") == "no-audio":
            current["status"] = "no-audio"
    return sorted(merged.values(), key=lambda row: row["path"])


def audio_path(audio_dir: Path, row: dict) -> Path:
    return audio_dir / f"{row['file']}{AUDIO_SUFFIX}"


def hardlink_replace(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


def seed_media_duplicates(rows: list[dict], audio_dir: Path) -> int:
    by_md5: dict[str, Path] = {}
    for row in rows:
        target = audio_path(audio_dir, row)
        digest = row.get("media_md5")
        if digest and target.exists() and target.stat().st_size:
            by_md5.setdefault(digest, target)
    linked = 0
    for row in rows:
        target = audio_path(audio_dir, row)
        source = by_md5.get(row.get("media_md5", ""))
        if source is None or target.exists() or row.get("status") == "no-audio":
            continue
        hardlink_replace(source, target)
        linked += 1
    return linked


def extract_one(
    row: dict, audio_dir: Path, mirrors: tuple[str, ...]
) -> tuple[str, str, str]:
    target = audio_path(audio_dir, row)
    if row.get("status") == "no-audio":
        return row["path"], "no-audio", ""
    if target.exists() and target.stat().st_size:
        return row["path"], "ok", "cached"
    errors = []
    for mirror in mirrors:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-user_agent",
            "Mozilla/5.0",
            "-headers",
            f"Referer: https://{mirror}/{row['board']}/\r\n",
            "-rw_timeout",
            "30000000",
            "-i",
            f"https://{mirror}{row['path']}",
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
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False
        )
        if result.returncode == 0 and temporary.exists() and temporary.stat().st_size:
            temporary.replace(target)
            return row["path"], "ok", mirror
        temporary.unlink(missing_ok=True)
        error = result.stderr.decode("utf-8", "replace").strip()
        if "matches no streams" in error or "does not contain any stream" in error:
            return row["path"], "no-audio", ""
        errors.append(error)
    return row["path"], "failed", " | ".join(filter(None, errors))[-1000:]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hash_audio(
    rows: list[dict], audio_dir: Path, previous_items: dict[str, dict], workers: int
) -> dict[tuple[int, int], str]:
    paths_by_inode: dict[tuple[int, int], Path] = {}
    cached: dict[tuple[int, int], str] = {}
    for row in rows:
        path = audio_path(audio_dir, row)
        if not path.exists() or not path.stat().st_size:
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        paths_by_inode.setdefault(identity, path)
        old = previous_items.get(row["path"], {})
        if old.get("audio_bytes") == stat.st_size and old.get("audio_sha256"):
            cached[identity] = old["audio_sha256"]
    missing = {
        identity: path
        for identity, path in paths_by_inode.items()
        if identity not in cached
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(file_sha256, path): identity
            for identity, path in missing.items()
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            cached[futures[future]] = future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"[hash {index}/{len(futures)}]", file=sys.stderr, flush=True)
    return cached


def deduplicate_audio(rows: list[dict], audio_dir: Path, hashes: dict) -> int:
    canonical: dict[tuple[int, str], Path] = {}
    linked = 0
    for row in rows:
        path = audio_path(audio_dir, row)
        if not path.exists() or not path.stat().st_size:
            continue
        stat = path.stat()
        digest = hashes[(stat.st_dev, stat.st_ino)]
        source = canonical.setdefault((stat.st_size, digest), path)
        if source == path or os.path.samefile(source, path):
            continue
        hardlink_replace(source, path)
        linked += 1
    return linked


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_index(
    rows: list[dict],
    audio_dir: Path,
    index_path: Path,
    mirrors: tuple[str, ...],
    workers: int,
    should_deduplicate: bool,
) -> tuple[dict, int]:
    previous = {}
    if index_path.exists():
        previous_payload = json.loads(index_path.read_text())
        previous = {item["path"]: item for item in previous_payload.get("items", [])}

    seeded = seed_media_duplicates(rows, audio_dir)
    if seeded:
        print(f"seeded {seeded} media-MD5 duplicates", file=sys.stderr)

    outcomes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_one, row, audio_dir, mirrors): row for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path, status, detail = future.result()
            outcomes[path] = (status, detail)
            if detail != "cached" or index % 50 == 0 or index == len(futures):
                print(
                    f"[audio {index}/{len(futures)}] {Path(path).name}: {status}"
                    + (f" ({detail})" if detail else ""),
                    file=sys.stderr,
                    flush=True,
                )

    hashes = hash_audio(rows, audio_dir, previous, workers)
    hash_by_path = {}
    for row in rows:
        path = audio_path(audio_dir, row)
        if path.exists() and path.stat().st_size:
            stat = path.stat()
            hash_by_path[row["path"]] = hashes[(stat.st_dev, stat.st_ino)]
    linked = deduplicate_audio(rows, audio_dir, hashes) if should_deduplicate else 0
    if linked:
        print(f"hard-linked {linked} duplicate audio payloads", file=sys.stderr)

    # Hard-link replacement changes inode identities. Re-index them from the hashes captured
    # immediately before replacement; content is unchanged, so no second full-corpus hash pass is
    # necessary.
    final_hashes = {}
    for row in rows:
        path = audio_path(audio_dir, row)
        if path.exists() and path.stat().st_size:
            stat = path.stat()
            final_hashes[(stat.st_dev, stat.st_ino)] = hash_by_path[row["path"]]

    groups: dict[tuple[int, str], list[tuple[dict, Path]]] = {}
    for row in rows:
        path = audio_path(audio_dir, row)
        if not path.exists() or not path.stat().st_size:
            continue
        stat = path.stat()
        digest = final_hashes[(stat.st_dev, stat.st_ino)]
        groups.setdefault((stat.st_size, digest), []).append((row, path))
    duplicate_of = {}
    for members in groups.values():
        canonical = min(row["path"] for row, _ in members)
        for row, _ in members:
            duplicate_of[row["path"]] = None if row["path"] == canonical else canonical

    items = []
    failures = 0
    for row in rows:
        target = audio_path(audio_dir, row)
        status, detail = outcomes.get(row["path"], (row.get("status", "failed"), ""))
        if status == "ok" and (not target.exists() or not target.stat().st_size):
            status, detail = "failed", "audio file missing after extraction"
        item = {
            key: row.get(key)
            for key in (
                "path",
                "file",
                "board",
                "thread",
                "media_md5",
                "source_bytes",
                "source_duration_s",
                "source",
            )
        }
        item["status"] = status
        if status == "ok":
            stat = target.stat()
            item.update(
                {
                    "audio_file": target.name,
                    "audio_bytes": stat.st_size,
                    "audio_sha256": final_hashes[(stat.st_dev, stat.st_ino)],
                    "duplicate_of": duplicate_of.get(row["path"]),
                }
            )
        elif status == "failed":
            failures += 1
            item["error"] = detail
        items.append(item)

    payload = {
        "schema": 1,
        "audio": {
            "sample_rate": 16000,
            "channels": 2,
            "sample_format": "pcm_f32le",
            "container": "wav",
        },
        "items": items,
    }
    atomic_json(index_path, payload)
    return payload, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources", nargs="*", type=Path, help="JSON manifests/API responses"
    )
    parser.add_argument("--thread", action="append", default=[], help="2ch thread URL")
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="user reviewed this thread; non-screamer attachments in this snapshot are negative",
    )
    parser.add_argument(
        "--screamer",
        action="append",
        default=[],
        help="confirmed audio screamer filename/path; repeat; timings can follow later",
    )
    parser.add_argument("--labels", type=Path, default=Path("corpus/labels.json"))
    parser.add_argument(
        "--audio-dir",
        "--output",
        dest="audio_dir",
        type=Path,
        default=Path("corpus/audio"),
    )
    parser.add_argument("--index", type=Path, default=Path("corpus/index.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mirrors", nargs="+", default=list(MIRRORS))
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument(
        "--replace-index",
        action="store_true",
        help="drop index entries not present in the supplied sources",
    )
    parser.add_argument("--deduplicate", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sources and not args.thread:
        raise SystemExit("provide at least one JSON source or --thread URL")
    if args.screamer and not args.reviewed:
        raise SystemExit("--screamer requires --reviewed")
    if args.reviewed and (len(args.thread) != 1 or args.sources or args.replace_index):
        raise SystemExit(
            "--reviewed requires exactly one --thread and no other sources/--replace-index"
        )
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.sources:
        rows.extend(rows_from_payload(json.loads(source.read_text()), source.name))
    for thread in args.thread:
        board, number, _ = thread_parts(thread)
        rows.extend(
            rows_from_payload(
                fetch_thread(thread, tuple(args.mirrors)), f"{board}/{number}"
            )
        )
    updated_labels = None
    if args.reviewed:
        from review_labels import reviewed_labels

        updated_labels = reviewed_labels(
            json.loads(args.labels.read_text()),
            args.thread[0],
            merge_rows(rows),
            args.screamer,
        )
    if args.index.exists() and not args.replace_index:
        supplied = {row["path"] for row in rows}
        previous = json.loads(args.index.read_text())
        rows.extend(
            item for item in previous.get("items", []) if item["path"] not in supplied
        )
    merged = merge_rows(rows)
    payload, failures = build_index(
        merged,
        args.audio_dir,
        args.index,
        tuple(args.mirrors),
        max(1, args.workers),
        not args.no_deduplicate,
    )
    if updated_labels is not None:
        atomic_json(args.labels, updated_labels)
        print(
            "Saved reviewed-thread labels. Failed downloads remain failed, not successful safe analyses; timings can be supplied later.",
            file=sys.stderr,
        )
    counts = {}
    for item in payload["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    unique_audio = len(
        {item["audio_sha256"] for item in payload["items"] if item["status"] == "ok"}
    )
    print(
        f"indexed {len(payload['items'])} media: {counts}; "
        f"{unique_audio} unique audio payloads",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
