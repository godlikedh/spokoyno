"""Locations for source code and local-only datasets, caches, and training runs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def corpus_path(name=""):
    return Path(os.environ.get("SPOKOYNO_CORPUS", ROOT / "corpus")) / name


def artifact_path(name=""):
    return Path(os.environ.get("SPOKOYNO_CACHE", ROOT / "artifacts/cache")) / name


def source_path(name):
    """Resolve source names stored in model manifests across the package layout."""
    direct = ROOT / name
    if direct.is_file():
        return direct
    matches = [
        p
        for base in (ROOT / "training", ROOT / "tools")
        for p in base.rglob(name)
        if p.is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one source file for {name}: {matches}")
    return matches[0]
