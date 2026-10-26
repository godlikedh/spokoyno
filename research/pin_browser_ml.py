#!/usr/bin/env python3
"""Pin userscript dependencies to immutable, checksummed browser ML assets."""

import argparse
import json
import re
from pathlib import Path

from extract_audio_context import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "revision", help="Git commit containing ml/assets, or LOCAL for development"
    )
    args = parser.parse_args()
    if args.revision != "LOCAL" and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("expected full Git revision")
    assets = Path("ml/assets")
    manifest = json.loads((assets / "manifest.json").read_text())
    resources = {
        "spokoynoMLRuntime": "runtime.js",
        "spokoynoMLModel": "model.json",
        "spokoynoMLWeights": "weights.bin.b64",
        "spokoynoMLWasm": "tf.wasm.b64",
        "spokoynoMLSimd": "tf-simd.wasm.b64",
    }
    for name in resources.values():
        if sha256(assets / name) != manifest["resources"][name]:
            raise ValueError("assets changed since build")
    base = f"https://raw.githubusercontent.com/godlikedh/spokoyno/{args.revision}/ml/assets/"
    header = [
        f"// @require      {base}runtime.js#sha256={manifest['resources']['runtime.js']}"
    ]
    header += [
        f"// @resource     {name} {base}{file}#sha256={manifest['resources'][file]}"
        for name, file in resources.items()
    ]
    script = Path("spokoyno.user.js")
    source = script.read_text()
    source = re.sub(r"^// @(?:require|resource)\s+.*\n", "", source, flags=re.M)
    source = source.replace("// @run-at", "\n".join(header) + "\n// @run-at", 1)
    hashes = {n: manifest["resources"][f] for n, f in resources.items()}
    source = re.sub(
        r"  const ML_RESOURCE_HASHES = .*?;[^\n]*",
        "  const ML_RESOURCE_HASHES = " + json.dumps(hashes) + ";",
        source,
        count=1,
        flags=re.S,
    )
    script.write_text(source)


if __name__ == "__main__":
    main()
