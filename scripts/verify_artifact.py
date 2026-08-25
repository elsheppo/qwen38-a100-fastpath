#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--allow-extra"):
        print("usage: verify_artifact.py MODEL_DIR MANIFEST [--allow-extra]", file=sys.stderr)
        return 2
    allow_extra = len(sys.argv) == 4
    model_dir = Path(sys.argv[1]).resolve()
    manifest = json.loads(Path(sys.argv[2]).read_text())
    expected_names = {item["path"] for item in manifest["files"]}
    actual_names = {str(path.relative_to(model_dir)) for path in model_dir.iterdir() if path.is_file()}
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing or (extra and not allow_extra):
        print(f"file set mismatch: missing={sorted(expected_names-actual_names)} extra={sorted(actual_names-expected_names)}", file=sys.stderr)
        return 1
    if extra:
        print(f"Ignoring repository metadata: {sorted(extra)}")
    for item in manifest["files"]:
        path = model_dir / item["path"]
        if path.stat().st_size != item["bytes"]:
            print(f"size mismatch: {item['path']}", file=sys.stderr)
            return 1
        if sha256(path) != item["sha256"]:
            print(f"sha256 mismatch: {item['path']}", file=sys.stderr)
            return 1
        print(f"PASS {item['path']}")
    print(f"Verified {len(expected_names)} model files and {manifest['totalBytes']} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
