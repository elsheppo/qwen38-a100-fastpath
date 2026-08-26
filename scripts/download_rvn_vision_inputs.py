#!/usr/bin/env python3
"""Download the pinned metadata and vision payload used by RVN vision assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def read_lock(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = read_lock(args.source_lock)
    metadata = lock["multimodalMetadata"]
    vision = lock["visionPayload"]

    snapshot_download(
        metadata["repository"],
        revision=metadata["revision"],
        local_dir=args.metadata_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "quantization_config.json",
            "processor_config.json",
            "preprocessor_config.json",
        ],
    )
    snapshot_download(
        vision["repository"],
        revision=vision["revision"],
        local_dir=args.vision_dir,
        allow_patterns=[
            "model.safetensors.index.json",
            vision["file"],
            "README.md",
        ],
    )

    record = {
        "schemaVersion": 1,
        "metadata": {
            "repository": metadata["repository"],
            "revision": metadata["revision"],
            "directory": str(args.metadata_dir),
        },
        "vision": {
            "repository": vision["repository"],
            "revision": vision["revision"],
            "file": vision["file"],
            "directory": str(args.vision_dir),
        },
    }
    record_path = args.metadata_dir.parent / "downloads.lock.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
