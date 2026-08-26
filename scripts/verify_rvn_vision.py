#!/usr/bin/env python3
"""Fast structural verification for an assembled RVN multimodal checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()

    source_lock = read_json(args.source_lock)
    config = read_json(args.model_dir / "config.json")
    index = read_json(args.model_dir / "model.safetensors.index.json")
    manifest = read_json(args.model_dir / "assembly-manifest.json")

    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise SystemExit("assembled target is not Qwen3_5ForConditionalGeneration")
    if manifest.get("status") != "assembled":
        raise SystemExit("assembly manifest is not complete")
    if manifest.get("sources") != source_lock:
        raise SystemExit("assembly source pins do not match this release")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("assembled index has no weight map")
    text_names = [
        name
        for name in weight_map
        if name.startswith("model.language_model.") or name.startswith("lm_head.")
    ]
    vision_names = [name for name in weight_map if name.startswith("model.visual.")]
    if len(text_names) != source_lock["rvnTarget"]["expectedIndexEntries"]:
        raise SystemExit(f"unexpected text entry count: {len(text_names)}")
    if len(vision_names) != source_lock["visionPayload"]["expectedEntries"]:
        raise SystemExit(f"unexpected vision entry count: {len(vision_names)}")
    if len(weight_map) != len(text_names) + len(vision_names):
        raise SystemExit("assembled index contains an unexpected tensor namespace")

    missing = sorted(
        filename
        for filename in set(weight_map.values())
        if not (args.model_dir / filename).is_file()
    )
    if missing:
        raise SystemExit(f"assembled target is missing weight files: {missing}")

    vision_file = args.model_dir / source_lock["visionPayload"]["file"]
    if vision_file.stat().st_size != source_lock["visionPayload"]["bytes"]:
        raise SystemExit("assembled vision payload has the wrong byte size")
    if manifest.get("visionSha256") != source_lock["visionPayload"]["sha256"]:
        raise SystemExit("assembly manifest has the wrong vision checksum")
    if manifest.get("textEntries") != len(text_names):
        raise SystemExit("assembly manifest text count disagrees with the index")
    if manifest.get("visionEntries") != len(vision_names):
        raise SystemExit("assembly manifest vision count disagrees with the index")

    print(
        "RVN vision target is structurally ready: "
        f"{len(text_names)} text/head tensors + {len(vision_names)} vision tensors."
    )


if __name__ == "__main__":
    main()
