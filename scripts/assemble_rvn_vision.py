#!/usr/bin/env python3
"""Recompose the exact RVN W4A16 language target with Qwen's vision tower.

The text checkpoint and multimodal checkpoint use the same tensor payloads but
different module namespaces:

    model.* -> model.language_model.*
    lm_head.* -> lm_head.*

Safetensors offsets are relative to the end of the JSON header. We can therefore
rename header keys and copy the tensor payload byte-for-byte without importing
PyTorch or deserializing 18 GB of weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


HEADER_LENGTH = struct.Struct("<Q")
COPY_CHUNK_BYTES = 16 * 1024 * 1024


class RepackError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafeTensorHeader:
    raw_prefix: bytes
    raw_header: bytes
    tensors: dict[str, dict[str, Any]]
    metadata: dict[str, str] | None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RepackError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_safetensors_header(path: Path) -> SafeTensorHeader:
    with path.open("rb") as handle:
        raw_prefix = handle.read(HEADER_LENGTH.size)
        if len(raw_prefix) != HEADER_LENGTH.size:
            raise RepackError(f"truncated safetensors prefix: {path}")
        (header_length,) = HEADER_LENGTH.unpack(raw_prefix)
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise RepackError(f"truncated safetensors header: {path}")
    try:
        decoded = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepackError(f"invalid safetensors header: {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise RepackError(f"safetensors header is not an object: {path}")
    metadata = decoded.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise RepackError(f"invalid safetensors metadata: {path}")
    tensors: dict[str, dict[str, Any]] = {}
    for name, descriptor in decoded.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise RepackError(f"invalid tensor descriptor in {path}")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise RepackError(f"invalid data offsets for {name} in {path}")
        tensors[name] = descriptor
    return SafeTensorHeader(raw_prefix, raw_header, tensors, metadata)


def encode_safetensors_header(
    tensors: dict[str, dict[str, Any]], metadata: dict[str, str] | None
) -> bytes:
    value: dict[str, Any] = dict(tensors)
    if metadata is not None:
        value["__metadata__"] = metadata
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return encoded


def map_text_name(name: str) -> str:
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    if name.startswith("lm_head."):
        return name
    raise RepackError(f"unexpected text tensor namespace: {name}")


def tensor_bytes(tensors: Iterable[dict[str, Any]]) -> int:
    return sum(item["data_offsets"][1] - item["data_offsets"][0] for item in tensors)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def copy_and_hash(
    source: BinaryIO,
    destination: BinaryIO,
    source_digest: hashlib._Hash,
    destination_digest: hashlib._Hash,
    payload_digest: hashlib._Hash,
) -> int:
    total = 0
    while chunk := source.read(COPY_CHUNK_BYTES):
        destination.write(chunk)
        source_digest.update(chunk)
        destination_digest.update(chunk)
        payload_digest.update(chunk)
        total += len(chunk)
    return total


def rewrite_text_shard(source: Path, destination: Path) -> dict[str, Any]:
    header = read_safetensors_header(source)
    mapped: dict[str, dict[str, Any]] = {}
    for name, descriptor in header.tensors.items():
        mapped_name = map_text_name(name)
        if mapped_name in mapped:
            raise RepackError(f"duplicate mapped tensor {mapped_name} in {source}")
        mapped[mapped_name] = descriptor
    encoded_header = encode_safetensors_header(mapped, header.metadata)
    output_prefix = HEADER_LENGTH.pack(len(encoded_header))

    source_digest = hashlib.sha256()
    source_digest.update(header.raw_prefix)
    source_digest.update(header.raw_header)
    output_digest = hashlib.sha256()
    output_digest.update(output_prefix)
    output_digest.update(encoded_header)
    payload_digest = hashlib.sha256()

    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        input_handle.seek(HEADER_LENGTH.size + len(header.raw_header))
        output_handle.write(output_prefix)
        output_handle.write(encoded_header)
        payload_size = copy_and_hash(
            input_handle,
            output_handle,
            source_digest,
            output_digest,
            payload_digest,
        )

    expected_payload = max(
        (descriptor["data_offsets"][1] for descriptor in header.tensors.values()),
        default=0,
    )
    if payload_size != expected_payload:
        raise RepackError(
            f"payload size mismatch for {source}: copied {payload_size}, expected {expected_payload}"
        )
    return {
        "source": str(source),
        "output": destination.name,
        "tensorCount": len(mapped),
        "tensorBytes": tensor_bytes(mapped.values()),
        "payloadBytes": payload_size,
        "sourceSha256": source_digest.hexdigest(),
        "outputSha256": output_digest.hexdigest(),
        "payloadSha256": payload_digest.hexdigest(),
    }


def load_weight_map(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index = read_json(path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
    ):
        raise RepackError(f"invalid weight_map: {path}")
    return weight_map, index


def require_architecture(path: Path, expected: str) -> dict[str, Any]:
    config = read_json(path)
    architectures = config.get("architectures")
    if architectures != [expected]:
        raise RepackError(
            f"unexpected architecture in {path}: {architectures!r}, expected {[expected]!r}"
        )
    return config


def plan_mapping(
    text_dir: Path, metadata_dir: Path, vision_dir: Path, source_lock: dict[str, Any]
) -> dict[str, Any]:
    require_architecture(text_dir / "config.json", "Qwen3_5ForCausalLM")
    require_architecture(metadata_dir / "config.json", "Qwen3_5ForConditionalGeneration")

    text_map, text_index = load_weight_map(text_dir / "model.safetensors.index.json")
    metadata_map, _ = load_weight_map(metadata_dir / "model.safetensors.index.json")
    vision_map, _ = load_weight_map(vision_dir / "model.safetensors.index.json")

    mapped_text = {map_text_name(name): filename for name, filename in text_map.items()}
    expected_text = {
        name
        for name in metadata_map
        if name.startswith("model.language_model.") or name.startswith("lm_head.")
    }
    expected_vision = {name for name in metadata_map if name.startswith("model.visual.")}
    supplied_vision = {name for name in vision_map if name.startswith("model.visual.")}

    missing_text = sorted(expected_text - mapped_text.keys())
    extra_text = sorted(mapped_text.keys() - expected_text)
    missing_vision = sorted(expected_vision - supplied_vision)
    extra_vision = sorted(supplied_vision - expected_vision)
    if missing_text or extra_text or missing_vision or extra_vision:
        raise RepackError(
            "mapping mismatch: "
            f"missing_text={missing_text[:5]} extra_text={extra_text[:5]} "
            f"missing_vision={missing_vision[:5]} extra_vision={extra_vision[:5]}"
        )

    expected_text_count = source_lock["rvnTarget"]["expectedIndexEntries"]
    expected_vision_count = source_lock["visionPayload"]["expectedEntries"]
    if len(mapped_text) != expected_text_count:
        raise RepackError(f"RVN entry count {len(mapped_text)} != {expected_text_count}")
    if len(supplied_vision) != expected_vision_count:
        raise RepackError(f"vision entry count {len(supplied_vision)} != {expected_vision_count}")

    collisions = sorted(mapped_text.keys() & supplied_vision)
    if collisions:
        raise RepackError(f"text/vision key collisions: {collisions[:5]}")

    return {
        "textEntries": len(mapped_text),
        "visionEntries": len(supplied_vision),
        "totalEntries": len(mapped_text) + len(supplied_vision),
        "textTensorBytes": text_index.get("metadata", {}).get("total_size"),
        "mappedText": mapped_text,
        "visionMap": {name: vision_map[name] for name in supplied_vision},
    }


def copy_support_files(text_dir: Path, metadata_dir: Path, output_dir: Path) -> list[str]:
    sources = {
        "config.json": metadata_dir,
        "quantization_config.json": metadata_dir,
        "processor_config.json": metadata_dir,
        "preprocessor_config.json": metadata_dir,
        "chat_template.jinja": text_dir,
        "generation_config.json": text_dir,
        "tokenizer.json": text_dir,
        "tokenizer_config.json": text_dir,
    }
    required = {
        "config.json",
        "quantization_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    copied: list[str] = []
    for filename, directory in sources.items():
        source = directory / filename
        if not source.exists():
            if filename in required:
                raise RepackError(f"required support file missing: {source}")
            continue
        shutil.copy2(source, output_dir / filename)
        copied.append(filename)
    return copied


def assemble(
    text_dir: Path,
    metadata_dir: Path,
    vision_dir: Path,
    output_dir: Path,
    source_lock_path: Path,
    plan_only: bool,
) -> dict[str, Any]:
    if not plan_only and output_dir.exists() and any(output_dir.iterdir()):
        raise RepackError(f"refusing nonempty output directory: {output_dir}")
    source_lock = read_json(source_lock_path)
    plan = plan_mapping(text_dir, metadata_dir, vision_dir, source_lock)
    public_plan = {key: value for key, value in plan.items() if key not in {"mappedText", "visionMap"}}
    if plan_only:
        return {"status": "plan-pass", **public_plan}

    output_dir.mkdir(parents=True, exist_ok=True)

    text_map: dict[str, str] = plan["mappedText"]
    source_text_map, _ = load_weight_map(text_dir / "model.safetensors.index.json")
    shards: list[dict[str, Any]] = []
    for filename in sorted(set(source_text_map.values())):
        source = text_dir / filename
        if not source.is_file():
            raise RepackError(f"missing text shard: {source}")
        shards.append(rewrite_text_shard(source, output_dir / filename))

    vision_files = sorted(set(plan["visionMap"].values()))
    if len(vision_files) != 1:
        raise RepackError(f"expected one separated vision shard, got {vision_files}")
    vision_filename = vision_files[0]
    source_vision = vision_dir / vision_filename
    destination_vision = output_dir / vision_filename
    if not source_vision.is_file():
        raise RepackError(f"missing vision shard: {source_vision}")
    observed_vision_sha = sha256_file(source_vision)
    expected_vision_sha = source_lock["visionPayload"]["sha256"]
    if observed_vision_sha != expected_vision_sha:
        raise RepackError(
            f"vision SHA-256 mismatch: {observed_vision_sha} != {expected_vision_sha}"
        )
    shutil.copy2(source_vision, destination_vision)

    support_files = copy_support_files(text_dir, metadata_dir, output_dir)
    output_weight_map = dict(text_map)
    output_weight_map.update(plan["visionMap"])
    vision_header = read_safetensors_header(source_vision)
    total_tensor_bytes = int(plan["textTensorBytes"]) + tensor_bytes(vision_header.tensors.values())
    write_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {
                "format": "safetensors",
                "total_size": total_tensor_bytes,
            },
            "weight_map": output_weight_map,
        },
    )

    output_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schemaVersion": 1,
        "kind": "rvn-qwen38-multimodal-w4a16",
        "status": "assembled",
        **public_plan,
        "totalTensorBytes": total_tensor_bytes,
        "visionSha256": observed_vision_sha,
        "payloadInvariant": "RVN tensor payload bytes unchanged; only safetensors header names changed",
        "sources": source_lock,
        "textShards": shards,
        "supportFiles": support_files,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
            if path.name != "assembly-manifest.json"
        ],
    }
    write_json(output_dir / "assembly-manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = assemble(
        text_dir=args.text_dir,
        metadata_dir=args.metadata_dir,
        vision_dir=args.vision_dir,
        output_dir=args.output_dir,
        source_lock_path=args.source_lock,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
