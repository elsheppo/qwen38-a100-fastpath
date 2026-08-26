from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "assemble_rvn_vision.py"
SPEC = importlib.util.spec_from_file_location("repack_rvn_vision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repack
SPEC.loader.exec_module(repack)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_fake_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class RepackTests(unittest.TestCase):
    def test_rewrites_only_header_names_and_adds_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text = root / "text"
            metadata = root / "metadata"
            vision = root / "vision"
            output = root / "output"
            for directory in (text, metadata, vision):
                directory.mkdir()

            text_tensors = {
                "model.layers.0.fake.weight": b"rvn-language",
                "lm_head.weight": b"rvn-head",
            }
            vision_tensors = {"model.visual.fake.weight": b"qwen-vision"}
            write_fake_safetensors(text / "text.safetensors", text_tensors)
            write_fake_safetensors(vision / "vision.safetensors", vision_tensors)

            write_json(
                text / "config.json",
                {"architectures": ["Qwen3_5ForCausalLM"]},
            )
            write_json(
                metadata / "config.json",
                {"architectures": ["Qwen3_5ForConditionalGeneration"]},
            )
            write_json(metadata / "quantization_config.json", {"format": "pack-quantized"})
            write_json(metadata / "processor_config.json", {"processor_class": "Fake"})
            for filename in (
                "chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (text / filename).write_text("{}", encoding="utf-8")

            text_map = {name: "text.safetensors" for name in text_tensors}
            metadata_map = {
                "model.language_model.layers.0.fake.weight": "base.safetensors",
                "lm_head.weight": "base.safetensors",
                "model.visual.fake.weight": "base.safetensors",
            }
            vision_map = {name: "vision.safetensors" for name in vision_tensors}
            write_json(
                text / "model.safetensors.index.json",
                {"metadata": {"total_size": 20}, "weight_map": text_map},
            )
            write_json(
                metadata / "model.safetensors.index.json",
                {"metadata": {"total_size": 31}, "weight_map": metadata_map},
            )
            write_json(
                vision / "model.safetensors.index.json",
                {"metadata": {"total_size": 11}, "weight_map": vision_map},
            )
            lock = root / "source-lock.json"
            write_json(
                lock,
                {
                    "rvnTarget": {"expectedIndexEntries": 2},
                    "visionPayload": {
                        "expectedEntries": 1,
                        "sha256": repack.sha256_file(vision / "vision.safetensors"),
                    },
                },
            )

            result = repack.assemble(text, metadata, vision, output, lock, False)
            self.assertEqual(result["textEntries"], 2)
            self.assertEqual(result["visionEntries"], 1)
            self.assertEqual(result["sources"], repack.read_json(lock))
            output_header = repack.read_safetensors_header(output / "text.safetensors")
            self.assertEqual(
                set(output_header.tensors),
                {"model.language_model.layers.0.fake.weight", "lm_head.weight"},
            )
            source_payload = (text / "text.safetensors").read_bytes()[
                8 + len(repack.read_safetensors_header(text / "text.safetensors").raw_header) :
            ]
            output_payload = (output / "text.safetensors").read_bytes()[
                8 + len(output_header.raw_header) :
            ]
            self.assertEqual(source_payload, output_payload)
            index = repack.read_json(output / "model.safetensors.index.json")
            self.assertEqual(len(index["weight_map"]), 3)

    def test_refuses_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "existing").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(repack.RepackError, "nonempty output"):
                repack.assemble(
                    Path("missing-text"),
                    Path("missing-metadata"),
                    Path("missing-vision"),
                    output,
                    Path("missing-lock"),
                    False,
                )


if __name__ == "__main__":
    unittest.main()
