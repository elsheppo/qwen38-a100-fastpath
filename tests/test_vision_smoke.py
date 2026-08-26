from __future__ import annotations

import importlib.util
import io
import json
import struct
import sys
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "vision_smoke.py"
SPEC = importlib.util.spec_from_file_location("vision_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vision_smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vision_smoke
SPEC.loader.exec_module(vision_smoke)


class FakeResponse:
    def __enter__(self):
        return io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "RED_SQUARE"}}]}
            ).encode("utf-8")
        )

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class VisionSmokeTests(unittest.TestCase):
    def test_fixture_is_a_centered_red_square(self) -> None:
        image = vision_smoke.built_in_fixture()
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        position = 8
        chunks = {}
        while position < len(image):
            length = struct.unpack(">I", image[position : position + 4])[0]
            kind = image[position + 4 : position + 8]
            data = image[position + 8 : position + 8 + length]
            chunks.setdefault(kind, bytearray()).extend(data)
            position += 12 + length
        width, height = struct.unpack(">II", chunks[b"IHDR"][:8])
        self.assertEqual((width, height), (256, 256))
        pixels = zlib.decompress(chunks[b"IDAT"])
        row_bytes = 1 + width * 3
        self.assertEqual(tuple(pixels[1:4]), (255, 255, 255))
        center = 128 * row_bytes + 1 + 128 * 3
        self.assertEqual(tuple(pixels[center : center + 3]), (220, 30, 30))

    def test_default_request_sends_image_and_checks_answer(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch.object(sys, "argv", ["vision_smoke.py"]), mock.patch.object(
            vision_smoke.urllib.request, "urlopen", side_effect=fake_urlopen
        ), redirect_stdout(io.StringIO()):
            vision_smoke.main()

        request = captured["request"]
        payload = json.loads(request.data)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(captured["timeout"], 180)


if __name__ == "__main__":
    unittest.main()
