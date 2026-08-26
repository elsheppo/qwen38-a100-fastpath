#!/usr/bin/env python3
"""Send a real image through the OpenAI-compatible RVN vision endpoint."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
import struct
import urllib.request
import zlib
from pathlib import Path


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(
        ">I", binascii.crc32(body) & 0xFFFFFFFF
    )


def built_in_fixture() -> bytes:
    width = height = 256
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            centered = 64 <= x < 192 and 64 <= y < 192
            rows.extend((220, 30, 30) if centered else (255, 255, 255))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + png_chunk(b"IEND", b"")
    )


def image_data_url(path: Path | None) -> str:
    if path is None:
        payload = built_in_fixture()
        mime = "image/png"
    else:
        payload = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--expect")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("QWEN38_API_BASE", "http://127.0.0.1:18020"),
    )
    args = parser.parse_args()

    using_fixture = args.image is None
    prompt = args.prompt or (
        "What colored shape is centered in this image? Reply exactly RED_SQUARE."
        if using_fixture
        else "Describe what is visible in this image in one concise sentence."
    )
    expected = args.expect if args.expect is not None else ("RED_SQUARE" if using_fixture else None)
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(args.image)},
                    },
                ],
            }
        ],
        "max_tokens": 128 if not using_fixture else 24,
        "temperature": 0,
        "top_p": 1,
        "seed": 1,
        "enable_thinking": False,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        request.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"].strip()
    print(text)
    if expected is not None and expected.casefold() not in text.casefold():
        raise SystemExit(f"vision smoke failed: expected {expected!r}")


if __name__ == "__main__":
    main()
