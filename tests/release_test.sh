#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for script in "$root"/scripts/*.sh "$root"/tests/*.sh; do
  bash -n "$script"
done
python3 -m json.tool "$root/release.json" >/dev/null
python3 -m json.tool "$root/results/results.json" >/dev/null
python3 -m json.tool "$root/model-card/artifact-manifest.json" >/dev/null
python3 -m json.tool "$root/sources/rvn-vision.json" >/dev/null
python3 -m py_compile "$root"/scripts/*.py
python3 -m unittest discover -s "$root/tests" -p 'test_*.py'
grep -q -- '--allow-extra' "$root/scripts/verify_artifact.py"
test "$(python3 -c 'import json; x=json.load(open("'$root'/model-card/artifact-manifest.json")); print(sum(i["bytes"] for i in x["files"]))')" = 17702015479
grep -q '138.38' "$root/README.md"
grep -q 'fixed 8 GiB' "$root/README.md"
grep -q 'core W4A16/vLLM/DFlash2 implementation comes from' "$root/README.md"
grep -q 'sheppo/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100' "$root/release.json"
grep -q 'a13afb3b0fc361e100c8352e925495832a0ed1a3' "$root/release.json"
grep -q 'RVN_HF_REVISION=a13afb3b0fc361e100c8352e925495832a0ed1a3' "$root/scripts/common.sh"
grep -q '140.84' "$root/README.md"
grep -q 'rvn-vision' "$root/scripts/bootstrap.sh"
grep -q -- '--env VISION=1' "$root/scripts/serve.sh"
test -x "$root/scripts/vision_smoke.sh"
python3 - "$root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
release = json.loads((root / "release.json").read_text())
source = json.loads((root / "sources/rvn-vision.json").read_text())
assert release["dependencies"]["rvnVision"]["visionPayload"]["sha256"] == source["visionPayload"]["sha256"]
assert release["dependencies"]["rvnVision"]["visionPayload"]["revision"] == source["visionPayload"]["revision"]
assert release["rvnModel"]["revision"] == source["rvnTarget"]["revision"]
PY
printf 'release package checks passed\n'
