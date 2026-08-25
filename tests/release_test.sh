#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for script in "$root"/scripts/*.sh "$root"/tests/*.sh; do
  bash -n "$script"
done
python3 -m json.tool "$root/release.json" >/dev/null
python3 -m json.tool "$root/results/results.json" >/dev/null
python3 -m json.tool "$root/model-card/artifact-manifest.json" >/dev/null
python3 -m py_compile "$root/scripts/verify_artifact.py"
grep -q -- '--allow-extra' "$root/scripts/verify_artifact.py"
test "$(python3 -c 'import json; x=json.load(open("'$root'/model-card/artifact-manifest.json")); print(sum(i["bytes"] for i in x["files"]))')" = 17702015479
grep -q '138.38' "$root/README.md"
grep -q 'fixed 8 GiB' "$root/README.md"
grep -q 'core W4A16/vLLM/DFlash2 implementation comes from' "$root/README.md"
grep -q 'sheppo/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100' "$root/release.json"
grep -q 'a13afb3b0fc361e100c8352e925495832a0ed1a3' "$root/release.json"
grep -q 'RVN_HF_REVISION=a13afb3b0fc361e100c8352e925495832a0ed1a3' "$root/scripts/common.sh"
printf 'release package checks passed\n'
