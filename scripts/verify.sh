#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
target="${1:-all}"
case "$target" in base|rvn|all) ;; *) printf 'Usage: %s [base|rvn|all]\n' "$0" >&2; exit 2 ;; esac

require_command docker
require_command git
test -d "$UPSTREAM_DIR/.git"
test "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "$UPSTREAM_REVISION"
git -C "$UPSTREAM_DIR" diff --check
docker image inspect "$IMAGE_TAG" >/dev/null

if [[ "$target" = base || "$target" = all ]]; then
  test -f "$BASE_MODEL_DIR/config.json"
  test -f "$BASE_MODEL_DIR/model.safetensors.index.json"
fi
if [[ "$target" = rvn || "$target" = all ]]; then
  test -f "$RVN_MODEL_DIR/config.json"
  test -f "$RVN_MODEL_DIR/model.safetensors.index.json"
fi
test -f "$DRAFTER_DIR/model.safetensors"

docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_TAG" \
  --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee /dev/stderr | grep -q A100
printf 'Release runtime and %s model assets are structurally ready.\n' "$target"

