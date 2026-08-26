#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

target="${1:-all}"
case "$target" in
  base|rvn|rvn-vision|all) ;;
  *) printf 'Usage: %s [base|rvn|rvn-vision|all]\n' "$0" >&2; exit 2 ;;
esac

require_command git
require_command docker
mkdir -p "$RUNTIME_ROOT" "$MODELS_DIR" "$CACHE_DIR"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone --filter=blob:none "$UPSTREAM_REPOSITORY" "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --depth=1 origin "$UPSTREAM_REVISION"
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_REVISION"
test "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "$UPSTREAM_REVISION"

pin_patch="$RELEASE_ROOT/patches/reproducibility.patch"
if git -C "$UPSTREAM_DIR" apply --check "$pin_patch"; then
  git -C "$UPSTREAM_DIR" apply "$pin_patch"
elif ! git -C "$UPSTREAM_DIR" apply --reverse --check "$pin_patch"; then
  printf 'Pinned upstream patch is neither cleanly applicable nor already applied.\n' >&2
  exit 1
fi
git -C "$UPSTREAM_DIR" diff --check

docker build --tag "$IMAGE_TAG" "$UPSTREAM_DIR"

if [[ "$target" = base || "$target" = all ]]; then
  docker run --rm \
    --volume "$MODELS_DIR:/app/models" \
    "$IMAGE_TAG" prepare
fi

if [[ "$target" = rvn || "$target" = rvn-vision || "$target" = all ]]; then
  hf_token_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then hf_token_args=(--env HF_TOKEN); fi
  docker run --rm \
    "${hf_token_args[@]}" \
    --volume "$MODELS_DIR:/app/models" \
    --entrypoint /app/venv/bin/hf \
    "$IMAGE_TAG" download "$RVN_HF_REPO" \
      --revision "$RVN_HF_REVISION" \
      --local-dir "/app/models/$(basename "$RVN_MODEL_DIR")"
  docker run --rm \
    "${hf_token_args[@]}" \
    --volume "$MODELS_DIR:/app/models" \
    --entrypoint /app/venv/bin/python \
    "$IMAGE_TAG" /app/prepare/fetch_dflash2.py \
      /app/models/Qwen3.8-27B-DFlash2-W4A16
fi

if [[ "$target" = rvn-vision || "$target" = all ]]; then
  docker run --rm \
    "${hf_token_args[@]}" \
    --volume "$MODELS_DIR:/app/models" \
    --volume "$RELEASE_ROOT:/release:ro" \
    --entrypoint /app/venv/bin/python \
    "$IMAGE_TAG" /release/scripts/download_rvn_vision_inputs.py \
      --source-lock /release/sources/rvn-vision.json \
      --metadata-dir /app/models/.rvn-vision-inputs/metadata \
      --vision-dir /app/models/.rvn-vision-inputs/vision

  if [[ -f "$RVN_VISION_MODEL_DIR/assembly-manifest.json" ]]; then
    printf 'Reusing assembled RVN vision target: %s\n' "$RVN_VISION_MODEL_DIR"
  elif [[ -d "$RVN_VISION_MODEL_DIR" ]] && find "$RVN_VISION_MODEL_DIR" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing nonempty unverified vision target: %s\nMove it aside and rerun bootstrap.\n' \
      "$RVN_VISION_MODEL_DIR" >&2
    exit 1
  else
    docker run --rm \
      --volume "$MODELS_DIR:/app/models" \
      --volume "$RELEASE_ROOT:/release:ro" \
      --entrypoint /app/venv/bin/python \
      "$IMAGE_TAG" /release/scripts/assemble_rvn_vision.py \
        --text-dir "/app/models/$(basename "$RVN_MODEL_DIR")" \
        --metadata-dir /app/models/.rvn-vision-inputs/metadata \
        --vision-dir /app/models/.rvn-vision-inputs/vision \
        --output-dir "/app/models/$(basename "$RVN_VISION_MODEL_DIR")" \
        --source-lock /release/sources/rvn-vision.json
  fi
fi

"$RELEASE_ROOT/scripts/verify.sh" "$target"
