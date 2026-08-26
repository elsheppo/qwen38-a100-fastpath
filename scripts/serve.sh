#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

profile="${1:-rvn}"
mode="${2:-dflash2}"
case "$profile" in
  base)
    profile_file="$RELEASE_ROOT/profiles/base-a100.env"
    target_host="$BASE_MODEL_DIR"
    target_container=/app/target
    ;;
  rvn)
    profile_file="$RELEASE_ROOT/profiles/rvn-a10080.env"
    target_host="$RVN_MODEL_DIR"
    target_container=/app/target
    ;;
  rvn-vision)
    profile_file="$RELEASE_ROOT/profiles/rvn-vision-a10080.env"
    target_host="$RVN_VISION_MODEL_DIR"
    target_container=/app/target
    ;;
  *) printf 'Profile must be base, rvn, or rvn-vision.\n' >&2; exit 2 ;;
esac
case "$mode" in raw|dflash2) ;; *) printf 'Mode must be raw or dflash2.\n' >&2; exit 2 ;; esac

require_command docker
test -f "$target_host/config.json" || {
  printf 'Target is not prepared: %s\nRun ./scripts/bootstrap.sh %s first.\n' "$target_host" "$profile" >&2
  exit 1
}
if [[ "$mode" = dflash2 ]]; then
  test -f "$DRAFTER_DIR/model.safetensors" || {
    printf 'DFlash2 drafter is not prepared: %s\n' "$DRAFTER_DIR" >&2
    exit 1
  }
fi
docker image inspect "$IMAGE_TAG" >/dev/null
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container %s already exists. Run ./scripts/stop.sh first.\n' "$CONTAINER_NAME" >&2
  exit 1
fi

set -a
source "$profile_file"
set +a
mkdir -p "$CACHE_DIR"

common_args=(
  --detach --name "$CONTAINER_NAME" --gpus all --network host --ipc host --shm-size 16g
  --volume "$target_host:$target_container:ro"
  --volume "$MODELS_DIR:/app/models:ro"
  --volume "$CACHE_DIR:/cache:rw"
)
if [[ -n "${VLLM_API_KEY:-}" ]]; then common_args+=(--env VLLM_API_KEY); fi

if [[ "$mode" = raw ]]; then
  if [[ "$profile" = rvn-vision ]]; then
    docker run "${common_args[@]}" \
      --env TARGET_MODEL="$target_container" \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --env VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$IMAGE_TAG" bash -lc \
      'export PATH=/app/venv/bin:$PATH; exec venv/bin/vllm serve "$TARGET_MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 18020 --gpu-memory-utilization 0.93 --kv-cache-memory 8589934592 --max-model-len 65536 --max-num-seqs 8 --limit-mm-per-prompt '\''{"image":{"count":1}}'\'' --mm-processor-kwargs '\''{"size":{"shortest_edge":65536,"longest_edge":2097152}}'\'' --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16 --async-scheduling --max-num-batched-tokens 2048 --compilation-config '\''{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}'\'' --reasoning-parser qwen3'
  else
    docker run "${common_args[@]}" \
      --env TARGET_MODEL="$target_container" \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --env VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$IMAGE_TAG" bash -lc \
      'export PATH=/app/venv/bin:$PATH; exec venv/bin/vllm serve "$TARGET_MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 18020 --gpu-memory-utilization 0.93 --max-model-len 65536 --max-num-seqs 8 --language-model-only --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16 --async-scheduling --max-num-batched-tokens 2048 --compilation-config '\''{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}'\'' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder'
  fi
else
  vision_args=()
  if [[ "$profile" = rvn-vision ]]; then
    vision_args=(--env VISION=1)
  fi
  docker run "${common_args[@]}" \
    "${vision_args[@]}" \
    --env MODEL="$target_container" \
    --env DRAFT=/app/models/Qwen3.8-27B-DFlash2-W4A16 \
    --env VERIFY=0 --env CTX --env SPEC --env LOOKUP --env TOOLS \
    --env DFLASH_TOKENS --env CUDAGRAPH_MODE --env KV_MEM \
    --env MAX_LEN --env MAX_SEQS --env GPU_UTIL \
    "$IMAGE_TAG" single
fi

printf 'Starting %s/%s on http://127.0.0.1:18020\n' "$profile" "$mode"
printf 'Follow startup with: docker logs -f %s\n' "$CONTAINER_NAME"
