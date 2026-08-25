#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

model_dir="${1:?usage: upload_model.sh MODEL_DIR}"
require_command hf
require_command python3
python3 "$RELEASE_ROOT/scripts/verify_artifact.py" \
  "$model_dir" "$RELEASE_ROOT/model-card/artifact-manifest.json"

HF_XET_HIGH_PERFORMANCE=1 hf upload "$RVN_HF_REPO" "$model_dir" .
hf upload "$RVN_HF_REPO" "$RELEASE_ROOT/model-card/README.md" README.md
hf upload "$RVN_HF_REPO" "$RELEASE_ROOT/LICENSE" LICENSE
hf upload "$RVN_HF_REPO" "$RELEASE_ROOT/model-card/artifact-manifest.json" artifact-manifest.json
printf 'Uploaded model and publication metadata to https://huggingface.co/%s\n' "$RVN_HF_REPO"

