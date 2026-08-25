#!/usr/bin/env bash

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${QWEN38_RUNTIME_ROOT:-$RELEASE_ROOT/.runtime}"
UPSTREAM_DIR="$RUNTIME_ROOT/qwen38-27b-rtx3090"
MODELS_DIR="${QWEN38_MODELS_DIR:-$RELEASE_ROOT/models}"
CACHE_DIR="${QWEN38_CACHE_DIR:-$RELEASE_ROOT/.cache}"

UPSTREAM_REPOSITORY=https://github.com/syv-ai/qwen38-27b-rtx3090.git
UPSTREAM_REVISION=2738b38fdd40d455b4cdbc35d7763f0d47203af0
IMAGE_TAG="qwen38-a100-fastpath:$UPSTREAM_REVISION"
CONTAINER_NAME=qwen38-a100-fastpath

BASE_MODEL_DIR="$MODELS_DIR/Qwen3.8-27B-W4A16-AutoRound-fast"
RVN_MODEL_DIR="$MODELS_DIR/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100"
DRAFTER_DIR="$MODELS_DIR/Qwen3.8-27B-DFlash2-W4A16"

RVN_HF_REPO=sheppo/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100
RVN_HF_REVISION=a13afb3b0fc361e100c8352e925495832a0ed1a3
DRAFTER_HF_REVISION=4d30ec736ffc6b8688dc2ae2b502d9b48bdec279

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  }
}
