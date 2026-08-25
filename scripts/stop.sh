#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker rm --force "$CONTAINER_NAME" >/dev/null
  printf 'Stopped and removed %s\n' "$CONTAINER_NAME"
else
  printf '%s is not running.\n' "$CONTAINER_NAME"
fi

