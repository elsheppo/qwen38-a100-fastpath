#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command curl
require_command python3
headers=(--header 'content-type: application/json')
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  headers+=(--header "authorization: Bearer $VLLM_API_KEY")
fi
response="$(curl --fail --silent --show-error "${headers[@]}" \
  --data '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with exactly: A100_OK"}],"max_tokens":16,"temperature":0,"top_p":1,"seed":1,"enable_thinking":false,"reasoning_effort":"none"}' \
  http://127.0.0.1:18020/v1/chat/completions)"
python3 -c 'import json,sys; r=json.load(sys.stdin); text=r["choices"][0]["message"]["content"].strip(); print(text); raise SystemExit(text != "A100_OK")' <<<"$response"
