# Method and claim boundary

## What this project contributes

The underlying serving implementation comes from
`syv-ai/qwen38-27b-rtx3090`. This project establishes and packages four things:

1. The pinned implementation runs correctly on NVIDIA A100 80 GB.
2. The same frozen single-stream profile fits and passes on A100 40 GB.
3. A fixed 8 GiB DFlash2 state/KV pool is stable through eight active streams
   on A100 80 GB; unbounded auto-sizing is rejected for this patched runtime.
4. An independently modified RVN checkpoint can be converted to W4A16 without
   a base-model overlay and use the same DFlash2 drafter with a measured 1.89x
   sampled-prose speedup.

This is an end-to-end deployment result across model weights, quantization,
runtime patches, speculative decoding, and configuration. It is not causal
evidence that one custom kernel is responsible for the gain.

## Measurement contract

- Hardware identity, target/drafter revisions, runtime revision, workload,
  sampling, concurrency, and output length were frozen per run.
- Correctness ran before throughput.
- Ranked C1 cells used three warmups and five measured trials unless the compact
  record explicitly identifies a separate fixture contract.
- Headline throughput counts committed streamed completion tokens after the
  first content byte.
- Concurrency-one and aggregate measurements are reported separately.
- Sampled writing and greedy code/JSON are not substituted for one another.

## RVN identity boundary

The RVN target comes from the exact `RVN-BF16.gguf` source at revision
`8581a3e4cd8cdeca9bb6709d81ebcbcdfc93fe43`, SHA-256
`fe3cb9c7d067f0016fb8ab0150e2b226c4dfb4e63b497e46ada1e7744b035cdc`.
All 851 tensors were mapped into the Transformers model consumed by AutoRound.
The conversion inverted the Qwen3.8 GGUF norm, `A_log`, and Gated DeltaNet head
layout transforms before quantization. No fast base-model tensor overlay was
applied, because that would replace the RVN-modified weights.

The retained reprint is functionally reproducible but not bit-reproducible.
Ten of fourteen total output-path hashes matched the qualification print. Four
safetensor shards differed while paths, sizes, source identity, mapped tensor
count, quantization recipe, vLLM load, and deterministic semantic smoke all
matched. AutoRound emitted its Flash Attention nondeterminism warning.

## Quality boundary

The blind evaluations support a narrow conclusion: no material quality
regression attributable to the tested DFlash2 serving path was observed. They
do not prove that speculative output is semantically identical in every
sampling regime, that the abliterated model is generally preferable, or that
an LLM judge substitutes for a user's own preference tests.

