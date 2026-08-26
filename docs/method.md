# Method and claim boundary

## What this project contributes

The underlying serving implementation comes from
`syv-ai/qwen38-27b-rtx3090`. This project establishes and packages five things:

1. The pinned implementation runs correctly on NVIDIA A100 80 GB.
2. The same frozen single-stream profile fits and passes on A100 40 GB.
3. A fixed 8 GiB DFlash2 state/KV pool is stable through eight active streams
   on A100 80 GB; unbounded auto-sizing is rejected for this patched runtime.
4. An independently modified RVN checkpoint can be converted to W4A16 without
   a base-model overlay and use the same DFlash2 drafter with a measured 1.89x
   sampled-prose speedup.
5. The exact RVN language target can be composed with Qwen's pinned vision
   tower and keep DFlash2 active after image conditioning, with a measured
   1.94x speedup on the frozen multimodal workload.

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

## Multimodal assembly and qualification

The published RVN W4A16 checkpoint is a causal language model. The optional
`rvn-vision` profile builds a local conditional-generation target from three
pinned inputs:

- the exact RVN W4A16 text/head tensors;
- multimodal configuration and processor metadata from the matching Qwen
  W4A16 architecture;
- the separated 333-tensor Qwen vision payload.

The assembler renames only the RVN safetensors header namespace from
`model.*` to `model.language_model.*`. It copies the 17,681,667,328-byte RVN
tensor payload unchanged, adds the 333 `model.visual.*` entries, and emits a
`Qwen3_5ForConditionalGeneration` checkpoint. It does not overlay base-Qwen
language weights.

The A100 qualification used a generated semantic battery, a frozen
image-conditioned sampled-generation benchmark, a clean restart, and a
separate 100-image breadth battery. The long-generation median was 72.5755
committed tok/s raw and 140.8400 with DFlash2. In the breadth battery, the
machine-scored DFlash path answered 88/100 with the correct images and 35/100
when images were deliberately permuted. Raw and DFlash answers agreed on 98%
of cases.

Manual review of all 100 fixtures confirmed 93 labels, marked six as
ambiguous, and found one definite duplicate-answer defect. Conservative
adjudicated scores were 88/100 raw and 89/100 DFlash2. This establishes that
the image path is live and that no broad DFlash-specific semantic regression
appeared in this bounded test. It is not an official MMBench score or a claim
of general vision quality.
