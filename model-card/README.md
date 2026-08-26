---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3
  - qwen3.8
  - text-generation
  - vllm
  - compressed-tensors
  - autoround
  - gptq
  - w4a16
  - a100
  - speculative-decoding
---

# Qwen3.8-27B Heretic Abliterated W4A16 — A100 tested

W4A16 AutoRound checkpoint derived from the exact RVN BF16 GGUF and tested on
one A100-SXM4-80GB. It measured 73.15 tok/s raw and 138.38 tok/s with the
companion DFlash2 runtime on the frozen sampled-prose benchmark, a 1.89x
speedup.

This Hugging Face repo contains the W4A16 target. The accelerated runtime,
DFlash2 drafter, optional local vision assembly, exact launch settings, and
base-Qwen A100 results live in
[`elsheppo/qwen38-a100-fastpath`](https://github.com/elsheppo/qwen38-a100-fastpath).

## What this model is

This is a text-only, vLLM-compatible W4A16 AutoRound derivative of the exact
`RVN-BF16.gguf` file from
[`0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF).

- Source revision: `8581a3e4cd8cdeca9bb6709d81ebcbcdfc93fe43`
- Source file: `RVN-BF16.gguf`
- Source size: `53,808,272,896` bytes
- Source SHA-256:
  `fe3cb9c7d067f0016fb8ab0150e2b226c4dfb4e63b497e46ada1e7744b035cdc`
- Quantization: symmetric W4A16, group size 128
- AutoRound revision: `96ce448039b3c36fa879b9f4c740a8ee50c0f9ba`
- Seed: 42
- Serving payload: 13 model/tokenizer/config files, `17,702,015,479` bytes
- Base-model fast overlay applied: **no**

That final line matters. The convenient base-Qwen overlay replaces tensors.
Applying it here could erase the RVN modifications we were trying to preserve.

## How the GGUF became this checkpoint

The first attempt matched all 851 tensor names and shapes, loaded in vLLM, and
still produced corrupted multilingual symbols.

The failure was upstream of quantization: the Qwen3.8 GGUF converter changes
several tensor representations. The corrected bridge inverted the non-GDN norm
shift, continuous-time `A_log` transform, and Gated DeltaNet value-head layout
before AutoRound saw the model. It then mapped all 851 source and target tensors,
checked that every value was finite, and produced the expected deterministic
BF16 answer before quantization resumed.

The user-facing model is Qwen3.8-27B. `Qwen3_5ForCausalLM` labels inside the
config and runtime are the architecture names used by the official release.

## The A100 result

The primary cell used one A100-SXM4-80GB, concurrency one, temperature 0.8,
top-p 0.95, top-k 20, thinking off, and a fixed eight-prompt prose workload.

| Serving mode | Median committed tok/s | Mean committed tok/s | Median TTFT |
| --- | ---: | ---: | ---: |
| Raw RVN W4A16 | 73.15 | 73.31 | 132 ms |
| RVN + DFlash2, k=7 | **138.38** | **148.89** | 155 ms |

The median speedup is **1.89x**. We had frozen a 140 tok/s target before the
run; this landed 1.62 tok/s short, so the experiment record says
`promising-below-target`.

A separate greedy story fixture measured 152.30 tok/s at C1 and 609.87 tok/s
aggregate at C8. Different workload, different measurement. The primary prose
number stays 138.38.

## Does it still behave like the model?

The checkpoint passed:

- six-shard vLLM load and finite-logit semantic smoke
- exact chat-template response
- structured JSON output
- automatic tool-call parsing
- completion and token-accounting checks
- clean-process restart
- a 36-pair blind sampled-writing comparison

Claude Fable judged all 36 pairs without knowing which side was raw or DFlash2,
then repeated the judgments after every A/B position was swapped. The same
content side won 35 of 36 calls across the swap. After unblinding, the preferred
responses were nearly evenly split between serving modes: raw led by two pairs
in the first presentation and three in the swapped presentation.

We observed no material writing-quality regression from the tested DFlash2
path. That conclusion belongs to this sampling profile and this bounded
battery.

## Run it

```bash
git clone https://github.com/elsheppo/qwen38-a100-fastpath
cd qwen38-a100-fastpath
./scripts/bootstrap.sh rvn
./scripts/serve.sh rvn dflash2
./scripts/smoke.sh
```

The fast profile uses
[`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
at revision `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`, seven draft
tokens, lookup drafting disabled, and a fixed 8 GiB state/KV pool.

Run the target without DFlash2 for comparison:

```bash
./scripts/serve.sh rvn raw
```

### Add vision locally

The model artifact on this page is still text-only. The companion repository
can combine these exact text weights with a pinned 921 MB Qwen vision payload
without replacing the RVN language tensors:

```bash
./scripts/bootstrap.sh rvn-vision
./scripts/serve.sh rvn-vision dflash2
./scripts/vision_smoke.sh
```

The assembled target measured 72.58 tok/s raw and 140.84 tok/s with DFlash2 on
the frozen image-conditioned generation benchmark, a 1.94x speedup. In a
separate 100-image breadth battery, the DFlash path scored 88/100 with the
correct images and 35/100 after the images were deliberately mismatched.

## Caveats

- This exact RVN artifact was verified on A100 80 GB. The companion project
  separately verified regular Qwen on A100 40 GB.
- The `rvn` serving profile is text-only and launches with
  `--language-model-only`. The optional `rvn-vision` profile assembles a
  separate local conditional-generation target.
- Code and repetitive structured output accept more speculative drafts than
  sampled prose. Throughput moves with the workload.
- Stochastic responses can differ between raw, speculative, and dynamically
  batched serving. The quality battery looks for degradation, while byte
  identity falls outside its job.
- The conversion reproduces functionally. A second print matched source
  identity, tensor mapping, file paths, sizes, vLLM load, and semantic smoke.
  Four of fourteen total output-path hashes changed after AutoRound warned that
  Flash Attention was nondeterministic.
- Abliteration changes model behavior. Test the checkpoint against your own
  prompts and requirements.

## Credit

The underlying work comes from Qwen, the RVN/Heretic source authors,
AutoRound, vLLM, the DFlash2 authors, and the syv-ai serving project. The
companion repository preserves exact revisions and links in its `NOTICE`.

Apache-2.0, following the Qwen base model and the source model card.
