# Qwen3.8-27B A100 fast path

Reproducible A100 serving profiles for regular Qwen3.8-27B and an RVN
abliterated derivative. The release includes pinned revisions, launch scripts,
benchmark records, and the converted RVN W4A16 checkpoint.

At concurrency one:

- Regular Qwen3.8-27B went from about **79 to 158 tok/s** on sampled writing.
- The exact abliterated RVN model went from **73 to 138 tok/s**.
- The same regular-Qwen profile also passed on an A100 40 GB.
- A fixed **8 GiB** state/KV pool turned out to matter enormously once requests
  became concurrent.

The speed comes from a W4A16 target, a DFlash2 drafter, patched vLLM, and an
A100-qualified serving profile. No custom CUDA kernel was added in this work.

## The numbers

These are committed completion-token rates at concurrency one. Speculative
decoding is deeply workload-shaped, so the prose number and the structured JSON
number answer different questions.

| Target | GPU | Workload | Raw | DFlash2 k=7 | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Base Qwen | A100 80 GB | sampled long-form writing | 78.77 tok/s | 156.02 tok/s | 1.98x |
| Base Qwen | A100 80 GB | greedy story | 79.80 tok/s | 168.08 tok/s | 2.11x |
| Base Qwen | A100 40 GB | greedy story | 69.87 tok/s | 148.93 tok/s | 2.13x |
| Exact RVN | A100 80 GB | sampled prose, primary median | 73.15 tok/s | 138.38 tok/s | 1.89x |

Regular Qwen reached 306.54 tok/s on CUDA code and 474.15 tok/s on repetitive
structured JSONL. Those are real single-stream measurements. They are also a
great demonstration of why speculative-decoding screenshots can get silly:
predictable output accepts drafts much more often than open-ended prose.

At eight active streams, the fixed-pool profile reached 625.66 aggregate tok/s
on story, 1,064.63 on code, and 1,772.09 on structured JSONL. Median per-stream
story speed at C8 was 79.55 tok/s. Aggregate speed and single-stream speed stay
separate throughout this repo because combining them makes every inference
project sound better than it is.

The compact measurement record is in
[results/results.json](results/results.json). The test boundaries live in
[docs/method.md](docs/method.md).

## What this repo adds

The core W4A16/vLLM/DFlash2 implementation comes from
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).
That project did the hard runtime work for Ampere consumer cards.

This repo carries the A100-specific work:

- Ran the pinned stack on an A100 80 GB and kept the full benchmark identity.
- Found that its 3090-sized state/KV pool left A100 concurrency on the table.
- Tried unbounded auto-sizing, watched it allocate 56.38 GiB and crash with a
  CUDA illegal memory access, then qualified a fixed 8 GiB pool through C8.
- Repeated the frozen profile on an A100 40 GB. It fit with roughly 6 GiB of
  physical headroom and cleared the throughput gates on every test fixture.
- Converted the exact RVN abliterated GGUF into a native W4A16 checkpoint
  without applying the base-model fast overlay over its modified weights.
- Ran chat-template, structured-output, tool-call, restart, mechanical, and
  blind-writing checks before keeping the RVN result.

The regular and abliterated results belong together. Regular Qwen shows that
the runtime path works broadly on A100. RVN shows that the same path survives a
real modified checkpoint instead of quietly swapping the interesting weights
back to base Qwen.

## Run it

You need Linux, one NVIDIA A100 40 GB or 80 GB, a CUDA-13-compatible driver,
Docker with NVIDIA Container Toolkit, and enough disk for the model.

### Regular Qwen

```bash
./scripts/bootstrap.sh base
./scripts/serve.sh base dflash2
./scripts/smoke.sh
```

### RVN abliterated Qwen

```bash
./scripts/bootstrap.sh rvn
./scripts/serve.sh rvn dflash2
./scripts/smoke.sh
```

The first bootstrap builds the pinned container and downloads the selected
target plus the public W4A16 DFlash2 drafter. The first server start can sit
there looking suspiciously inert for several minutes while torch compiles and
CUDA graphs are captured. Follow it with:

```bash
docker logs -f qwen38-a100-fastpath
```

The OpenAI-compatible API listens on port `18020`. Stop the server with:

```bash
./scripts/stop.sh
```

Run the same target without speculative decoding when you want the raw
comparison:

```bash
./scripts/serve.sh rvn raw
```

## The settings worth copying

This is the profile that passed the A100 tests:

```text
SPEC=dflash2
DFLASH_TOKENS=7
LOOKUP=0
KV_MEM=8589934592
MAX_LEN=65536
MAX_SEQS=8
GPU_UTIL=0.93
CTX=fast
CUDAGRAPH_MODE=FULL
```

The weirdly consequential line is `KV_MEM=8589934592`. Unbounded auto-sizing
allocated 56.38 GiB and died with an illegal memory access.

`MAX_SEQS=8` describes server capacity. It does not turn a concurrency-one
benchmark into an eight-stream result by osmosis.

## Does it still write well?

For generative writing, predictable structured-output peaks are weak evidence.
The quality checks therefore used sampled prose in addition to mechanical and
structured-output tests.

For regular Qwen, we captured raw and DFlash2 responses across a 36-pair
writing battery, then repeated the ceiling-affected prompts at 1,280 output
tokens. Two independent model judges found ordinary errors in both arms and no
recurring DFlash-specific collapse. DFlash2 averaged 156.02 tok/s versus 78.77
raw in the longer replication.

For RVN, Claude Fable judged 36 blinded pairs and repeated the full judgment
after every A/B presentation was swapped. The same content side won 35 of 36
times across the swap, which ruled out a simple position preference. After
unblinding, the wins were nearly even between raw and DFlash2.

That supports the profile we tested: temperature 0.8, top-p 0.95, top-k 20,
thinking off, seven DFlash tokens. Change the sampling regime, thinking mode,
draft width, attention backend, or runtime revision and you have created a new
quality question.

## What is in here

- `scripts/` — bootstrap, serve, smoke, stop, artifact verification, and upload
- `profiles/` — the frozen A100 settings
- `patches/` — immutable pins applied to the tested upstream source
- `results/` — compact benchmark records
- `model-card/` — the Hugging Face card and RVN artifact manifest
- `docs/` — the exact measurement and reproducibility boundaries

The 17.7 GB RVN checkpoint lives on Hugging Face:
[`sheppo/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100`](https://huggingface.co/sheppo/Qwen3.8-27B-Heretic-Abliterated-W4A16-A100).
Regular-Qwen weights and the DFlash2 drafter already existed publicly, so this
repo pins them instead of uploading duplicate copies under a new name.

## Caveats

- The exact RVN artifact was qualified on A100 80 GB. The separate A100 40 GB
  replication used regular Qwen.
- The packaged RVN path is text-only and launches with
  `--language-model-only`.
- Stochastic responses can differ across serving modes and dynamic batches.
  The quality tests look for material degradation; they do not promise byte
  identity.
- The RVN conversion reproduces functionally. A second print matched the source,
  tensor mapping, file layout, sizes, vLLM load, and semantic smoke. Four
  weight-shard hashes changed after AutoRound emitted its Flash Attention
  nondeterminism warning.
- The 138.38 tok/s RVN result missed our precommitted 140 tok/s line by 1.62.
  The experiment therefore says `promising-below-target`, even though the
  practical speedup is large.

## Credit where it is due

Qwen, AutoRound, vLLM, the DFlash2 authors, and the syv-ai serving project made
the underlying stack possible. [NOTICE](NOTICE) carries the exact links and
attribution.

Apache-2.0. Model weights retain their model-card provenance and license
notices.
