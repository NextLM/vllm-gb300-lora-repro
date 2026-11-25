# vLLM LoRA kernel measurements on GB300

Reproduction material for two comments on `vllm-project/vllm`: PR 61087 (cap LoRA shrink split-K) and issue 50602 (LoRA path non-determinism).
Everything here is public material: a public checkpoint, public or synthetic adapters, prompts built from the Python standard library.

## Environment
- 1x NVIDIA GB300, aarch64, Ubuntu 24.04, driver 595.99.02 (CUDA 13.2)
- vLLM 0.34.0 (PyPI wheel) and nightly 0.34.1rc1.dev90+g2d08d35d4; torch 2.14.0+cu130, triton 3.8.0, flashinfer 0.7.3 with precompiled kernels
- Full package list: `env/pip-freeze-vllm-0.34.0-env.txt`

## Models and adapters
- `RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8` at revision `38effd54b334bfda8fce082f4290aa56d86b7533`.
  Served from a directory of links to the snapshot with one edited file: `config.json` gets two extra entries in `quantization_config.ignore`
  (`re:.*embed_tokens$`, `re:.*embeddings$`). Weights untouched.
- Synthetic adapters: `scripts/make_adapters.py` (seeded random weights, bf16). Seeds, sizes and SHA-256 per adapter are in `env/adapters-*manifest.json`; regenerating gives the same hashes.
- `unsloth/Llama-3.2-3B-Instruct` with the trained adapter `jeeejeee/llama32-3b-text2sql-spider`.

## Workloads
- Throughput: `scripts/stream_bench.py ctx`. Each request is a seeded word-list prompt padded to exactly 3,072 chat-templated tokens and forced to 512 output tokens.
- Determinism: `scripts/dprobe.py`. Prompts are in `prompts/` with SHA-256 in `prompts/manifest.json` (text of CPython 3.12.3 standard-library modules, PSF licence).

## Scripts (as run)
- `chain_pub.sh`: five arms on the hybrid model (default, tuned split_k=1, batch-invariant, split-K cap, tuned for speed)
- `chain_min.sh`: minimal launch command, speculative decoding off
- `chain_var.sh`: repeat runs of default and cap, and the pair without speculative decoding
- `chain_dense.sh`: Llama-3.2-3B with the trained adapter
- `logcfg.py`: records the kernel settings vLLM resolves under each arm's environment
- `xfer.py`, `shapes.py`: kernel-level timing of shrink/expand across model shapes
- The split-K cap arm applies the one changed line of PR 61087 to the installed 0.34.0 wheel behind the environment switch `KV_CAP_SPLITK=1`.
- Output directories named `private/` are only a local naming habit.

## Config files (`configs/`)
- `gb300-nemotron-speed/`: shrink and expand settings tuned on GB300 for this model's shapes at rank 256, 16 slots
- `gb300-nemotron-split1/`: the same with shrink restricted to `split_k=1`
- `default-blocks-split1/`: vLLM's default shrink settings with `split_k=1`, for any shape

## Results
- `results/SUMMARY.md`: tables
- `results/throughput.csv`, `results/determinism.csv`: one row per run / probe
- `results/runs/<run>/`: client summary, per-request timings (no text), GPU telemetry
- `results/servers/<arm>/`: launch arguments, environment switches, resolved kernel settings, start-up log excerpt
- `comments/`: the text posted upstream

Full server logs are available on request.
