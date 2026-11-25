**Reproduces on v0.34.0 and nightly on GB300; turning shrink split-K off removes it on two models**

We see the same run-to-run drift on the current release, on different hardware and a different hybrid checkpoint from the original report, and also on a plain dense model with a trained adapter. On both, setting shrink `split_k=1` removes it. That differs from the matrix in the original report, so I'm sharing the numbers and a public reproduction rather than claiming a root cause for that setup.

**Setup**

- 1x NVIDIA GB300 (SM103), aarch64, Ubuntu 24.04, driver 595.99.02 (CUDA 13.2), torch built for CUDA 13.0
- vLLM 0.34.0 wheel, torch 2.14.0+cu130, triton 3.8.0, flashinfer 0.7.3
- Model A: `RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8` at `38effd54`, text only, with one synthetic bf16 LoRA (rank 256, alpha 128, 192 linear modules); 16 adapter slots
- Model B: `unsloth/Llama-3.2-3B-Instruct` with the trained adapter `jeeejeee/llama32-3b-text2sql-spider` (rank 8)
- Probe: the design from this issue. 16 byte-identical chat requests, temperature 0, seed 0, 256 output tokens, prefix caching off, outputs byte-compared. Four prompts from 29 to 7,110 tokens, built from Python 3.12.3 standard-library source so anyone can regenerate them.

Minimal launch command for model A (speculative decoding off):

```
vllm serve <model> --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --gpu-memory-utilization 0.89 --max-model-len 32768 \
  --no-enable-prefix-caching \
  --enable-lora --max-loras 16 --max-lora-rank 256 \
  --lora-target-modules qkv_proj q_proj k_proj v_proj o_proj in_proj out_proj up_proj down_proj
```

(`--lora-target-modules` is needed for memory: without it adapter slots are also reserved on the routed experts and start-up runs out of GPU memory.)

**Model A: distinct outputs out of 16 identical requests, sent one at a time (four prompts)**

| Shrink kernel settings | Speculative decoding | Adapter | Base model, same server |
|---|---|---|---|
| Default (`split_k=64` at decode batch sizes) | off | 16, 15, 15, 16 | 1, 1, 1, 1 |
| Default | MTP, 3 draft tokens | 16, 16, 16, 16 | 1, 1, 1, 1 |
| Split-K capped at the number of K blocks (#61087): 16 to 32 | MTP | 16, 16, 16, 16 | 1, 1, 1, 1 |
| Tuned block sizes, split-K still on | MTP | 16, 16, 15, 16 | 1, 1, 1, 1 |
| Tuned block sizes, `split_k=1` | off | 1, 1, 1, 1 | 1, 1, 1, 1 |
| Tuned block sizes, `split_k=1` | MTP | 1, 1, 1, 1 | 1, 1, 1, 1 |

The last row and the default row were repeated on nightly `0.34.1rc1.dev90+g2d08d35d4` with the same pattern (default 16, 15, 15, 15; `split_k=1` 1, 1, 1, 1).

**Model B (trained adapter): same probe**

| Settings | Adapter, one at a time | Adapter, sent together | Adapter throughput, 16 clients |
|---|---|---|---|
| Default (split-K 64 over 12 K blocks) | 7, 7, 12, 5 | 7, 11, 11, 6 | 5.86 requests/s |
| Default block sizes, `split_k=1` | 1, 1, 1, 1 | 6, 5, 3, 2 | 3.87 |
| `VLLM_BATCH_INVARIANT=1` | 1, 1, 1, 1 | 1, 1, 1, 1 | 2.22 |

The base model gave 1 distinct output in every one-at-a-time row on both models.

So on this hardware the drift is confined to the LoRA path, does not depend on speculative decoding, persists with fewer splits and with tuned block sizes, and disappears with split reduction off.

**How this relates to existing work**

- The docs for batch-invariant mode already describe replacing the atomic split-K reduction with a fixed-order one, so the mechanism is not news. What may be useful here is the measurement on a hybrid model and with a trained adapter.
- `VLLM_BATCH_INVARIANT=1` does not start on model A in 0.34.0 ("not supported for MAMBA2_ATTN"); that is tracked in #27433 and #54993. Until that work lands, `split_k=1` through a config file is the only way we found to get repeatable adapter output on a Mamba2 hybrid.
- Cost of the workaround depends on tuning. On model A, with block sizes tuned per shape, `split_k=1` runs mixed 16-adapter traffic at 1.58 requests/s against 0.99 on default (+59%); tuned for speed with split-K on gives 1.63 (+65%) but is not repeatable. On model B, changing only `split_k` and keeping default block sizes cost 34% of adapter throughput.
- `split_k=1` gives run-to-run repeatability for a request sent on its own. It does not give batch invariance: sent together, outputs still vary on both models. Batch-invariant mode does, at a larger cost (model B base traffic 9.30 → 2.69 requests/s).
- The original report found `split_k=1` insufficient on GB10 with an NVFP4 checkpoint. We have not reproduced that setup, so there may be a second effect there that this does not explain.

**Caveats**

- Model A's adapter is synthetic: seeded random values, not trained. Random weights push generation into unstable text, so a small numeric difference shows up within a few tokens; the trained adapter on model B drifts less often (5 to 12 of 16).
- Adapter magnitude is not the cause on model A: weights ten times smaller (std 0.001) gave 15, 15, 15, 16 under default settings and 1, 1, 1, 1 with `split_k=1`.
- Model A needs two extra `ignore` patterns for its embedding layers to load on 0.34.0. Weights are unchanged.
- Prompt token counts differ slightly between 0.34.0 and the nightly for the same text (for example 6,524 and 6,537).

Probe script, exact prompts with checksums, adapter generator with seeds, config files and per-arm results: https://github.com/NextLM/vllm-gb300-lora-repro. We have this GB300 until 7 December and GB10 machines after that, and can run other arms if useful.

<sub>AI assistance was used for the harness, the runs and this draft; I reviewed the results.</sub>
