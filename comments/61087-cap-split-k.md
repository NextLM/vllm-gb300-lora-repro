**End-to-end results on GB300 (hybrid 120B model, rank 256, 16 adapters), plus a small-model control**

@linitra24 asked for end-to-end numbers. Here is a serving comparison on different hardware and a much larger model than the earlier results in this thread, with only this PR's change applied.

**Setup**

- 1x NVIDIA GB300 (SM103), aarch64, Ubuntu 24.04, driver 595.99.02 (CUDA 13.2), torch built for CUDA 13.0
- vLLM 0.34.0 wheel, torch 2.14.0+cu130, triton 3.8.0, flashinfer 0.7.3
- Model: `RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8` (hybrid Mamba2 / attention / MoE), text only
- 16 bf16 LoRA adapters, rank 256, on 192 linear modules each; `--max-loras 16 --max-lora-rank 256`
- Workload: 16 concurrent streaming clients, each request exactly 3,072 prompt tokens and 512 output tokens (`ignore_eos`), temperature 0, prefix caching off. Speculative decoding (MTP, 3 draft tokens) on unless stated.
- One server per run, same flags. The only difference between arms is this PR's line: `SPLIT_K = min(kernel_config["split_k"], triton.cdiv(K, BLOCK_K))`.

How the patch was applied: I did not build this branch. I added the one changed line to `lora_shrink_op.py` in the 0.34.0 wheel behind a local switch and ran with it off and on. On this model the shrink kernel sees K = 4,096, 5,376 and 8,192, so at decode batch sizes the cap takes split-K from 64 to 16, 21 and 32.

**Results, three runs per arm (min to max)**

| Run | Requests valid | Inter-token p50, 0.34.0 → 0.34.0 + this change | Requests/s, 0.34.0 → 0.34.0 + this change |
|---|---|---|---|
| 16 adapters mixed, 16 clients | 400 / 400 each | 28.2–28.5 → 22.2–22.4 ms | 0.987–0.992 → 1.236–1.257 (+27%) |
| One adapter, 16 clients | 192 / 192 each | 18.4–18.8 → 16.6 ms | 1.413–1.460 → 1.639–1.670 (+14%) |
| One adapter, 1 client | 40 / 40 each | 4.6–4.8 → 3.6 ms | 0.325–0.343 → 0.426–0.435 (+27%) |
| Base model (no adapter), 16 clients | 192 / 192 each | 9.3 → 9.3 ms | 2.965–2.970 → 2.966–2.969 |

Percentages compare medians. Mean GPU power on the mixed run rose from 627 W to 698 W, consistent with less time in empty programs.

With speculative decoding off (one run per arm): mixed 16 adapters 0.444 → 0.610 requests/s (+37%), one adapter single client 0.124 → 0.167 requests/s (+35%).

**Small-model control**

`unsloth/Llama-3.2-3B-Instruct` with the trained adapter `jeeejeee/llama32-3b-text2sql-spider` (rank 8), same workload, one run per arm: no measurable change and no regression. Adapter at 16 clients 5.86 → 5.78 requests/s (4.7 ms/token in both), base model 9.30 → 9.32 requests/s. The cap did apply (split-K 64 → 12 on the 3,072-wide layers), but at rank 8 on a 3B model the shrink kernel is a small share of the step. That matches the PR description's own note about small-model end-to-end runs.

So the change is a clear end-to-end win where adapter kernels are a large share of the decode step, and neutral where they are not, with no effect on base-model traffic in either case.

**Two notes for context**

- Repeatability is unchanged from default on both models. With 16 byte-identical greedy requests sent one at a time, the adapter gave the same number of distinct outputs with and without the cap (16 of 16 on the hybrid model). This PR does not claim to change that.
- For reference only, and outside this PR's scope: per-shape tuned settings via `VLLM_TUNED_CONFIG_FOLDER` go further on the hybrid model's mixed workload. Tuned for speed they reach 1.63 requests/s (+65% over 0.34.0); with `split_k=1` they reach 1.58 requests/s (+59%) and the adapter output is identical on every repeat of a request sent on its own. Most of the gain comes from the 5-slice Mamba2 input projection.

**Caveats**

- The adapters on the hybrid model are synthetic: seeded random weights (std 0.01), not trained. That does not affect kernel timing, but the generated text is not meaningful.
- The hybrid checkpoint needs two extra `ignore` patterns for its embedding layers to load on 0.34.0. Weights are unchanged.
- The default arm was also run on nightly `0.34.1rc1.dev90+g2d08d35d4` with the same result (0.991 requests/s mixed); the cap was not tested on the nightly.

Scripts, launch flags, per-run summaries and request timings: https://github.com/NextLM/vllm-gb300-lora-repro

<sub>AI assistance was used for the harness, the runs and this draft; I reviewed the results.</sub>
