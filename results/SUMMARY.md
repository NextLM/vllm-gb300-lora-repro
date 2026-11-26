## Nemotron Super FP8, 16 synthetic rank-256 adapters, vLLM 0.34.0, speculative decoding on

| Arm | Mixed 16 adapters: req/s | ITL p50 ms | One adapter w16: req/s | One adapter w1: req/s | Base w16: req/s | GPU W (mixed) | Adapter distinct of 16, one at a time | Base |
|---|---|---|---|---|---|---|---|---|
| Default (3 runs) | 0.987 to 0.992 | 28.2 to 28.5 | 1.413 to 1.460 | 0.325 to 0.343 | 2.965 to 2.970 | 627 to 628 | 16, 16, 16, 16 | 1, 1, 1, 1 |
| Split-K cap, PR 61087 (3 runs) | 1.236 to 1.257 | 22.2 to 22.4 | 1.639 to 1.670 | 0.426 to 0.435 | 2.966 to 2.969 | 697 to 699 | 16, 16, 16, 16 | 1, 1, 1, 1 |
| Tuned, split on | 1.632 | 17.4 | 1.989 | 0.497 | 2.967 | 752 | 16, 16, 15, 16 | 1, 1, 1, 1 |
| Tuned, split_k=1 | 1.575 | 17.8 | 1.918 | 0.446 | 2.963 | 742 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| Fixed-order shrink reduce, LoRA op only (`KV_LORA_DET=1`, default blocks, 1 run) | 1.276 | 22.1 | 1.615 | 0.439 | 2.968 | 691 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| Nightly 2d08d35d4, default | 0.991 | 28.2 | 1.452 | 0.333 | 2.951 | 624 | 16, 15, 15, 15 | 1, 1, 1, 1 |
| Nightly 2d08d35d4, tuned split_k=1 | 1.578 | 17.9 | 1.852 | 0.441 | 2.975 | 740 | 1, 1, 1, 1 | 1, 1, 1, 1 |

Median mixed throughput: default 0.990, cap 1.255 (+27%), tuned 1.632 (+65%), tuned split_k=1 1.575 (+59%).
Median-one-adapter-w16: default 1.449, cap 1.645 (+14%).
Median-one-adapter-w1: default 0.339, cap 0.430 (+27%).

## Speculative decoding off (one run each)

| Arm | Mixed req/s | ITL p50 ms | One adapter w1 req/s | ITL p50 ms |
|---|---|---|---|---|
| var-nomtp-default | 0.444 | 62.8 | 0.124 | 15.1 |
| var-nomtp-cap | 0.61 | 43.5 | 0.167 | 11.0 |

## Minimal launch command, probe only

| Arm | Adapter std 0.01 | Adapter std 0.001 | Base |
|---|---|---|---|
| min-default | 16, 15, 15, 16 | 15, 15, 15, 16 | 1, 1, 1, 1 |
| min-default-mtp | 16, 16, 16, 16 | 16, 15, 16, 14 | 1, 1, 1, 1 |
| min-nosplit | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| min-bi | server did not start: batch_invariant mode is not supported for MAMBA2_ATTN | | |

## Llama-3.2-3B-Instruct with trained adapter jeeejeee/llama32-3b-text2sql-spider (rank 8), vLLM 0.34.0

| Arm | Adapter w16 req/s | ITL p50 ms | Adapter w1 ITL ms | Base w16 req/s | ITL ms | Adapter distinct, one at a time | together | Base one at a time | together |
|---|---|---|---|---|---|---|---|---|---|
| dense-default | 5.862 | 4.7 | 3.7 | 9.303 | 3.0 | 7, 7, 12, 5 | 7, 11, 11, 6 | 1, 1, 1, 1 | 3, 2, 4, 1 |
| dense-cap | 5.777 | 4.7 | 3.8 | 9.322 | 3.0 | 7, 9, 13, 3 | 6, 9, 11, 7 | 1, 1, 1, 1 | 3, 2, 4, 1 |
| dense-split1 | 3.869 | 7.4 | 6.5 | 9.311 | 3.0 | 1, 1, 1, 1 | 6, 5, 3, 2 | 1, 1, 1, 1 | 3, 2, 4, 1 |
| dense-bi | 2.219 | 13.0 | 12.9 | 2.685 | 11.0 | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| dense-det (fixed-order shrink reduce, LoRA op only) | 5.424 | 5.1 | 4.1 | 9.330 | 3.0 | 1, 1, 1, 1 | 7, 4, 4, 2 | 1, 1, 1, 1 | 3, 2, 4, 1 |

The `det` arms enable the fixed-order split-K reduce that `VLLM_BATCH_INVARIANT=1` uses (PR 59377) for the LoRA shrink op only, through a local switch (`scripts/patch_det.py`); global batch-invariant mode stays off. A third arm combining it with the tuned block sizes did not start: the path forces `BLOCK_K=256`, and with the tuned `block_m`/`block_n` Triton ran out of shared memory.

## PR 70197 (`--lora-split-k-reduction`), its `vllm/` changes at ca6e2b9 applied to the 0.34.0 wheel, one run per arm

| Arm | Adapter req/s (w16) | Adapter w1 req/s | Base w16 req/s | Adapter distinct of 16, one at a time | together | Base one at a time | together |
|---|---|---|---|---|---|---|---|
| dense-pr-kernel (`kernel`) | 5.429 | 0.464 | 9.321 | 1, 1, 1, 1 | 7, 4, 3, 2 | 1, 1, 1, 1 | 3, 2, 3, 1 |
| dense-pr-atomic (`atomic`, probe only) | | | | 7, 9, 10, 6 | 8, 9, 7, 5 | 1, 1, 1, 1 | 3, 2, 4, 1 |
| dense-pr-kernel-tuned (`kernel` + tuned folder, probe only) | | | | 5, 1, 6, 6 | 11, 4, 6, 2 | 1, 1, 1, 1 | 3, 2, 4, 1 |
| dense-pr-bi (`VLLM_BATCH_INVARIANT=1`, probe only) | | | | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| dense-pr-bi-tuned (`VLLM_BATCH_INVARIANT=1` + tuned folder, probe only) | | | | 5, 2, 10, 4 | 4, 2, 4, 3 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| pub-pr-kernel (Nemotron, `kernel`): mixed 1.287, one adapter w16 1.607 | | 0.406 | 2.935 | 1, 1, 1, 1 | 12, 13, 4, 2 | 1, 1, 1, 1 | 12, 14, 4, 2 |

The tuned folder in these arms is `configs/gb300-nemotron-speed/`. The PR's unit-test file `tests/lora/test_lora_shrink_batch_invariant.py` passed on the same GPU (18 passed).

Runs with any invalid request: none.
