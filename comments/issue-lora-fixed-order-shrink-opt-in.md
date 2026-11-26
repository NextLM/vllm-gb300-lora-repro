Title: [Feature]: Opt-in fixed-order LoRA shrink reduction without global batch-invariant mode

### The feature, motivation and pitch

**Ask:** an opt-in that selects the fixed-order split-K reduction for the LoRA shrink op (`_lora_shrink_reduce_kernel`, added in #59377) on its own, without turning on `VLLM_BATCH_INVARIANT` for the whole model. No change to the default.

**Why:** with default settings, a LoRA request repeated one at a time at temperature 0 does not return the same text (#50602, #50059). The cause on the setups below is the atomic split-K reduction in shrink. The fix already exists in tree, but it is only reachable through global batch-invariant mode, which

- does not start on Mamba2 hybrids ("not supported for MAMBA2_ATTN", tracked in #54993 and #27433), and
- slows base-model traffic as well (Llama-3.2-3B base, 16 clients: 9.30 → 2.69 requests/s).

For many uses, the weaker property is enough: the same adapter request gives the same answer when repeated. That needs the shrink reduction to be fixed-order and nothing else.

**Measurement.** vLLM 0.34.0, 1x GB300, 16 clients, 3,072 prompt / 512 output tokens, prefix caching off. "Repeatable" is the probe from #50602: 16 byte-identical greedy requests sent one at a time, four prompts, distinct outputs counted. For the new row I enabled the existing reduce path for the LoRA shrink op only with a one-line local switch in the 0.34.0 wheel; global batch-invariant mode stayed off.

`unsloth/Llama-3.2-3B-Instruct` + trained adapter `jeeejeee/llama32-3b-text2sql-spider` (rank 8):

| Setting | Adapter distinct of 16 | Adapter requests/s | Base requests/s |
|---|---|---|---|
| Default | 7, 7, 12, 5 | 5.86 | 9.30 |
| **Fixed-order reduce, LoRA shrink only** | **1, 1, 1, 1** | **5.42 (−7%)** | **9.33** |
| Tuned-config file with `split_k=1`, default block sizes | 1, 1, 1, 1 | 3.87 (−34%) | 9.31 |
| `VLLM_BATCH_INVARIANT=1` | 1, 1, 1, 1 | 2.22 (−62%) | 2.69 |

`RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8` (Mamba2 hybrid) + 16 synthetic rank-256 adapters, mixed traffic, MTP on:

| Setting | Adapter distinct of 16 | Mixed requests/s | Base requests/s |
|---|---|---|---|
| Default (0.34.0) | 16, 16, 16, 16 | 0.99 | 2.97 |
| Default + split-K cap (#61087, now on main) | 16, 16, 16, 16 | 1.24 to 1.26 | 2.97 |
| **Fixed-order reduce, LoRA shrink only** | **1, 1, 1, 1** | **1.28** | **2.97** |
| `VLLM_BATCH_INVARIANT=1` | does not start | | |

With the fixed-order path on, the adapter's variation when the 16 requests are sent together (12, 13, 4, 2 distinct) matches the base model's on the same server (11, 14, 4, 2), against 16, 15, 16, 14 by default. So the adapter path stops adding run-to-run variation; what remains is ordinary batching variation that only full batch-invariant mode removes.

So on these two models the existing kernel gives repeatable adapter output at −7% on a small dense model, and on a large hybrid it matches or slightly exceeds the default with the #61087 cap that is now on main (one run), with base traffic untouched.

### Alternatives

- **`split_k=1` by default.** Proposed in #50278 and declined there; I agree, and this is not that. Defaults stay as they are.
- **A tuned-config file with `split_k=1`** (the route suggested in #50278). It works and I use it, but it needs a per-GPU JSON that covers every `max_loras` / slice count / shape the model uses, and with default block sizes it costs 34% on the small model. It only gets fast after per-shape tuning (1.58 requests/s on the hybrid above).
- **Wait for batch-invariant Mamba2 (#54993).** That solves the hybrid start-up, not the cost to base traffic.

### Open questions for maintainers

1. Is an opt-in like this wanted at all, or is "use a tuned config" the intended answer?
2. If wanted, what form: an env var next to `VLLM_BATCH_INVARIANT`, or a LoRA config field?
3. The reduce path forces `BLOCK_K=256` and an 8-way split. Combining it with tuned `block_m`/`block_n` from a config file ran Triton out of shared memory on the hybrid's shapes, so the opt-in probably has to ignore tuned shrink configs the way batch-invariant mode does.
4. FP8 shrink (`lora_shrink_fp8_op.py`) would need the same treatment, with the `BLOCK_K` caveat from #50278.

If the direction is acceptable I can send a small PR with a test based on `tests/v1/determinism/test_lora_shrink_batch_invariant.py`.

### Additional context

- One GPU type and one run per new arm. The hybrid's adapters are synthetic (seeded random weights), the dense model's adapter is trained.
- This does not explain the GB10 / NVFP4 reports in #50602 where `split_k=1` was not sufficient and only an fp32 interior was stable.
- Scripts, the local switch, prompts with checksums and per-run results: https://github.com/NextLM/vllm-gb300-lora-repro (`scripts/chain_det.sh`, `scripts/patch_det.py`, `results/SUMMARY.md`).

cc @jeejeelee

<sub>AI assistance was used for the harness, the runs and this draft; I reviewed the results.</sub>
