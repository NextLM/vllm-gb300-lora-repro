Title: [Bug]: VLLM_BATCH_INVARIANT=1 loads tuned LoRA shrink configs since #70197

### Your current environment

1x NVIDIA GB300 (SM103), aarch64, Ubuntu 24.04, driver 595.99.02. Tested with the `vllm/` changes of #70197 (`ca6e2b9`) applied to the 0.34.0 wheel. `use_lora_kernel_reduction` and `load_lora_op_config` on main are identical to what I tested.

### Describe the bug

Thanks again for #70197. `--lora-split-k-reduction kernel` works well here: repeatable adapter output on both models from #70061, at 5.43 requests/s on Llama-3.2-3B (5.42 with my local switch) and 1.29 on the Nemotron hybrid (1.28).

One side effect I wanted to check. Since #70197, a shrink file in `VLLM_TUNED_CONFIG_FOLDER` takes precedence over `VLLM_BATCH_INVARIANT=1`. Before, batch-invariant mode skipped tuned LoRA configs. With a tuned shrink file that uses `split_k` above 1, adapter output under batch-invariant mode is no longer repeatable:

| `VLLM_BATCH_INVARIANT=1` | Adapter, one at a time | Adapter, 16 sent together | Base, both ways |
|---|---|---|---|
| no tuned folder | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 |
| tuned folder set | 5, 2, 10, 4 | 4, 2, 4, 3 | 1, 1, 1, 1 |

Distinct outputs out of 16 identical greedy requests, four prompts. The log in the second case shows `Using tuned LoRA kernel configs from …/NVIDIA_GB300_SHRINK.json`, with no warning that the fixed-order reduction is off.

The same holds for the new flag: with the folder set, `--lora-split-k-reduction kernel` starts normally and the adapter gives 5, 1, 6, 6 distinct. The docstring documents that case; the batch-invariant one is the change in existing behaviour.

**Is this precedence intended?** If it is, a start-up warning when a tuned shrink config overrides either switch would stop the guarantee from being lost unnoticed.

### To reproduce

```
VLLM_BATCH_INVARIANT=1 VLLM_TUNED_CONFIG_FOLDER=<folder> \
vllm serve unsloth/Llama-3.2-3B-Instruct --enable-lora --max-loras 4 --max-lora-rank 8 \
  --lora-modules sql=jeeejeee/llama32-3b-text2sql-spider --no-enable-prefix-caching
```

- Folder: `configs/gb300-nemotron-speed/` in https://github.com/NextLM/vllm-gb300-lora-repro. Its shrink file has `split_k` values of 1, 8 and 64 depending on shape; it was tuned for another model and reaches Llama's shapes through the nearest-key lookup. Any shrink file for the local GPU with `split_k` above 1 should behave the same.
- Probe: `scripts/dprobe.py` (temperature 0, seed 0, 256 output tokens), 16 requests per prompt, one at a time and then together.
- Script and per-run results: `scripts/chain_pr70197.sh`, `results/SUMMARY.md` (arms `dense-pr-bi`, `dense-pr-bi-tuned`, `dense-pr-kernel-tuned`).

One run per arm, one GPU type.

cc @jeejeelee

<sub>AI assistance was used for the test runs and this draft; I reviewed the results.</sub>
