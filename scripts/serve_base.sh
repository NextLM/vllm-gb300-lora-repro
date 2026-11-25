#!/bin/bash
# Base-only serve, loopback only. Publisher recipe minus --tensor-parallel-size (single GPU).
M=$HOME/bench/model-view
export CPATH=${CPATH:-$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec ${VLLM_BIN:-vllm} serve "$M" \
  --served-model-name nemotron-super-vl-fp8 \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --gpu-memory-utilization 0.89 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --prefix-match-unit 64 --block-size 64 \
  --compilation_config.max_cudagraph_capture_size 192 \
  --enable-expert-parallel \
  --enable-prefix-caching --async-scheduling \
  --speculative_config.method mtp --speculative_config.num_speculative_tokens 3 \
  --reasoning-parser nemotron_v3 "$@"
