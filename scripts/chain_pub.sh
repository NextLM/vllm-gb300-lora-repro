#!/bin/bash
# vLLM 0.34.0, Nemotron Super FP8 (public Red Hat checkpoint), 16 synthetic rank-256 adapters, public prompts only.
# One server per arm, prefix caching off. Per arm: throughput replay (synthetic prompts of exactly 3,072 tokens, 512 forced output tokens)
# and the identical-request determinism probe.
cd $HOME/bench; source lib4.sh
export VLLM_BIN=$HOME/kv/venv034/bin/vllm; export PATH=$HOME/kv/venv034/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12
arm() { # tag config-folder|none batch-invariant(0/1) cap(0/1)
  CUR=pub-$1; SR=$HOME/bench/runs/$CUR; stop_server
  if [ "$2" != none ]; then export VLLM_TUNED_CONFIG_FOLDER=$HOME/bench/tune/$2; else unset VLLM_TUNED_CONFIG_FOLDER; fi
  if [ "$3" = 1 ]; then export VLLM_BATCH_INVARIANT=1; else unset VLLM_BATCH_INVARIANT; fi
  if [ "$4" = 1 ]; then export KV_CAP_SPLITK=1; else unset KV_CAP_SPLITK; fi
  export EXTRA="--no-enable-prefix-caching"
  setsid nohup ./serve_y.sh 16 256 0.89 $SR > /dev/null 2>&1 &
  ok=0; for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u $USER -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    date -u +%FT%T.%3NZ > $SR/serve-ready.txt; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $SR/mem-at-ready.txt
    load_set cap-r256
    ./run_stream.sh _warm-$CUR ctx 16 2 3072 64 $(names cap-r256); pkill -x nvidia-smi
    run $CUR-mixed16-w16 ctx 16 25 3072 512 $(names cap-r256)
    run $CUR-one-adapter-w16 ctx 16 12 3072 512 cap-r256-01
    run $CUR-one-adapter-w1 ctx 1 40 3072 512 cap-r256-01
    run $CUR-base-w16 ctx 16 12 3072 512 nemotron-super-vl-fp8
    mkdir -p private/runs/dprobe
    python3 $HOME/kv/dprobe.py private/runs/dprobe/$CUR.json 16 256 cap-r256-01 nemotron-super-vl-fp8 > private/runs/dprobe/$CUR.log 2>&1
    echo "$(date -u +%T) arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: $1 $(grep -E "Error" $SR/vllm.log | tail -1 | cut -c1-160)" >> logs/chain4.log; fi
}
if [ -n "$1" ]; then arm "$@"; stop_server; exit; fi
arm default none 0 0
arm nosplit configs-nosplit-full 0 0
arm bi none 1 0
arm cap none 0 1
arm tuned configs-full 0 0
stop_server; echo done > logs/chain_pub.done
# kernel-level transfer matrix once the GPU is free
cd $HOME/kv && venv034/bin/python xfer.py shapes.json xfer-full.jsonl $HOME/bench/tune/configs-full $HOME/bench/tune/configs-nosplit-full --oracle > logs/xfer.log 2>&1; echo done > logs/xfer.done
