#!/bin/bash
# Minimal reproduction arms for vllm issue 50602: fewest flags that still load this checkpoint, speculative decoding (MTP) OFF.
# --lora-target-modules is required: without it vLLM also reserves adapter slots on the routed experts and runs out of memory at start-up.
# Probe only (16 identical requests, adapter and base). Arms: default kernels | shrink split_k=1 configs | batch-invariant | default with MTP on.
cd $HOME/bench; source lib4.sh
export PATH=$HOME/kv/venv034/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
arm() { # tag config-folder|none batch-invariant(0/1) extra-flags
  CUR=min-$1; SR=$HOME/bench/runs/$CUR; mkdir -p $SR; stop_server
  if [ "$2" != none ]; then export VLLM_TUNED_CONFIG_FOLDER=$HOME/bench/tune/$2; else unset VLLM_TUNED_CONFIG_FOLDER; fi
  if [ "$3" = 1 ]; then export VLLM_BATCH_INVARIANT=1; else unset VLLM_BATCH_INVARIANT; fi
  unset KV_CAP_SPLITK
  CMD="vllm serve $HOME/bench/model-view --served-model-name nemotron-super-vl-fp8 --host 127.0.0.1 --port 8000 --trust-remote-code --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0} --gpu-memory-utilization 0.89 --max-model-len 32768 --no-enable-prefix-caching --enable-lora --max-loras 16 --max-lora-rank 256 --lora-target-modules qkv_proj q_proj k_proj v_proj o_proj in_proj out_proj up_proj down_proj $4"
  echo "$CMD" > $SR/serve-command.txt; env | grep -E "^VLLM_" > $SR/serve-env.txt
  python $HOME/kv/logcfg.py $SR/chosen-kernel-config.json 16 256 "4096,5;4096,3;4096,1;5376,1;8192,1" "8192,5;4096,3;4096,1;5376,1;256,1" > $SR/chosen-kernel-config.log 2>&1
  setsid nohup $CMD > $SR/vllm.log 2>&1 &
  ok=0; for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u $USER -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    for a in cap-r256-01 cap-s001-01; do [ -d adapters/$a ] && curl -s -o /dev/null -X POST http://127.0.0.1:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d "{\"lora_name\":\"$a\",\"lora_path\":\"$HOME/bench/adapters/$a\"}"; done
    M="cap-r256-01 nemotron-super-vl-fp8"; [ -d adapters/cap-s001-01 ] && M="cap-r256-01 cap-s001-01 nemotron-super-vl-fp8"
    python3 $HOME/kv/dprobe.py private/runs/dprobe/$CUR.json 16 256 $M > private/runs/dprobe/$CUR.log 2>&1
    echo "$(date -u +%T) min arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: min $1 $(grep -E "Error" $SR/vllm.log | tail -1 | cut -c1-160)" >> logs/chain4.log; fi
}
arm default none 0 ""
arm nosplit configs-nosplit-full 0 ""
arm bi none 1 ""
arm default-mtp none 0 "--speculative_config.method mtp --speculative_config.num_speculative_tokens 3"
stop_server; echo done > logs/chain_min.done
