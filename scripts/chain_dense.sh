#!/bin/bash
# Second model, where batch-invariant mode is supported: Llama-3.2-3B-Instruct (public mirror) with a public TRAINED adapter
# (jeeejeee/llama32-3b-text2sql-spider, rank 8). Arms: default | default with shrink split_k=1 | batch-invariant | split-K cap (PR 61087).
cd $HOME/bench; source lib4.sh
export PATH=$HOME/kv/venv034/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
B=$(ls -d $HOME/kv/models/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/*); A=$(ls -d $HOME/kv/models/hub/models--jeeejeee--llama32-3b-text2sql-spider/snapshots/*)
python - <<PY > logs/mk_split1.log 2>&1
import runpy,sys; sys.argv=["mk_split1.py","$HOME/bench/tune/configs-default-split1"]; runpy.run_path("$HOME/kv/mk_split1.py")
PY
arm() { # tag config-folder|none batch-invariant(0/1) cap(0/1)
  CUR=dense-$1; SR=$HOME/bench/runs/$CUR; mkdir -p $SR; stop_server
  if [ "$2" != none ]; then export VLLM_TUNED_CONFIG_FOLDER=$HOME/bench/tune/$2; else unset VLLM_TUNED_CONFIG_FOLDER; fi
  if [ "$3" = 1 ]; then export VLLM_BATCH_INVARIANT=1; else unset VLLM_BATCH_INVARIANT; fi
  if [ "$4" = 1 ]; then export KV_CAP_SPLITK=1; else unset KV_CAP_SPLITK; fi
  CMD="vllm serve $B --served-model-name llama32 --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.3 --max-model-len 16384 --no-enable-prefix-caching --enable-lora --max-loras 4 --max-lora-rank 8 --lora-modules sql=$A"
  echo "$CMD" > $SR/serve-command.txt; env | grep -E "^VLLM_|^KV_" > $SR/serve-env.txt
  python $HOME/kv/logcfg.py $SR/chosen-kernel-config.json 4 8 "3072,3;3072,1;3072,2;8192,1" "3072,3;3072,1;8192,2" > $SR/chosen-kernel-config.log 2>&1
  setsid nohup $CMD > $SR/vllm.log 2>&1 &
  ok=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u $USER -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    ./run_stream.sh _warm-$CUR ctx 16 2 3072 64 sql; pkill -x nvidia-smi
    run $CUR-adapter-w16 ctx 16 12 3072 512 sql
    run $CUR-adapter-w1 ctx 1 30 3072 512 sql
    run $CUR-base-w16 ctx 16 12 3072 512 llama32
    python3 $HOME/kv/dprobe.py private/runs/dprobe/$CUR.json 16 256 sql llama32 > private/runs/dprobe/$CUR.log 2>&1
    echo "$(date -u +%T) dense arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: dense $1 $(grep -E "Error" $SR/vllm.log | tail -1 | cut -c1-160)" >> logs/chain4.log; fi
}
arm default none 0 0
arm split1 configs-default-split1 0 0
arm bi none 1 0
arm cap none 0 1
stop_server; echo done > logs/chain_dense.done
