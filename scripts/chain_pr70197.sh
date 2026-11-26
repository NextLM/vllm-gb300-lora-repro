#!/bin/bash
# vLLM PR 70197 (--lora-split-k-reduction) tested on GB300.
# Code under test: the PR's changes to vllm/ (head ca6e2b91d) applied with patch(1) to an untouched copy of the
# vLLM 0.34.0 wheel's package, loaded ahead of the installed wheel through PYTHONPATH. Nothing else differs from the
# 0.34.0 arms in chain_dense.sh / chain_pub.sh / chain_det.sh.
cd $HOME/bench; source lib4.sh
export VLLM_BIN=$HOME/kv/venv034/bin/vllm; export PATH=$HOME/kv/venv034/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset VLLM_BATCH_INVARIANT KV_CAP_SPLITK KV_LORA_DET
OVL=$HOME/kv/pr70197
B=$(ls -d $HOME/kv/models/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/*); A=$(ls -d $HOME/kv/models/hub/models--jeeejeee--llama32-3b-text2sql-spider/snapshots/*)
mkdir -p private/runs/dprobe
setenv() { # config-folder|none  bi(0|1)
  if [ "$1" != none ]; then export VLLM_TUNED_CONFIG_FOLDER=$HOME/bench/tune/$1; else unset VLLM_TUNED_CONFIG_FOLDER; fi
  if [ "$2" = 1 ]; then export VLLM_BATCH_INVARIANT=1; else unset VLLM_BATCH_INVARIANT; fi
}
record() { env | grep -E "^VLLM_|^KV_" > $SR/serve-env.txt; PYTHONPATH=$OVL python -c "import vllm; print('vllm package:', vllm.__file__, vllm.__version__)" >> $SR/serve-env.txt 2>/dev/null; }
darm() { # tag config-folder|none reduction(atomic|kernel) bi(0|1) full|probe
  CUR=dense-$1; SR=$HOME/bench/runs/$CUR; mkdir -p $SR; stop_server; setenv $2 $4
  CMD="vllm serve $B --served-model-name llama32 --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.3 --max-model-len 16384 --no-enable-prefix-caching --enable-lora --max-loras 4 --max-lora-rank 8 --lora-modules sql=$A --lora-split-k-reduction $3"
  echo "$CMD" > $SR/serve-command.txt; record
  PYTHONPATH=$OVL setsid nohup $CMD > $SR/vllm.log 2>&1 &
  ok=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u gb300 -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    ./run_stream.sh _warm-$CUR ctx 16 2 3072 64 sql; pkill -x nvidia-smi
    if [ "$5" = full ]; then
      run $CUR-adapter-w16 ctx 16 12 3072 512 sql
      run $CUR-adapter-w1 ctx 1 30 3072 512 sql
      run $CUR-base-w16 ctx 16 12 3072 512 llama32
    fi
    python3 $HOME/kv/dprobe.py private/runs/dprobe/$CUR.json 16 256 sql llama32 > private/runs/dprobe/$CUR.log 2>&1
    echo "$(date -u +%T) dense arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: dense $1 $(grep -E "Error" $SR/vllm.log | tail -1 | cut -c1-160)" >> logs/chain4.log; fi
}
parm() { # tag config-folder|none reduction
  CUR=pub-$1; SR=$HOME/bench/runs/$CUR; mkdir -p $SR; stop_server; setenv $2 0
  export EXTRA="--no-enable-prefix-caching --lora-split-k-reduction $3"
  PYTHONPATH=$OVL setsid nohup ./serve_y.sh 16 256 0.89 $SR > /dev/null 2>&1 &
  ok=0; for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u gb300 -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    date -u +%FT%T.%3NZ > $SR/serve-ready.txt; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $SR/mem-at-ready.txt
    record
    load_set cap-r256
    ./run_stream.sh _warm-$CUR ctx 16 2 3072 64 $(names cap-r256); pkill -x nvidia-smi
    run $CUR-mixed16-w16 ctx 16 25 3072 512 $(names cap-r256)
    run $CUR-one-adapter-w16 ctx 16 12 3072 512 cap-r256-01
    run $CUR-one-adapter-w1 ctx 1 40 3072 512 cap-r256-01
    run $CUR-base-w16 ctx 16 12 3072 512 nemotron-super-vl-fp8
    python3 $HOME/kv/dprobe.py private/runs/dprobe/$CUR.json 16 256 cap-r256-01 nemotron-super-vl-fp8 > private/runs/dprobe/$CUR.log 2>&1
    echo "$(date -u +%T) arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: $1 $(grep -E "Error" $SR/vllm.log | tail -1 | cut -c1-160)" >> logs/chain4.log; fi
}
echo "$(date -u +%T) chain_pr70197 start" >> logs/chain4.log
darm pr-kernel none kernel 0 full
darm pr-atomic none atomic 0 probe
darm pr-kernel-tuned configs-full kernel 0 probe
darm pr-bi-tuned configs-full atomic 1 probe
darm pr-bi none atomic 1 probe
parm pr-kernel none kernel
stop_server; echo "$(date -u +%T) chain_pr70197 finished" >> logs/chain4.log; echo done > logs/chain_pr70197.done
