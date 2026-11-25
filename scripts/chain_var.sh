#!/bin/bash
# Repeat runs for a spread on the default-vs-cap comparison (vllm PR 61087), plus one default/cap pair with speculative decoding off.
# Throughput replays only; same flags and workload as chain_pub.sh.
cd $HOME/bench; source lib4.sh
export VLLM_BIN=$HOME/kv/venv034/bin/vllm; export PATH=$HOME/kv/venv034/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.16-linux-aarch64-gnu/include/python3.12
grep -v "speculative_config" serve_base.sh > serve_base_nomtp.sh; sed 's#./serve_base.sh#./serve_base_nomtp.sh#' serve_y.sh > serve_y_nomtp.sh; chmod +x serve_base_nomtp.sh serve_y_nomtp.sh
arm() { # tag cap(0/1) serve-script full(0/1)
  grep -q "var arm $1 done" logs/chain4.log && return
  CUR=var-$1; SR=$HOME/bench/runs/$CUR; stop_server
  unset VLLM_TUNED_CONFIG_FOLDER VLLM_BATCH_INVARIANT
  if [ "$2" = 1 ]; then export KV_CAP_SPLITK=1; else unset KV_CAP_SPLITK; fi
  export EXTRA="--no-enable-prefix-caching"
  setsid nohup ./$3 16 256 0.89 $SR > /dev/null 2>&1 &
  ok=0; for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/health > /dev/null && { ok=1; break; }; pgrep -u $USER -x "VLLM::EngineCor" > /dev/null || [ $i -lt 24 ] || break; sleep 5; done
  if [ $ok = 1 ]; then
    date -u +%FT%T.%3NZ > $SR/serve-ready.txt; load_set cap-r256
    ./run_stream.sh _warm-$CUR ctx 16 2 3072 64 $(names cap-r256); pkill -x nvidia-smi
    run $CUR-mixed16-w16 ctx 16 25 3072 512 $(names cap-r256)
    run $CUR-one-adapter-w1 ctx 1 40 3072 512 cap-r256-01
    if [ "$4" = 1 ]; then
      run $CUR-one-adapter-w16 ctx 16 12 3072 512 cap-r256-01
      run $CUR-base-w16 ctx 16 12 3072 512 nemotron-super-vl-fp8
    fi
    echo "$(date -u +%T) var arm $1 done" >> logs/chain4.log
  else echo "$(date -u +%T) server failed: var $1" >> logs/chain4.log; fi
}
arm default-r2 0 serve_y.sh 1
arm cap-r2 1 serve_y.sh 1
arm cap-r3 1 serve_y.sh 1
arm default-r3 0 serve_y.sh 1
arm nomtp-default 0 serve_y_nomtp.sh 0
arm nomtp-cap 1 serve_y_nomtp.sh 0
stop_server; echo done > logs/chain_var.done
