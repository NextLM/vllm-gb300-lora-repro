cd $HOME/bench
start_server() { # N RK U name ; returns 0 if ready
  stop_server
  SR=$HOME/bench/runs/$4
  setsid nohup ./serve_x.sh $1 $2 $3 $SR > /dev/null 2>&1 &
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null; then date -u +%FT%T.%3NZ > $SR/serve-ready.txt; sleep 5
      nvidia-smi --query-gpu=memory.used --format=csv,noheader > $SR/mem-at-ready.txt
      nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > $SR/compute-apps-at-ready.txt; return 0; fi
    if ! pgrep -f "[v]llm serve" > /dev/null; then echo "server died: $4" >> logs/chain4.log; return 1; fi
    sleep 5
  done
  echo "server not ready: $4" >> logs/chain4.log; return 1
}
stop_server() { pkill -f "[v]llm serve"; sleep 3; pkill -x nvidia-smi; for i in $(seq 1 60); do pgrep -f "[v]llm serve" > /dev/null || break; sleep 2; done; sleep 5; }
load_set() { # prefix
  for i in $(seq -f "%02g" 1 16); do
    curl -s -o /dev/null -w "$1-$i load %{http_code} %{time_total}s\n" -X POST http://127.0.0.1:8000/v1/load_lora_adapter -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$1-$i\",\"lora_path\":\"$HOME/bench/adapters/$1-$i\"}" >> logs/chain4.log
  done
}
unload_set() { for i in $(seq -f "%02g" 1 16); do curl -s -o /dev/null -X POST http://127.0.0.1:8000/v1/unload_lora_adapter -H "Content-Type: application/json" -d "{\"lora_name\":\"$1-$i\"}"; done; }
names() { seq -f "$1-%02g" 1 16; }
run() { # name args...   (with CPU thread sampling)
  n=$1; shift; echo "$(date -u +%T) run $n" >> logs/chain4.log
  mkdir -p private/runs/$n
  ( top -b -H -d 5 -u $USER -w 200 | awk '$1 ~ /^[0-9]+$/ && $9+0 > 20 {print systime(), $1, $9, $12, $13}' > private/runs/$n/cpu-threads.txt ) &
  ./run_stream.sh $n "$@"
  pkill -x top; pkill -x nvidia-smi
  cp runs/$CUR/compute-apps-at-ready.txt private/runs/$n/ 2>/dev/null
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > private/runs/$n/mem-after.txt
  sleep 8
}
