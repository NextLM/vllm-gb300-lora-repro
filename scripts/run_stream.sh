#!/bin/bash
# Usage: run_stream.sh <run-name> <stream_bench args after out_dir...>   e.g. run_stream.sh name ctx 16 25 3072 512 model...
R=$HOME/bench/private/runs/$1; MODE=$2; shift 2
mkdir -p "$R"
curl -s http://127.0.0.1:8000/metrics > "$R/metrics-before.prom"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,nounits -l 1 > "$R/gpu-telemetry.csv" &
TP=$!
date -u +%FT%T.%3NZ > "$R/timer-start.txt"
python3 $HOME/bench/stream_bench.py "$MODE" "$R" "$@" > "$R/client.log" 2>&1
echo "exit=$?" > "$R/client-exit.txt"
date -u +%FT%T.%3NZ > "$R/timer-end.txt"
kill $TP
curl -s http://127.0.0.1:8000/metrics > "$R/metrics-after.prom"
