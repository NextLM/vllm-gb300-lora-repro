#!/bin/bash
# Usage: serve_x.sh <max-loras> <max-lora-rank> <gpu-memory-utilization> <run-dir>
N=$1; RK=$2; U=$3; R=$4; mkdir -p "$R"; cd $HOME/bench
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,nounits -l 1 > "$R/gpu-telemetry-startup.csv" &
date -u +%FT%T.%3NZ > "$R/serve-start.txt"
echo "max_loras=$N max_lora_rank=$RK gpu_memory_utilization=$U max_cpu_loras=20" > "$R/serve-args.txt"
./serve_base.sh --gpu-memory-utilization $U --enable-lora --max-loras $N --max-cpu-loras 20 --max-lora-rank $RK \
  $EXTRA --lora-target-modules qkv_proj q_proj k_proj v_proj o_proj in_proj out_proj up_proj down_proj > "$R/vllm.log" 2>&1
