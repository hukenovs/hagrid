#!/bin/bash

gpu_indexes=""
command=""
config_path=""

while getopts "g:c:p:" opt; do
  case $opt in
    g) gpu_indexes="$OPTARG" ;;
    c) command="$OPTARG" ;;
    p) config_path="$OPTARG" ;;
    \?) echo "Invalid Argument: -$OPTARG" >&2; exit 1 ;;
    :) echo "Need value for argument: -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$gpu_indexes" ] || [ -z "$command" ] || [ -z "$config_path" ]; then
  echo "Usage: $0 -g gpu_indexes -c command -p config_path"
  exit 1
fi

IFS=',' read -r -a gpu_indices <<< "$gpu_indexes"
num_gpus=${#gpu_indices[@]}

gpu_indexes_str=$(IFS=,; echo "${gpu_indices[*]}")

CUDA_VISIBLE_DEVICES=$gpu_indexes_str torchrun --standalone --nproc_per_node=$num_gpus run.py -c "$command" -p "$config_path" --n_gpu=$num_gpus
