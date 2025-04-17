#!/bin/bash

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=27500
TRAIN_NODES=("${NODELIST[@]}")

echo "Nodes allocated:"
for node in "${TRAIN_NODES[@]}"; do
    echo "  - $node"
done

NO_PROXY_LIST=""
for node in "${TRAIN_NODES[@]}"; do
    if [ -z "$NO_PROXY_LIST" ]; then
        NO_PROXY_LIST="$node"
    else
        NO_PROXY_LIST="$NO_PROXY_LIST,$node"
    fi
done

export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

PROJECT_HOME=/mnt/petrelfs/lisiheng/24Game

MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/lisiheng/checkpoints
MODEL_NAME=Qwen2.5-0.5B-24_game_100000_direct_sft_0.01-32768-5e-5-256
MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

TENSOR_PARALLEL_SIZE=2

TEMPERATURE=0.6
TOP_K=32
MAX_TOKENS=1000

DATASET_DIR=./data
DATASET_NAME=24_game_100000_direct
DATASET_NAME_OR_PATH=$DATASET_DIR/$DATASET_NAME

OUTPUT_DIR=$PROJECT_HOME/results/$MODEL_NAME-$MAX_TOKENS

python evaluate.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --max_tokens $MAX_TOKENS \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --input_path $DATASET_NAME_OR_PATH \
    --output_path $OUTPUT_DIR