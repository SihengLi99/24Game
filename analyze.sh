#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=eval-passk
#SBATCH --output=logs/%x-%j.out

# Number of GPUs on this single node
GPUS_PER_NODE=8

echo "Running on node: $(hostname)"
echo "GPUs per node: $GPUS_PER_NODE"

# Build NO_PROXY list for local node
NO_PROXY_LIST=$(hostname)
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

PROJECT_HOME=/mnt/petrelfs/lisiheng/24Game

MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/lisiheng/checkpoints
MODEL_NAME=Qwen2.5-1.5B-24_game_100000_direct_sft_0.2_0.2_0.01-32768-5e-5-128-24_game_100000_direct-2500-format_correctness-512-4096-8-0.6-32-0.05-5e-6-128-GRPO

# MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen
# MODEL_NAME=QwQ-32B
# MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/deepseek-ai
# MODEL_NAME=DeepSeek-R1-Distill-Qwen-14B
MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

TEMPERATURE=0.6
TOP_K=32
MAX_TOKENS=4096
MAX_THINK_TOKENS=4096
NUM_SAMPLES=16

DATASET_DIR=./data
DATASET_NAME=24_game_100000_direct
INPUT_PATH=$DATASET_DIR/$DATASET_NAME

OUTPUT_DIR=$PROJECT_HOME/results/$MODEL_NAME-$MAX_TOKENS-$MAX_THINK_TOKENS
mkdir -p $OUTPUT_DIR

# Launch on single node, multi-GPU via accelerate
accelerate launch \
    --num_processes $GPUS_PER_NODE \
    analyze.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --max_tokens $MAX_TOKENS \
    --max_think_tokens $MAX_THINK_TOKENS \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \