#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety
#SBATCH --job-name=GRPO
#SBATCH --output=log/job_%j.out
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-22,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-39

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

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export ACCELERATE_LOG_LEVEL=info
export WANDB_API_KEY=85a3c5af1814c40a13d5d9e64783857cf260b506
export WANDB_PROJECT=24Game_GRPO

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
OUTPUT_HOME=/mnt/lustrenew/mllm_safety-shared/lisiheng

# MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen
# MODEL_NAME=Qwen2.5-1.5B
MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/lisiheng/checkpoints
MODEL_NAME=Qwen2.5-Math-1.5B-24_game_100000_direct_sft_0.2_0.2_0.01-32768-5e-5-128-4096

MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

DATASET_DIR=./data
DATASET_NAME=24_game_100000_direct
DATASET_CONFIG=none
DATASET_NAME_OR_PATH=$DATASET_DIR/$DATASET_NAME

NUM_EXAMPLES=2500
REWARD_FUNCS=format_correctness
MAX_PROMPT_LENGTH=512
MAX_COMPLETION_LENGTH=4096
NUM_GENERATIONS=8
TEMPERATURE=0.6
TOP_K=32
BETA=0.05

# if using vllm
USE_VLLM=true
if [[ "$USE_VLLM" == "true" ]]; then
    # Reserve last node to serve via vllm
    TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
    VLLM_NODE=${NODELIST[-1]}
    WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
    NUM_NODES=$((NUM_NODES - 1))
    VLLM_SERVER_HOST=$VLLM_NODE
    VLLM_SERVER_PORT=27800
    srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE \
         trl vllm-serve \
         --model $MODEL_NAME_OR_PATH \
         --tensor_parallel_size 4 \
         --port $VLLM_SERVER_PORT &
else
    # Keep WORLD_SIZE and NUM_NODES unchanged
    # Provide default values for vLLM server variables
    VLLM_SERVER_HOST="localhost"
    VLLM_SERVER_PORT=27800
    echo "Starting evaluation on all $NUM_NODES nodes (no vLLM server)."
    echo "Default vLLM host: $VLLM_SERVER_HOST, port: $VLLM_SERVER_PORT"
fi

PER_DEVICE_TRAIN_BATCH_SIZE=16
LEARNING_RATE=5e-6
TOTAL_BATCH_SIZE=$(expr $PER_DEVICE_TRAIN_BATCH_SIZE \* $WORLD_SIZE)

OUTPUT_DIR=$OUTPUT_HOME/checkpoints/$MODEL_NAME-$DATASET_NAME-$NUM_EXAMPLES-$REWARD_FUNCS-$MAX_PROMPT_LENGTH-$MAX_COMPLETION_LENGTH-$NUM_GENERATIONS-$TEMPERATURE-$TOP_K-$BETA-$LEARNING_RATE-$TOTAL_BATCH_SIZE-GRPO

srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --nodelist=$TRAIN_NODES accelerate launch \
    --config_file configs/accelerate_ds.yaml \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    grpo.py \
    --config configs/grpo.yaml \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --dataset_name=$DATASET_NAME_OR_PATH \
    --dataset_config=$DATASET_CONFIG \
    --output_dir=$OUTPUT_DIR \
    --fsdp_save_full_state_dict=true \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --num_examples=$NUM_EXAMPLES \
    --reward_funcs=$REWARD_FUNCS \
    --max_prompt_length=$MAX_PROMPT_LENGTH \
    --max_completion_length=$MAX_COMPLETION_LENGTH \
    --num_generations=$NUM_GENERATIONS \
    --temperature=$TEMPERATURE \
    --top_k=$TOP_K \
    --beta=$BETA \
    --use_vllm=$USE_VLLM \
    --vllm_server_host=$VLLM_SERVER_HOST \
    --vllm_server_port=$VLLM_SERVER_PORT \
    --vllm_server_timeout=360.0 