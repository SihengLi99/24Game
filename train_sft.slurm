#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety
#SBATCH --job-name=SFT
#SBATCH --output=log/job_%j.out

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
export WANDB_PROJECT=SFT_24Game

PROJECT_HOME=/mnt/petrelfs/lisiheng/24Game
OUTPUT_HOME=/mnt/lustrenew/mllm_safety-shared/lisiheng

MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen
MODEL_NAME=Qwen2.5-3B

MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

DATASET_DIR=./data
DATASET_NAME=24_game_sft_0.01
DATASET_CONFIG=none
DATASET_NAME_OR_PATH=$DATASET_DIR/$DATASET_NAME
PROCESS_REASONING_CONVERSATION=false

PER_DEVICE_TRAIN_BATCH_SIZE=16
MAX_SEQ_LENGTH=32768
LEARNING_RATE=5e-5
TOTAL_BATCH_SIZE=$(expr $PER_DEVICE_TRAIN_BATCH_SIZE \* $WORLD_SIZE)

OUTPUT_DIR=$OUTPUT_HOME/checkpoints/$MODEL_NAME-$DATASET_NAME-$MAX_SEQ_LENGTH-$LEARNING_RATE-$TOTAL_BATCH_SIZE

srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --nodelist=$TRAIN_NODES accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    sft.py \
    --config configs/sft.yaml \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --dataset_name=$DATASET_NAME_OR_PATH \
    --dataset_config=$DATASET_CONFIG \
    --output_dir=$OUTPUT_DIR \
    --learning_rate=$LEARNING_RATE \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --process_reasoning_conversation=$PROCESS_REASONING_CONVERSATION \
    --fsdp_save_full_state_dict=true 