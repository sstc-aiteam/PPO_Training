#!/bin/bash
set -e

export PYTHONWARNINGS="ignore:Trainer\.tokenizer is now deprecated:UserWarning"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORK_DIR="/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory"
PARENT_DIR="/home/itrib30156/llm_vision/"
DATA_DIR="/home/itrib30156/llm_vision/LLaMA-Factory/data/ppo_data"
RUN_NAME="qwen_ppo_lora_training_$(date +%Y%m%d_%H%M%S)"

cd $WORK_DIR

# Base model path
BASE_MODEL="/home/itrib30156/llm_vision/qwen3b"
ADAPTER_PATH="$PARENT_DIR/ppo_model/checkpoint-650"

echo "🚀 Starting PPO training with proper data format..."
echo "📂 Base model: ${BASE_MODEL}"
echo "🔧 Adapter path: ${ADAPTER_PATH}"
echo "📊 Data directory: ${DATA_DIR}"

# Verify data files exist
if [ ! -f "${DATA_DIR}/ppo_data_train.json" ]; then
    echo "❌ Training data file not found: ${DATA_DIR}/ppo_data_train.json"
    echo "Please run the data conversion script first!"
    exit 1
fi

if [ ! -f "${DATA_DIR}/dataset_info.json" ]; then
    echo "❌ Dataset info file not found: ${DATA_DIR}/dataset_info.json"
    echo "Please run the data conversion script first!"
    exit 1
fi

echo "✅ Data files verified"

# PPO training with proper configuration
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/train.py \
    --stage ppo \
    --model_name_or_path ${BASE_MODEL} \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --reward_model $PARENT_DIR/rm_model/checkpoint-600 \
    --ref_model /home/itrib30156/llm_vision/qwen3b \
    --trust_remote_code True \
    --do_train \
    --dataset multimodal_ppo_data \
    --dataset_dir ${DATA_DIR} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --output_dir $PARENT_DIR/ppo_training_continued_fixed \
    --logging_dir $PARENT_DIR/logs/ppo_training_continued_fixed \
    --run_name ${RUN_NAME} \
    --report_to tensorboard \
    --cache_dir ./cache \
    --plot_loss \
    --bf16 true \
    --weight_decay 0.01 \
    --temperature 0.7 \
    --max_length 512 \
    --max_new_tokens 128 \
    --do_sample \
    --seed 42 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --ddp_find_unused_parameters false

echo "✅ PPO training completed!"
echo "📁 Model saved in: ${PARENT_DIR}/ppo_training_continued_fixed"
echo "📊 View training progress:"
echo "tensorboard --logdir=${PARENT_DIR}/logs/ppo_training_continued_fixed"