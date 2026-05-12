#!/bin/bash
# ============================================================
# ActRep-R1: SFT Training (Stage 1)
# Pipeline: CoT Data -> [SFT] -> GRPO -> Eval
#                        ^^^^^
# Usage:
#   1. Edit the paths below
#   2. bash scripts/finetune/sft_qwen.sh
# ============================================================

export WANDB_PROJECT=ActRep-R1
export EXP_NAME=sft_qwen3vl_8b
export PYTHONPATH=".:$PYTHONPATH"
export DECORD_EOF_RETRY_MAX=40960
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MODE="true"

# ============================================================
# Paths (EDIT BEFORE RUNNING)
# ============================================================
OUTDIR=${OUTDIR:-./outputs/sft_qwen3vl_8b}
export LOG_PATH="$OUTDIR/$EXP_NAME.txt"
mkdir -p $OUTDIR

# Base MLLM checkpoint (HuggingFace ID or local path)
BASE_MODEL_NAME_OR_PATH=${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

# CoT training data (JSONL with video_id / action / gt_count / cot / split)
TRAIN_DATA=${TRAIN_DATA:-/path/to/cot_dataset.jsonl}
TRAIN_VIDEO=${TRAIN_VIDEO:-/path/to/videos}

DEEPSPEED="scripts/zero3.json"

# ============================================================
# Training
# ============================================================
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12380" \
    sft_main.py \
    --deepspeed $DEEPSPEED \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path $TRAIN_DATA \
    --video_folder $TRAIN_VIDEO \
    --dataset_name repcount_sft \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --num_train_epochs 6 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --run_name $EXP_NAME \
    --report_to wandb \
    --logging_dir $OUTDIR \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 4 \
    --is_curriculum_learning false
