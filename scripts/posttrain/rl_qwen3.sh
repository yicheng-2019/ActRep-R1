#!/bin/bash
# ============================================================
# ActRep-R1: GRPO Reinforcement Learning Training (Stage 2)
# Pipeline: CoT Data -> SFT -> [GRPO] -> Eval
#                              ^^^^^^
# Usage:
#   1. Edit the paths below
#   2. bash scripts/posttrain/rl_qwen3.sh
# ============================================================

export WANDB_PROJECT=ActRep-R1
export EXP_NAME=rl_qwen3vl_8b
export PYTHONPATH=".:$PYTHONPATH"
export DECORD_EOF_RETRY_MAX=40960
export DEBUG_MODE="true"

OUTDIR=${OUTDIR:-./outputs/rl_qwen3vl_8b}
export LOG_PATH="$OUTDIR/$EXP_NAME.txt"
mkdir -p $OUTDIR

# Initialize from the SFT checkpoint
BASE_MODEL_NAME_OR_PATH=${BASE_MODEL:-./outputs/sft_qwen3vl_8b/checkpoint-best}

# RL training data (CSV: video_id, split, action, count) — train split only
TRAIN_DATA_PATH=${TRAIN_DATA:-/path/to/train.csv}
TRAIN_VIDEO=${TRAIN_VIDEO:-/path/to/videos}

DEEPSPEED="scripts/zero3_offload_false.json"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
    main.py \
    --deepspeed $DEEPSPEED \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --video_folder $TRAIN_VIDEO \
    --dataset_name repcount_rl \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 2 \
    --run_name $EXP_NAME \
    --report_to wandb \
    --reward_funcs count_hybrid count_format_8b \
    --temperature 1.0 \
    --rl_algorithm grpo \
    --prompt_type v1 \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 50 \
    --save_only_model true
