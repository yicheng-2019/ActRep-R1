#!/bin/bash
# ============================================================
# ActRep-R1: Evaluation
# Pipeline: CoT Data -> SFT -> GRPO -> [Eval]
#                                       ^^^^
# Usage:
#   1. Edit the paths below
#   2. bash scripts/test.sh
# ============================================================

export PYTHONPATH=".:$PYTHONPATH"
export DECORD_EOF_RETRY_MAX=40960

# ============================================================
# Paths (EDIT BEFORE RUNNING)
# ============================================================

# Trained model checkpoint (SFT or RL)
MODEL_PATH=${MODEL_PATH:-./outputs/rl_qwen3vl_8b/checkpoint-best}

# Test data and videos
DATA_PATH=${DATA_PATH:-/path/to/test_data.jsonl}
VIDEO_DIR=${VIDEO_DIR:-/path/to/videos}

# Split: test / train / val
SPLIT=${SPLIT:-test}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/eval}
mkdir -p $OUTPUT_DIR

# ============================================================
# Run
# ============================================================
python eval.py \
    --model_path ${MODEL_PATH} \
    --test_data $DATA_PATH \
    --video_folder $VIDEO_DIR \
    --output_dir ${OUTPUT_DIR} \
    --split $SPLIT \
    --max_new_tokens 1024 \
    --flash_attn \
    --verbose
