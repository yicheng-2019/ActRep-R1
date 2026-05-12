"""
SFT Training Entry Point for RepCount

Pipeline:  CoT Data → [SFT] → GRPO → Eval
                       ^^^^^

Usage:
    bash scripts/finetune/train_sft_rep.sh
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from src.trainer.sft_trainer import QwenSFTTrainer
from src.dataset.repcount import load_json_dataset_repcount


# ============================================================
# Config — 与 main.py 的 MY_GRPOConfig / GRPOScriptArguments 对应
# ============================================================

@dataclass
class SFTConfig(TrainingArguments):
    """
    SFT 训练配置
    保留与 QwenSFTTrainer / QwenGRPOTrainer 共用的参数
    """
    fix_vit: bool = field(
        default=True,
        metadata={"help": "Whether to freeze ViT (keep merger trainable)"},
    )
    slide_window: bool = field(
        default=False,
        metadata={"help": "Whether to use sliding window (Qwen2.5-VL)"},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Max total sequence length (prompt + response)"},
    )


@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    SFT 脚本参数
    与 main.py 的 GRPOScriptArguments 对应
    """
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    train_data_path: str = field(
        default="",
        metadata={"help": "Path to CoT JSONL file (gen_cot_dataset.py output)"},
    )
    video_folder: str = field(
        default="",
        metadata={"help": "Path to video directory"},
    )
    is_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Sort by count (easy → hard)"},
    )
    is_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping"},
    )


# ============================================================
# Callbacks — 与 main.py 一致
# ============================================================

class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return
            ckpt_dir = os.path.join(args.output_dir, f"epoch-{int(state.epoch)}")
            print(
                f"\n{'='*20} Saving epoch {int(state.epoch)} → {ckpt_dir} {'='*20}\n"
            )
            trainer.save_model(ckpt_dir)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, num_epochs_to_train=1):
        super().__init__()
        self.num_epochs_to_train = num_epochs_to_train
        print(f"[SFT] Will stop after {self.num_epochs_to_train} epoch(s).")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.num_epochs_to_train:
            print(f"Epoch {state.epoch:.0f} done. Stopping.")
            control.should_training_stop = True


# ============================================================
# Seed
# ============================================================

def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# ============================================================
# Main — 参考 main.py 结构
# ============================================================

def main(script_args, training_args, model_args):

    set_global_seed(42)

    # ---- 1. Load dataset (复用 GRPO 的数据加载) ----
    dataset = load_json_dataset_repcount(
        script_args.train_data_path,
        script_args.video_folder,
        is_curriculum_learning=script_args.is_curriculum_learning,
    )

    # SFT 需要 cot 字段，过滤空 cot
    valid_indices = [i for i in range(len(dataset)) if dataset[i].get("cot", "")]
    if len(valid_indices) < len(dataset):
        print(f"[SFT] Filtered {len(dataset) - len(valid_indices)} samples without CoT")
        dataset = dataset.select(valid_indices)

    print(f"[SFT] Dataset size: {len(dataset)}")

    # ---- 2. Callbacks ----
    callbacks_list = [SaveEpochEndCallback()]
    if script_args.is_early_stopping:
        callbacks_list.append(StopAfterNEpochsCallback())

    # ---- 3. Initialize trainer ----
    print(f"[SFT] Using: QwenSFTTrainer")
    print(f"[SFT] Model: {model_args.model_name_or_path}")

    trainer = QwenSFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks_list,
    )

    # ---- 4. Train (resume 逻辑与 main.py 一致) ----
    if training_args.resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(
            training_args.resume_from_checkpoint, "trainer_state.json"
        )
        if os.path.exists(trainer_state_path):
            print(f"Loading trainer state from: {trainer_state_path}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            resumed_global_step = trainer_state.get("global_step", 0)

            num_micro_batches_per_epoch_per_gpu = len(trainer.get_train_dataloader())
            max_step = math.ceil(
                trainer.args.num_train_epochs
                * num_micro_batches_per_epoch_per_gpu
                / trainer.args.gradient_accumulation_steps
            )
            trainer.args.max_steps = resumed_global_step + max_step

            if hasattr(trainer, "state") and hasattr(trainer.state, "max_steps"):
                trainer.state.max_steps = max_step

        print(f"Resuming from: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # ---- 5. Save final model ----
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)