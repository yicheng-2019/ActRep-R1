"""
GRPO / DAPO Training Entry Point — RepCount

Supports:
  - rl_algorithm: "grpo" | "grpo_clip" | "dapo"
  - All Qwen-VL models via unified QwenGRPOTrainer

Usage:
  bash scripts/posttrain/train_rl_rep_qwen3_8B_A100.sh
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from transformers import TrainerCallback
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from src.trainer import QwenGRPOTrainer
from src.dataset.repcount import load_json_dataset_repcount
from src.reward.count_reward import REPCOUNT_REWARD_FUNCS, REPCOUNT_REWARD_FUNCS_8B

torch.serialization.add_safe_globals([ZeroStageEnum])
torch.serialization.add_safe_globals([LossScaler])


# ============================================================
# Reward Registry
# ============================================================

reward_funcs_registry = {
    **REPCOUNT_REWARD_FUNCS,
    **REPCOUNT_REWARD_FUNCS_8B,
}


# ============================================================
# Config
# ============================================================

@dataclass
class MY_GRPOConfig(GRPOConfig):
    # ---- 模型相关 ----
    fix_vit: bool = field(default=False)                # 是否冻结 ViT 视觉编码器（保留 merger 可训练）
    slide_window: bool = field(default=False)            # 是否启用滑动窗口注意力（Qwen2.5-VL 特有）
    max_window_layers: int = field(default=2)            # 滑动窗口覆盖的底部 layer 数
    sliding_window_length: int = field(default=4096)     # 滑动窗口长度（token 数）
    prompt_type: str = field(default="v1")               # prompt 模板版本

    # ---- RL 算法选择 ----
    use_grpo: bool = field(default=False)                # 旧参数，True=标准GRPO（无clip），建议用 rl_algorithm 代替
    rl_algorithm: str = field(
        default="grpo_clip",
        metadata={"help":
            "RL 算法选择: "
            "'grpo' = 标准 GRPO（DeepSeekMath，不裁剪 ratio）; "
            "'grpo_clip' = GRPO + PPO 风格的 ratio 裁剪（默认，更稳定）; "
            "'dapo' = DAPO（ByteDance，非对称裁剪+动态采样+token级loss，无KL）"
        },
    )

    # ---- DAPO 专用参数 ----
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={"help":
            "DAPO 非对称裁剪的上界 ε_high（默认 0.28）。"
            "标准 GRPO/grpo_clip 的 ε_low 和 ε_high 相同，由 --epsilon 控制。"
            "DAPO 中 ε_high > ε_low 鼓励探索。"
        },
    )
    overlong_buffer_len: int = field(
        default=0,
        metadata={"help":
            "DAPO 超长惩罚缓冲区长度（0=不启用）。"
            "在 max_completion_length 前的最后 N 个 token 施加线性惩罚，"
            "防止模型生成被截断的无效输出。推荐值: 64"
        },
    )
    overlong_penalty_factor: float = field(
        default=1.0,
        metadata={"help": "DAPO 超长惩罚系数，越大惩罚越重"},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    # ---- 奖励函数 ----
    reward_funcs: list[str] = field(
        default_factory=lambda: ["count", "count_format"],
        metadata={"help":
            "奖励函数列表，可选: "
            "'count' = 归一化高斯（σ随GT缩放，推荐）; "
            "'count_hybrid' = 高斯 + OBO/精确匹配 bonus; "
            "'count_linear' = CrowdVLM-R1 线性归一化; "
            "'count_fixed' = 固定σ=3.0 高斯（消融对照）; "
            "'count_format' / 'count_format_8b' = 格式检查（<think>/<thinking>标签）"
        },
    )

    # ---- 视频像素控制 ----
    max_pixels: Optional[int] = field(default=12845056)  # 视频最大像素数（影响帧采样数量和显存占用）
    min_pixels: Optional[int] = field(default=3136)       # 视频最小像素数

    # ---- 数据路径 ----
    train_data_path: str = field(default="")              # 训练数据文件路径（.csv 或 .jsonl）
    video_folder: str = field(default="")                  # 视频文件目录

    # ---- 训练策略 ----
    is_curriculum_learning: bool = field(default=False)    # 课程学习：按计数从少到多排序（简单→困难）
    is_early_stopping: bool = field(default=False)         # 是否在 1 个 epoch 后停止训练


# ============================================================
# Callbacks
# ============================================================

class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer:
                ckpt = os.path.join(args.output_dir, f"epoch-{int(state.epoch)}")
                print(f"\n{'='*20} Saving epoch {int(state.epoch)} → {ckpt} {'='*20}\n")
                trainer.save_model(ckpt)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, n=1):
        super().__init__()
        self.n = n

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.n:
            control.should_training_stop = True


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main(script_args, training_args, model_args):
    set_global_seed(42)

    print(f"[Main] model_name_or_path: {model_args.model_name_or_path}")
    print(f"[Main] train_data_path: {script_args.train_data_path}")
    print(f"[Main] video_folder: {script_args.video_folder}")
    print(f"[Main] output_dir: {training_args.output_dir}")

    reward_funcs = [reward_funcs_registry[f] for f in script_args.reward_funcs]

    dataset = load_json_dataset_repcount(
        script_args.train_data_path,
        script_args.video_folder,
        is_curriculum_learning=script_args.is_curriculum_learning,
    )

    callbacks_list = []
    if script_args.is_early_stopping:
        callbacks_list.append(StopAfterNEpochsCallback())

    algo = getattr(training_args, 'rl_algorithm', 'grpo_clip')
    print(f"[Main] algo={algo}, reward_funcs={script_args.reward_funcs}")

    trainer = QwenGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks_list,
    )

    if training_args.resume_from_checkpoint is not None:
        state_path = os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                ts = json.load(f)
            resumed = ts.get("global_step", 0)
            n_batches = len(trainer.get_train_dataloader())
            max_step = math.ceil(
                trainer.args.num_train_epochs * n_batches / trainer.args.gradient_accumulation_steps
            )
            trainer.args.max_steps = resumed + max_step
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)