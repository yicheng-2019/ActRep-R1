"""
Unified RL Trainer for RepCount

Replaces: qwen3_trainer.py (QwenGRPOTrainer) + timer1_trainer.py (TimeR1_Trainer)

Supports:
  - All Qwen-VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL Dense/MoE)
  - Multiple RL algorithms: GRPO, GRPO-Clip (PPO-style), DAPO

Algorithm Variants (--rl_algorithm):
  "grpo":      Standard GRPO (DeepSeekMath)
  "grpo_clip": GRPO with PPO-style symmetric clipping
  "dapo":      DAPO (ByteDance) — asymmetric clip + dynamic sampling + token-level loss

DAPO Reference: https://arxiv.org/abs/2503.14476
  1. Clip-Higher: ε_high > ε_low for exploration (default: 0.2/0.28)
  2. Dynamic Sampling: skip groups where all rewards identical
  3. Token-Level PG Loss: normalize across all tokens, not per-sequence
  4. Overlong Reward Shaping: soft penalty for truncated outputs
  5. No KL penalty (β=0 enforced)
"""

import json
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from src.model_utils import (
    detect_model_info, load_model, load_processor, freeze_vit, is_qwen3,
)
from src.video_utils import process_video, run_processor
from .prompts_config import build_repcount_messages

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class QwenGRPOTrainer(Trainer):
    """
    RL Trainer supporting GRPO / DAPO for video understanding.
    
    Supports all Qwen-VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL Dense/MoE).
    Model-type-specific logic is handled by src.model_utils and src.video_utils.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset=None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes=None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # ============================================================
        # 1. Config defaults
        # ============================================================
        if args is None:
            name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{name.split('/')[-1]}-GRPO")

        # RL algorithm config
        self.rl_algorithm = getattr(args, 'rl_algorithm', 'grpo_clip')
        
        # Epsilon config (asymmetric for DAPO)
        epsilon = getattr(args, 'epsilon', 0.2)
        epsilon_high = getattr(args, 'epsilon_high', None)
        
        if self.rl_algorithm == 'dapo':
            # DAPO defaults: ε_low=0.2, ε_high=0.28
            self.epsilon_low = epsilon
            self.epsilon_high = epsilon_high if epsilon_high is not None else 0.28
            # DAPO enforces no KL
            args.beta = 0.0
            print(f"[DAPO] ε_low={self.epsilon_low}, ε_high={self.epsilon_high}, β=0")
        else:
            self.epsilon_low = epsilon
            self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        
        # Overlong penalty config (DAPO feature)
        self.overlong_buffer_len = getattr(args, 'overlong_buffer_len', 0)
        self.overlong_penalty_factor = getattr(args, 'overlong_penalty_factor', 1.0)

        # ============================================================
        # 2. Model loading (unified via model_utils)
        # ============================================================
        if isinstance(model, str):
            model_id = model
            self.model_type, self.model_size = detect_model_info(model_id)
            print(f"[Trainer] model_type={self.model_type}, model_size={self.model_size}")
            print(f"[Trainer] rl_algorithm={self.rl_algorithm}")
            
            model = load_model(
                model_id, self.model_type,
                attn_implementation=attn_implementation,
                slide_window=getattr(args, 'slide_window', False),
                max_window_layers=getattr(args, 'max_window_layers', 4),
                sliding_window_length=getattr(args, 'sliding_window_length', 32768),
            )
        else:
            model_id = model.config._name_or_path
            self.model_type, self.model_size = detect_model_info(model_id)

        # Gradient checkpointing
        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def _make_grads(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(_make_grads)

        # LoRA
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # ViT freezing
        fix_vit = getattr(args, 'fix_vit', False)
        if fix_vit and peft_config is None:
            freeze_vit(model)

        # ============================================================
        # 3. Reference model
        # ============================================================
        self.beta = args.beta
        # GRPO vs clip mode (backward compat with use_grpo flag)
        self.use_grpo = getattr(args, 'use_grpo', False)
        
        if self.rl_algorithm == 'grpo':
            self.use_grpo = True  # force GRPO mode
        elif self.rl_algorithm == 'grpo_clip':
            self.use_grpo = False  # force clip mode
        # DAPO uses clip mode internally (with asymmetric epsilon)
        
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = load_model(
                model_id, self.model_type,
                attn_implementation=attn_implementation,
                slide_window=getattr(args, 'slide_window', False),
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # ============================================================
        # 4. Processor
        # ============================================================
        self.video_total_pixels = max_pixels if max_pixels != 12845056 else None
        if processing_class is None:
            processing_class = load_processor(
                model_id, max_pixels=max_pixels, min_pixels=min_pixels,
                padding_side="left",
            )

        pad_token_id = processing_class.tokenizer.pad_token_id

        # ============================================================
        # 5. Reward functions
        # ============================================================
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, rf in enumerate(reward_funcs):
            if isinstance(rf, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    rf, num_labels=1,
                )
        self.reward_funcs = reward_funcs

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        for i, (rpc, rf) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(rf, PreTrainedModel) and rpc is None:
                rpc = AutoTokenizer.from_pretrained(rf.config._name_or_path)
                if rpc.pad_token_id is None:
                    rpc.pad_token = rpc.eos_token
                rf.config.pad_token_id = rpc.pad_token_id
                reward_processing_classes[i] = rpc
        self.reward_processing_classes = reward_processing_classes

        # ============================================================
        # 6. Generation config
        # ============================================================
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        self.prompt_type = getattr(args, 'prompt_type', 'v1')
        print(f"[Trainer] beta={self.beta}, temperature={self.temperature}")
        print(f"[Trainer] rl_algorithm={self.rl_algorithm}, use_grpo={self.use_grpo}")

        # Suppress FLOPs warning
        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        def data_collator(features):
            return features

        # ============================================================
        # 7. Parent init
        # ============================================================
        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=processing_class, callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        # Prepare reward models
        for i, rf in enumerate(self.reward_funcs):
            if isinstance(rf, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    rf, evaluation_mode=True
                )

    # ============================================================
    # Required overrides
    # ============================================================

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs):
        return inputs

    # ============================================================
    # Per-token log probabilities + entropy
    # ============================================================

    def _get_per_token_logps(self, model, input_ids, attention_mask,
                             pixel_values_videos, video_grid_thw):
        logits = model(
            input_ids, attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        ).logits

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        per_token_logps = []
        per_token_entropy = []
        for logits_row, ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_lp = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_lp)

            probs = torch.exp(log_probs)
            ent = -torch.sum(probs * log_probs, dim=-1)
            per_token_entropy.append(ent)

        return torch.stack(per_token_logps), torch.stack(per_token_entropy)

    # ============================================================
    # Prompt Construction
    # ============================================================

    def _make_prompt(self, example):
        action = example.get("action", "repetitive action")
        return build_repcount_messages(action, example["video_path"], self.model_size)

    # ============================================================
    # Loss Computation Methods
    # ============================================================

    def _compute_grpo_loss(self, per_token_logps, advantages, completion_mask,
                           per_token_kl=None):
        """Standard GRPO loss (DeepSeekMath)."""
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = -(ratio * advantages.unsqueeze(1))

        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Sequence-level: mean over tokens per sequence, then mean over sequences
        loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1)
        ).mean()
        return loss

    def _compute_clip_loss(self, per_token_logps, advantages, completion_mask,
                           per_token_kl=None):
        """GRPO with PPO-style symmetric clipping."""
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        clamped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss = -torch.min(
            ratio * advantages.unsqueeze(1),
            clamped * advantages.unsqueeze(1),
        )

        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Sequence-level aggregation (original behavior)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss

    def _compute_dapo_loss(self, per_token_logps, advantages, completion_mask,
                           rewards=None):
        """
        DAPO loss (ByteDance, arxiv 2503.14476).
        
        Key differences from standard GRPO:
        1. Asymmetric clipping (ε_high > ε_low)
        2. Dynamic sampling: skip groups where all rewards are identical
        3. Token-level loss: normalize across ALL tokens, not per-sequence
        4. No KL penalty (enforced in __init__)
        
        Reference: verl/recipe/dapo — pg_losses = torch.maximum(pg_losses1, pg_losses2)
        """
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        clamped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # DAPO uses maximum (equivalent to -min for negated losses)
        pg_losses1 = -advantages.unsqueeze(1) * ratio
        pg_losses2 = -advantages.unsqueeze(1) * clamped
        per_token_loss = torch.maximum(pg_losses1, pg_losses2)

        # ---- Dynamic Sampling: mask out all-same-reward groups ----
        if rewards is not None:
            G = self.num_generations
            grouped_rewards = rewards.view(-1, G)  # (B, G)
            group_std = grouped_rewards.std(dim=1)  # (B,)
            # Groups with zero std → no learning signal
            valid_groups = (group_std > 1e-6).float()  # (B,)
            # Expand to per-sample mask
            valid_mask = valid_groups.repeat_interleave(G)  # (B*G,)

            n_filtered = (valid_groups < 0.5).sum().item()
            if n_filtered > 0:
                self._metrics["dapo/filtered_groups"].append(n_filtered)

            # Apply group validity mask
            completion_mask = completion_mask * valid_mask.unsqueeze(1)

        # ---- Token-level loss: sum / total_valid_tokens ----
        total_tokens = completion_mask.sum()
        if total_tokens > 0:
            loss = (per_token_loss * completion_mask).sum() / total_tokens
        else:
            # All groups filtered → zero loss
            loss = per_token_loss.sum() * 0.0

        return loss

    # ============================================================
    # Overlong Reward Shaping (DAPO feature)
    # ============================================================

    def _apply_overlong_penalty(self, rewards, completion_mask):
        """
        DAPO overlong reward shaping: soft linear penalty for sequences
        approaching max_completion_length.
        
        Penalty increases linearly in the last `overlong_buffer_len` tokens.
        """
        if self.overlong_buffer_len <= 0:
            return rewards

        seq_lengths = completion_mask.sum(dim=1)  # actual completion lengths
        expected_len = self.max_completion_length - self.overlong_buffer_len
        exceed_len = seq_lengths - expected_len
        
        penalty = torch.clamp(
            -exceed_len / self.overlong_buffer_len * self.overlong_penalty_factor,
            max=0.0,
        )
        
        return rewards + penalty

    # ============================================================
    # Completion Logging (训练时保存生成结果，用于调试和分析)
    # ============================================================

    def _log_completions(self, inputs, completions, rewards, rewards_per_func):
        """
        保存每个 step 的生成结果到 LOG_PATH 文件。
        仅在 rank 0 且 DEBUG_MODE=true 时执行。
        
        输出格式（每个 step）:
          ── Step XXX | video_id | action | GT=N ──
          [Gen 0] reward=0.95 | pred=7
            <thinking>模型的推理过程...</thinking>
            <answer>7</answer>
          [Gen 1] reward=0.61 | pred=8
            ...
        """
        if not self.accelerator.is_main_process:
            return
        if os.getenv("DEBUG_MODE") != "true":
            return
        
        log_path = os.getenv("LOG_PATH")
        if not log_path:
            return

        from src.reward.count_reward import parse_count_output
        from datetime import datetime

        G = self.num_generations
        step = self.state.global_step if hasattr(self, 'state') else 0
        ts = datetime.now().strftime("%m-%d %H:%M:%S")

        with open(log_path, "a", encoding="utf-8") as f:
            for batch_idx, example in enumerate(inputs):
                video_id = example.get("video_id", "?")
                action = example.get("action", "?")
                gt = example.get("gt_count", example.get("solution", "?"))

                f.write(f"\n{'═'*70}\n")
                f.write(f"Step {step} | {ts} | {video_id} | {action} | GT={gt}\n")
                f.write(f"{'─'*70}\n")

                for gen_idx in range(G):
                    flat_idx = batch_idx * G + gen_idx
                    if flat_idx >= len(completions):
                        break

                    comp = completions[flat_idx]
                    # 如果是 conversational 格式，提取文本
                    if isinstance(comp, list) and len(comp) > 0:
                        comp = comp[0].get("content", str(comp))

                    r_total = rewards[flat_idx].item()
                    pred = parse_count_output(comp)

                    # 每个 reward 函数的分项
                    r_parts = []
                    for func_idx, rf in enumerate(self.reward_funcs):
                        name = rf.__name__ if hasattr(rf, '__name__') else str(func_idx)
                        val = rewards_per_func[flat_idx, func_idx].item()
                        r_parts.append(f"{name}={val:.3f}")

                    f.write(f"[Gen {gen_idx}] reward={r_total:.4f} ({', '.join(r_parts)}) | pred={pred}\n")
                    
                    # 生成文本（截断过长的，避免 log 文件爆炸）
                    comp_display = comp[:800] + "..." if len(comp) > 800 else comp
                    f.write(f"  {comp_display}\n")

                f.write(f"{'═'*70}\n")
            f.flush()

    # ============================================================
    # Main compute_loss
    # ============================================================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("QwenGRPOTrainer does not support return_outputs")

        device = self.accelerator.device

        # 1. Build prompts
        prompts = [self._make_prompt(ex) for ex in inputs]
        prompts_text = [
            self.processing_class.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

        # 2. Process video (unified via video_utils)
        video_info = process_video(self.model_type, inputs[0]["video_path"],
                                   total_pixels=self.video_total_pixels)

        # 3. Tokenize (unified via video_utils.run_processor)
        prompt_inputs = run_processor(
            self.processing_class, [prompts_text[0]], video_info,
            self.model_type, padding=True, padding_side="left",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values_videos = prompt_inputs["pixel_values_videos"]
        video_grid_thw = prompt_inputs["video_grid_thw"]

        # 4. Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            prompt_completion_ids = unwrapped.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # 5. EOS masking
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # 6. Repeat visual inputs for num_generations
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values_videos = prompt_inputs["pixel_values_videos"].repeat(
            self.num_generations, 1
        )
        video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(
            self.num_generations, dim=0
        )

        # 7. Compute log probabilities
        per_token_logps, per_token_entropy = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask,
            pixel_values_videos, video_grid_thw,
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        entropy_completion = per_token_entropy[:, prompt_length - 1:]

        # 8. Reference model KL (if needed)
        per_token_kl = None
        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_logps, _ = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask,
                        pixel_values_videos, video_grid_thw,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_logps, _ = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask,
                            pixel_values_videos, video_grid_thw,
                        )
                ref_logps = ref_logps[:, prompt_length - 1:]
                per_token_kl = (
                    torch.exp(ref_logps - per_token_logps)
                    - (ref_logps - per_token_logps) - 1
                )

        # 9. Decode completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": c}] for c in completions
            ]

        # 10. Compute rewards
        prompts_expanded = [p for p in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(prompts_expanded), len(self.reward_funcs), device=device
        )

        for i, (rf, rpc) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(rf, PreTrainedModel):
                if is_conversational(inputs[0]):
                    msgs = [{"messages": p + c} for p, c in zip(prompts_expanded, completions)]
                    texts = [apply_chat_template(x, rpc)["text"] for x in msgs]
                else:
                    texts = [p + c for p, c in zip(prompts_expanded, completions)]
                reward_inputs = rpc(
                    texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = rf(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {
                    k: [] for k in inputs[0].keys()
                    if k not in ["prompt", "completion"]
                }
                for k in reward_kwargs:
                    for ex in inputs:
                        reward_kwargs[k].extend([ex[k]] * self.num_generations)
                output = rf(
                    prompts=prompts_expanded, completions=completions,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(
                    output, dtype=torch.float32, device=device
                )

        rewards = rewards_per_func.sum(dim=1)

        # 10.5 Log completions (训练时保存生成结果，仅 rank0 + DEBUG_MODE)
        self._log_completions(inputs, completions, rewards, rewards_per_func)

        # 11. Overlong reward shaping (DAPO)
        if self.rl_algorithm == 'dapo':
            rewards = self._apply_overlong_penalty(rewards, completion_mask)

        # 12. Compute advantages (group-normalized)
        G = self.num_generations
        mean_r = rewards.view(-1, G).mean(dim=1).repeat_interleave(G)
        std_r = rewards.view(-1, G).std(dim=1).repeat_interleave(G)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # 13. Compute loss based on algorithm
        if self.rl_algorithm == 'dapo':
            loss = self._compute_dapo_loss(
                per_token_logps, advantages, completion_mask, rewards=rewards
            )
        elif self.use_grpo or self.rl_algorithm == 'grpo':
            loss = self._compute_grpo_loss(
                per_token_logps, advantages, completion_mask, per_token_kl
            )
        else:  # grpo_clip
            loss = self._compute_clip_loss(
                per_token_logps, advantages, completion_mask, per_token_kl
            )

        # ============================================================
        # 14. Metrics
        # ============================================================
        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(
                completion_mask.sum(1)
            ).float().mean().item()
        )

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, rf in enumerate(self.reward_funcs):
            name = (rf.config._name_or_path.split("/")[-1]
                    if isinstance(rf, PreTrainedModel) else rf.__name__)
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_r).mean().item()
        )

        if self.beta != 0.0 and per_token_kl is not None:
            mean_kl = (
                (per_token_kl * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        # Entropy
        comp_lens = completion_mask.sum(dim=1).clamp(min=1)
        mean_ent = (entropy_completion * completion_mask).sum(dim=1) / comp_lens
        self._metrics["generation_entropy"].append(
            self.accelerator.gather_for_metrics(mean_ent.mean()).mean().item()
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    # ============================================================
    # Logging & Model Card
    # ============================================================

    def log(self, logs, start_time=None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items() if v}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(self, model_name=None, dataset_name=None, tags=None):
        if not self.is_world_process_zero():
            return
        base = getattr(self.model.config, "_name_or_path", None)
        if base and os.path.isdir(base):
            base = None
        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        citation = textwrap.dedent("""\
            @article{zhihong2024deepseekmath,
                title={{DeepSeekMath}}, author={Zhihong Shao et al.},
                year=2024, eprint={arXiv:2402.03300}}""")
        card = generate_model_card(
            base_model=base, model_name=model_name,
            hub_model_id=self.hub_model_id, dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO", trainer_citation=citation,
            paper_title="DeepSeekMath", paper_id="2402.03300",
        )
        card.save(os.path.join(self.args.output_dir, "README.md"))

