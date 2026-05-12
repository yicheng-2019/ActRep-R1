"""
QwenSFTTrainer — SFT Trainer for Qwen-VL series

Pipeline:  CoT Data → [SFT] → GRPO → Eval
                       ^^^^^

Cross-entropy loss on assistant response tokens only.
Uses shared model_utils / video_utils / prompts_config.

References:
  - Official: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py
"""

import os
from collections import defaultdict
from typing import Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from src.model_utils import detect_model_info, load_model, load_processor, freeze_vit
from src.video_utils import process_video, run_processor
from .prompts_config import build_repcount_messages, get_think_tag

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

IGNORE_INDEX = -100


class QwenSFTTrainer(Trainer):
    """
    SFT Trainer for Qwen-VL series.
    
    Supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL (Dense & MoE).
    Uses shared model_utils / video_utils / prompts_config to ensure
    preprocessing is identical to QwenGRPOTrainer and eval.py.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset=None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        peft_config=None,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # ============================================================
        # 1. Model loading (shared via model_utils)
        # ============================================================
        if isinstance(model, str):
            model_id = model
            self.model_type, self.model_size = detect_model_info(model_id)
            print(f"[SFT] model_type={self.model_type}, model_size={self.model_size}")

            model = load_model(
                model_id, self.model_type,
                attn_implementation=attn_implementation,
                slide_window=getattr(args, 'slide_window', False),
            )
        else:
            model_id = model.config._name_or_path
            self.model_type, self.model_size = detect_model_info(model_id)

        # ============================================================
        # 2. Gradient Checkpointing
        # ============================================================
        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def _make_grads(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(_make_grads)

        # ============================================================
        # 3. LoRA (optional)
        # ============================================================
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # ============================================================
        # 4. ViT Freezing (shared via model_utils)
        # ============================================================
        fix_vit = getattr(args, "fix_vit", False)
        if fix_vit and peft_config is None:
            freeze_vit(model, self.model_type)

        # ============================================================
        # 5. Processor (shared via model_utils)
        # ============================================================
        if processing_class is None:
            processing_class = load_processor(
                model_id, max_pixels=max_pixels, min_pixels=min_pixels,
                padding_side="right",  # SFT uses right padding
            )

        self.max_seq_length = getattr(args, 'max_seq_length', 8192)

        # ============================================================
        # 6. Think tag (for building assistant response)
        # ============================================================
        self.think_open, self.think_close = get_think_tag(self.model_size)

        # ============================================================
        # 7. Data Collator + Parent init
        # ============================================================
        def data_collator(features):
            return features  # Processed in compute_loss

        self._metrics = defaultdict(list)
        model.warnings_issued["estimate_tokens"] = True
        args.remove_unused_columns = False

        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=processing_class, callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

    # ============================================================
    # Prompt Construction (shared via prompts_config)
    # ============================================================

    def _make_prompt_messages(self, example):
        """Build system + user messages (same as GRPO trainer)."""
        action = example.get("action", "repetitive action")
        return build_repcount_messages(action, example["video_path"], self.model_size)

    def _build_messages(self, example):
        """Full conversation: prompt + assistant response (for SFT)."""
        messages = self._make_prompt_messages(example)

        cot_text = example["cot"]
        count = example.get("gt_count", "")
        response = f"{self.think_open}\n{cot_text}\n{self.think_close}\n<answer>\n{count}\n</answer>"

        messages.append({"role": "assistant", "content": response})
        return messages

    # ============================================================
    # Core: compute_loss
    # ============================================================

    def _prepare_inputs(self, inputs):
        return inputs

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """SFT Loss: Cross-entropy on assistant tokens only."""
        if return_outputs:
            raise ValueError("QwenSFTTrainer does not support return_outputs")

        device = self.accelerator.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for example in inputs:
            try:
                loss_i = self._compute_single_loss(model, example, device)
                if loss_i is not None:
                    total_loss = total_loss + loss_i
                    count += 1
            except Exception as e:
                print(f"[SFT] Skipping sample {example.get('video_id', '?')}: {e}")
                continue

        if count == 0:
            print("[SFT] WARNING: All samples in batch failed!")
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = total_loss / count
        self._metrics["sft_loss"].append(
            self.accelerator.gather_for_metrics(loss.detach()).mean().item()
        )
        return loss

    def _compute_single_loss(self, model, example, device):
        """Compute SFT loss for a single example (CE on assistant tokens)."""

        # 1. Build messages
        full_messages = self._build_messages(example)
        prompt_messages = self._make_prompt_messages(example)

        # 2. Apply chat template
        full_text = self.processing_class.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processing_class.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        # 3. Process video (shared via video_utils — same as GRPO + eval)
        video_info = process_video(self.model_type, example["video_path"])

        # 4. Tokenize (shared via video_utils.run_processor)
        full_inputs = run_processor(
            self.processing_class, [full_text], video_info,
            self.model_type, padding=False, padding_side="right",
        )
        prompt_inputs = run_processor(
            self.processing_class, [prompt_text], video_info,
            self.model_type, padding=False, padding_side="right",
        )

        # 5. Move to device
        full_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in full_inputs.items()}

        input_ids = full_inputs["input_ids"]        # (1, seq_len)
        attention_mask = full_inputs["attention_mask"]

        # 6. Truncate
        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_length:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            seq_len = self.max_seq_length

        # 7. Build labels: mask prompt tokens with IGNORE_INDEX
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = IGNORE_INDEX

        # 8. Forward (always pass labels → CE loss computed internally)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=full_inputs.get("pixel_values_videos"),
            video_grid_thw=full_inputs.get("video_grid_thw"),
            labels=labels,
        )
        loss = outputs.loss

        # Metrics
        self._metrics["prompt_length"].append(prompt_len)
        self._metrics["response_length"].append(max(0, seq_len - prompt_len))

        del full_inputs, prompt_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    # ============================================================
    # Logging
    # ============================================================

    def log(self, logs, start_time=None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items() if v}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    # ============================================================
    # Save — with processor (official pattern)
    # ============================================================

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir

        if self.is_deepspeed_enabled:
            torch.cuda.synchronize()
        super().save_model(output_dir, _internal_call=_internal_call)

        # Save processor alongside model
        if self.args.should_save and self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Restore use_cache for inference
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
            self.model.config.use_cache = False