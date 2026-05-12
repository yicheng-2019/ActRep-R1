# ActRep-R1

**Reasoning for Video Repetitive Action Counting via Multimodal Large Language Models with Reinforcement Learning**

ActRep-R1 is a post-training framework that adapts Multimodal Large Language Models (MLLMs) to repetitive action counting (RAC) through structured reasoning and reinforcement learning. The framework consists of:

- A structured reasoning format with temporal-aware verification
- **Random Count Sampling (RCS)** — a data augmentation strategy that addresses count distribution imbalance via stratified ratio sampling
- A two-stage training pipeline: **SFT → GRPO** with a count-normalized hybrid reward

## Pipeline

```
CoT Data Generation → SFT (Supervised Fine-Tuning) → GRPO (RL) → Evaluation
```

Base model: any [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL) family checkpoint (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

## Project Structure

```
ActRep-R1/
├── sft_main.py                    # SFT training entry point
├── main.py                        # GRPO RL training entry point
├── eval.py                        # Local-checkpoint evaluation
├── eval_multigpu.py               # Multi-GPU parallel evaluation
├── eval_api.py                    # API-based evaluation (OpenAI / Gemini)
├── src/
│   ├── model_utils.py             # VLM backend registry (Qwen2/2.5/3-VL)
│   ├── video_utils.py             # Video preprocessing
│   ├── dataset/
│   │   └── repcount.py            # Dataset loader (CSV / JSONL)
│   ├── trainer/
│   │   ├── sft_trainer.py         # SFT trainer
│   │   ├── grpo_trainer.py        # GRPO trainer
│   │   └── prompts_config.py      # Prompt templates
│   ├── reward/
│   │   └── count_reward.py        # Count-normalized hybrid reward
│   └── utils/                     # Vision/data preprocessing utilities
└── scripts/
    ├── finetune/sft_qwen.sh       # SFT training script (multi-GPU)
    ├── posttrain/rl_qwen3.sh      # GRPO RL training script
    ├── test.sh                    # Evaluation script
    ├── zero2.json                 # DeepSpeed ZeRO-2 config
    ├── zero3.json                 # DeepSpeed ZeRO-3 config
    └── zero3_offload_false.json   # ZeRO-3 with CPU offload
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with PyTorch ≥ 2.4, transformers ≥ 4.46, deepspeed ≥ 0.15, trl ≥ 0.13, and flash-attn 2.

## Data Format

The dataset loader (`src/dataset/repcount.py`) accepts both CSV and JSONL.

**CSV** (one row per sample):
```
video_id, split, action, count
v001.mp4, train, push_up, 12
```

**JSONL**:
```json
{"video_id": "v001.mp4", "split": "train", "action": "push_up", "gt_count": 12, "cot": "..."}
```

The `cot` field contains structured reasoning text used for SFT. RL training only requires the count.

## Step 1 — SFT

Edit the paths at the top of `scripts/finetune/sft_qwen.sh` (base model, training data, video directory), then run:

```bash
bash scripts/finetune/sft_qwen.sh
```

Defaults: 4 GPUs, batch=1, grad_accum=4, 6 epochs, lr=1e-5, cosine schedule, ViT frozen.

## Step 2 — GRPO RL

Initialize from the best SFT checkpoint and edit the paths in `scripts/posttrain/rl_qwen3.sh`:

```bash
bash scripts/posttrain/rl_qwen3.sh
```

Defaults: 4 GPUs, num_generations=8, temperature=1.0, 2 epochs, GRPO algorithm, `count_hybrid` + `count_format` rewards.

## Step 3 — Evaluation

Single checkpoint:

```bash
bash scripts/test.sh
```

Multi-GPU parallel evaluation:

```bash
python eval_multigpu.py --model_path <ckpt> --test_data <data> --video_folder <videos>
```

API model evaluation (OpenAI / Gemini):

```bash
export OPENAI_API_KEY=...   # or GOOGLE_API_KEY=...
python eval_api.py --provider openai --model gpt-4o \
    --test_data <data> --video_folder <videos>
```

Metrics: **OBO** (Off-By-One accuracy), **Exact** match, **MAE** (normalized), **RMSE**.

## Video Resolution & OOM Prevention

GRPO replicates each video tensor `num_generations` times. For high-resolution datasets (1080p), control memory usage with `--max_pixels`:

| `--max_pixels` | 1080p resized | Tokens | Use case |
|----------------|---------------|--------|----------|
| (default)      | 1920×1088     | ~20K   | Low-resolution datasets |
| 393216         | 832×448       | ~3.6K  | Mixed-resolution datasets |
| 262144         | 672×384       | ~2.5K  | Conservative |

## Important Notes

- DeepSpeed ZeRO-3 with generation-based eval during training causes hangs. Evaluate checkpoints separately after training finishes.
- Equivalent batch size = `num_gpus × per_device_train_batch_size × gradient_accumulation_steps`.
- For long-CoT data, set `max_new_tokens=1024` during evaluation.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{actrep-r1,
  title   = {Reasoning for Video Repetitive Action Counting via Multimodal Large Language Models with Reinforcement Learning},
  author  = {Yicheng Qiu and Junwen Chen and Hanzhe Gao and Feng Sha and Keiji Yanai},
  year    = {2026}
}
```

## License

See [LICENSE](LICENSE).
