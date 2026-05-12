"""
RepCount Multi-GPU Evaluation Script

For large models (e.g., 32B) that don't fit on a single GPU.
Uses device_map="auto" to distribute across multiple GPUs.
Uses skip_special_tokens=False to preserve XML tags from models
that treat <think>/<tool_call> as special tokens (e.g., Qwen3-VL-32B).

Usage:
    python eval_multigpu.py --model_path <path> --test_data <csv_or_jsonl> --video_folder <dir>
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

from eval import compute_metrics, load_test_data, discover_models, print_comparison_table
from src.model_utils import detect_model_info, get_backend
from src.video_utils import process_video, run_processor
from src.trainer.prompts_config import build_repcount_messages
from src.reward.count_reward import parse_count_output


# ==================== Multi-GPU Inference ====================

def run_multigpu_inference(args, samples, model_type, model_size, model_path,
                           output_files=None):
    """
    Run inference with device_map="auto" for multi-GPU support.
    """
    backend = get_backend(model_path)

    model = backend.load_model(
        model_path,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        device_map="auto",
    )
    model.eval()
    print(f"  Model distributed across devices: {set(model.hf_device_map.values())}")

    processor = backend.load_processor(model_path, padding_side="left")

    predictions, raw_outputs = [], []

    for sample in tqdm(samples, desc=f"Evaluating {os.path.basename(model_path)}"):
        messages = build_repcount_messages(
            sample['action'], sample['video_path'], model_size,
        )

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        video_info = process_video(model_type, sample['video_path'])

        inputs = run_processor(
            processor, [text], video_info, model_type,
            padding=True, padding_side="left",
        )
        # For device_map="auto", use the device of the first parameter
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        # Use skip_special_tokens=False to preserve XML tags (<think>, <answer>, etc.)
        # Some models (e.g., Qwen3-VL-32B) treat these as special tokens
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # Clean up real special tokens but keep XML-like tags
        for special_tok in [processor.tokenizer.eos_token, processor.tokenizer.pad_token,
                            '<|endoftext|>', '<|im_end|>', '<|im_start|>']:
            if special_tok:
                output_text = output_text.replace(special_tok, '')
        output_text = output_text.strip()

        pred_count = parse_count_output(output_text)
        if pred_count is None:
            pred_count = -1
        predictions.append(pred_count)
        raw_outputs.append(output_text)

        # Incremental save
        if output_files:
            detail = {
                "video_id": sample['video_id'],
                "action": sample['action'],
                "gt_count": sample['gt_count'],
                "pred_count": pred_count,
                "error": abs(pred_count - sample['gt_count']) if pred_count >= 0 else None,
                "output": output_text,
            }
            output_files['jsonl'].write(json.dumps(detail, ensure_ascii=False) + "\n")
            output_files['jsonl'].flush()

            output_files['txt'].write(f"{'='*60}\n")
            output_files['txt'].write(f"Video: {sample['video_id']}\n")
            output_files['txt'].write(f"Action: {sample['action']}\n")
            output_files['txt'].write(f"GT: {sample['gt_count']}, Pred: {pred_count}\n")
            output_files['txt'].write(f"Output:\n{output_text}\n\n")
            output_files['txt'].flush()

        if args.verbose:
            tqdm.write(f"  {sample['video_id']} | GT={sample['gt_count']} Pred={pred_count}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, raw_outputs


# ==================== Single Model Eval ====================

def evaluate_single_model(args, model_path, samples, ground_truths):
    """Evaluate a single model with multi-GPU. Returns (metrics, elapsed)."""

    model_type, model_size = detect_model_info(model_path)
    model_name = os.path.basename(model_path.rstrip("/"))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = os.path.join(args.output_dir, f"details_{model_name}_{timestamp}.jsonl")
    log_path = os.path.join(args.output_dir, f"outputs_{model_name}_{timestamp}.txt")

    t0 = time.time()
    with open(details_path, 'w', encoding='utf-8') as f_jsonl, \
         open(log_path, 'w', encoding='utf-8') as f_txt:
        output_files = {'jsonl': f_jsonl, 'txt': f_txt}
        predictions, raw_outputs = run_multigpu_inference(
            args, samples, model_type, model_size, model_path, output_files
        )
    elapsed = time.time() - t0

    metrics = compute_metrics(predictions, ground_truths)

    # Save metrics
    result = {
        "model_path": model_path,
        "model_name": model_name,
        "test_data": args.test_data,
        "model_type": model_type,
        "model_size": model_size,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
    }
    metrics_path = os.path.join(args.output_dir, f"metrics_{model_name}_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return metrics, elapsed


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RepCount with multi-GPU support (device_map=auto)"
    )

    parser.add_argument("--model_path", type=str, nargs="+", default=None)
    parser.add_argument("--model_dir", type=str, default=None)

    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.no_flash_attn:
        args.flash_attn = False

    # Resolve model paths
    model_paths = []
    if args.model_dir:
        model_paths = discover_models(args.model_dir)
        if not model_paths:
            print(f"[Error] No checkpoints found in {args.model_dir}")
            return
        print(f"Discovered {len(model_paths)} models in {args.model_dir}:")
        for p in model_paths:
            print(f"  - {os.path.basename(p)}")
    elif args.model_path:
        model_paths = args.model_path
    else:
        parser.error("Must specify --model_path or --model_dir")

    # Load data
    samples = load_test_data(args.test_data, args.video_folder, split=args.split)
    ground_truths = [s['gt_count'] for s in samples]

    # Evaluate
    all_results = []
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path.rstrip("/"))
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(model_paths)}] Evaluating: {model_name}")
        print(f"  Path: {model_path}")
        print(f"  device_map: auto")
        print(f"{'='*60}")

        metrics, elapsed = evaluate_single_model(
            args, model_path, samples, ground_truths
        )

        all_results.append({
            "model_name": model_name,
            "model_path": model_path,
            "metrics": metrics,
            "elapsed": elapsed,
        })

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")
        print(f"{'='*60}")
        print(f"  MAE:        {metrics['MAE']}  (normalized)")
        print(f"  AbsMAE:     {metrics['AbsMAE']}")
        print(f"  OBO (%):    {metrics['OBO']}")
        print(f"  Exact (%):  {metrics['Exact']}")
        print(f"  RMSE:       {metrics['RMSE']}")
        print(f"  Parse rate: {metrics['parse_rate']}% ({metrics['parsed']}/{metrics['total']})")
        print(f"{'='*60}")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
