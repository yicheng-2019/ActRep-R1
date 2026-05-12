"""
RepCount Evaluation Script

Uses shared model_utils / video_utils / prompts_config to ensure
preprocessing is identical to training (GRPO / SFT).

Usage:
    # 单模型评估
    python eval.py --model_path <path> --test_data <csv_or_jsonl> --video_folder <dir>

    # 批量评估: 自动发现目录下所有 checkpoint
    python eval.py --model_dir <result_dir> --test_data <csv_or_jsonl> --video_folder <dir>

    # 批量评估: 指定多个路径
    python eval.py --model_path ckpt1 ckpt2 ckpt3 --test_data <csv_or_jsonl> --video_folder <dir>
"""

import argparse
import glob
import json
import math
import os
import re
import time
from datetime import datetime

import torch
from tqdm import tqdm

from src.model_utils import detect_model_info, get_backend
from src.video_utils import process_video, run_processor
from src.trainer.prompts_config import build_repcount_messages
from src.reward.count_reward import parse_count_output


# ==================== Metrics ====================

def compute_metrics(predictions, ground_truths):
    """
    Compute RepCount evaluation metrics (aligned with TransRAC).

    MAE: Normalized MAE = mean(|pred - gt| / (gt + 0.1))
    OBO: Off-By-One accuracy (|pred - gt| <= 1)
    """
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return {}

    norm_errors, abs_errors = [], []
    obo_correct = exact_correct = parse_success = 0

    for pred, gt in zip(predictions, ground_truths):
        if pred is None or pred < 0:
            continue
        parse_success += 1
        abs_err = abs(pred - gt)
        abs_errors.append(abs_err)
        norm_errors.append(abs_err / (gt + 1e-1))
        if abs_err <= 1:
            obo_correct += 1
        if abs_err == 0:
            exact_correct += 1

    if parse_success == 0:
        return {"MAE": float('inf'), "OBO": 0, "Exact": 0, "RMSE": float('inf'),
                "parse_rate": 0, "total": n, "parsed": 0}

    mae = sum(norm_errors) / parse_success
    abs_mae = sum(abs_errors) / parse_success
    rmse = math.sqrt(sum(e ** 2 for e in abs_errors) / parse_success)
    median_ae = sorted(abs_errors)[parse_success // 2]

    return {
        "MAE": round(mae, 4),
        "AbsMAE": round(abs_mae, 4),
        "MedianAE": round(median_ae, 4),
        "OBO": round(obo_correct / parse_success * 100, 2),
        "Exact": round(exact_correct / parse_success * 100, 2),
        "RMSE": round(rmse, 4),
        "parse_rate": round(parse_success / n * 100, 2),
        "total": n,
        "parsed": parse_success,
    }


# ==================== Data Loading ====================

def load_test_data(test_data_path, video_folder, split="test"):
    """
    Load test data — delegates to shared dataset loader.
    Supports CSV and JSONL (auto-detect by extension).
    """
    from src.dataset.repcount import load_json_dataset_repcount

    dataset = load_json_dataset_repcount(
        test_data_path, video_folder, split=split,
    )
    return [dataset[i] for i in range(len(dataset))]


# ==================== Model Discovery ====================

def _ckpt_sort_key(path):
    """Extract numeric key for sorting: checkpoint-100 -> 100, epoch-2 -> 2."""
    name = os.path.basename(path.rstrip("/"))
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else 0


def discover_models(model_dir):
    """
    Auto-discover model checkpoints under a directory.
    Finds: checkpoint-*, epoch-*, best, last
    Returns sorted list of paths.
    """
    patterns = ["checkpoint-*", "epoch-*", "best", "last"]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(model_dir, pat)))

    # Filter: only dirs that contain model files
    valid = []
    for p in found:
        if not os.path.isdir(p):
            continue
        # Must contain config.json or adapter_config.json
        has_config = (
            os.path.exists(os.path.join(p, "config.json"))
            or os.path.exists(os.path.join(p, "adapter_config.json"))
        )
        if has_config:
            valid.append(p)

    valid.sort(key=_ckpt_sort_key)
    return valid


# ==================== HF Inference ====================

def run_hf_inference(args, samples, model_type, model_size, model_path,
                     output_files=None):
    """
    Run inference using HuggingFace transformers.
    Uses shared model_utils / video_utils / prompts_config (same as training).
    """
    backend = get_backend(model_path)

    model = backend.load_model(
        model_path,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    model = model.to("cuda")
    model.eval()

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
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
    """Evaluate a single model. Returns (metrics, elapsed)."""

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
        predictions, raw_outputs = run_hf_inference(
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


# ==================== Comparison Table ====================

def print_comparison_table(all_results):
    """Print a comparison table for multiple models."""
    if len(all_results) <= 1:
        return

    print(f"\n{'='*90}")
    print(f"{'Model':<30s} {'OBO%':>8s} {'Exact%':>8s} {'MAE':>8s} {'RMSE':>8s} {'Parse%':>8s} {'Time':>8s}")
    print(f"{'-'*90}")

    best_obo = max(r['metrics'].get('OBO', 0) for r in all_results)

    for r in all_results:
        m = r['metrics']
        name = r['model_name']
        if len(name) > 28:
            name = "..." + name[-25:]
        obo = m.get('OBO', 0)
        mark = " *" if obo == best_obo else ""
        print(
            f"{name:<30s} "
            f"{obo:>7.2f}{mark:1s} "
            f"{m.get('Exact', 0):>8.2f} "
            f"{m.get('MAE', 0):>8.4f} "
            f"{m.get('RMSE', 0):>8.4f} "
            f"{m.get('parse_rate', 0):>7.1f}% "
            f"{r['elapsed']:>7.0f}s"
        )

    print(f"{'='*90}")
    print(f"  * = best OBO")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RepCount counting performance")

    # Model selection (三种方式)
    parser.add_argument("--model_path", type=str, nargs="+", default=None,
                        help="单个或多个模型路径")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="自动发现该目录下所有 checkpoint-*/epoch-*/best/last")

    # Data
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.no_flash_attn:
        args.flash_attn = False

    # ---- Resolve model paths ----
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

    # ---- Load data (once) ----
    samples = load_test_data(args.test_data, args.video_folder, split=args.split)
    ground_truths = [s['gt_count'] for s in samples]

    # ---- Evaluate each model ----
    all_results = []

    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path.rstrip("/"))
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(model_paths)}] Evaluating: {model_name}")
        print(f"  Path: {model_path}")
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

        # Print individual results
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

    # ---- Comparison table ----
    print_comparison_table(all_results)

    # ---- Save summary ----
    if len(all_results) > 1:
        summary_path = os.path.join(
            args.output_dir,
            f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
