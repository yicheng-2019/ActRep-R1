"""
RepCount API Evaluation Script

Evaluate commercial API models (OpenAI GPT-4o, Google Gemini, etc.)
on the RepCount video repetition counting benchmark.

Reuses metrics, prompts, parsing, and data loading from the existing
HuggingFace evaluation pipeline (eval.py).

Usage:
    # OpenAI GPT-4o
    python eval_api.py --provider openai --model gpt-4o \
        --test_data <csv_or_jsonl> --video_folder <dir>

    # Gemini 3 Flash
    python eval_api.py --provider gemini --model gemini-3-flash-preview \
        --test_data <csv_or_jsonl> --video_folder <dir>

    # Resume from previous run (skip already-completed samples)
    python eval_api.py --provider gemini --model gemini-3-flash-preview \
        --test_data <csv_or_jsonl> --video_folder <dir> \
        --resume <output_dir>

Environment variables:
    OPENAI_API_KEY   — Required for --provider openai
    GOOGLE_API_KEY   — Required for --provider gemini (or GEMINI_API_KEY)
"""

import argparse
import base64
import glob
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm

from eval import compute_metrics, load_test_data
from src.reward.count_reward import parse_count_output
from src.trainer.prompts_config import get_prompts


# ==================== Video Processing ====================

def extract_frames(video_path: str, num_frames: int = 16,
                   resize_max: int = 768, fps: float = 0) -> list:
    """
    Extract uniformly sampled frames from a video.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample (used when fps=0).
        resize_max: Max dimension (longest side) for each frame.
        fps: If > 0, sample at this fps (dynamic frame count).
             Overrides num_frames. E.g., fps=2 for a 30s video = 60 frames.

    Returns:
        List of PIL.Image.Image objects.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    # Dynamic fps-based sampling or fixed num_frames
    if fps > 0:
        video_fps = vr.get_avg_fps()
        duration = total / video_fps
        num_frames = max(int(duration * fps), 4)  # at least 4 frames

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)

    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        w, h = img.size
        if max(w, h) > resize_max:
            scale = resize_max / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(img)
    return images


def frames_to_base64(frames: list, fmt: str = "JPEG",
                     quality: int = 85) -> list:
    """Convert PIL Images to base64-encoded strings."""
    result = []
    for img in frames:
        buf = BytesIO()
        img.save(buf, format=fmt, quality=quality)
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


# ==================== Resume Support ====================

def load_completed_results(output_dir: str) -> dict:
    """
    Load successfully completed results from existing JSONL files.

    Returns:
        dict: {video_id: {detail_dict}} for samples with pred_count >= 0
    """
    completed = {}
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "details_*.jsonl")))
    for fpath in jsonl_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    # Only count as completed if prediction was successful
                    if d.get('pred_count', -1) >= 0:
                        completed[d['video_id']] = d
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ==================== API Providers ====================

class APIProvider(ABC):
    """Abstract base class for API providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def call(self, system_prompt: str, user_content,
             max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Send a request and return the text response."""
        ...


class OpenAIProvider(APIProvider):
    """OpenAI-compatible API provider (GPT-4o, etc.)."""

    @property
    def name(self):
        return "openai"

    def __init__(self, model: str, api_key: str = None,
                 base_url: str = None, max_retries: int = 3,
                 timeout: float = 120.0):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def call(self, system_prompt, user_content,
             max_tokens=1024, temperature=0.0):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        # Capture reasoning_content from thinking models (e.g. qwen-thinking)
        reasoning = getattr(msg, 'reasoning_content', None) or ""
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    @staticmethod
    def build_user_content(frames_b64: list, question_text: str) -> list:
        """Build OpenAI-format user content with images + text."""
        content = []
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "auto",
                },
            })
        content.append({"type": "text", "text": question_text})
        return content


class GeminiProvider(APIProvider):
    """Google Gemini API provider (2.0/2.5/3.0 Flash/Pro, etc.)."""

    @property
    def name(self):
        return "gemini"

    def __init__(self, model: str, api_key: str = None,
                 max_retries: int = 3, timeout: float = 120.0,
                 use_native_video: bool = False):
        from google import genai
        self.model = model
        self.use_native_video = use_native_video
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
        )
        self._uploaded_files = {}  # cache: video_path -> File object

    def call(self, system_prompt, user_content,
             max_tokens=1024, temperature=0.0):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text

    def upload_video(self, video_path: str):
        """Upload video via File API. Results are cached."""
        if video_path in self._uploaded_files:
            return self._uploaded_files[video_path]

        video_file = self.client.files.upload(file=video_path)
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {video_path}")

        self._uploaded_files[video_path] = video_file
        return video_file

    def build_user_content_frames(self, frames_b64: list,
                                  question_text: str) -> list:
        """Build Gemini-format content from base64 frames."""
        from google.genai import types

        parts = []
        for b64 in frames_b64:
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64),
                mime_type="image/jpeg",
            ))
        parts.append(types.Part.from_text(text=question_text))
        return parts

    def build_user_content_video(self, video_file,
                                 question_text: str) -> list:
        """Build Gemini-format content from uploaded video."""
        from google.genai import types
        return [video_file, types.Part.from_text(text=question_text)]

    def cleanup_uploaded_files(self):
        """Delete all uploaded files from Gemini servers."""
        for path, f in self._uploaded_files.items():
            try:
                self.client.files.delete(name=f.name)
            except Exception:
                pass
        self._uploaded_files.clear()


# ==================== Prompt Building ====================

def build_api_prompt(action: str, model_size: str = "default") -> tuple:
    """
    Build (system_prompt, question_text) for API calls.

    Reuses prompts_config.get_prompts() to ensure consistency
    with HF evaluation.

    Args:
        action: Action type (e.g., "push_up").
        model_size: "default" or "8b" — controls think/thinking tags.

    Returns:
        (system_prompt, question_text)
    """
    system_prompt, question_template = get_prompts(model_size)

    if isinstance(action, str):
        action = action.replace("_", " ")

    question_text = question_template.replace("[ACTION]", action)
    return system_prompt, question_text


# ==================== Inference ====================

def _is_quota_error(error: Exception) -> bool:
    """Check if an exception is a quota/rate-limit error."""
    err_str = str(error)
    return "429" in err_str or "RESOURCE_EXHAUSTED" in err_str


def run_api_inference(args, provider, samples, output_files=None):
    """
    Run inference via API provider.
    Mirrors run_hf_inference() in eval.py.
    Supports --stop_on_quota to gracefully stop when quota is exhausted.
    """
    predictions, raw_outputs = [], []
    consecutive_quota_errors = 0

    for sample in tqdm(samples, desc=f"Evaluating {provider.name}/{args.model}"):
        system_prompt, question_text = build_api_prompt(
            sample['action'], args.model_size
        )

        # Build visual content
        try:
            if isinstance(provider, GeminiProvider) and provider.use_native_video:
                video_file = provider.upload_video(sample['video_path'])
                user_content = provider.build_user_content_video(
                    video_file, question_text
                )
            else:
                frames = extract_frames(
                    sample['video_path'],
                    num_frames=args.num_frames,
                    resize_max=args.resize_max,
                    fps=args.fps,
                )
                frames_b64 = frames_to_base64(frames)

                if isinstance(provider, OpenAIProvider):
                    user_content = provider.build_user_content(
                        frames_b64, question_text
                    )
                else:
                    user_content = provider.build_user_content_frames(
                        frames_b64, question_text
                    )

            output_text = provider.call(
                system_prompt, user_content,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            consecutive_quota_errors = 0  # Reset on success
        except Exception as e:
            tqdm.write(f"  [Error] {sample['video_id']}: {e}")
            output_text = f"[ERROR] {e}"

            # Check for quota exhaustion
            if _is_quota_error(e):
                consecutive_quota_errors += 1
                if args.stop_on_quota and consecutive_quota_errors >= 2:
                    tqdm.write(
                        f"\n  [Quota exhausted] Stopping after {len(predictions)} samples. "
                        f"Use --resume to continue later."
                    )
                    # Still record this failed sample
                    pred_count = -1
                    predictions.append(pred_count)
                    raw_outputs.append(output_text)
                    if output_files:
                        _write_detail(output_files, sample, pred_count, output_text)
                    break
            else:
                consecutive_quota_errors = 0

        # Parse
        pred_count = parse_count_output(output_text)
        if pred_count is None:
            pred_count = -1
        predictions.append(pred_count)
        raw_outputs.append(output_text)

        # Incremental save
        if output_files:
            _write_detail(output_files, sample, pred_count, output_text)

        if args.verbose:
            tqdm.write(
                f"  {sample['video_id']} | "
                f"GT={sample['gt_count']} Pred={pred_count}"
            )

        # Rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

    return predictions, raw_outputs


def _write_detail(output_files, sample, pred_count, output_text):
    """Write a single sample's results to JSONL and TXT files."""
    detail = {
        "video_id": sample['video_id'],
        "action": sample['action'],
        "gt_count": sample['gt_count'],
        "pred_count": pred_count,
        "error": abs(pred_count - sample['gt_count']) if pred_count >= 0 else None,
        "output": output_text,
    }
    output_files['jsonl'].write(
        json.dumps(detail, ensure_ascii=False) + "\n"
    )
    output_files['jsonl'].flush()

    output_files['txt'].write(f"{'=' * 60}\n")
    output_files['txt'].write(f"Video: {sample['video_id']}\n")
    output_files['txt'].write(f"Action: {sample['action']}\n")
    output_files['txt'].write(
        f"GT: {sample['gt_count']}, Pred: {pred_count}\n"
    )
    output_files['txt'].write(f"Output:\n{output_text}\n\n")
    output_files['txt'].flush()


# ==================== Evaluation ====================

def evaluate_api_model(args, samples, ground_truths):
    """Evaluate an API model. Returns (metrics, elapsed)."""

    # Create provider
    if args.provider == "openai":
        provider = OpenAIProvider(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
    elif args.provider == "gemini":
        provider = GeminiProvider(
            model=args.model,
            api_key=args.api_key,
            max_retries=args.max_retries,
            timeout=args.timeout,
            use_native_video=args.use_native_video,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # Resume: load completed results and filter samples
    completed = {}
    if args.resume:
        completed = load_completed_results(args.resume)
        if completed:
            print(f"[Resume] Found {len(completed)} completed samples, skipping them")

    # Filter out already-completed samples
    if completed:
        remaining_samples = [s for s in samples if s['video_id'] not in completed]
        print(f"[Resume] {len(remaining_samples)} samples remaining")
    else:
        remaining_samples = samples

    # Output paths — use resume dir or create new
    model_name = args.model.replace("/", "_")
    output_dir = args.resume if args.resume else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = os.path.join(
        output_dir, f"details_{model_name}_{timestamp}.jsonl"
    )
    log_path = os.path.join(
        output_dir, f"outputs_{model_name}_{timestamp}.txt"
    )

    t0 = time.time()
    if remaining_samples:
        with open(details_path, 'w', encoding='utf-8') as f_jsonl, \
             open(log_path, 'w', encoding='utf-8') as f_txt:
            output_files = {'jsonl': f_jsonl, 'txt': f_txt}
            new_predictions, new_outputs = run_api_inference(
                args, provider, remaining_samples, output_files
            )
    else:
        new_predictions, new_outputs = [], []
        print("[Resume] All samples already completed!")
    elapsed = time.time() - t0

    # Cleanup Gemini uploaded files
    if isinstance(provider, GeminiProvider):
        provider.cleanup_uploaded_files()

    # Merge: build full predictions list in original sample order
    # New results from this run
    new_results = {}
    for i, s in enumerate(remaining_samples[:len(new_predictions)]):
        new_results[s['video_id']] = new_predictions[i]

    # Build final predictions in original order
    all_predictions = []
    for s in samples:
        vid = s['video_id']
        if vid in new_results:
            all_predictions.append(new_results[vid])
        elif vid in completed:
            all_predictions.append(completed[vid]['pred_count'])
        else:
            all_predictions.append(-1)  # Not reached yet

    metrics = compute_metrics(all_predictions, ground_truths)

    # Count completion status
    n_done = sum(1 for p in all_predictions if p >= 0)
    n_total = len(samples)

    # Save metrics JSON
    result = {
        "provider": args.provider,
        "model": args.model,
        "model_name": model_name,
        "test_data": args.test_data,
        "num_frames": args.num_frames,
        "use_native_video": getattr(args, 'use_native_video', False),
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 1),
        "completed": f"{n_done}/{n_total}",
        "metrics": metrics,
    }
    metrics_path = os.path.join(
        output_dir, f"metrics_{model_name}_{timestamp}.json"
    )
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also save a merged details file if resuming
    if completed and new_results:
        merged_path = os.path.join(output_dir, f"details_merged_{model_name}.jsonl")
        with open(merged_path, 'w', encoding='utf-8') as f:
            for s in samples:
                vid = s['video_id']
                if vid in completed:
                    f.write(json.dumps(completed[vid], ensure_ascii=False) + "\n")
                elif vid in new_results:
                    pred = new_results[vid]
                    detail = {
                        "video_id": vid,
                        "action": s['action'],
                        "gt_count": s['gt_count'],
                        "pred_count": pred,
                        "error": abs(pred - s['gt_count']) if pred >= 0 else None,
                        "output": new_outputs[list(new_results.keys()).index(vid)]
                            if vid in new_results else "",
                    }
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
        print(f"[Resume] Merged results saved: {merged_path}")

    return metrics, elapsed, n_done, n_total


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate API models (OpenAI/Gemini) on RepCount"
    )

    # Provider & Model
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "gemini"],
                        help="API provider")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., gpt-4o, gemini-3-flash-preview)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: OPENAI_API_KEY or GOOGLE_API_KEY)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Custom base URL (OpenAI only, for Azure/proxy)")

    # Data
    parser.add_argument("--test_data", type=str, required=True,
                        help="Test data file (.csv or .jsonl)")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0=all)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from output dir: skip completed samples, "
                             "append new results")

    # Video processing
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames to extract per video (default: 16)")
    parser.add_argument("--resize_max", type=int, default=768,
                        help="Max frame dimension in pixels (default: 768)")
    parser.add_argument("--fps", type=float, default=0,
                        help="Sample at this fps (dynamic frame count). "
                             "0=use fixed --num_frames. E.g., --fps 2")
    parser.add_argument("--use_native_video", action="store_true",
                        help="Gemini: upload video natively via File API")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=deterministic)")
    parser.add_argument("--model_size", type=str, default="default",
                        choices=["default", "8b"],
                        help="Prompt variant: default=<think>, 8b=<thinking>")

    # API settings
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Request timeout in seconds")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests (rate limiting)")
    parser.add_argument("--stop_on_quota", action="store_true", default=True,
                        help="Stop gracefully when quota is exhausted (default: True)")
    parser.add_argument("--no_stop_on_quota", action="store_true",
                        help="Don't stop on quota errors, keep retrying")

    # Output
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.no_stop_on_quota:
        args.stop_on_quota = False

    # Print config
    print(f"{'=' * 60}")
    print(f"RepCount API Evaluation")
    print(f"{'=' * 60}")
    print(f"  Provider:     {args.provider}")
    print(f"  Model:        {args.model}")
    print(f"  Test data:    {args.test_data}")
    if args.fps > 0:
        print(f"  Sampling:     {args.fps} fps (dynamic)")
    else:
        print(f"  Frames:       {args.num_frames} (fixed)")
    print(f"  Resize:       {args.resize_max}px")
    if args.provider == "gemini" and args.use_native_video:
        print(f"  Video mode:   native upload")
    else:
        print(f"  Video mode:   frame extraction")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Delay:        {args.delay}s")
    if args.resume:
        print(f"  Resume from:  {args.resume}")
    print(f"  Stop on quota: {args.stop_on_quota}")
    print(f"{'=' * 60}")

    # Load data
    samples = load_test_data(args.test_data, args.video_folder, split=args.split)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    ground_truths = [s['gt_count'] for s in samples]
    print(f"Loaded {len(samples)} test samples")

    # Evaluate
    metrics, elapsed, n_done, n_total = evaluate_api_model(
        args, samples, ground_truths
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Model: {args.provider}/{args.model}")
    print(f"Progress: {n_done}/{n_total} samples completed")
    print(f"Time (this run): {elapsed:.1f}s")
    print(f"{'=' * 60}")
    if n_done > 0:
        print(f"  MAE:        {metrics['MAE']}  (normalized)")
        print(f"  AbsMAE:     {metrics['AbsMAE']}")
        print(f"  OBO (%):    {metrics['OBO']}")
        print(f"  Exact (%):  {metrics['Exact']}")
        print(f"  RMSE:       {metrics['RMSE']}")
        print(f"  Parse rate: {metrics['parse_rate']}% ({metrics['parsed']}/{metrics['total']})")
    else:
        print(f"  No successful predictions yet.")
    print(f"{'=' * 60}")

    if n_done < n_total:
        resume_dir = args.resume if args.resume else args.output_dir
        print(f"\n  To continue, run with: --resume {resume_dir}")


if __name__ == "__main__":
    main()
