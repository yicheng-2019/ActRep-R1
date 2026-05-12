"""
RepCount Dataset Loader
支持 CSV / JSONL 两种格式，自动检测
"""
import csv
import json
import os
import random
from tqdm import tqdm
from datasets import Dataset


# ============================================================
# 单条样本构建
# ============================================================

def _build_example(video_id, action, count, video_dir, cot=''):
    video_path = os.path.join(video_dir, video_id)
    if not os.path.isfile(video_path):
        return None

    action_name = action.replace('_', ' ')
    return {
        'problem': f"Count the number of '{action_name}' repetitions in this video.",
        'solution': count,
        'video_path': video_path,
        'action': action,
        'video_id': video_id,
        'gt_count': count,
        'cot': cot,
    }


# ============================================================
# CSV:  video_id, split, action, count, [frames...]
# ============================================================

def _load_csv(file_path, video_dir, split):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, row in enumerate(tqdm(csv.reader(f), desc="Loading CSV")):
            try:
                if len(row) < 4:
                    continue
                video_id, row_split, action, count = (
                    row[0].strip(), row[1].strip(), row[2].strip(), int(row[3].strip())
                )
                if split and row_split != split:
                    continue
                ex = _build_example(video_id, action, count, video_dir)
                if ex:
                    examples.append(ex)
            except (ValueError, IndexError) as e:
                print(f"[Error] CSV line {idx}: {e}")
    return examples


# ============================================================
# JSONL (兼容两种)
#   旧: {"video_id": ..., "ground_truth": {"action": ..., "count": N}}
#   新: {"video_id": ..., "action": ..., "gt_count": N, "split": "train"}
# ============================================================

def _load_jsonl(file_path, video_dir, split):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Loading JSONL")):
            try:
                item = json.loads(line.strip())
                video_id = item.get('video_id')

                if 'ground_truth' in item:
                    action = item['ground_truth']['action']
                    count = item['ground_truth']['count']
                else:
                    action = item['action']
                    count = item['gt_count']

                if split and item.get('split', 'train') != split:
                    continue

                cot = item.get('cot', '')
                ex = _build_example(video_id, action, count, video_dir, cot)
                if ex:
                    examples.append(ex)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Error] JSONL line {idx}: {e}")
    return examples


# ============================================================
# 主入口
# ============================================================

def load_json_dataset_repcount(
    train_data_path: str,
    video_folder: str,
    is_curriculum_learning: bool = False,
    split: str = None,
) -> Dataset:
    """
    加载 RepCount 数据集（自动检测 CSV / JSONL）

    Args:
        train_data_path: 数据文件 (.csv 或 .jsonl)
        video_folder: 视频目录
        is_curriculum_learning: 按 count 排序（简单→困难）
        split: "train" / "test" / "val" / None（不过滤）
    """
    if train_data_path.endswith('.csv'):
        examples = _load_csv(train_data_path, video_folder, split)
    else:
        examples = _load_jsonl(train_data_path, video_folder, split)

    if not examples:
        raise ValueError(f"No valid examples from {train_data_path} (split={split})")

    if is_curriculum_learning:
        examples.sort(key=lambda x: x['gt_count'])
        print("[RepCount] Curriculum learning: sorted by count")
    else:
        random.shuffle(examples)

    print(f"[RepCount] Loaded {len(examples)} examples (split={split or 'all'})")
    for i, ex in enumerate(examples[:3]):
        print(f"  {i+1}. {ex['video_id']} | {ex['action']} | count={ex['gt_count']}")

    return Dataset.from_list(examples)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python repcount.py <data_file> <video_dir>")
        sys.exit(1)
    ds = load_json_dataset_repcount(sys.argv[1], sys.argv[2])
    print(f"Dataset size: {len(ds)}")