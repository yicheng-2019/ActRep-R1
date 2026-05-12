"""
RepCount Reward Functions — 分析与改进

============================================================
文献参考:
  - CrowdVLM-R1 (FGRPR, arxiv 2504.03724): fuzzy reward for counting
  - R1-V: binary reward for counting (被证明效果不如 fuzzy)
  - DRA-GRPO: diversity-aware reward + cosine length scaling

============================================================
当前设计反思:

你当前用的是固定 σ=3.0 的高斯 reward:
    r = exp(-error² / (2 * 3.0²))

问题:
  1. σ 固定 → GT=3 差3 和 GT=50 差3 获得相同 reward (0.607)
     但前者是100%误差，后者仅6%
  2. error > 6 时 reward < 0.14，对高计数样本梯度稀疏
  3. 无法区分"方向对但差一些"vs"完全不沾边"

============================================================
推荐的 Reward 选项 (建议做消融实验):

Option A: 归一化高斯 (推荐，简单且合理)
    σ = max(1.0, gt * 0.2)
    r = exp(-error² / (2σ²))
  
  → GT=3:  σ=1.0, error=3 → r=0.011 (正确惩罚)
  → GT=50: σ=10,  error=3 → r=0.956 (正确宽容)

Option B: CrowdVLM-R1 风格 (线性归一化)
    r = max(0, 1 - |pred-gt| / max(gt, 1))
  
  → GT=3:  error=3 → r=0.0
  → GT=50: error=3 → r=0.94
  → 简单，但没有平滑过渡

Option C: 混合奖励 (accuracy + OBO bonus)
    base = exp(-error² / (2σ²))   # σ = max(1, gt*0.2)
    obo_bonus = 0.2 if error <= 1 else 0.0
    exact_bonus = 0.3 if error == 0 else 0.0
    r = base + obo_bonus + exact_bonus
  
  → 额外激励 OBO (论文核心指标) 和 exact match

============================================================
"""

import re
import os
import math
from typing import List, Optional
from datetime import datetime


# ==================== Parsing (不变) ====================

def parse_count_output(output_string: str) -> Optional[int]:
    """Parse count from <answer> tags. Returns None if failed."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    if not answer_matches:
        return None
    last = answer_matches[-1].strip()
    if last.isdigit():
        return int(last)
    numbers = re.findall(r'\b(\d+)\b', last)
    if numbers:
        try:
            count = int(numbers[-1])
            if 0 <= count <= 1000:
                return count
        except ValueError:
            pass
    return None


# ==================== Option A: 归一化高斯 (推荐) ====================

def count_reward_normalized_gaussian(
    completions: List[str],
    solution: List[int],
    sigma_ratio: float = 0.2,
    sigma_min: float = 1.0,
    **kwargs
) -> List[float]:
    """
    归一化高斯 reward — σ 与 GT 成比例
    
    σ = max(sigma_min, gt_count * sigma_ratio)
    r = exp(-error² / (2σ²))
    
    优点:
      - GT=3 时 σ=1.0: 差1就明显惩罚 (r=0.61), 差3基本为0 (r=0.01)
      - GT=50 时 σ=10: 差3还有 r=0.96, 差10有 r=0.61
      - 对所有 GT 量级都有有效梯度
    
    默认 sigma_ratio=0.2: 约 20% 的 GT 作为一个标准差
    """
    rewards = []
    for content, gt_count in zip(completions, solution):
        reward = 0.0
        pred_count = parse_count_output(content)
        if pred_count is not None:
            sigma = max(sigma_min, gt_count * sigma_ratio)
            error = abs(pred_count - gt_count)
            reward = math.exp(-(error ** 2) / (2 * sigma ** 2))
        rewards.append(reward)
        _debug_log(content, pred_count, gt_count, reward, "norm_gauss")
    return rewards


# ==================== Option B: CrowdVLM-R1 风格 ====================

def count_reward_linear(
    completions: List[str],
    solution: List[int],
    **kwargs
) -> List[float]:
    """
    线性归一化 reward — CrowdVLM-R1 (FGRPR) 风格
    
    r = max(0, 1 - |pred-gt| / max(gt, 1))
    
    优点: 简单、可解释
    缺点: 不平滑，pred=gt±1 时 reward 突变
    """
    rewards = []
    for content, gt_count in zip(completions, solution):
        reward = 0.0
        pred_count = parse_count_output(content)
        if pred_count is not None:
            error = abs(pred_count - gt_count)
            reward = max(0.0, 1.0 - error / max(gt_count, 1))
        rewards.append(reward)
        _debug_log(content, pred_count, gt_count, reward, "linear")
    return rewards


# ==================== Option C: 混合奖励 (推荐实验) ====================

def count_reward_hybrid(
    completions: List[str],
    solution: List[int],
    sigma_norm: float = 0.20,
    obo_bonus: float = 0.5,
    **kwargs
) -> List[float]:
    """
    混合 reward = MAE归一化高斯 + OBO bonus

    设计思路:
      1. 高斯部分采用 MAE 风格归一化: norm_error = |pred-gt| / (gt+0.1)
         用固定 σ_norm，使相同相对误差在所有 GT 下得到相同 reward。
         高计数区间区分度显著优于 σ=gt*0.2 方案。
      2. OBO bonus: |error| ≤ 1 时额外 +0.5，与核心评价指标对齐。

    σ_norm=0.20 含义: 20% 相对误差 = 1σ
      - 10% 相对误差 → base=0.88
      - 20% 相对误差 → base=0.61
      - 30% 相对误差 → base=0.32
      - 50% 相对误差 → base=0.04

    总 reward 范围: [0, 1.5]

    示例:
      GT=10, pred=11 → norm_err=0.10, base=0.88 + bonus=0.5 = 1.38  (OBO)
      GT=10, pred=12 → norm_err=0.20, base=0.61             = 0.61  (断崖)
      GT=50, pred=51 → norm_err=0.02, base=1.00 + bonus=0.5 = 1.50  (OBO)
      GT=50, pred=55 → norm_err=0.10, base=0.88             = 0.88
      GT=50, pred=60 → norm_err=0.20, base=0.61             = 0.61
    """
    rewards = []
    for content, gt_count in zip(completions, solution):
        reward = 0.0
        pred_count = parse_count_output(content)
        if pred_count is not None:
            error = abs(pred_count - gt_count)
            norm_error = error / (gt_count + 0.1)

            base = math.exp(-(norm_error ** 2) / (2 * sigma_norm ** 2))
            bonus = obo_bonus if error <= 1 else 0.0

            reward = base + bonus

        rewards.append(reward)
        _debug_log(content, pred_count, gt_count, reward, "hybrid")
    return rewards


# ==================== 原始 reward (保留对照) ====================

def count_accuracy_reward(
    completions: List[str],
    solution: List[int],
    **kwargs
) -> List[float]:
    """原始固定 σ=3.0 高斯 reward (保留用于消融实验对照)"""
    sigma = 3.0
    rewards = []
    for content, gt_count in zip(completions, solution):
        reward = 0.0
        pred_count = parse_count_output(content)
        if pred_count is not None:
            error = abs(pred_count - gt_count)
            reward = math.exp(-(error ** 2) / (2 * sigma ** 2))
        rewards.append(reward)
        _debug_log(content, pred_count, gt_count, reward, "gauss_fixed")
    return rewards


# ==================== Format Rewards ====================

def count_format_reward(completions: List[str], **kwargs) -> List[float]:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return [1.0 if re.search(pattern, c.strip()) else 0.0 for c in completions]


def count_format_reward_8b(completions: List[str], **kwargs) -> List[float]:
    pattern = re.compile(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", re.DOTALL)
    return [1.0 if re.search(pattern, c.strip()) else 0.0 for c in completions]


# ==================== Interval Consistency Reward ====================

def _parse_interval_count(content: str) -> Optional[int]:
    """
    Parse the number of intervals from the think block.

    Supported formats:
      1. "rep1: Xs-Ys, rep2: Xs-Ys, ..." → count repN patterns
      2. "reps 1-20: Xs-Ys (...), reps 21-60: ..." → sum group sizes
      3. Legacy "Intervals: X.X-Y.Y, ..." → count time pairs
    """
    # Find think block content
    think_match = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', content, re.DOTALL)
    if not think_match:
        return None

    think_text = think_match.group(1)

    # Format 1: count "repN:" patterns (new natural format)
    rep_matches = re.findall(r'rep\d+:', think_text)
    if rep_matches:
        return len(rep_matches)

    # Format 2: grouped "reps X-Y:" → sum up group sizes
    groups = re.findall(r'reps\s+(\d+)-(\d+):', think_text)
    if groups:
        total = sum(int(end) - int(start) + 1 for start, end in groups)
        return total

    # Format 3: legacy "Intervals: ..." or "Intervals(grouped): ..."
    grouped_match = re.search(r'Intervals\(grouped\):\s*(.+?)(?:\n|$)', think_text)
    if grouped_match:
        groups = re.findall(r'reps\s+(\d+)-(\d+)', grouped_match.group(1))
        if groups:
            return sum(int(end) - int(start) + 1 for start, end in groups)

    intervals_match = re.search(r'Intervals:\s*(.+?)(?:\n|$)', think_text)
    if intervals_match:
        intervals_str = intervals_match.group(1)
        count = len(re.findall(r'\d+\.?\d*\s*-\s*\d+\.?\d*', intervals_str))
        return count if count > 0 else None

    return None


def count_interval_consistency_reward(
    completions: List[str],
    solution: List[int],
    **kwargs
) -> List[float]:
    """
    Interval consistency reward: 奖励区间数量与最终 count 的一致性。

    设计思路:
      模型输出 Intervals 列表 + Total count: N
      如果列出的区间数量 == 声明的 count → reward = 1.0
      如果不一致 → partial reward based on ratio
      如果没有 Intervals → reward = 0.0

    这迫使模型真正"数"每一次动作，而不是猜一个数字。
    """
    rewards = []
    for content, gt_count in zip(completions, solution):
        reward = 0.0
        pred_count = parse_count_output(content)
        interval_count = _parse_interval_count(content)

        if pred_count is not None and interval_count is not None:
            if interval_count == pred_count:
                reward = 1.0  # 完全一致
            elif interval_count > 0 and pred_count > 0:
                # Partial reward: how close interval count is to declared count
                ratio = min(interval_count, pred_count) / max(interval_count, pred_count)
                reward = ratio * 0.5

        rewards.append(reward)
        _debug_log(content, pred_count, gt_count, reward, "interval_consistency")
    return rewards


# ==================== Debug Logging ====================

def _debug_log(content, pred, gt, reward, method):
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        if log_path:
            ts = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{method}] pred={pred}, gt={gt}, reward={reward:.4f} | {ts}\n")


# ==================== Registry ====================

REPCOUNT_REWARD_FUNCS = {
    # 推荐
    'count': count_reward_normalized_gaussian,  # 默认推荐
    'count_hybrid': count_reward_hybrid,         # 带 OBO/exact bonus
    'count_linear': count_reward_linear,         # CrowdVLM-R1 风格
    # 对照
    'count_fixed': count_accuracy_reward,        # 原始 σ=3.0 (消融对照)
    # format
    'count_format': count_format_reward,
    # interval consistency (v4)
    'count_interval': count_interval_consistency_reward,
}

REPCOUNT_REWARD_FUNCS_8B = {
    'count_format_8b': count_format_reward_8b,
}