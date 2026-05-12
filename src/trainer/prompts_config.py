"""
Prompt configurations — Single source of truth.

Used by: grpo_trainer.py, sft_trainer.py, eval.py
DO NOT duplicate these templates anywhere else.

Model-size variants:
  - default (2B/4B/32B etc.): uses <think> tag
  - 8b: uses <thinking> tag (Qwen3-VL-8B native format)

v4: Updated to temporal interval CoT format.
  Model is instructed to output:
    Each rep consists of [description].
    Tracking each repetition: rep1: s1-e1, rep2: s2-e2, ...
    Total count: N.
"""


# ==================== System Prompts ====================

SYSTEM_PROMPT_DEFAULT = """You are a video analysis expert.
You MUST follow the output format exactly and include BOTH tags:
<think>...</think>
<answer>...</answer>
Do not use any other tags."""

SYSTEM_PROMPT_8B = """You are a video analysis expert.
You MUST follow the output format exactly and include BOTH tags:
<thinking>...</thinking>
<answer>...</answer>
Do not use any other tags."""


# ==================== RepCount Question Templates ====================

QUESTION_TEMPLATE_REPCOUNT_v1_DEFAULT = """Watch the video carefully and count the number of "[ACTION]" repetitions performed.

Think through this step by step:
- Briefly describe what you see in the video
- Describe what constitutes one complete "[ACTION]"
- For each repetition, identify its time interval (start and end in seconds)
- List all intervals to track each repetition
- Count the total number of repetitions

Provide your reasoning in <think> </think> tags, then give the final count in <answer> </answer> tags.
The answer should be a single integer.

EXAMPLE OUTPUT FORMAT:
<think>
The video shows a person lying on a bench in a gym performing dumbbell presses.
Each rep consists of lowering the barbell to chest and pressing it back up.
Tracking each repetition: rep1: 1.0s-5.0s, rep2: 5.0s-8.8s, rep3: 8.8s-11.8s
Total count: 3.
</think>
<answer>
3
</answer>"""


QUESTION_TEMPLATE_REPCOUNT_v1_8B = """Watch the video carefully and count the number of "[ACTION]" repetitions performed.

Think through this step by step:
- Briefly describe what you see in the video
- Describe what constitutes one complete "[ACTION]"
- For each repetition, identify its time interval (start and end in seconds)
- List all intervals to track each repetition
- Count the total number of repetitions

Provide your reasoning in <thinking> </thinking> tags, then give the final count in <answer> </answer> tags.
The answer should be a single integer.

EXAMPLE OUTPUT FORMAT:
<thinking>
The video shows a person lying on a bench in a gym performing dumbbell presses.
Each rep consists of lowering the barbell to chest and pressing it back up.
Tracking each repetition: rep1: 1.0s-5.0s, rep2: 5.0s-8.8s, rep3: 8.8s-11.8s
Total count: 3.
</thinking>
<answer>
3
</answer>"""


# ==================== Helper Functions ====================

def get_prompts(model_size: str):
    """
    Get (system_prompt, question_template) for given model size.

    Args:
        model_size: "8b" or "default"

    Returns:
        (system_prompt, question_template)
    """
    if model_size == "8b":
        return SYSTEM_PROMPT_8B, QUESTION_TEMPLATE_REPCOUNT_v1_8B
    else:
        return SYSTEM_PROMPT_DEFAULT, QUESTION_TEMPLATE_REPCOUNT_v1_DEFAULT


def get_think_tag(model_size: str) -> tuple:
    """
    Get the think tag pair for the model size.

    Returns:
        (open_tag, close_tag)
    """
    if model_size == "8b":
        return "<thinking>", "</thinking>"
    return "<think>", "</think>"


def build_repcount_messages(action: str, video_path: str, model_size: str):
    """
    Build conversation messages for RepCount task.

    Used by: grpo_trainer, sft_trainer, eval.py
    Ensures prompt format consistency across train/eval.
    """
    system_prompt, question_template = get_prompts(model_size)

    if isinstance(action, str):
        action = action.replace("_", " ")

    question = question_template.replace("[ACTION]", action)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "video", "video": video_path},
        ]},
    ]
