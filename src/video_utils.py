"""
Unified Video Processing — delegates to model-specific backends

用法不变:
    from src.video_utils import process_video, run_processor
    video_info = process_video(model_type, video_path)
    inputs = run_processor(processor, texts, video_info, model_type)

内部实现: 通过 model_utils.py 的 Backend Registry 自动分发
将来新增 InternVL: 只需在 model_utils.py 注册 InternVLBackend，此文件无需修改
"""

from src.model_utils import _BACKENDS


def _find_backend(model_type: str):
    """Find backend by name."""
    for b in _BACKENDS:
        if b.name == model_type:
            return b
    raise ValueError(
        f"No backend found for model_type='{model_type}'. "
        f"Available: {[b.name for b in _BACKENDS]}"
    )


def process_video(model_type: str, video_path: str, total_pixels: int = None) -> dict:
    """
    Process a single video file using the appropriate backend.

    Args:
        model_type: e.g. "qwen3vl", "qwen25vl", "internvl" (future)
        video_path: path to video file
        total_pixels: max total pixels per video (None = backend default)

    Returns:
        dict: model-specific video info, passed to run_processor()
    """
    backend = _find_backend(model_type)
    return backend.process_video(video_path, total_pixels=total_pixels)


def run_processor(processor, text_list: list, video_info: dict, model_type: str,
                  padding=True, padding_side="left", **kwargs):
    """
    Call processor with model-specific arguments.
    
    Args:
        processor: AutoProcessor instance
        text_list: list of tokenized text strings
        video_info: dict from process_video()
        model_type: backend name
    """
    backend = _find_backend(model_type)
    return backend.run_processor(
        processor, text_list, video_info,
        padding=padding, padding_side=padding_side, **kwargs,
    )