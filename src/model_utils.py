"""
Model Registry & Utilities — Extensible Architecture

现有实现: Qwen2-VL, Qwen2.5-VL, Qwen3-VL (Dense/MoE)
扩展方式: 新建 backend 文件 → 继承 VLMBackend → 调用 register_backend()

Example (将来添加 InternVL):
    # src/backends/internvl_backend.py
    class InternVLBackend(VLMBackend):
        name = "internvl"
        def load_model(self, ...): ...
        def process_video(self, ...): ...
    
    register_backend(InternVLBackend())

Reference: VLM-R1 (om-ai-lab) vlm_modules pattern
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoProcessor


# ============================================================
# Abstract Backend Interface
# ============================================================

class VLMBackend(ABC):
    """
    Abstract interface for VLM model families.
    
    Each backend handles:
    1. Model detection (from path)
    2. Model loading (with correct parameters)
    3. Video preprocessing (model-specific)
    4. Processor calling (model-specific arguments)
    
    To add a new model family, subclass this and implement all methods,
    then call register_backend(YourBackend()).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'qwen3vl', 'internvl'"""
        ...

    @abstractmethod
    def match(self, model_path: str) -> bool:
        """Return True if this backend handles the given model path."""
        ...

    @abstractmethod
    def detect_model_size(self, model_path: str) -> str:
        """Detect model size variant. Returns e.g. '8b', 'default'."""
        ...

    @abstractmethod
    def load_model(self, model_path: str, attn_implementation: str = "flash_attention_2", **kwargs):
        """Load and return the model."""
        ...

    @abstractmethod
    def process_video(self, video_path: str, **kwargs) -> dict:
        """
        Process a single video file.
        Returns dict ready to be passed to run_processor().
        """
        ...

    @abstractmethod
    def run_processor(self, processor, text_list: list, video_info: dict,
                      padding=True, padding_side="left", **kwargs):
        """Call processor with model-specific arguments."""
        ...

    def load_processor(self, model_path: str, max_pixels=12845056, min_pixels=3136,
                       padding_side="left"):
        """Load processor. Default implementation works for most HF models."""
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = padding_side
        
        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        
        if hasattr(processor, 'image_processor'):
            processor.image_processor.max_pixels = max_pixels
            processor.image_processor.min_pixels = min_pixels
        
        return processor

    def freeze_vit(self, model, keep_merger=True):
        """Freeze ViT backbone. Default implementation for Qwen-style models."""
        if hasattr(model, "visual"):
            print(f"[{self.name}] Freezing ViT (keep_merger={keep_merger})")
            model.visual.requires_grad_(False)
            if keep_merger and hasattr(model.visual, "merger"):
                model.visual.merger.requires_grad_(True)
        else:
            print(f"[{self.name}] WARNING: model.visual not found")


# ============================================================
# Backend Registry
# ============================================================

_BACKENDS: list[VLMBackend] = []


def register_backend(backend: VLMBackend):
    """Register a VLM backend. Backends are matched in registration order."""
    _BACKENDS.append(backend)
    # print(f"[Registry] Registered backend: {backend.name}")


def get_backend(model_path: str) -> VLMBackend:
    """Find the appropriate backend for a model path."""
    for backend in _BACKENDS:
        if backend.match(model_path):
            return backend
    raise ValueError(
        f"No backend registered for model: {model_path}\n"
        f"Available backends: {[b.name for b in _BACKENDS]}\n"
        f"To add support, implement VLMBackend and call register_backend()."
    )


def list_backends() -> list[str]:
    """List all registered backend names."""
    return [b.name for b in _BACKENDS]


# ============================================================
# Convenience Functions (backward compatible)
# ============================================================

def detect_model_info(model_path: str) -> tuple:
    """
    Detect model type and size from path.
    Returns: (model_type, model_size) — e.g. ("qwen3vl", "8b")
    """
    backend = get_backend(model_path)
    model_size = backend.detect_model_size(model_path)
    return backend.name, model_size


def load_model(model_path: str, model_type: str = None, **kwargs):
    """Load model using the appropriate backend."""
    backend = get_backend(model_path)
    return backend.load_model(model_path, **kwargs)


def load_processor(model_path: str, model_type: str = None, **kwargs):
    """Load processor using the appropriate backend."""
    backend = get_backend(model_path)
    return backend.load_processor(model_path, **kwargs)


def freeze_vit(model, model_type: str = None, **kwargs):
    """Freeze ViT using the appropriate backend."""
    # model_type needed here since we don't have path
    if model_type:
        for b in _BACKENDS:
            if b.name == model_type:
                return b.freeze_vit(model, **kwargs)
    # fallback: try default Qwen-style freezing
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
        if hasattr(model.visual, "merger"):
            model.visual.merger.requires_grad_(True)


def is_qwen3(model_type: str) -> bool:
    """Check if model type is Qwen3-VL family."""
    return model_type in ("qwen3vl", "qwen3vl_moe")


# ============================================================
# Qwen Backend Implementation
# ============================================================

# Dynamic imports
try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN3_VL = True
except ImportError:
    _QWEN3_VL = False

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    _QWEN3_VL_MOE = True
except ImportError:
    _QWEN3_VL_MOE = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    _QWEN25_VL = True
except ImportError:
    _QWEN25_VL = False

try:
    from transformers import Qwen2VLForConditionalGeneration
    _QWEN2_VL = True
except ImportError:
    _QWEN2_VL = False


class Qwen3VLBackend(VLMBackend):
    """Backend for Qwen3-VL (Dense) models."""
    
    name = "qwen3vl"
    
    def match(self, model_path: str) -> bool:
        path = os.path.abspath(model_path).lower()
        if "qwen3" not in path:
            return False
        # Exclude MoE variants (handled by Qwen3VLMoEBackend)
        parts = path.replace("-", "_").split("/")
        is_moe = any("a" in p and any(c.isdigit() for c in p.split("a")[-1][:2])
                      for p in parts if "qwen3" in p)
        return not is_moe
    
    def detect_model_size(self, model_path: str) -> str:
        return "8b" if "8b" in os.path.abspath(model_path).lower() else "default"
    
    def load_model(self, model_path, attn_implementation="flash_attention_2", **kwargs):
        assert _QWEN3_VL, "Qwen3VLForConditionalGeneration not available"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            **kwargs,
        )
        model.config.use_cache = False
        return model

    def process_video(self, video_path: str, total_pixels: int = None, **kwargs) -> dict:
        from qwen_vl_utils import process_vision_info

        if total_pixels is None:
            total_pixels = 3584 * 32 * 32
        video_messages = [{"role": "user", "content": [{
            "type": "video", "video": video_path,
            "total_pixels": total_pixels, "min_pixels": 16 * 32 * 32,
        }]}]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            video_messages, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True,
        )
        
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        return {
            "images": image_inputs, "videos": video_inputs,
            "video_metadata": video_metadatas, "video_kwargs": video_kwargs,
        }
    
    def run_processor(self, processor, text_list, video_info,
                      padding=True, padding_side="left", **kwargs):
        return processor(
            text=text_list,
            images=video_info["images"],
            videos=video_info["videos"],
            video_metadata=video_info["video_metadata"],
            padding=padding, return_tensors="pt",
            padding_side=padding_side, add_special_tokens=False,
            **video_info["video_kwargs"],
        )


class Qwen3VLMoEBackend(Qwen3VLBackend):
    """Backend for Qwen3-VL MoE models (e.g., Qwen3-VL-4B-A3B)."""
    
    name = "qwen3vl_moe"
    
    def match(self, model_path: str) -> bool:
        path = os.path.abspath(model_path).lower()
        if "qwen3" not in path:
            return False
        parts = path.replace("-", "_").split("/")
        return any("a" in p and any(c.isdigit() for c in p.split("a")[-1][:2])
                    for p in parts if "qwen3" in p)
    
    def load_model(self, model_path, attn_implementation="flash_attention_2", **kwargs):
        assert _QWEN3_VL_MOE, "Qwen3VLMoeForConditionalGeneration not available"
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        return model


class Qwen25VLBackend(VLMBackend):
    """Backend for Qwen2.5-VL models."""
    
    name = "qwen25vl"
    
    def match(self, model_path: str) -> bool:
        path = os.path.abspath(model_path).lower()
        return "qwen2.5" in path or "qwen2_5" in path
    
    def detect_model_size(self, model_path: str) -> str:
        return "8b" if "8b" in os.path.abspath(model_path).lower() else "default"
    
    def load_model(self, model_path, attn_implementation="flash_attention_2",
                   slide_window=False, max_window_layers=4,
                   sliding_window_length=32768, **kwargs):
        assert _QWEN25_VL, "Qwen2_5_VLForConditionalGeneration not available"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            use_sliding_window=slide_window,
            max_window_layers=max_window_layers,
            sliding_window=sliding_window_length,
        )
        model.config.use_cache = False
        return model
    
    def process_video(self, video_path: str, total_pixels: int = None, **kwargs) -> dict:
        from src.utils import process_vision_info_v3

        if total_pixels is None:
            total_pixels = 3584 * 28 * 28
        video_messages = [{"role": "user", "content": [{
            "type": "video", "video": video_path,
            "total_pixels": total_pixels, "min_pixels": 16 * 28 * 28,
        }]}]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info_v3(
            [video_messages], return_video_kwargs=True,
        )
        
        return {
            "images": None,
            "videos": [video_inputs[0]],
            "fps": [video_kwargs["fps"][0]],
        }
    
    def run_processor(self, processor, text_list, video_info,
                      padding=True, padding_side="left", **kwargs):
        return processor(
            text=text_list,
            images=video_info.get("images"),
            videos=video_info["videos"],
            fps=video_info["fps"],
            padding=padding, return_tensors="pt",
            padding_side=padding_side, add_special_tokens=False,
        )


class Qwen2VLBackend(VLMBackend):
    """Backend for Qwen2-VL models (fallback)."""
    
    name = "qwen2vl"
    
    def match(self, model_path: str) -> bool:
        path = os.path.abspath(model_path).lower()
        return "qwen2" in path and "qwen2.5" not in path and "qwen3" not in path
    
    def detect_model_size(self, model_path: str) -> str:
        return "default"
    
    def load_model(self, model_path, attn_implementation="flash_attention_2", **kwargs):
        assert _QWEN2_VL, "Qwen2VLForConditionalGeneration not available"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        model.config.use_cache = False
        return model
    
    def process_video(self, video_path: str, **kwargs) -> dict:
        # Same as Qwen2.5-VL
        return Qwen25VLBackend.process_video(self, video_path, **kwargs)
    
    def run_processor(self, processor, text_list, video_info,
                      padding=True, padding_side="left", **kwargs):
        return Qwen25VLBackend.run_processor(
            self, processor, text_list, video_info, padding, padding_side
        )


# ============================================================
# Auto-register all Qwen backends
# ============================================================
# Order matters: more specific backends first
register_backend(Qwen3VLMoEBackend())
register_backend(Qwen3VLBackend())
register_backend(Qwen25VLBackend())
register_backend(Qwen2VLBackend())


# ============================================================
# Template: How to add InternVL (未来)
# ============================================================
#
# class InternVLBackend(VLMBackend):
#     name = "internvl"
#
#     def match(self, model_path):
#         return "internvl" in model_path.lower()
#
#     def detect_model_size(self, model_path):
#         return "default"
#
#     def load_model(self, model_path, **kwargs):
#         from transformers import AutoModel
#         model = AutoModel.from_pretrained(model_path, trust_remote_code=True, ...)
#         return model
#
#     def process_video(self, video_path):
#         # InternVL uses its own video processing
#         ...
#
#     def run_processor(self, processor, text_list, video_info, **kwargs):
#         ...
#
# register_backend(InternVLBackend())