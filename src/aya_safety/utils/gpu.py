"""
GPU/VRAM detection and automatic NLLB model variant selection.

Model VRAM requirements (FP16):
  - nllb-200-3.3B:            ~8 GB
  - nllb-200-distilled-1.3B:  ~3-4 GB
  - nllb-200-distilled-600M:  ~2 GB
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Model variants ordered by quality (best first)
NLLB_VARIANTS = [
    ("facebook/nllb-200-3.3B", 8.0),
    ("facebook/nllb-200-distilled-1.3B", 4.0),
    ("facebook/nllb-200-distilled-600M", 2.0),
]


def get_available_vram_gb() -> float:
    """Return available VRAM in GB on the first CUDA device, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            vram_gb = free / (1024 ** 3)
            logger.info(
                "GPU: %s, VRAM available: %.1f GB / %.1f GB total",
                torch.cuda.get_device_name(device),
                vram_gb,
                total / (1024 ** 3),
            )
            return vram_gb
    except Exception as e:
        logger.debug("CUDA not available: %s", e)
    return 0.0


def select_nllb_model(requested: str = "auto") -> tuple[str, str]:
    """
    Select the best NLLB model variant that fits in available VRAM.

    Args:
        requested: "auto" for automatic selection, or explicit model name.

    Returns:
        (model_name, device) tuple.
    """
    if requested != "auto":
        # Explicit model requested — determine device
        vram = get_available_vram_gb()
        device = "cuda" if vram > 1.0 else "cpu"
        logger.info("Using requested model: %s on %s", requested, device)
        return requested, device

    vram = get_available_vram_gb()

    if vram <= 0:
        model_name = "facebook/nllb-200-distilled-600M"
        logger.info("No GPU detected, using %s on CPU", model_name)
        return model_name, "cpu"

    for model_name, required_gb in NLLB_VARIANTS:
        if vram >= required_gb * 1.1:  # 10% headroom
            logger.info("Auto-selected %s (%.1f GB available, %.1f GB needed)", model_name, vram, required_gb)
            return model_name, "cuda"

    # Fallback: smallest model on CPU
    model_name = "facebook/nllb-200-distilled-600M"
    logger.warning("Insufficient VRAM (%.1f GB), falling back to %s on CPU", vram, model_name)
    return model_name, "cpu"


def select_device(requested: str = "auto") -> str:
    """Select compute device."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
