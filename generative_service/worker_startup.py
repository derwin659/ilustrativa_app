"""Validate deployment resources before accepting jobs; no downloads or credentials in logs."""
import os
from pathlib import Path


def validate_model_files():
    for name in ("SD_MODEL", "CONTROLNET_MODEL"):
        path = Path(os.environ.get(name, ""))
        if not os.environ.get(name) or not path.is_dir():
            raise RuntimeError(f"{name}: configure an existing local model directory")
        expected = "model_index.json" if name == "SD_MODEL" else "config.json"
        if not (path / expected).is_file():
            raise RuntimeError(f"{name}: missing {expected}")
    if os.environ.get("REQUIRE_LORA", "true").lower() == "true":
        lora = Path(os.environ.get("LORA_MID_FADE_PATH", ""))
        if not lora.is_file() or lora.stat().st_size == 0:
            raise RuntimeError("LORA_MID_FADE_PATH: trained LoRA weights are missing")


def initialize_worker():
    import torch
    if os.environ.get("REQUIRE_GPU", "true").lower() == "true" and not torch.cuda.is_available():
        raise RuntimeError("This worker requires an available CUDA GPU")
    validate_model_files()
    from services.face_detector import initialize_face_detector
    initialize_face_detector()
    if not os.environ.get("CLOUDINARY_URL"):
        raise RuntimeError("CLOUDINARY_URL is required for storing results")
    # Import and warm the pipeline once before RunPod accepts requests.
    from services.sd_services import get_pipe_inpaint, aplicar_lora
    aplicar_lora(get_pipe_inpaint(), "mid_fade")