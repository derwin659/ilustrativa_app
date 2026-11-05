import os
import threading
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from services.face_detector import get_face_bbox
from services.image_preparation import prepare_inference_image, preserve_unmasked_pixels, ensure_visible_edit

# =========================
# FLAGS
# =========================
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
ENABLE_VAE_OPT = os.getenv("ENABLE_VAE_OPT", "false").lower() == "true"

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SD_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny")

# Generator se crea por request (evita race conditions)
def make_generator():
    return torch.Generator(device=DEVICE).manual_seed(42)

# =========================
# LORA
# =========================
LORA_MID_FADE_PATH = os.getenv(
    "LORA_MID_FADE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora", "output", "mid_fade_sd15.safetensors")
)
LORA_SCALE = float(os.getenv("LORA_SCALE", "0.85"))

# =========================
# PROMPTS
# =========================
NEGATIVE = (
    "cartoon, anime, painting, illustration, fabric, textile, mesh, pattern, "
    "noise, grain, artifacts, blurry, melted hair, wig, helmet hair, "
    "deformed head, extra ear, extra face, watermark, text, logo"
)

NEGATIVE_ZERO = (
    "long hair, thick hair, beard, mustache, "
    "dark hair shadow, dirty skin, "
    "purple, magenta, violet, "
    "artifacts, noise, blotches"
)

# =========================
# GLOBALS (LAZY)
# =========================
pipe_inpaint = None
pipe_cn_inpaint = None
controlnet = None

PIPE_LOCK = threading.Lock()
PIPE_LORA_READY = set()

# =========================
# FACE ANALYSIS (CPU)
# =========================
# =========================
# PIPELINE UTILS
# =========================
def _apply_stable_settings(pipe):
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        # SDPA is already provided by modern PyTorch; slicing serializes attention.
        if os.getenv("ENABLE_ATTENTION_SLICING", "false").lower() == "true":
            pipe.enable_attention_slicing()

        if ENABLE_VAE_OPT and hasattr(pipe, "vae"):
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass

    return pipe


def get_pipe_inpaint():
    global pipe_inpaint
    if pipe_inpaint:
        return pipe_inpaint

    with PIPE_LOCK:
        if pipe_inpaint is None:
            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                SD_MODEL,
                local_files_only=os.getenv("MODEL_LOCAL_FILES_ONLY", "false").lower() == "true",
                torch_dtype=DTYPE,
                safety_checker=None,
            ).to(DEVICE)

            pipe_inpaint = _apply_stable_settings(pipe_inpaint)

    return pipe_inpaint


def get_pipe_cn_inpaint():
    global pipe_cn_inpaint, controlnet
    if pipe_cn_inpaint:
        return pipe_cn_inpaint

    with PIPE_LOCK:
        if pipe_cn_inpaint is None:
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL,
                local_files_only=os.getenv("MODEL_LOCAL_FILES_ONLY", "false").lower() == "true",
                torch_dtype=DTYPE,
            ).to(DEVICE)

            pipe_cn_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                SD_MODEL,
                controlnet=controlnet,
                local_files_only=os.getenv("MODEL_LOCAL_FILES_ONLY", "false").lower() == "true",
                torch_dtype=DTYPE,
                safety_checker=None,
            ).to(DEVICE)

            pipe_cn_inpaint = _apply_stable_settings(pipe_cn_inpaint)

    return pipe_cn_inpaint


# =========================
# LORA
# =========================
def aplicar_lora(pipe, key: str):
    if key in PIPE_LORA_READY:
        return

    if not os.path.exists(LORA_MID_FADE_PATH):
        if os.getenv("REQUIRE_LORA", "false").lower() == "true":
            raise RuntimeError("Missing trained LoRA at LORA_MID_FADE_PATH")
        return

    pipe.load_lora_weights(
        os.path.dirname(LORA_MID_FADE_PATH),
        weight_name=os.path.basename(LORA_MID_FADE_PATH),
    )

    pipe.fuse_lora(lora_scale=LORA_SCALE)

    PIPE_LORA_READY.add(key)

# =========================
# MASKS
# =========================
def frontal_mask(img, bbox):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    if not bbox:
        return mask
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    d = ImageDraw.Draw(mask)
    d.ellipse([max(0, x1 - .18*bw), max(0, y1 - .65*bh), min(W-1, x2 + .18*bw), min(H-1, y1 + .45*bh)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(5))
    # Hard exclusion AFTER feathering preserves eyes, glasses, nose and mouth.
    ImageDraw.Draw(mask).rectangle([x1, y1 + .08*bh, x2, y2], fill=0)
    return mask


def lateral_mask(img):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    # Framing prior, not semantic hair segmentation. Guided capture is required.
    ImageDraw.Draw(mask).ellipse([W*.18, H*.06, W*.82, H*.57], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(5))


def trasera_mask(img):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse([W*.16, H*.07, W*.84, H*.75], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(5))


# =========================
# GENERACION
# =========================
def generar_sd(prompt: str, base_image: Image.Image, vista: str, corte_tipo: str):
    if DEVICE != "cuda":
        raise RuntimeError("CUDA GPU is required for image generation")

    vista = vista.lower()
    if vista not in {"frontal", "lateral", "trasera"}:
        raise ValueError("Unsupported generation view")
    original = base_image.convert("RGB")
    base_image = prepare_inference_image(original)
    bbox = get_face_bbox(base_image) if vista in {"frontal", "trasera"} else None
    if vista == "frontal" and bbox is None:
        raise ValueError("No se detecto un rostro frontal unico. Mira a la camara y mejora la iluminacion.")
    if vista == "trasera" and bbox is not None:
        raise ValueError("La foto trasera muestra un rostro frontal. Captura la parte posterior de la cabeza.")

    pipe = get_pipe_inpaint()
    if corte_tipo.upper() == "MID_FADE":
        aplicar_lora(pipe, "mid_fade")
    if vista == "frontal":
        mask = frontal_mask(base_image, bbox)
        suffix, negative, strength = ", natural hairline, soft barber lineup", NEGATIVE, 0.68
    elif vista == "lateral":
        mask = lateral_mask(base_image)
        suffix, negative, strength = ", side profile, natural fade transition", NEGATIVE, 0.78
    else:
        mask = trasera_mask(base_image)
        suffix, negative, strength = ", rear view, nape visible, no face visible", NEGATIVE, 0.78

    generated = pipe(
        prompt=prompt + suffix,
        negative_prompt=negative,
        image=base_image,
        mask_image=mask,
        strength=strength,
        guidance_scale=7,
        num_inference_steps=24,
        generator=make_generator(),
    ).images[0]
    ensure_visible_edit(base_image, generated, mask)
    result = preserve_unmasked_pixels(original, generated, mask)
    return result
