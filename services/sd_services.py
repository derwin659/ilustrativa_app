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

from insightface.app import FaceAnalysis

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
    "/workspace/ilustrativa_app/loras/mid_fade_sd15.safetensors"
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
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1)

# =========================
# PIPELINE UTILS
# =========================
def _apply_stable_settings(pipe):
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
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
                torch_dtype=DTYPE,
            ).to(DEVICE)

            pipe_cn_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                SD_MODEL,
                controlnet=controlnet,
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
        return

    pipe.load_lora_weights(
        os.path.dirname(LORA_MID_FADE_PATH),
        weight_name=os.path.basename(LORA_MID_FADE_PATH),
    )

    try:
        pipe.fuse_lora(lora_scale=LORA_SCALE)
    except Exception:
        pass

    PIPE_LORA_READY.add(key)

# =========================
# FACE BBOX CACHE
# =========================
def get_face_bbox(img: Image.Image):
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_np)

    if not faces:
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return [int(v) for v in face.bbox]

# =========================
# MASKS
# =========================
def frontal_mask(img, bbox):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)

    if bbox:
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        d.rectangle([0, max(0, y1 - int(0.6 * bh)), W, int(H * 0.85)], fill=255)
        d.rectangle([x1, y1 + int(0.15 * bh), x2, y2], fill=0)
    else:
        d.rectangle([0, 0, W, int(H * 0.55)], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(14))


def lateral_mask(img):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([int(W * 0.3), int(H * 0.15), W, H], fill=255)
    d.rectangle([0, int(H * 0.7), W, H], fill=0)
    return mask.filter(ImageFilter.GaussianBlur(14))


def trasera_mask(img):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([0, int(H * 0.25), W, H], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(16))


# =========================
# GENERACION
# =========================
def generar_sd(prompt: str, base_image: Image.Image, vista: str, corte_tipo: str):
    if DEVICE != "cuda":
        return base_image

    vista = vista.lower()
    corte_tipo = corte_tipo.lower()
    generator = make_generator()

    bbox = get_face_bbox(base_image)

    pipe = get_pipe_inpaint()
    aplicar_lora(pipe, "mid_fade")

    if vista == "frontal":
        return pipe(
            prompt=prompt + ", natural hairline, soft barber lineup",
            negative_prompt=NEGATIVE,
            image=base_image,
            mask_image=frontal_mask(base_image, bbox),
            strength=0.45,
            guidance_scale=7,
            num_inference_steps=28,
            generator=generator
        ).images[0]

    if vista == "lateral":
        return pipe(
            prompt=prompt + ", professional mid fade, skin fade to zero",
            negative_prompt=NEGATIVE_ZERO,
            image=base_image,
            mask_image=lateral_mask(base_image),
            strength=0.75,
            guidance_scale=7,
            num_inference_steps=28,
            generator=generator
        ).images[0]

    if vista == "trasera":
        return pipe(
            prompt=prompt + ", clean mid fade on the nape",
            negative_prompt=NEGATIVE_ZERO,
            image=base_image,
            mask_image=trasera_mask(base_image),
            strength=0.75,
            guidance_scale=7,
            num_inference_steps=28,
            generator=generator
        ).images[0]

    return base_image
