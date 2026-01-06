import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

USE_GPU = os.getenv("USE_GPU", "false") == "true"

def generar_sd(...):
    if not USE_GPU:
        print("⚠️ GPU DESACTIVADA (DEV)")
        return base_image

    # código real SD aquí


from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SD_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny")

# ✅ Ruta a tu LoRA (pon aquí tu LoRA del fade)
# Ejemplo: /workspace/loras/mid_fade_sd15.safetensors
LORA_MID_FADE_PATH = os.getenv("LORA_MID_FADE_PATH", "/workspace/loras/mid_fade_sd15.safetensors")
LORA_SCALE = float(os.getenv("LORA_SCALE", "0.85"))

NEGATIVE = (
    "cartoon, anime, painting, illustration, fabric, textile, mesh, pattern, "
    "noise, grain, artifacts, blurry, melted hair, wig, helmet hair, "
    "deformed head, extra ear, extra face, watermark, text, logo"
)

# =========================
# SINGLETONS
# =========================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

# Frontal: INPAINT normal (sin controlnet)
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

pipe_inpaint.scheduler = UniPCMultistepScheduler.from_config(pipe_inpaint.scheduler.config)

# Lateral/Trasera: CONTROLNET INPAINT
pipe_cn_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    SD_MODEL,
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

pipe_cn_inpaint.scheduler = UniPCMultistepScheduler.from_config(pipe_cn_inpaint.scheduler.config)

# xformers (opcional)
if DEVICE == "cuda":
    for p in (pipe_inpaint, pipe_cn_inpaint):
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

# =========================
# LoRA LOADING (una sola vez)
# =========================
def _load_lora_once(pipe, lora_path: str, scale: float):
    """
    Carga y aplica LoRA de forma segura.
    - Si el archivo no existe, no rompe: levanta error claro.
    """
    if not lora_path or not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA no encontrado en: {lora_path}")

    # Evitar recargar si ya está cargado
    if getattr(pipe, "_gods_lora_loaded", None) == lora_path:
        return

    # Limpia loras previas si existían
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    # load_lora_weights acepta:
    # - carpeta con loras o archivo .safetensors
    pipe.load_lora_weights(lora_path)
    try:
        pipe.fuse_lora(lora_scale=scale)
    except Exception:
        # Algunas versiones no traen fuse_lora; igual funciona sin fuse (más lento)
        pass

    pipe._gods_lora_loaded = lora_path


def aplicar_lora_por_corte(pipe, corte_tipo: str):
    """
    Aquí mapeas corte -> LoRA.
    Por ahora: mid_fade usa LORA_MID_FADE_PATH
    """
    if corte_tipo in ("mid_fade", "fade_mid", "midfade"):
        _load_lora_once(pipe, LORA_MID_FADE_PATH, LORA_SCALE)
    else:
        # Si no tienes lora para otros cortes, no aplicar nada.
        # O puedes levantar error si quieres obligarlo.
        pass


# =========================
# UTILS
# =========================
def make_canny(pil_img: Image.Image):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


def _get_main_face_bbox(pil_img: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return [int(v) for v in face.bbox]


# =========================
# MASCARAS
# =========================
def build_frontal_hair_mask(pil_img: Image.Image):
    """
    Máscara frontal SOLO cabello/parte superior.
    Protege rostro (no se toca).
    Heurística basada en bbox del rostro.
    """
    W, H = pil_img.size
    bbox = _get_main_face_bbox(pil_img)
    if bbox is None:
        # fallback: tocar zona superior (pero es peor)
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, 0, W, int(H * 0.55)], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(12))

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Zona de pelo: arriba del rostro + un poco de frente
    hair_top = max(0, int(y1 - 0.55 * bh))
    hair_bottom = max(0, int(y1 + 0.18 * bh))
    hair_left = max(0, int(x1 - 0.25 * bw))
    hair_right = min(W - 1, int(x2 + 0.25 * bw))

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # Pintamos área de cabello
    draw.rectangle([hair_left, hair_top, hair_right, int(H * 0.85)], fill=255)

    # Protegemos cara: quitamos un bloque grande del rostro
    face_protect_top = max(0, int(y1 + 0.10 * bh))
    draw.rectangle([max(0, x1 - 0.10*bw), face_protect_top, min(W-1, x2 + 0.10*bw), min(H-1, y2 + 0.20*bh)], fill=0)

    return mask.filter(ImageFilter.GaussianBlur(14))


def build_mid_fade_mask_lateral(pil_img: Image.Image):
    """
    Máscara lateral: lados / patilla / degradado.
    Usa bbox del rostro como referencia.
    """
    W, H = pil_img.size
    bbox = _get_main_face_bbox(pil_img)
    if bbox is None:
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([int(W*0.35), int(H*0.20), W, H], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(14))

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    x_cut = int(x1 + 0.45 * bw)
    temple_y = int(y1 + 0.18 * bh)
    mid_y = int(y1 + 0.55 * bh)

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # zona derecha (asumiendo lateral derecha; si fuera izquierda invierte según tu cámara)
    draw.polygon([(x_cut, temple_y), (W, temple_y), (W, H), (x_cut, H)], fill=255)

    grad = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        if y < temple_y:
            v = 0
        elif y < mid_y:
            v = 40 + (y - temple_y) / max(1, (mid_y - temple_y)) * 140
        else:
            v = 255
        grad[y, :] = v

    out = np.minimum(np.array(mask), grad)
    return Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))


def build_mid_fade_mask_trasera(pil_img: Image.Image):
    W, H = pil_img.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, int(H * 0.25), W, H], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(16))


# =========================
# GENERACION PRINCIPAL
# =========================
def generar_sd(
    prompt: str,
    base_image: Image.Image,
    vista: str,
    corte_tipo: str,
):
    """
    Genera una vista usando LoRA:
    - frontal: inpaint normal con máscara de cabello
    - lateral/trasera: controlnet inpaint con canny + máscara
    """
    
    vista = vista.lower().strip()

    if vista == "frontal":
        aplicar_lora_por_corte(pipe_inpaint, corte_tipo)
        mask = build_frontal_hair_mask(base_image)

        out = pipe_inpaint(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=base_image,
            mask_image=mask,
            strength=0.65,
            guidance_scale=7.2,
            num_inference_steps=28,
        ).images[0]
        return out

    if vista == "lateral":
        aplicar_lora_por_corte(pipe_cn_inpaint, corte_tipo)
        mask = build_mid_fade_mask_lateral(base_image)
        canny = make_canny(base_image)

        out = pipe_cn_inpaint(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=base_image,
            mask_image=mask,
            control_image=canny,
            strength=0.72,
            guidance_scale=7.2,
            num_inference_steps=28,
        ).images[0]
        return out

    if vista == "trasera":
        aplicar_lora_por_corte(pipe_cn_inpaint, corte_tipo)
        mask = build_mid_fade_mask_trasera(base_image)
        canny = make_canny(base_image)

        out = pipe_cn_inpaint(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=base_image,
            mask_image=mask,
            control_image=canny,
            strength=0.72,
            guidance_scale=7.2,
            num_inference_steps=28,
        ).images[0]
        
    return out
            print("⚠️ GPU DESACTIVADA (DEV)")
    return base_image

