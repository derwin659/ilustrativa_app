import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

from insightface.app import FaceAnalysis

# =========================
# FLAGS
# =========================
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SD_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny")

PIPE_LORA_READY = {
    "frontal": False,
    "side": False
}

GENERATOR = torch.Generator(device=DEVICE).manual_seed(42)

LORA_MID_FADE_PATH = os.getenv(
    "LORA_MID_FADE_PATH",
    "/workspace/ilustrativa_app/loras/mid_fade_sd15.safetensors"
)
LORA_SCALE = float(os.getenv("LORA_SCALE", "0.85"))

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
# FACE ANALYSIS
# =========================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

# =========================
# CONTROLNET
# =========================
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

# =========================
# PIPELINES
# =========================
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

pipe_cn_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    SD_MODEL,
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

for p in (pipe_inpaint, pipe_cn_inpaint):
    p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
    if DEVICE == "cuda":
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

# =========================
# LORA MANAGEMENT
# =========================
def _load_lora_once(pipe, lora_path: str, scale: float):
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA no encontrado: {lora_path}")

    if getattr(pipe, "_gods_lora_loaded", None) == lora_path:
        return

    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    pipe.load_lora_weights(os.path.dirname(lora_path),
                           weight_name=os.path.basename(lora_path))

    try:
        pipe.fuse_lora(lora_scale=0.65)
    except Exception:
        pass

    pipe._gods_lora_loaded = lora_path


def aplicar_lora_por_corte(pipe, corte_tipo: str, key: str):
    if PIPE_LORA_READY[key]:
        return

    if corte_tipo.lower() in ("mid_fade", "fade_mid", "midfade"):
        _load_lora_once(pipe, LORA_MID_FADE_PATH, LORA_SCALE)
        PIPE_LORA_READY[key] = True

# =========================
# UTILS
# =========================
def make_canny(pil_img: Image.Image):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 140)
    return Image.fromarray(np.stack([edges]*3, axis=-1))


def _get_main_face_bbox(pil_img: Image.Image):
    faces = face_app.get(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return [int(v) for v in face.bbox]

# =========================
# MASKS
# =========================
def build_frontal_hair_mask(img: Image.Image):
    W, H = img.size
    bbox = _get_main_face_bbox(img)

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    if bbox:
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        draw.rectangle([0, max(0, y1 - int(0.6 * bh)), W, int(H * 0.85)], fill=255)
        draw.rectangle([x1, y1 + int(0.15 * bh), x2, y2], fill=0)
    else:
        draw.rectangle([0, 0, W, int(H * 0.55)], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(14))


def build_mid_fade_mask_lateral(img: Image.Image):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    # zona fade
    draw.rectangle([
    int(W * 0.30),   # más adentro
    int(H * 0.15),   # más arriba
    W,
    H               # hasta abajo completo
    ], fill=255)


    # BLOQUEAR barba / mandíbula
    draw.rectangle([0, int(H * 0.70), W, H], fill=0)

    return mask.filter(ImageFilter.GaussianBlur(14))


def build_mid_fade_mask_trasera(pil_img: Image.Image):
    W, H = pil_img.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, int(H * 0.25), W, H], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(16))
    
# def build_mid_fade_mask_trasera(img: Image.Image):
#     """
#     FUNCIÓN DESACTIVADA
#
#     Esta función construía una máscara para la vista TRASERA
#     de un corte MID FADE usando Pillow (PIL).
#
#     Actualmente está totalmente comentada para que:
#     - NO se ejecute
#     - NO interfiera con el pipeline
#     - Sirva solo como documentación / referencia
#     """

#     # ==========================================================
#     # OBTENER DIMENSIONES DE LA IMAGEN
#     # ==========================================================
#     # W = ancho de la imagen
#     # H = alto de la imagen
#     W, H = img.size

#     # ==========================================================
#     # CREAR MÁSCARA BASE (ESCALA DE GRISES)
#     # ==========================================================
#     # "L" = grayscale
#     # 0   = negro (zona bloqueada, no editable)
#     # 255 = blanco (zona editable por SD Inpaint)
#     mask = Image.new("L", (W, H), 0)

#     # Objeto para dibujar formas sobre la máscara
#     draw = ImageDraw.Draw(mask)

#     # ==========================================================
#     # ZERO MÁS ALTO EN EL CENTRO (MID FADE CORE)
#     # ==========================================================
#     # Este rectángulo representaba:
#     # - El rapado a piel (zero)
#     # - Elevado al centro para simular un mid fade real
#     # - Evita que parezca low fade
#     draw.rectangle(
#         [
#             int(W * 0.38),  # límite izquierdo del área central
#             int(H * 0.28),  # altura del zero (subido)
#             int(W * 0.62),  # límite derecho del área central
#             int(H * 0.50),  # hasta mitad de la cabeza
#         ],
#         fill=255           # blanco = editable
#     )

#     # ==========================================================
#     # ZONA DE TRANSICIÓN INFERIOR
#     # ==========================================================
#     # Esta zona suavizaba la transición:
#     # zero → mid → cabello
#     # Sin esto el corte se veía duro y falso
#     draw.rectangle(
#         [
#             int(W * 0.30),  # más ancho que el zero
#             int(H * 0.45),  # inicia donde termina el zero
#             int(W * 0.70),
#             int(H * 0.65),
#         ],
#         fill=255
#     )

#     # ==========================================================
#     # BLOQUEO DE OREJAS (LADOS)
#     # ==========================================================
#     # Evitaba que la IA:
#     # - invente cabello en orejas
#     # - genere deformaciones laterales
#
#     # Lado izquierdo
#     draw.rectangle(
#         [0, 0, int(W * 0.28), H],
#         fill=0
#     )
#
#     # Lado derecho
#     draw.rectangle(
#         [int(W * 0.72), 0, W, H],
#         fill=0
#     )

#     # ==========================================================
#     # BLOQUEO DE CUELLO / NUCA
#     # ==========================================================
#     # Zona crítica:
#     # - evita manchas marrones
#     # - evita cabello falso en piel
#     # - mantiene la nuca limpia
#     draw.rectangle(
#         [0, int(H * 0.78), W, H],
#         fill=0
#     )

#     # ==========================================================
#     # DESENFOQUE GAUSSIANO FINAL
#     # ==========================================================
#     # Suavizaba los bordes de la máscara
#     # Permitía un degradado natural
#     # 18 era un valor equilibrado para vista trasera
#     return mask.filter(ImageFilter.GaussianBlur(18))



def generar_sd(prompt, base_image, vista, corte_tipo):

    if not USE_GPU:
        print("⚠️ GPU DESACTIVADA (DEV)")
        return base_image

    vista = vista.lower()
    corte_tipo = corte_tipo.lower()

    # =========================
    # FRONTAL → INPAINT LIMPIO
    # =========================
    if vista == "frontal":

        aplicar_lora_por_corte(pipe_inpaint, corte_tipo, "frontal")

        return pipe_inpaint(
            generator=GENERATOR,
            prompt=prompt + ", natural hairline, soft barber lineup",
            negative_prompt=NEGATIVE + ", hard hairline, straight hairline",
            image=base_image,
            mask_image=build_frontal_hair_mask(base_image),
            strength=0.45,                 # 🔒 no subir más
            guidance_scale=7.0,
            num_inference_steps=30,
        ).images[0]

    # =========================
    # LATERAL → ZERO / MID FADE REAL (SIN CONTROLNET)
    # =========================
    if vista == "lateral" and corte_tipo in ("mid_fade", "fade_mid", "midfade"):

        aplicar_lora_por_corte(pipe_inpaint, corte_tipo, "side")

        return pipe_inpaint(
            generator=GENERATOR,
            prompt=prompt + ", side view, professional skin fade, shaved to the skin, skin fade to zero",
            negative_prompt=NEGATIVE_ZERO,
            image=base_image,
            mask_image=build_mid_fade_mask_lateral(base_image),
            strength=0.75,                 # 🔑 clave para zero fade
            guidance_scale=7.0,
            num_inference_steps=30,
        ).images[0]

    # =========================
    # TRASERA → ZERO FADE REAL (SIN CONTROLNET)
    # =========================
    if vista == "trasera" and corte_tipo in ("mid_fade", "fade_mid", "midfade", "MID_FADE"):

        aplicar_lora_por_corte(pipe_inpaint, corte_tipo, "side")

        return pipe_inpaint(
            generator=GENERATOR,
            prompt=prompt + ", back view of a professional mid fade haircut, clean skin fade on the nape, slightly textured short hair on the crown",
            negative_prompt=NEGATIVE_ZERO + ", background, wall, mirror, chair",
            image=base_image,
            mask_image=build_mid_fade_mask_trasera(base_image),
            strength=0.75,                  # 🔒 trasera necesita menos
            guidance_scale=7.0,
            num_inference_steps=30,
        ).images[0]

    # =========================
    # OTROS CORTES → CONTROLNET
    # =========================
    aplicar_lora_por_corte(pipe_cn_inpaint, corte_tipo, "side")

    mask = (
        build_mid_fade_mask_lateral(base_image)
        if vista == "lateral"
        else build_mid_fade_mask_trasera(base_image)
    )

    return pipe_cn_inpaint(
        generator=GENERATOR,
        prompt=prompt,
        negative_prompt=NEGATIVE,
        image=base_image,
        mask_image=mask,
        control_image=make_canny(base_image),
        strength=0.65,
        guidance_scale=7.2,
        num_inference_steps=28,
    ).images[0]
