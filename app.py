import os
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List

from services.sd_services import generar_sd
from services.prompt_builder import build_prompt

app = FastAPI()

# =========================
# STATIC (archivos públicos)
# =========================
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Monta /static para servir imágenes
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Base URL pública (PRODUCCIÓN)
# Ejemplos:
#   PUBLIC_BASE_URL=https://api.gods-tech.ai
#   PUBLIC_BASE_URL=https://<tu-ip>:8003   (si expones puerto directo)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


# =========================
# MODELOS
# =========================
class Corte(BaseModel):
    nombre: str
    tipo: str  # ej: "MID_FADE"

class Tinte(BaseModel):
    aplicar: bool
    color: str | None = None

class Ondulado(BaseModel):
    aplicar: bool
    tipo: str | None = None

class RequestGenerar(BaseModel):
    sesionId: str
    imagenes: Dict[str, str]   # frontal / lateral / trasera (base64)
    corte: Corte
    tinte: Tinte | None = None
    ondulado: Ondulado | None = None
    vistas: List[str]          # ["frontal","lateral","trasera"]

MAPEO_VISTAS = {"frontal": "frontal", "lateral": "lateral", "trasera": "trasera"}


# =========================
# UTILS
# =========================
def base64_to_pil(base64_str: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(base64_str)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")


def get_public_base_url(request: Request) -> str:
    """
    Devuelve la base URL pública.
    Prioridad:
      1) PUBLIC_BASE_URL (env) -> recomendado para producción
      2) Construir con headers del request (soporta proxys)
    """
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL

    # Si estás detrás de proxy (Nginx/Cloudflare), suelen venir estos headers
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}".rstrip("/")


# =========================
# ENDPOINT
# =========================
@app.post("/generar")
def generar(req: RequestGenerar, request: Request):
    resultados = {}
    corte_tipo = req.corte.tipo

    base_url = get_public_base_url(request)

    for vista in req.vistas:
        vista_norm = MAPEO_VISTAS.get(vista)
        if not vista_norm:
            raise HTTPException(status_code=400, detail=f"Vista inválida: {vista}")

        imagen_b64 = req.imagenes.get(vista_norm)
        if not imagen_b64:
            raise HTTPException(status_code=400, detail=f"No se recibió imagen para '{vista_norm}'")

        imagen_base = base64_to_pil(imagen_b64)

        # Prompt específico por vista
        prompt = build_prompt(req, vista_norm)

        try:
            imagen_resultado = generar_sd(
                prompt=prompt,
                base_image=imagen_base,
                vista=vista_norm,      # frontal | lateral | trasera
                corte_tipo=corte_tipo
            )
        except Exception as e:
            print("⚠️ Error generación:", e)
            imagen_resultado = imagen_base

        # Guardar en /static
        nombre_archivo = f"{req.sesionId}_{vista_norm}.png"
        ruta = os.path.join(STATIC_DIR, nombre_archivo)
        imagen_resultado.save(ruta)

        # ✅ URL ABSOLUTA para el frontend
        resultados[vista_norm] = f"{base_url}/static/{nombre_archivo}"

    return {"sesionId": req.sesionId, "imagenes": resultados}
