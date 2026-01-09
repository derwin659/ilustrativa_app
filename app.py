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
# STATIC
# =========================
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# =========================
# MODELOS
# =========================
class Corte(BaseModel):
    nombre: str
    tipo: str

class Tinte(BaseModel):
    aplicar: bool
    color: str | None = None

class Ondulado(BaseModel):
    aplicar: bool
    tipo: str | None = None

class RequestGenerar(BaseModel):
    sesionId: str
    imagenes: Dict[str, str]   # base64
    corte: Corte
    tinte: Tinte | None = None
    ondulado: Ondulado | None = None
    vistas: List[str]

MAPEO_VISTAS = {
    "frontal": "frontal",
    "lateral": "lateral",
    "trasera": "trasera"
}

# =========================
# UTILS
# =========================
def base64_to_pil(base64_str: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")

def get_public_base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return f"{proto}://{host}".rstrip("/")

# =========================
# ENDPOINT
# =========================
@app.post("/generar")
def generar(req: RequestGenerar, request: Request):
    resultados = {}
    base_url = get_public_base_url(request)
    corte_tipo = req.corte.tipo.lower()

    # =========================
    # 1️⃣ GUARDAR IMÁGENES ANTES
    # =========================
    if "frontal" not in req.imagenes:
        raise HTTPException(status_code=400, detail="Imagen frontal requerida")

    # ANTES FRONTAL
    imagen_antes = base64_to_pil(req.imagenes["frontal"])
    nombre_antes = f"{req.sesionId}_antes.png"
    ruta_antes = os.path.join(STATIC_DIR, nombre_antes)
    imagen_antes.save(ruta_antes)

    resultados["antes"] = f"{base_url}/static/{nombre_antes}"

    # ANTES LATERAL (opcional)
    if "lateral" in req.imagenes:
        img_lat = base64_to_pil(req.imagenes["lateral"])
        nombre_lat = f"{req.sesionId}_anteslateral.png"
        ruta_lat = os.path.join(STATIC_DIR, nombre_lat)
        img_lat.save(ruta_lat)

        resultados["anteslateral"] = f"{base_url}/static/{nombre_lat}"

    # =========================
    # 2️⃣ GENERAR IMÁGENES IA
    # =========================
    for vista in req.vistas:
        vista_norm = MAPEO_VISTAS.get(vista)
        if not vista_norm:
            continue

        imagen_base = base64_to_pil(req.imagenes[vista_norm])
        prompt = build_prompt(req, vista_norm)

        try:
            imagen_resultado = generar_sd(
                prompt=prompt,
                base_image=imagen_base,
                vista=vista_norm,
                corte_tipo=corte_tipo
            )
        except Exception as e:
            print("⚠️ Error generación:", e)
            imagen_resultado = imagen_base

        nombre_archivo = f"{req.sesionId}_{vista_norm}.png"
        ruta = os.path.join(STATIC_DIR, nombre_archivo)
        imagen_resultado.save(ruta)

        resultados[vista_norm] = f"{base_url}/static/{nombre_archivo}"

    # =========================
    # 3️⃣ RESPUESTA FINAL
    # =========================
    return {
        "sesionId": req.sesionId,
        "imagenes": resultados
    }
