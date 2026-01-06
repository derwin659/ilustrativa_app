import os
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from services.sd_services import generar_sd
from services.prompt_builder import build_prompt

app = FastAPI()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# =========================
# MODELOS
# =========================
class Corte(BaseModel):
    nombre: str
    tipo: str  # ej: "mid_fade"

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

# =========================
# ENDPOINT
# =========================
@app.post("/generar")
def generar(req: RequestGenerar):
    resultados = {}
    corte_tipo = req.corte.tipo

    # ✅ Recomendación realista:
    # - Para generar "frontal", necesitas SI o SI la imagen frontal.
    # - Para lateral/trasera, lo ideal es su propia foto real.
    #   (si no la mandas, el resultado será menos fiel si intentas inventar el ángulo)

    for vista in req.vistas:
        vista_norm = MAPEO_VISTAS.get(vista)
        if not vista_norm:
            raise HTTPException(status_code=400, detail=f"Vista inválida: {vista}")

        imagen_b64 = req.imagenes.get(vista_norm)
        if not imagen_b64:
            raise HTTPException(status_code=400, detail=f"No se recibió imagen para '{vista_norm}'")

        imagen_base = base64_to_pil(imagen_b64)

        # Prompt específico por vista (tu builder decide)
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

        # Guardar
        nombre_archivo = f"{req.sesionId}_{vista_norm}.png"
        ruta = os.path.join(STATIC_DIR, nombre_archivo)
        imagen_resultado.save(ruta)

        resultados[vista_norm] = f"/static/{nombre_archivo}"

    return {"sesionId": req.sesionId, "imagenes": resultados}
