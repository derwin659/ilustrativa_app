import os
from fastapi import APIRouter
from models.requests import GenerarImagenRequest
from utils.image_utils import decode_base64
from services.prompt_builder import build_prompt
from services.ip_adapter_service import extract_identity
from services.sd_service import generar_imagen_sd

router = APIRouter()

@router.post("/generar")
def generar(req: GenerarImagenRequest):

    imagen_base = decode_base64(req.imagenBase64)
    identidad = extract_identity(imagen_base)

    os.makedirs("static", exist_ok=True)
    resultados = {}

    for vista in req.vistas:
        prompt = build_prompt(req, vista)

        imagen = generar_imagen_sd(
            prompt=prompt,
            control_image=imagen_base,
            identity=identidad
        )

        path = f"static/{req.sesionId}_{vista}.png"
        imagen.save(path)
        resultados[vista] = path

    return {
        "sesionId": req.sesionId,
        "imagenes": resultados
    }
