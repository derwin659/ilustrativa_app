import os
import base64
from io import BytesIO
from typing import Dict, Any

from PIL import Image
def run_generation(req, base_url: str | None = None, save_files: bool = True):
    from services.sd_services import generar_sd
    from services.prompt_builder import build_prompt
    ...


STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

MAPEO_VISTAS = {
    "frontal": "frontal",
    "lateral": "lateral",
    "trasera": "trasera",
}


def base64_to_pil(base64_str: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Imagen base64 inválida") from e


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image(image: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def run_generation(req, base_url: str | None = None, save_files: bool = True) -> Dict[str, Any]:
    resultados = {}
    corte_tipo = req.corte.tipo.lower()

    if "frontal" not in req.imagenes:
        raise ValueError("Imagen frontal requerida")

    # Guardar / devolver imagen antes frontal
    imagen_antes = base64_to_pil(req.imagenes["frontal"])

    if save_files:
        nombre_antes = f"{req.sesionId}_antes.png"
        ruta_antes = os.path.join(STATIC_DIR, nombre_antes)
        save_image(imagen_antes, ruta_antes)

        if base_url:
            resultados["antes"] = f"{base_url}/static/{nombre_antes}"
        else:
            resultados["antes"] = ruta_antes
    else:
        resultados["antes"] = pil_to_base64(imagen_antes)

    # Guardar / devolver imagen antes lateral si existe
    if "lateral" in req.imagenes:
        img_lat = base64_to_pil(req.imagenes["lateral"])

        if save_files:
            nombre_lat = f"{req.sesionId}_anteslateral.png"
            ruta_lat = os.path.join(STATIC_DIR, nombre_lat)
            save_image(img_lat, ruta_lat)

            if base_url:
                resultados["anteslateral"] = f"{base_url}/static/{nombre_lat}"
            else:
                resultados["anteslateral"] = ruta_lat
        else:
            resultados["anteslateral"] = pil_to_base64(img_lat)

    # Generación por vistas
    for vista in req.vistas:
        vista_norm = MAPEO_VISTAS.get(vista)
        if not vista_norm:
            continue

        if vista_norm not in req.imagenes:
            # si pidieron una vista que no llegó, la saltamos sin romper todo
            continue

        imagen_base = base64_to_pil(req.imagenes[vista_norm])
        prompt = build_prompt(req, vista_norm)

        try:
            imagen_resultado = generar_sd(
                prompt=prompt,
                base_image=imagen_base,
                vista=vista_norm,
                corte_tipo=corte_tipo,
            )
        except Exception as e:
            print(f"⚠️ Error generación en vista {vista_norm}: {e}")
            imagen_resultado = imagen_base

        if save_files:
            nombre_archivo = f"{req.sesionId}_{vista_norm}.png"
            ruta = os.path.join(STATIC_DIR, nombre_archivo)
            save_image(imagen_resultado, ruta)

            if base_url:
                resultados[vista_norm] = f"{base_url}/static/{nombre_archivo}"
            else:
                resultados[vista_norm] = ruta
        else:
            resultados[vista_norm] = pil_to_base64(imagen_resultado)

    return {
        "sesionId": req.sesionId,
        "imagenes": resultados,
    }