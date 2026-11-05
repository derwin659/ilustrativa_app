import base64
import os
import logging
import time
from io import BytesIO
from typing import Any, Dict

from PIL import Image

from services.image_transport import load_input_image
from services.prompt_builder import build_prompt
from services.sd_services import generar_sd

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

MAPEO_VISTAS = {
    "frontal": "frontal",
    "lateral": "lateral",
    "trasera": "trasera",
}


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image(image: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def serialize_image(image, name, req, base_url, save_files, output_uploader):
    if save_files:
        filename = f"{req.sesionId}_{name}.png"
        path = os.path.join(STATIC_DIR, filename)
        save_image(image, path)
        return f"{base_url}/static/{filename}" if base_url else path
    if output_uploader:
        return output_uploader(image, name)
    return pil_to_base64(image)


def run_generation(
    req,
    base_url: str | None = None,
    save_files: bool = True,
    output_uploader=None,
) -> Dict[str, Any]:
    if not req.vistas:
        raise ValueError("Se requiere al menos una vista")
    resultados = {}
    corte_tipo = req.corte.tipo.lower()

    if "frontal" not in req.imagenes:
        raise ValueError("Imagen frontal requerida")

    loaded = {view: load_input_image(source) for view, source in req.imagenes.items()}
    imagen_antes = loaded["frontal"]
    resultados["antes"] = serialize_image(
        imagen_antes, "antes", req, base_url, save_files, output_uploader
    )

    if "lateral" in req.imagenes:
        imagen_lateral = loaded["lateral"]
        resultados["anteslateral"] = serialize_image(
            imagen_lateral, "anteslateral", req, base_url, save_files, output_uploader
        )

    for vista in req.vistas:
        vista_norm = MAPEO_VISTAS.get(vista)
        if not vista_norm or vista_norm not in req.imagenes:
            raise ValueError(f'Vista requerida inválida o ausente: {vista}')

        imagen_base = loaded[vista_norm]
        started = time.perf_counter()
        prompt = build_prompt(req, vista_norm)
        # Never present the original photo as a successfully generated view.
        imagen_resultado = generar_sd(
            prompt=prompt,
            base_image=imagen_base,
            vista=vista_norm,
            corte_tipo=corte_tipo,
        )

        logging.getLogger(__name__).info("generation_view=%s elapsed_seconds=%.2f", vista_norm, time.perf_counter() - started)
        resultados[vista_norm] = serialize_image(
            imagen_resultado, vista_norm, req, base_url, save_files, output_uploader
        )

    return {"sesionId": req.sesionId, "imagenes": resultados}