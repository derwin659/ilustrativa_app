from __future__ import annotations

import base64
import binascii
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from recommendations import recommend_detailed

app = FastAPI(title="Super Gods IA Analitica", version="1.0.0")


class Contexto(BaseModel):
    tenantId: int | None = None
    sucursalId: int | None = None


class AnalizarRequest(BaseModel):
    imagen_base64: str = Field(min_length=32)
    contexto: Contexto | None = None


def decode_image(value: str) -> np.ndarray:
    raw = value.split(",", 1)[-1].strip()
    try:
        payload = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("La imagen no contiene un Base64 valido") from exc
    if len(payload) > 12 * 1024 * 1024:
        raise ValueError("La imagen supera el limite de 12 MB")
    try:
        pil = Image.open(BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("El archivo enviado no es una imagen valida") from exc
    if min(pil.size) < 240:
        raise ValueError("Acerca el rostro y usa una imagen de al menos 240 px")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def face_geometry(image: np.ndarray) -> tuple[str, str, float, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alternative = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    min_side = max(60, min(image.shape[:2]) // 10)
    attempts = (
        (cascade, gray, 1.08, 6),
        (cascade, enhanced, 1.05, 4),
        (alternative, enhanced, 1.05, 3),
    )
    faces = ()
    for detector, source, scale_factor, neighbors in attempts:
        detected = detector.detectMultiScale(
            source,
            scaleFactor=scale_factor,
            minNeighbors=neighbors,
            minSize=(min_side, min_side),
        )
        if len(detected) > 0:
            faces = detected
            break
    if len(faces) == 0:
        raise ValueError("No se detecto un rostro frontal. Mejora la luz y mira a la camara")
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    ratio = h / max(w, 1)
    # Clasificacion geometrica conservadora. La confianza se limita porque Haar no ve la mandibula completa.
    if ratio >= 1.34:
        principal, alternativa = "alargado", "ovalado"
        confidence = min(.82, .62 + (ratio - 1.34))
    elif ratio <= 1.08:
        principal, alternativa = "redondo", "cuadrado"
        confidence = min(.78, .61 + (1.08 - ratio))
    elif ratio <= 1.18:
        principal, alternativa = "cuadrado", "redondo"
        confidence = .68
    else:
        principal, alternativa = "ovalado", "alargado"
        confidence = .72
    return principal, alternativa, round(confidence, 3), (int(x), int(y), int(w), int(h))


def image_quality(image: np.ndarray, face: tuple[int, int, int, int]) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, _, w, h = face
    face_coverage = (w * h) / float(image.shape[0] * image.shape[1])
    warnings = []
    if brightness < 55:
        warnings.append("iluminacion_baja")
    elif brightness > 220:
        warnings.append("sobreexposicion")
    if sharpness < 45:
        warnings.append("imagen_borrosa")
    if face_coverage < .035:
        warnings.append("rostro_alejado")
    return {"apta": not warnings, "advertencias": warnings, "brillo": round(brightness, 1), "nitidez": round(sharpness, 1), "cobertura_rostro": round(face_coverage, 3)}


def hair_features(image: np.ndarray, face: tuple[int, int, int, int]) -> dict:
    x, y, w, h = face
    top = image[max(0, y - int(h * .32)):y + int(h * .22), max(0, x):min(image.shape[1], x + w)]
    if top.size == 0:
        return {"densidad": "media", "textura": "no_determinada", "largo": "no_determinado", "confianza": .0}
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    edges = float((cv2.Canny(gray, 60, 140) > 0).mean())
    density = "alta" if edges > .19 else "media" if edges > .10 else "baja"
    texture = "rizado" if edges > .24 and contrast > 58 else "ondulado" if contrast > 48 and edges > .13 else "lacio"
    hair_height = max(0, y - max(0, y - int(h * .32))) / max(h, 1)
    length = "medio" if hair_height > .24 and edges > .12 else "corto"
    confidence = round(min(.78, .48 + min(edges, .25) + min(contrast / 400, .15)), 2)
    return {"densidad": density, "textura": texture, "largo": length, "confianza": confidence}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ia-analitica", "version": "geometry-v1"}


@app.post("/analizar")
def analizar(request: AnalizarRequest):
    started = time.perf_counter()
    try:
        image = decode_image(request.imagen_base64)
        principal, alternativa, confianza, face = face_geometry(image)
        quality = image_quality(image, face)
        hair = hair_features(image, face)
        cuts, tint_services = recommend_detailed(principal, hair["densidad"], hair["textura"], hair["largo"])
        legacy_cuts = cuts[:3]
        legacy_tints = [service["name"] for service in tint_services[:3]]
        ondulado_apto = hair["densidad"] != "baja" and hair["largo"] in {"medio", "largo"}
        return {
            "forma_rostro": {"principal": principal, "alternativa": alternativa, "confianza": confianza},
            "cabello": {"densidad": hair["densidad"], "textura": hair["textura"], "largo": hair["largo"], "confianza": hair["confianza"], "ondulado": {"apto": ondulado_apto, "tipo": hair["textura"]}},
            "recomendaciones": {"top_recomendado": legacy_cuts[0], "cortes": legacy_cuts, "tintes": legacy_tints},
            "analisis_v2": {
                "calidad_captura": quality,
                "rasgos_capilares": hair,
                "cortes": cuts,
                "servicios_tinte": tint_services,
                "aviso_quimicos": "La foto solo orienta. Tinte, decoloracion u ondulado requieren evaluacion profesional y prueba de sensibilidad.",
            },
            "meta": {"versionModelo": "ia-analitica-hybrid-v2", "catalogoVersion": "2026.08", "procesadoMs": round((time.perf_counter() - started) * 1000)},
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
