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

from recommendations import recommend

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
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(120, 120))
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


def hair_features(image: np.ndarray, face: tuple[int, int, int, int]) -> tuple[str, str, bool]:
    x, y, w, h = face
    top = image[max(0, y - int(h * .32)):y + int(h * .22), max(0, x):min(image.shape[1], x + w)]
    if top.size == 0:
        return "media", "sin_largo", False
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    edges = float((cv2.Canny(gray, 60, 140) > 0).mean())
    density = "alta" if edges > .19 else "media" if edges > .10 else "baja"
    texture = "ondulado" if contrast > 52 and edges > .14 else "lacio"
    return density, texture, density != "baja"


@app.get("/health")
def health():
    return {"status": "ok", "service": "ia-analitica", "version": "geometry-v1"}


@app.post("/analizar")
def analizar(request: AnalizarRequest):
    started = time.perf_counter()
    try:
        image = decode_image(request.imagen_base64)
        principal, alternativa, confianza, face = face_geometry(image)
        densidad, textura, ondulado_apto = hair_features(image, face)
        cuts, tints = recommend(principal, densidad)
        return {
            "forma_rostro": {"principal": principal, "alternativa": alternativa, "confianza": confianza},
            "cabello": {"densidad": densidad, "ondulado": {"apto": ondulado_apto, "tipo": textura}},
            "recomendaciones": {"top_recomendado": cuts[0], "cortes": cuts, "tintes": tints},
            "meta": {"versionModelo": "ia-analitica-geometry-v1", "procesadoMs": round((time.perf_counter() - started) * 1000)},
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
