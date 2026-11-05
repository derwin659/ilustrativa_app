import base64
import os
import re
import uuid
from io import BytesIO
from urllib.parse import urlparse

import cloudinary
import cloudinary.uploader
import requests
from cloudinary import CloudinaryImage
from PIL import Image, ImageOps

MAX_IMAGE_BYTES = int(os.getenv("AI_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("AI_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "30"))
DEFAULT_ALLOWED_HOSTS = "res.cloudinary.com"


def load_input_image(value: str) -> Image.Image:
    if not value or not value.strip():
        raise ValueError("Imagen vacía")

    normalized = value.strip()
    if normalized.startswith(("https://", "http://")):
        image_bytes = _download_image(normalized)
    else:
        image_bytes = _decode_base64(normalized)
    return _open_image(image_bytes)


def upload_output_image(image: Image.Image, tenant_id: int, session_id: str, view: str) -> dict:
    if not os.getenv("CLOUDINARY_URL"):
        raise RuntimeError("CLOUDINARY_URL es obligatorio para almacenar resultados Serverless")

    cloudinary.config(secure=True)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen generada supera el tamaño máximo permitido")

    safe_session = _safe_segment(session_id, "session")
    safe_view = _safe_segment(view, "view")
    public_id = f"{safe_view}_{uuid.uuid4().hex}"
    result = cloudinary.uploader.upload(
        payload,
        folder=f"super-gods/tenants/{tenant_id}/ai/outputs/{safe_session}",
        public_id=public_id,
        resource_type="image",
        type="authenticated",
        overwrite=False,
    )
    stored_public_id = result["public_id"]
    signed_url = CloudinaryImage(stored_public_id).build_url(
        resource_type="image",
        type="authenticated",
        sign_url=True,
        secure=True,
        format=result.get("format", "png"),
    )
    return {"url": signed_url, "publicId": stored_public_id}


def delete_output_assets(public_ids) -> None:
    for public_id in public_ids:
        try:
            cloudinary.uploader.destroy(
                public_id,
                resource_type="image",
                type="authenticated",
                invalidate=True,
            )
        except Exception:
            pass

def _download_image(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("La URL de imagen no es HTTPS válida")

    allowed_hosts = {
        host.strip().lower()
        for host in os.getenv("AI_ALLOWED_IMAGE_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",")
        if host.strip()
    }
    hostname = parsed.hostname.lower()
    if hostname not in allowed_hosts:
        raise ValueError("El host de la imagen no está autorizado")

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, allow_redirects=False) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_IMAGE_BYTES:
            raise ValueError("La imagen remota supera el tamaño máximo permitido")

        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("La imagen remota supera el tamaño máximo permitido")
            chunks.append(chunk)
        return b"".join(chunks)


def _decode_base64(value: str) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception as error:
        raise ValueError("Imagen base64 inválida") from error
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el tamaño máximo permitido")
    return decoded


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Formato de imagen no permitido")
        image.load()
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.info.pop("exif", None)
        return normalized
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("El contenido recibido no es una imagen válida") from error


def _safe_segment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", value or "")[:80]
    return normalized or fallback