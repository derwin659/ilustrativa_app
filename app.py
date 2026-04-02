import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List

from services.generation_runner import run_generation

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


# =========================
# UTILS
# =========================
def get_public_base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return f"{proto}://{host}".rstrip("/")


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generar")
def generar(req: RequestGenerar, request: Request):
    try:
        base_url = get_public_base_url(request)
        return run_generation(req=req, base_url=base_url, save_files=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")