from pydantic import BaseModel
from typing import Dict, List


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
    imagenes: Dict[str, str]
    corte: Corte
    tinte: Tinte | None = None
    ondulado: Ondulado | None = None
    vistas: List[str]