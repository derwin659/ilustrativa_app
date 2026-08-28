from pydantic import BaseModel
from typing import List, Optional

class Corte(BaseModel):
    nombre: str
    tipo: str

class Tinte(BaseModel):
    aplicar: bool
    color: Optional[str] = None

class Ondulado(BaseModel):
    aplicar: bool
    tipo: Optional[str] = None

class GenerarImagenRequest(BaseModel):
    sesionId: str
    imagenBase64: str
    vistas: List[str]  # ["frontal", "lateral", "atras"]

    corte: Corte
    tinte: Optional[Tinte] = None
    ondulado: Optional[Ondulado] = None
