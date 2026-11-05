from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class Corte(BaseModel):
    nombre: str
    tipo: str


class Tinte(BaseModel):
    aplicar: bool
    color: str | None = None


class Ondulado(BaseModel):
    aplicar: bool
    tipo: str | None = None


class StorageContext(BaseModel):
    tenantId: int = Field(gt=0)


class RequestGenerar(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sesionId: str
    imagenes: Dict[str, str]
    corte: Corte
    tinte: Tinte | None = None
    ondulado: Ondulado | None = None
    vistas: List[str]
    storage: StorageContext | None = None