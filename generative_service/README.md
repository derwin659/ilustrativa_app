# Super Gods IA Generativa

Servicio GPU que genera vistas ilustrativas del corte elegido después del análisis.

## FastAPI

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /generar`

## RunPod Serverless

El worker se inicia con:

```text
python handler.py
```

Construya usando `DockerfileHandler` con `generative_service` como contexto. Para FastAPI use `Dockerfile` con la misma carpeta como contexto.

## Contrato de cortes

Los valores de `corte.tipo` deben coincidir con `services/prompt_builder.py`, por ejemplo `MID_FADE`, `LOW_FADE`, `TAPER`, `BUZZ` o `FADE_MODERNO`.
