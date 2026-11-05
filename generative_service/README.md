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
### Variables del worker Serverless

- `CLOUDINARY_URL`: credencial Cloudinary del worker para guardar resultados autenticados.
- `AI_ALLOWED_IMAGE_HOSTS`: hosts HTTPS separados por coma desde los que se aceptan entradas; por defecto `res.cloudinary.com`.
- `AI_MAX_IMAGE_BYTES`: límite por entrada y salida; por defecto 5 MB.
- `AI_IMAGE_DOWNLOAD_TIMEOUT_SECONDS`: timeout de descarga; por defecto 30 segundos.

El input Serverless conserva `sesionId`, `imagenes`, `corte`, `tinte`, `ondulado` y `vistas`, y agrega `storage.tenantId`. `imagenes` acepta Base64 o URLs HTTPS firmadas. La respuesta devuelve URLs autenticadas en `imagenes` y los identificadores Cloudinary en `assets`.

Las descargas no siguen redirecciones, sólo aceptan hosts autorizados y validan tamaño y formato. Si una generación falla después de subir una vista, el worker elimina los resultados parciales.
## Contenedor Serverless y modelos

Consulte [DEPLOY_SERVERLESS.md](DEPLOY_SERVERLESS.md) para construir la imagen,
configurar el volumen de pesos y validar el endpoint de prueba. DockerfileHandler
usa requirements-serverless.txt; requirements.txt continúa siendo del servicio
FastAPI. El worker requiere GPU, modelos locales y LoRA entrenado antes de aceptar
trabajos. No incluye las fotografías del dataset en la imagen.