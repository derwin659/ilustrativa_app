# Super Gods IA Analítica

Microservicio Python compatible con `IaAnaliticaClient` del backend Spring.

## Capacidades actuales

- Valida Base64, formato, peso y resolución mínima.
- Detecta un rostro frontal y rechaza fotografías no aptas.
- Estima forma principal y alternativa con confianza conservadora.
- Estima densidad/textura capilar para orientar opciones.
- Devuelve cortes cuyos códigos existen en la IA generativa.
- Nunca sustituye errores por recomendaciones simuladas.

## Ejecutar localmente

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Configure Spring con:

```text
IA_ANALITICA_URL=http://host:8000
```

## API

- `GET /health`
- `POST /analizar`

Entrada:

```json
{
  "imagen_base64": "...",
  "contexto": {"tenantId": 73, "sucursalId": 1}
}
```

La salida conserva `forma_rostro`, `cabello`, `recomendaciones` y `meta`, exactamente como esperan los DTO del backend.

## Alcance de geometry-v1

Esta primera versión usa detección y geometría visual conservadora. No es identificación biométrica ni diagnóstico médico. Antes de producción debe calibrarse con fotografías autorizadas y etiquetas de especialistas; el contrato HTTP no cambiará cuando se sustituya por landmarks/modelo entrenado.

## Pruebas

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
