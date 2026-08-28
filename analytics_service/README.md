# Super Gods IA Analítica

Microservicio Python compatible con `IaAnaliticaClient` del backend Spring.

## Capacidades actuales

- Valida Base64, formato, peso y resolución mínima.
- Detecta un rostro frontal y rechaza fotografías no aptas.
- Estima forma principal y alternativa con confianza conservadora.
- Estima densidad/textura capilar para orientar opciones.
- Devuelve cortes cuyos códigos existen en la IA generativa.
- Nunca sustituye errores por recomendaciones simuladas.
- Evalúa brillo, nitidez y tamaño relativo del rostro.
- Estima densidad, textura y largo capilar con una confianza separada.
- Puntúa un catálogo inicial de 26 estilos y explica los factores del ranking.
- Distingue estilos con vista generativa preparada de recomendaciones todavía informativas.
- Marca todo servicio químico como sujeto a evaluación profesional.

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

La salida conserva `forma_rostro`, `cabello`, `recomendaciones` y `meta`, exactamente como esperan los DTO del backend. También incluye `analisis_v2` con calidad de captura, rasgos capilares, ranking ampliado, razones y servicios sugeridos.

## Alcance de hybrid-v2

Esta versión combina detección, geometría visual y reglas explícitas. No es identificación biométrica, diagnóstico médico ni un modelo capilar entrenado. Las puntuaciones expresan compatibilidad del motor, no probabilidades clínicas o científicas.

Para convertirla en un modelo calibrado se necesita un conjunto autorizado y diverso de fotografías frontales/laterales, con etiquetas independientes de al menos dos profesionales sobre forma facial, textura, largo, densidad y estilos recomendados. El contrato HTTP está versionado para sustituir las heurísticas sin romper backend o móvil.

## Pruebas

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
