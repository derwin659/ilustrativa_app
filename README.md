# Super Gods AI

Monorepo de servicios de inteligencia artificial de Super Gods.

## Servicios

### `analytics_service`

IA analítica de baja latencia. Recibe la fotografía frontal, valida el rostro, estima características faciales/capilares y devuelve recomendaciones compatibles con el catálogo generativo.

- Puerto sugerido: `8001`
- Salud: `GET /health`
- Análisis: `POST /analizar`
- Backend: `IA_ANALITICA_URL=http://analytics-service:8001`

### `generative_service`

IA ilustrativa con GPU. Recibe las vistas frontal, lateral y trasera junto con el corte seleccionado y genera la simulación visual. Puede ejecutarse como FastAPI o worker RunPod.

- Puerto sugerido: `8000`
- Salud: `GET /health`
- Generación: `POST /generar`
- Backend: `IA_ILUSTRATIVA_URL=http://generative-service:8000`

Los servicios comparten el contrato funcional, pero no el proceso, dependencias ni despliegue. La analítica puede permanecer disponible cuando la GPU generativa esté apagada.
