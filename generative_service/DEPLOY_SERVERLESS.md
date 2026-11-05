# Estado actualizado: imagen de prueba v2 (2026-11-04)

La nueva imagen Serverless utiliza el detector frontal Haar incluido en OpenCV 4.10.0. No necesita pesos InsightFace/buffalo_l ni embeddings de identidad. El codigo legacy ip_adapter_services.py queda fuera del runtime v2 y sus paquetes InsightFace/ONNX se desinstalan en esta imagen. OPENCV_FACE_LICENSE.txt conserva el aviso del clasificador distribuido.

La v2 se construye con DockerfileServerlessV2 sobre el digest exacto de la primera imagen; reutiliza CUDA y las capas descargadas. La primera imagen se conserva para trazabilidad, pero no se usa directamente como endpoint nuevo.

Referencia: derwin659/supergods-illustrative-serverless:20261104-v2
REQUIRE_LORA=false es el modo de prueba explicito de esta imagen. No hay pesos de cortes personalizados confirmados. No equivale a validar calidad de cortes ni preservacion de identidad; requiere inferencia GPU con fotografias autorizadas y revision visual antes de ofrecerla a clientes.

Volumen global existente: supergods-modelos-prueba. Archivos copiados por el operador:
- /runpod-volume/models/sd15: stable-diffusion-v1-5/stable-diffusion-inpainting (Diffusers, pesos .bin)
- /runpod-volume/models/controlnet-canny: lllyasviel/sd-controlnet-canny (config.json y diffusion_pytorch_model.safetensors)
En el Pod de carga, el mismo volumen se monta en /workspace. Solo se verificaron nombres y tamanos; falta lectura y carga real en el worker GPU.

Variables de prueba:
REQUIRE_GPU=true
REQUIRE_LORA=false
MODEL_LOCAL_FILES_ONLY=true
SD_MODEL=/runpod-volume/models/sd15
CONTROLNET_MODEL=/runpod-volume/models/controlnet-canny
HF_HOME=/tmp/huggingface
AI_ALLOWED_IMAGE_HOSTS=res.cloudinary.com
CLOUDINARY_URL=<secreto de Cloudinary del entorno de prueba>

Mantener HF_HOME fuera del volumen global para evitar escrituras de cache que dependan de semantica POSIX. Los modelos se cargan directamente desde las rutas locales del volumen. No instalar paquetes ni crear entornos virtuales dentro del volumen global.

Crear endpoint Queue, 0 Active Workers, maximo 1 worker y 1 GPU compatible con CUDA 12.4. Global volume seleccionado y Network volumes vacio. El Pod temporal de descarga esta detenido; no es el endpoint Serverless y no necesita reiniciarse para probar. Nunca borrar el volumen como parte de detener el Pod.

Global Volume puede reflejar archivos solo al montar un worker nuevo: cargar los modelos antes de iniciar el worker. Si se actualizan modelos, reemplazar workers. No ejecutar cp/copytree con preservacion de permisos: usar shutil.copyfile por archivo; el volumen no admite todas las operaciones POSIX.

Validacion local v2: 18 pruebas dentro de la imagen y comprobacion de importaciones sin red. Pendiente: GPU real, carga de los archivos en volumen, consentimiento/fotos de prueba, calidad visual, tiempos, Cloudinary, seguimiento y recuperacion de trabajos en movil/backend. Mantener la configuracion RunPod de produccion sin cambios hasta esa validacion.

---

# Preparación de RunPod Serverless

## Construcción local

Desde generative_service:

    docker build -f DockerfileHandler -t supergods-illustrative-serverless:local .
    docker run --rm --entrypoint python supergods-illustrative-serverless:local -m unittest discover -s tests -v
    docker run --rm --entrypoint python supergods-illustrative-serverless:local -m pip check

La imagen usa PyTorch 2.6.0 con CUDA 12.4 y versiones fijadas de las dependencias
directas. installed-requirements.txt dentro de la imagen registra todas las
versiones resueltas. Antes de distribuir, publicar y desplegar por digest.
Las dependencias transitivas y paquetes Debian aún no están bloqueados por hash.

El build verifica importaciones, sintaxis y pruebas sin GPU. Esto NO valida memoria,
calidad, carga de pesos ni rendimiento CUDA: esas pruebas requieren RunPod.

## Volumen de modelos

Montar un volumen en /runpod-volume:

    /runpod-volume/models/sd15/model_index.json
    /runpod-volume/models/sd15/... (todos los pesos y subdirectorios del modelo)
    /runpod-volume/models/controlnet-canny/config.json
    /runpod-volume/models/controlnet-canny/... (pesos del modelo)
    /runpod-volume/insightface/models/buffalo_l/*.onnx
    /runpod-volume/loras/mid_fade_sd15.safetensors

SD_MODEL y CONTROLNET_MODEL apuntan a directorios locales completos compatibles
con Diffusers, no a archivos sueltos. Deben ser los modelos probados por el proyecto.
No sustituir por modelos de otra arquitectura. Conservar sus revisiones y hashes.

No se encontró un LoRA entrenado en el proyecto local al preparar esta imagen.
La carpeta lora/dataset contiene material de entrenamiento, no pesos utilizables.
No se incluyen fotos de entrenamiento ni archivos de salida en la imagen.

## Arranque

- REQUIRE_GPU=true: rechaza un entorno sin CUDA en lugar de generar lentamente en CPU.
- MODEL_LOCAL_FILES_ONLY=true: evita descargar Stable Diffusion y ControlNet por solicitud.
- REQUIRE_LORA=true: rechaza pesos LoRA ausentes o vacíos.
- CLOUDINARY_URL: configurar como secreto en RunPod, nunca en el Dockerfile.
- AI_ALLOWED_IMAGE_HOSTS: mantener la lista de hosts de entrada autorizados.

El worker valida archivos, inicializa el detector facial y carga el pipeline y LoRA
antes de empezar a aceptar trabajos. La comprobación estructural no sustituye
la carga real del modelo, que detecta archivos corruptos o incompatibles.

No desactivar REQUIRE_LORA para ocultar pesos faltantes en producción. Para una
prueba exploratoria sin el modelo personalizado se puede establecer false, dejando
claro que no valida la calidad del corte entrenado.

## Validación en GPU antes de activar

1. Preparar el volumen y confirmar integridad de los pesos.
2. Publicar la imagen en el registro elegido y guardar su digest.
3. Crear un endpoint de prueba (0 Active Workers, máximo 1 worker inicialmente).
4. Configurar secretos y montar el volumen en la misma ubicación.
5. Confirmar arranque: CUDA disponible, pesos cargados, worker listo.
6. Ejecutar una solicitud con fotos autorizadas y comprobar las tres vistas.
7. Medir primer arranque, generación posterior y consumo de memoria.
8. Probar error de generación y limpieza de archivos parciales.
9. Conectar Railway de pruebas; mantener producción en POD hasta validación completa.

Pendientes independientes: reconciliación persistente de trabajos no confirmados
en el backend y recuperación del trabajo desde la app móvil.
## Reintentos de construcción

DockerfileHandler mantiene una caché BuildKit de pip y separa la instalación de
PyTorch/CUDA de las dependencias de la aplicación. Si la red falla, repetir el mismo
docker build; los paquetes descargados completos se reutilizan y pip puede reanudar
descargas interrumpidas. No ejecutar docker builder prune entre reintentos.

El equipo Windows de preparación no tiene una GPU visible en Docker/WSL. El build
y las pruebas sin GPU son locales; la comprobación CUDA requiere el endpoint GPU.
## Resultado de la validación local

- Imagen construida: supergods-illustrative-serverless:local
- Image ID local: sha256:e9c6906772d15ccdf33b36196f2eb76d58e2650f1f12d777b667b5ad257ffa16
- Tamaño: 6.822.685.687 bytes (aproximadamente 6,8 GB).
- pip check: sin dependencias rotas.
- Importaciones de PyTorch, Diffusers, Transformers, PEFT, InsightFace, Cloudinary y RunPod: correctas.
- PyTorch compilado con CUDA 12.4.
- 13 pruebas unitarias correctas dentro del contenedor.
- Arranque sin GPU: rechazado explícitamente antes de aceptar trabajos (resultado esperado).
- requirements-serverless.resolved.txt registra todas las versiones instaladas.

El ID local anterior no es un digest de publicación en un registro. Aún no se ha
publicado esta imagen, creado un endpoint ni ejecutado inferencia con GPU.
Falta localizar los pesos LoRA entrenados y preparar el volumen de modelos.