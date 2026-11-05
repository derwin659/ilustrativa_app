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

# Primera prueba: backend -> RunPod -> Cloudinary

Objetivo: comprobar transporte e inferencia con el modelo base antes de entrenar el LoRA. No habilitar para clientes ni evaluar esto como un corte personalizado terminado.

## Recursos pendientes

- Publicar supergods-illustrative-serverless:local en un registro accesible por RunPod y registrar el digest publicado. El ID local no sirve como referencia remota.
- Preparar un volumen de modelos siguiendo DEPLOY_SERVERLESS.md. Descargar los pesos en RunPod: el equipo local tiene poco espacio disponible.
- Crear endpoint de prueba con 0 workers activos y máximo 1 worker. Un trabajo de prueba puede generar cobros por GPU.
- Montar el volumen en /runpod-volume. Seleccionar una GPU compatible con CUDA 12.4; verificar memoria durante la prueba real.

## Variables del worker de PRUEBA

```dotenv
REQUIRE_GPU=true
REQUIRE_LORA=false
MODEL_LOCAL_FILES_ONLY=true
SD_MODEL=/runpod-volume/models/sd15
CONTROLNET_MODEL=/runpod-volume/models/controlnet-canny
INSIGHTFACE_ROOT=/runpod-volume/insightface
LORA_MID_FADE_PATH=/runpod-volume/loras/mid_fade_sd15.safetensors
HF_HOME=/runpod-volume/huggingface
AI_ALLOWED_IMAGE_HOSTS=res.cloudinary.com
```

Agregar CLOUDINARY_URL como secreto del endpoint con la cuenta correspondiente al entorno de pruebas. No pegar claves en el repositorio ni en el chat. REQUIRE_LORA=false permite probar sin pesos personalizados, pero todavía exige los modelos base y faciales. Para validar un despliegue entrenado, cargar el LoRA y volver a REQUIRE_LORA=true.

## Variables del backend de PRUEBA

Solo después de verificar que el worker arranca y termina un trabajo:

```dotenv
RUNPOD_MODE=SERVERLESS
RUNPOD_ENDPOINT_ID=<ID_REAL_DEL_ENDPOINT>
RUNPOD_SERVERLESS_API_BASE_URL=https://api.runpod.ai/v2
RUNPOD_SERVERLESS_WAIT_MILLIS=600000
RUNPOD_SERVERLESS_EXECUTION_TIMEOUT_MILLIS=600000
RUNPOD_SERVERLESS_TTL_MILLIS=900000
```

Configurar RUNPOD_API_KEY como secreto en Railway. Conservar la configuración anterior para volver a POD si falla. Revisar la migración SQL y demás pendientes en C:/godssass/docs/runpod_serverless_migration.md antes de habilitar clientes.

## Evidencia para considerar conectada la prueba

1. Worker listo: CUDA disponible y pipeline cargado.
2. Generación desde el backend autenticado con fotos autorizadas: RunPod devuelve jobId y termina COMPLETED.
3. Resultado con las tres vistas solicitadas, URLs de Cloudinary accesibles y sin devolver las fotos originales como sustituto de un fallo.
4. Registrar tiempos y memoria. Confirmar limpieza de entradas al terminar.
5. Verificar desde el móvil. Una interrupción de red todavía requiere revisar el trabajo; falta recuperación automática persistente antes de producción.

Estado: imagen construida y 13 pruebas locales aprobadas. Registro remoto, volumen, endpoint e inferencia GPU pendientes. Entrenamiento LoRA pendiente de confirmar; no hay pesos entrenados en el proyecto local.
## Correccion v3 preparada (pendiente de publicar)

DockerfileServerlessV3 reutiliza el digest de v2 y actualiza solo codigo/pruebas.
- Aplica ImageOps.exif_transpose antes de convertir y elimina la etiqueta EXIF; todas las vistas pasan por este lector.
- Limita la imagen de inferencia a 512 px en su lado mayor, dimensiones multiplos de 8. Devuelve el tamano original componiendo resultado solo bajo la mascara de edicion; no conserva detalle sintetico de alta resolucion.
- Corrige el prompt trasera (antes solo existia atras) y rechaza una foto trasera si se detecta un rostro frontal unico. Esto no clasifica perfiles ni garantiza una captura posterior correcta.
- Reduce fuerza de edicion lateral/trasera a .55 para reducir cambios innecesarios. Agrega el tinte Oscurecimiento natural al prompt.
- El mapeo de los tres campos movil->backend fue inspeccionado: son independientes y mantienen los nombres; no se encontro un intercambio de frontal/lateral/trasera.

23 pruebas locales pasaron, incluidas las 8 orientaciones EXIF y la preservacion de pixeles fuera de mascara. No hay inferencia GPU nueva ni medida de latencia/validacion visual de este parche.

Construccion Docker intentada sin respuesta; interrumpida. C: tiene aproximadamente 80 MB libres. v3 NO esta publicada; RunPod sigue en v2. Recuperar espacio y comprobar que Docker responde, luego construir/publicar :20261104-v3 y cambiar la imagen del endpoint manteniendo volumen y secretos. No eliminar volumen ni modelos. No lanzar mas trabajos para medir calidad de v2 mientras se prepara esta correccion.

Actualizacion: v3 publicada y verificada en Docker Hub como :20261104-v3. 23 pruebas dentro del contenedor aprobadas. Digest y siguientes pasos en RUNPOD_IMAGEN_PUBLICADA.md. La nota anterior de bloqueo de construccion queda superada; la validacion GPU/visual de v3 sigue pendiente.


## Revisión v4 — captura y cambios visibles (2026-11-04)

Imagen local: `derwin659/supergods-illustrative-serverless:20261104-v4`, construida con DockerfileServerlessV4.
Publicación autorizada por el usuario y completada el 2026-11-04. Digest remoto: `sha256:94cdb12838fc59e626c155e565634419cd8f68ab0bdad44b85824ccf1099a75f`. No se cambió el endpoint activo.

- SDPA predeterminado: no activar ENABLE_ATTENTION_SLICING con PyTorch moderno salvo medición concreta de memoria.
- 24 pasos; fuerza 0.68 frontal / 0.78 perfil y nuca. Más fuerza puede cambiar más el cabello; requiere validación visual para descartar deformaciones.
- Máscara frontal acotada con exclusión dura de ojos/gafas/rostro inferior. Perfil y nuca usan una guía geométrica aproximada, NO segmentación semántica del cabello.
- Se rechaza una salida prácticamente idéntica dentro de la máscara. Es un control de diferencia de píxeles, NO un verificador de que el corte sea mid fade.
- Reutilización de fotos descargadas y tiempos registrados por vista; no se han medido latencias GPU de v4.

Despliegue pendiente: seleccionar tag v4 en endpoint existente conservando volumen/variables, publicar analytics_service en Railway y actualizar móvil. Backend Java sin cambio adicional para este flujo.

Prueba piloto: enviar la misma sesión de tres fotos bien encuadradas, medir cola/arranque/generación por vista en frío y en caliente, y hacer revisar el degradado por un profesional. Repetir con cabello lacio, rizado, gafas y diferentes luces. Volver a v3 si aparecen deformaciones o demasiados rechazos; no aumentar workers para ocultar errores de imagen.

Colección para la fase LoRA: ../datasets/haircuts/README.md (desde la raíz del repositorio: datasets/haircuts/README.md). No hay un adaptador entrenado nuevo.
