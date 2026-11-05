# Imagen publicada para prueba RunPod

Fecha: 2026-11-04
Imagen: docker.io/derwin659/supergods-illustrative-serverless:20261104
Digest: sha256:05baee505ebaef61af003fa7367f9ed6684e7fae180a910b607fdc3e8a0c2c5c
Referencia fija: docker.io/derwin659/supergods-illustrative-serverless@sha256:05baee505ebaef61af003fa7367f9ed6684e7fae180a910b607fdc3e8a0c2c5c
Plataforma local verificada: linux/amd64
13 archivos Python de runtime coinciden por SHA256 con la imagen local. 13 pruebas pasaron dentro de la imagen.

La publicacion no valida inferencia GPU ni calidad. Pendientes: volumen y modelos, endpoint Queue de prueba, secretos, prueba real y configuracion del backend. Seguir RUNPOD_PRUEBA.md y DEPLOY_SERVERLESS.md.

## Publicacion v2 - 2026-11-04

Imagen: docker.io/derwin659/supergods-illustrative-serverless:20261104-v2
Digest: sha256:ec53f95f58730c55bc2dd6a7677a0f5da5c373d03b4e32d55d71cbc9b3b8be06
Referencia fija: docker.io/derwin659/supergods-illustrative-serverless@sha256:ec53f95f58730c55bc2dd6a7677a0f5da5c373d03b4e32d55d71cbc9b3b8be06
18 pruebas aprobadas dentro del contenedor. Importacion real de services.sd_services y carga del clasificador OpenCV comprobadas sin red; InsightFace no esta instalado en el runtime.
Usar esta version para la siguiente prueba Serverless. La v1 se conserva como base historica. No se ejecuto inferencia GPU ni se modifico Railway.

## Publicacion v3 confirmada - 2026-11-04

Imagen: docker.io/derwin659/supergods-illustrative-serverless:20261104-v3
Digest: sha256:6a6aed836f1e03a31186bb3ac37c01977a345a18d06996505532372934c55f15
Referencia fija: docker.io/derwin659/supergods-illustrative-serverless@sha256:6a6aed836f1e03a31186bb3ac37c01977a345a18d06996505532372934c55f15
23 pruebas pasaron dentro del contenedor sin red. Incluyen ocho orientaciones EXIF, limites de resolucion, composicion fuera de mascara y vista trasera. El motor Docker fue recuperado con reinicio; no se borraron imagenes ni volumenes.
La publicacion sustituye el bloqueo anterior: v3 ya existe en Docker Hub. Pendiente actualizar el endpoint existente manteniendo ID/volumen/secretos y validar una generacion real desde movil. No se cambio Railway ni el endpoint remoto desde estas herramientas.


## v4 publicada — 2026-11-04

Autorizada expresamente por el usuario.

Imagen: `derwin659/supergods-illustrative-serverless:20261104-v4`

Digest: `sha256:94cdb12838fc59e626c155e565634419cd8f68ab0bdad44b85824ccf1099a75f`

25 pruebas pasaron dentro del contenedor. El endpoint no fue modificado. Pendiente seleccionar v4 en RunPod y probar con fotos reales; velocidad y calidad visual aún requieren medición GPU. La validación analítica necesita su despliegue separado en Railway y el flujo nuevo necesita actualización móvil.
