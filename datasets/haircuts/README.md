# Colección de cortes SuperGods

Estado: colección por iniciar. No hay pesos entrenados ni promesa de calidad por corte.

## Primera colección

Empieza con un piloto de 20 sesiones por corte: MID_FADE, LOW_FADE y TAPER. Revisa el resultado antes de ampliar. Esta es una meta de organización, no un mínimo científico que garantice entrenamiento.

Por sesión fotografía ANTES y DESPUÉS: frente, perfil del mismo lado y nuca (6 fotos). Misma cámara, distancia, orientación, luz y fondo liso. Cabeza completa y orejas visibles; sin filtros, capturas de pantalla ni fotos de internet. El profesional confirma el corte real y la altura de transición. Incluye variedad de densidad, textura y tono de piel; no selecciones solo un tipo de cliente.

## Permiso y archivos

Recopila únicamente fotos propias con autorización expresa para entrenamiento y evaluación del modelo. Lleva el registro de autorización separado; usa identificadores de sesión, no nombres, teléfonos ni documentos en metadatos o nombres de archivos. No recopiles una sesión si falta autorización. Para este piloto usa participantes adultos.

Estructura local (fuera del control de versiones):

raw/MID_FADE/S0001/before/front.jpg
raw/MID_FADE/S0001/before/profile.jpg
raw/MID_FADE/S0001/before/back.jpg
raw/MID_FADE/S0001/after/front.jpg
raw/MID_FADE/S0001/after/profile.jpg
raw/MID_FADE/S0001/after/back.jpg

Completa catalog.csv: una fila por par antes/después y ángulo. consent_ref es solo un identificador del registro guardado por el negocio. Marca professional_verified y consent_training solo tras revisarlos. Usa una columna split (train/validation/test) asignada por PERSONA; sus diferentes vistas y sesiones nunca deben caer en grupos distintos. No publiques raw ni los permisos en GitHub.

## Evaluación antes de activar un LoRA

1. Validar encuadre, autorización y etiqueta con un barbero.
2. Reservar personas no vistas para evaluación; separar fotos de referencia y prueba.
3. Comparar el modelo base y el adaptado con los mismos ejemplos y semillas.
4. Evaluar si se reconoce la altura mid/low/taper, cambios visibles, rostro y gafas conservados, fondo, orejas y nuca sin deformaciones. Registrar rechazos y tiempos por vista, arranque frío y caliente.
5. Probar cada tipo y textura; activar solamente el adaptador aprobado para el corte correspondiente.

El entrenamiento del corte no entrena la identidad del cliente. El rostro se protege en la edición; la segmentación precisa del cabello y un modelo de rasgos faciales requerirán una fase adicional con modelos y licencias revisados. No es suficiente escribir el nombre del corte en un prompt.
