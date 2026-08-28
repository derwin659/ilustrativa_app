# services/hair_segmentation.py
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def generar_mascara_cabello(roi: Image.Image, vista: str = "lateral") -> Image.Image:
    """
    Máscara basada en MediaPipe Selfie Segmentation.
    Devuelve L (0..255):
      - Blanco = zona a editar (cabello: incluimos zona superior + laterales)
      - Negro  = proteger (cara / oreja / cuello)
    OJO: MediaPipe no segmenta "cabello" exacto, segmenta "persona".
         Aquí hacemos una heurística robusta: persona -> recortamos solo región de cabello
         usando un "hair band" dependiendo de vista.
    """

    vista = (vista or "").lower().strip()
    roi_rgb = np.array(_to_rgb(roi))

    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

    if res.segmentation_mask is None:
        raise RuntimeError("MediaPipe no devolvió segmentation_mask")

    # persona mask (0..255)
    person = (res.segmentation_mask > 0.15).astype(np.uint8) * 255

    h, w = person.shape[:2]

    # --- Heurística "solo cabello" (bandas por vista) ---
    hair = np.zeros_like(person)

    if vista == "lateral":
        # arriba + laterales (evita mucho la cara)
        y_top = int(h * 0.00)
        y_mid = int(h * 0.78)   # hasta donde baja el pelo (ajustable)
        x1 = int(w * 0.18)      # evita parte frontal de cara
        x2 = int(w * 0.98)
        hair[y_top:y_mid, x1:x2] = person[y_top:y_mid, x1:x2]

        # protege oreja y cara (negro)
        cv2.ellipse(hair, (int(w*0.22), int(h*0.42)), (int(w*0.30), int(h*0.30)), 0, 0, 360, 0, -1)  # cara
        cv2.ellipse(hair, (int(w*0.52), int(h*0.58)), (int(w*0.14), int(h*0.20)), 0, 0, 360, 0, -1)  # oreja

    else:  # trasera
        # arriba y laterales atrás
        y_top = int(h * 0.00)
        y_bot = int(h * 0.82)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        hair[y_top:y_bot, x1:x2] = person[y_top:y_bot, x1:x2]

        # protege cuello bajo
        cv2.rectangle(hair, (0, int(h*0.78)), (w, h), 0, -1)

    # --- Limpieza robusta ---
    hair = cv2.medianBlur(hair, 7)
    hair = cv2.GaussianBlur(hair, (0, 0), 3)

    # reforzar bordes suaves
    _, hair = cv2.threshold(hair, 10, 255, cv2.THRESH_BINARY)
    hair = cv2.GaussianBlur(hair, (0, 0), 2)

    return Image.fromarray(hair, mode="L")
