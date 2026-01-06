# services/controlnet_services.py
import numpy as np
from PIL import Image

def aplicar_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """
    Devuelve una imagen PIL (RGB) con bordes tipo Canny para ControlNet.
    """
    import cv2  # mantener import aquí para evitar problemas si no está instalado al importar el módulo

    img = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low, high)

    # ControlNet espera 3 canales (RGB)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)
