from controlnet_aux import SamDetector
from PIL import Image
import numpy as np

# ✅ REPO CORRECTO + ARCHIVO CORRECTO
sam = SamDetector.from_pretrained(
    "segment-anything/sam-vit-huge",
    filename="sam_vit_h_4b8939.pth"
)

def segmentar_cabello(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("segmentar_cabello espera PIL.Image")

    masks = sam(image)

    if not masks:
        raise RuntimeError("SAM no detectó ninguna máscara")

    # tomamos la máscara más grande (normalmente cabello + cabeza)
    mask = max(masks, key=lambda m: m.sum())
    mask = (mask * 255).astype(np.uint8)

    return Image.fromarray(mask).convert("L")
