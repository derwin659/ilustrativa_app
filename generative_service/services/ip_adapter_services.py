import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


# Inicializar UNA sola vez
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

def extraer_identidad(imagen: Image.Image):
    """
    Retorna:
    - embedding: torch.Tensor (1, 512)
    - face_img: PIL.Image (rostro alineado)
    """

    # InsightFace usa BGR
    img_np = np.array(imagen.convert("RGB"))[:, :, ::-1]

    faces = app.get(img_np)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen")

    face = faces[0]

    # 🔹 Embedding FaceID
    embedding = torch.tensor(face.embedding).unsqueeze(0).to("cuda")

    # 🔹 Rostro ALINEADO (FORMA CORRECTA)
    face_aligned = norm_crop(img_np, face.kps)   # np.ndarray BGR
    face_img = Image.fromarray(face_aligned[:, :, ::-1]).convert("RGB")

    return embedding, face_img