import base64
from io import BytesIO
from PIL import Image

def decode_base64(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes)).convert("RGB")
