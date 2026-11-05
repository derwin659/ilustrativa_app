"""Bound inference resolution without rotating or cropping the supplied view."""
from PIL import Image

def prepare_inference_image(image, max_side=512):
    if min(image.size) < 64:
        raise ValueError("La imagen debe medir al menos 64 pixeles por lado")
    scale = min(1.0, max_side / max(image.size))
    size = tuple(max(8, int(side * scale) // 8 * 8) for side in image.size)
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)

def preserve_unmasked_pixels(original, generated, mask):
    original = original.convert("RGB")
    generated = generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    mask = mask.resize(original.size, Image.Resampling.BILINEAR)
    return Image.composite(generated, original, mask)


def ensure_visible_edit(original, generated, mask):
    """Reject near-identical output, not a semantic haircut-quality score."""
    from PIL import ImageChops, ImageStat
    size = mask.size
    difference = ImageChops.difference(original.convert("RGB").resize(size), generated.convert("RGB").resize(size)).convert("L")
    region = mask.point(lambda value: 255 if value > 128 else 0)
    if region.getbbox() is None:
        raise ValueError("No se pudo delimitar la zona de edicion. Repite la captura.")
    mean = ImageStat.Stat(difference, region).mean[0]
    histogram = difference.histogram(mask=region)
    changed = sum(histogram[5:]) / max(1, sum(histogram))
    if mean < 1.0 or changed < .02:
        raise ValueError("El cambio generado es demasiado leve. No se guardo como una simulacion valida; revisa el encuadre o elige otro estilo.")
