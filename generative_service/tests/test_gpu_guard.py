"""Exercise the real generation entry function without loading model dependencies."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

class GpuGuardTest(unittest.TestCase):
    def function(self, device):
        source = (Path(__file__).resolve().parents[1] / "services/sd_services.py").read_text()
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generar_sd")
        scope = {"DEVICE": device, "Image": SimpleNamespace(Image=object)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<generation>", "exec"), scope)
        return scope["generar_sd"]
    def test_missing_gpu_raises_instead_of_echoing_photo(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.function("cpu")("prompt", object(), "frontal", "mid_fade")
    def test_unknown_view_is_rejected(self):
        with self.assertRaises(ValueError):
            self.function("cuda")("prompt", object(), "unknown", "mid_fade")
    def test_frontal_without_face_rejects_before_loading_diffusion(self):
        source = (Path(__file__).resolve().parents[1] / "services/sd_services.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "generar_sd")
        scope = {"DEVICE": "cuda", "Image": SimpleNamespace(Image=object),
                 "make_generator": lambda: None, "prepare_inference_image": lambda image: image, "get_face_bbox": lambda image: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<generation>", "exec"), scope)
        with self.assertRaisesRegex(ValueError, "rostro frontal"):
            scope["generar_sd"]("prompt", SimpleNamespace(convert=lambda mode: object()), "frontal", "mid_fade")
