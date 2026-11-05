import ast
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps
from services.image_transport import load_input_image
from services.image_preparation import prepare_inference_image, preserve_unmasked_pixels, ensure_visible_edit
from services.prompt_builder import build_prompt

class ImageOrientationTest(unittest.TestCase):
    def test_all_exif_orientations_are_baked_into_pixels_once(self):
        image = Image.new("RGB", (160, 240), "blue")
        ImageDraw.Draw(image).rectangle((0, 0, 70, 90), fill="red")
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = image.getexif()
                exif[274] = orientation
                stream = BytesIO()
                image.save(stream, "JPEG", exif=exif)
                payload = stream.getvalue()
                expected = ImageOps.exif_transpose(Image.open(BytesIO(payload))).convert("RGB")
                result = load_input_image(base64.b64encode(payload).decode())
                self.assertEqual(result.size, expected.size)
                self.assertIsNone(ImageChops.difference(result, expected).getbbox())
                self.assertIsNone(result.getexif().get(274))

    def test_inference_resolution_is_bounded_and_divisible_by_eight(self):
        for size in [(1600, 1200), (1200, 1600), (1920, 1080), (300, 300)]:
            result = prepare_inference_image(Image.new("RGB", size))
            self.assertLessEqual(max(result.size), 512)
            self.assertEqual(result.width % 8, 0)
            self.assertEqual(result.height % 8, 0)
            self.assertLess(abs(result.width / result.height - size[0] / size[1]), .04)

    def test_compositing_preserves_unmasked_pixels_and_dimensions(self):
        original = Image.new("RGB", (1000, 1500), "blue")
        generated = Image.new("RGB", (256, 384), "red")
        mask = Image.new("L", generated.size, 0)
        ImageDraw.Draw(mask).rectangle((30, 30, 100, 100), fill=255)
        result = preserve_unmasked_pixels(original, generated, mask)
        self.assertEqual(result.size, original.size)
        self.assertEqual(result.getpixel((0, 0)), original.getpixel((0, 0)))
        self.assertEqual(result.getpixel((250, 250)), (255, 0, 0))

    def test_frontal_edit_preserves_eyes_and_background_exactly(self):
        source = (Path(__file__).resolve().parents[1]/"services/sd_services.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "frontal_mask")
        scope = dict(Image=Image, ImageDraw=ImageDraw, ImageFilter=ImageFilter)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<mask>", "exec"), scope)
        original = Image.new("RGB", (512,768), "blue")
        mask = scope["frontal_mask"](original, (180,250,340,430))
        edited = preserve_unmasked_pixels(original, Image.new("RGB",original.size,"red"),mask)
        self.assertEqual(edited.getpixel((240,290)), (0,0,255))
        self.assertEqual(edited.getpixel((0,0)), (0,0,255))
        self.assertNotEqual(edited.getpixel((240,210)), (0,0,255))

    def test_identical_output_is_not_a_successful_edit(self):
        original = Image.new("RGB", (128, 128), "blue")
        with self.assertRaisesRegex(ValueError, "demasiado leve"):
            ensure_visible_edit(original, original.copy(), Image.new("L", original.size, 255))

    def test_rear_prompt_and_requested_tint_are_explicit(self):
        req = SimpleNamespace(corte=SimpleNamespace(tipo="MID_FADE"),
            tinte=SimpleNamespace(aplicar=True, color="Oscurecimiento natural"), ondulado=None)
        prompt = build_prompt(req, "trasera")
        self.assertIn("back view", prompt)
        self.assertIn("face not visible", prompt)
        self.assertIn("natural dark brown", prompt)

    def test_view_generation_keeps_slot_and_size_without_loading_gpu(self):
        source = (Path(__file__).resolve().parents[1]/"services/sd_services.py").read_text()
        names = {"generar_sd", "frontal_mask", "lateral_mask", "trasera_mask"}
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
        pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (384, 512), "red")]))
        scope = dict(DEVICE="cuda", Image=Image, ImageDraw=ImageDraw, ImageFilter=ImageFilter,
            prepare_inference_image=prepare_inference_image, preserve_unmasked_pixels=preserve_unmasked_pixels, ensure_visible_edit=ensure_visible_edit,
            get_face_bbox=lambda image: None, get_pipe_inpaint=lambda: pipe,
            aplicar_lora=lambda *args: None, make_generator=lambda: None,
            NEGATIVE="negative", NEGATIVE_ZERO="negative")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<real-generation>", "exec"), scope)
        original = Image.new("RGB", (1200, 1600), "blue")
        for view in ("lateral", "trasera"):
            result = scope["generar_sd"]("prompt", original, view, "mid_fade")
            self.assertEqual(result.size, original.size)
            self.assertLessEqual(max(pipe.call_args.kwargs["image"].size), 512)
        scope["get_face_bbox"] = lambda image: [40, 40, 160, 160]
        pipe.reset_mock()
        with self.assertRaisesRegex(ValueError, "foto trasera"):
            scope["generar_sd"]("prompt", original, "trasera", "mid_fade")
        pipe.assert_not_called()
