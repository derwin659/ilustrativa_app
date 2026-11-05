import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

class GenerationFailureTest(unittest.TestCase):
    def setUp(self):
        transport = ModuleType("services.image_transport")
        transport.load_input_image = Mock(return_value=object())
        prompts = ModuleType("services.prompt_builder")
        prompts.build_prompt = Mock(return_value="haircut")
        sd = ModuleType("services.sd_services")
        sd.generar_sd = Mock()
        with patch.dict(sys.modules, {
            "services.image_transport": transport,
            "services.prompt_builder": prompts,
            "services.sd_services": sd,
        }):
            spec = importlib.util.spec_from_file_location(
                "runner_test", Path(__file__).resolve().parents[1] / "services/generation_runner.py")
            self.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.runner)
        self.sd = sd.generar_sd
        self.req = SimpleNamespace(
            sesionId="test-session", corte=SimpleNamespace(tipo="MID_FADE"),
            imagenes={"frontal": "input"}, vistas=["frontal"])
    def test_gpu_failure_propagates_instead_of_returning_original(self):
        self.sd.side_effect = RuntimeError("GPU failed")
        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            self.runner.run_generation(self.req, save_files=False,
                output_uploader=lambda *args: "https://example.test/image.png")
    def test_missing_requested_view_fails(self):
        self.req.vistas = ["trasera"]
        with self.assertRaises(ValueError):
            self.runner.run_generation(self.req, save_files=False,
                output_uploader=lambda *args: "https://example.test/image.png")
    def test_success_uploads_generated_image(self):
        generated = object()
        self.sd.return_value = generated
        uploader = Mock(return_value="https://example.test/generated.png")
        result = self.runner.run_generation(self.req, save_files=False, output_uploader=uploader)
        self.assertEqual(result["imagenes"]["frontal"], "https://example.test/generated.png")
        uploader.assert_any_call(generated, "frontal")

if __name__ == "__main__":
    unittest.main()