import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from worker_startup import validate_model_files

class StartupValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sd = self.root / "sd"
        self.cn = self.root / "cn"
        self.face = self.root / "face"
        self.lora = self.root / "mid.safetensors"
        self.sd.mkdir()
        self.cn.mkdir()
        (self.sd / "model_index.json").write_text("{}")
        (self.cn / "config.json").write_text("{}")
        face_models = self.face / "models" / "buffalo_l"
        face_models.mkdir(parents=True)
        for i in range(5):
            (face_models / f"{i}.onnx").write_bytes(b"fixture")
        self.lora.write_bytes(b"fixture")
        self.env = patch.dict(os.environ, {
            "SD_MODEL": str(self.sd), "CONTROLNET_MODEL": str(self.cn),
            "INSIGHTFACE_ROOT": str(self.face), "LORA_MID_FADE_PATH": str(self.lora),
            "REQUIRE_LORA": "true",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
    def test_complete_layout_passes_structural_check(self):
        validate_model_files()
    def test_missing_lora_is_rejected(self):
        self.lora.unlink()
        with self.assertRaisesRegex(RuntimeError, "LoRA"):
            validate_model_files()
    def test_empty_lora_is_rejected(self):
        self.lora.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "LoRA"):
            validate_model_files()
    def test_missing_model_is_rejected(self):
        (self.sd / "model_index.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "model_index"):
            validate_model_files()
    def test_base_trial_does_not_require_face_weights_or_lora(self):
        self.lora.unlink()
        with patch.dict(os.environ, {"REQUIRE_LORA": "false"}):
            validate_model_files()
