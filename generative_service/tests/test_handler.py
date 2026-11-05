import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

class HandlerTest(unittest.TestCase):
    def setUp(self):
        self.runpod = ModuleType("runpod")
        self.runpod.serverless = SimpleNamespace(start=Mock())
        self.schemas = ModuleType("schemas_runpod")
        self.schemas.RequestGenerar = Mock(return_value=SimpleNamespace(
            storage=SimpleNamespace(tenantId=1), sesionId="test"))
        self.transport = ModuleType("services.image_transport")
        self.transport.delete_output_assets = Mock()
        self.transport.upload_output_image = Mock(return_value={
            "publicId": "test/asset", "url": "https://example.test/result.png"})
        self.runner = ModuleType("services.generation_runner")
        self.runner.run_generation = Mock()
        self.patch = patch.dict(sys.modules, {
            "runpod": self.runpod, "schemas_runpod": self.schemas,
            "services.image_transport": self.transport,
            "services.generation_runner": self.runner,
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        spec = importlib.util.spec_from_file_location(
            "handler_test", Path(__file__).resolve().parents[1] / "handler.py")
        self.handler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.handler)
    def test_import_does_not_start_worker(self):
        self.runpod.serverless.start.assert_not_called()
    def test_partial_failure_deletes_uploaded_outputs(self):
        def fail(**kwargs):
            kwargs["output_uploader"](object(), "antes")
            raise RuntimeError("generation failure")
        self.runner.run_generation.side_effect = fail
        with self.assertLogs("handler_test", level="ERROR"):
            result = self.handler.handler({"input": {}})
        self.assertFalse(result["success"])
        ids = self.transport.delete_output_assets.call_args.args[0]
        self.assertEqual(list(ids), ["test/asset"])
    def test_missing_storage_rejected_before_generation(self):
        self.schemas.RequestGenerar.return_value.storage = None
        self.assertFalse(self.handler.handler({"input": {}})["success"])
        self.runner.run_generation.assert_not_called()