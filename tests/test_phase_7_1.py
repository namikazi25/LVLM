"""Full system integration test for MMFakeBench.

This test runs the BenchmarkRunner with a simple pipeline using a
mock model client. It verifies that dataset loading, pipeline module
execution, and result saving work together end-to-end.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path

from mmfakebench.core.runner import BenchmarkRunner
from mmfakebench.models.base_client import BaseModelClient
from mmfakebench.models.registry import get_registry
from mmfakebench.core.base import BasePipelineModule


class DummyClient(BaseModelClient):
    """Minimal mock client returning deterministic responses."""

    def _init_client(self):
        return None

    def create_multimodal_message(self, system_prompt, text_prompt, image_path=None):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_prompt},
            {"role": "image", "path": image_path},
        ]

    def _generate_impl(self, prompt, image_data=None, system_prompt=None, **kwargs):
        text = " ".join(part for part in [system_prompt, prompt] if part)
        if "fake" in text.lower():
            return "This appears fake."
        if "real" in text.lower():
            return "This appears real."
        return "Uncertain response."

    def estimate_cost(self, prompt: str, response: str = "") -> float:
        return 0.0


class DummyRouter:
    """Minimal router used for integration testing."""

    def get_usage_stats(self):
        return {"estimated_cost": 0.0}

    def get_info(self):
        return {"name": "dummy-router"}


class DummyModule(BasePipelineModule):
    """Simple pipeline module that records its execution."""

    def initialize(self):
        pass

    def validate_input(self, data: dict) -> bool:
        return True

    def process(self, data: dict) -> dict:
        data[self.name] = True
        return data

    def get_output_schema(self) -> dict:
        return {"flag": "bool"}


class TestFullIntegration(unittest.TestCase):
    """Run a small benchmark with all core components."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.images_dir = Path(self.temp_dir) / "images"
        self.images_dir.mkdir()
        for name in ["img1.jpg", "img2.jpg"]:
            with open(self.images_dir / name, "wb") as f:
                f.write(b"\x00")
        dataset = [
            {
                "image_path": "/img1.jpg",
                "text": "Real news event",
                "gt_answers": "True",
                "fake_cls": "original",
                "text_source": "test",
                "image_source": "test",
            },
            {
                "image_path": "/img2.jpg",
                "text": "Fake news event",
                "gt_answers": "Fake",
                "fake_cls": "mismatch",
                "text_source": "test",
                "image_source": "test",
            },
        ]
        self.json_path = Path(self.temp_dir) / "dataset.json"
        with open(self.json_path, "w") as f:
            json.dump(dataset, f)

        registry = get_registry()
        try:
            registry.register_provider("mock", DummyClient)
        except ValueError:
            pass
        try:
            registry.register_model_mapping("mock-model", "mock")
        except ValueError:
            pass

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_run(self):
        config = {
            "model": {"name": "mock-model", "provider_name": "mock", "api_key": "x"},
            "dataset": {
                "type": "custom",
                "data_path": str(self.json_path),
                "params": {"images_base_dir": str(self.images_dir)},
            },
            "pipeline": [
                {"type": "dummy", "name": "step1"},
                {"type": "dummy", "name": "step2"},
            ],
            "output": {"directory": str(self.output_dir)},
        }

        runner = BenchmarkRunner(config, output_dir=self.output_dir)
        runner.model_router = DummyRouter()
        runner.setup_dataset(config["dataset"])
        runner.pipeline_manager.register_module("dummy", DummyModule)
        runner.setup_pipeline(config["pipeline"])
        summary = runner.run_benchmark(limit=2)

        self.assertEqual(summary["statistics"]["processed_items"], 2)
        csv_files = list(self.output_dir.glob("results_*.csv"))
        json_files = list(self.output_dir.glob("results_*.json"))
        self.assertTrue(csv_files)
        self.assertTrue(json_files)


if __name__ == "__main__":
    unittest.main(verbosity=2)
