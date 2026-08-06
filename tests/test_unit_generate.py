import json

from mlx_moe.lazy_experts.generate import _requires_strict_false_load


class TestQwen35VisionPackage:

    def test_requires_strict_false_for_qwen35_vision_package(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "vision_config": {"model_type": "qwen3_5_moe"},
            "text_config": {"model_type": "qwen3_5_moe_text"},
        }))

        assert _requires_strict_false_load(tmp_path)

    def test_does_not_affect_text_only_qwen35_package(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text"},
        }))

        assert not _requires_strict_false_load(tmp_path)

    def test_requires_strict_false_for_dense_qwen35_vision_package(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "vision_config": {"model_type": "qwen3_5"},
            "text_config": {"model_type": "qwen3_5_text"},
        }))

        assert _requires_strict_false_load(tmp_path)
