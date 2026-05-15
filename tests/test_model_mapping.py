"""
Tests for model mapping registration.

Verifies all multimodal models are properly registered in AUTO_EMBEDDER_MAPPING
and AUTO_RERANKER_MAPPING, and that FlagAutoModel/FlagAutoReranker dispatch correctly.
"""
import pytest

from FlagEmbedding.inference.embedder.model_mapping import (
    AUTO_EMBEDDER_MAPPING,
    EMBEDDER_CLASS_MAPPING,
    EmbedderModelClass,
)
from FlagEmbedding.inference.reranker.model_mapping import (
    AUTO_RERANKER_MAPPING,
    RERANKER_CLASS_MAPPING,
    RerankerModelClass,
)


class TestEmbedderMapping:
    EXPECTED_MULTIMODAL_MODELS = [
        "BGE-VL-MLLM-S1",
        "BGE-VL-MLLM-S2",
        "Qwen3-VL-Embedding-2B",
        "Qwen3-VL-Embedding-8B",
    ]

    EXPECTED_TEXT_MODELS = [
        "Qwen3-Embedding-0.6B",
        "Qwen3-Embedding-4B",
        "Qwen3-Embedding-8B",
    ]

    def test_multimodal_models_registered(self):
        for name in self.EXPECTED_MULTIMODAL_MODELS:
            assert name in AUTO_EMBEDDER_MAPPING, f"{name} not in AUTO_EMBEDDER_MAPPING"

    def test_text_models_registered(self):
        for name in self.EXPECTED_TEXT_MODELS:
            assert name in AUTO_EMBEDDER_MAPPING, f"{name} not in AUTO_EMBEDDER_MAPPING"

    def test_model_class_enum_complete(self):
        assert EmbedderModelClass.MULTIMODAL_MLLM.value == "multimodal-mllm"
        assert EmbedderModelClass.QWEN3_VL_EMBEDDING.value == "qwen3-vl-embedding"

    def test_class_mapping_has_all_enums(self):
        for cls in EmbedderModelClass:
            assert cls in EMBEDDER_CLASS_MAPPING, f"{cls} not in EMBEDDER_CLASS_MAPPING"

    def test_mapping_config_has_model_class(self):
        for name, config in AUTO_EMBEDDER_MAPPING.items():
            assert hasattr(config, "model_class"), f"{name} config missing model_class"
            assert config.model_class is not None


class TestRerankerMapping:
    EXPECTED_MULTIMODAL_RERANKERS = [
        "jina-reranker-m0",
        "Qwen3-VL-Reranker-2B",
        "Qwen3-VL-Reranker-8B",
    ]

    EXPECTED_TEXT_RERANKERS = [
        "Qwen3-Reranker-0.6B",
        "Qwen3-Reranker-4B",
        "Qwen3-Reranker-8B",
    ]

    def test_multimodal_rerankers_registered(self):
        for name in self.EXPECTED_MULTIMODAL_RERANKERS:
            assert name in AUTO_RERANKER_MAPPING, f"{name} not in AUTO_RERANKER_MAPPING"

    def test_text_rerankers_registered(self):
        for name in self.EXPECTED_TEXT_RERANKERS:
            assert name in AUTO_RERANKER_MAPPING, f"{name} not in AUTO_RERANKER_MAPPING"

    def test_model_class_enum_complete(self):
        assert RerankerModelClass.MULTIMODAL_BASE.value == "multimodal-base"
        assert RerankerModelClass.QWEN3_VL_RERANKER.value == "qwen3-vl-reranker"
        assert RerankerModelClass.QWEN3_RERANKER.value == "qwen3-reranker"

    def test_class_mapping_has_all_enums(self):
        for cls in RerankerModelClass:
            assert cls in RERANKER_CLASS_MAPPING, f"{cls} not in RERANKER_CLASS_MAPPING"

    def test_mapping_config_has_model_class(self):
        for name, config in AUTO_RERANKER_MAPPING.items():
            assert hasattr(config, "model_class"), f"{name} config missing model_class"
            assert config.model_class is not None
