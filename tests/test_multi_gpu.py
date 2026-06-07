"""
Tests for multi-GPU inference.

Verifies that models can distribute workload across multiple GPUs.
"""
import os
import pytest
import numpy as np

from conftest import HF_CACHE_DIR

pytestmark = pytest.mark.multi_gpu


def _model_path(name):
    return os.path.join(HF_CACHE_DIR, name)


def _skip_if_missing(name):
    path = _model_path(name)
    if not os.path.isdir(path):
        pytest.skip(f"Model not cached: {path}")


def _get_two_free_gpus():
    """Find two GPUs with enough free memory."""
    import torch
    devices = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > 4 * 1024**3:  # at least 4GB free
            devices.append(f"cuda:{i}")
        if len(devices) >= 2:
            break
    if len(devices) < 2:
        pytest.skip("Cannot find 2 GPUs with enough free memory")
    return devices


class TestQwen3VLEmbeddingMultiGPU:
    MODEL_NAME = "Qwen3-VL-Embedding-2B"

    @pytest.fixture(scope="class")
    def devices(self):
        return _get_two_free_gpus()

    @pytest.fixture(scope="class")
    def model(self, devices):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoModel
        return FlagAutoModel.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices=devices,
            trust_remote_code=True,
        )

    def test_multi_gpu_text_encode(self, model, devices):
        assert len(model.target_devices) == 2
        texts = ["hello world", "deep learning", "machine learning", "neural networks"]
        emb = model.encode(texts)
        assert emb.shape[0] == 4
        assert not np.isnan(emb).any()

    def test_multi_gpu_consistency(self, model, devices):
        """Results should be the same regardless of GPU count."""
        texts = ["test sentence one", "test sentence two"]
        emb_multi = model.encode(texts)

        # Single device encode for comparison
        emb_single = model.encode_single_device(
            texts, device=devices[0], convert_to_numpy=True,
        )
        np.testing.assert_allclose(emb_multi, emb_single, atol=1e-4)


class TestQwen3RerankerMultiGPU:
    MODEL_NAME = "Qwen3-Reranker-0.6B"

    @pytest.fixture(scope="class")
    def devices(self):
        return _get_two_free_gpus()

    @pytest.fixture(scope="class")
    def model(self, devices):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoReranker
        return FlagAutoReranker.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices=devices,
        )

    def test_multi_gpu_compute_score(self, model, devices):
        assert len(model.target_devices) == 2
        pairs = [
            ("What is AI?", "Artificial intelligence is a field of CS."),
            ("What is deep learning?", "Deep learning uses neural networks."),
            ("What is Python?", "Python is a programming language."),
            ("What is gravity?", "Dogs are cute."),
        ]
        scores = model.compute_score(pairs)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] > scores[3]

    def test_multi_gpu_consistency(self, model, devices):
        """Scores should match single-GPU results (within FP16 tolerance)."""
        pairs = [
            ("test query", "test document one"),
            ("test query", "test document two"),
        ]
        scores_multi = model.compute_score(pairs)
        scores_single = model.compute_score_single_gpu(
            pairs, device=devices[0],
        )
        for sm, ss in zip(scores_multi, scores_single):
            assert abs(sm - ss) < 5e-3


class TestJinaRerankerMultiGPU:
    MODEL_NAME = "jina-reranker-m0"

    @pytest.fixture(scope="class")
    def devices(self):
        return _get_two_free_gpus()

    @pytest.fixture(scope="class")
    def model(self, devices):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoReranker
        return FlagAutoReranker.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices=devices,
        )

    def test_multi_gpu_text_pairs(self, model, devices):
        assert len(model.target_devices) == 2
        pairs = [
            ("deep learning", "Deep learning uses neural networks."),
            ("deep learning", "Gardening is a hobby."),
            ("Paris", "Paris is the capital of France."),
            ("Paris", "Dogs are animals."),
        ]
        scores = model.compute_score(pairs, query_type="text", doc_type="text")
        assert len(scores) == 4
        assert scores[0] > scores[1]
        assert scores[2] > scores[3]
