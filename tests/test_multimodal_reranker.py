"""
Tests for multimodal and text reranker inference.

Covers jina-reranker-m0, Qwen3-VL-Reranker-2B, and Qwen3-Reranker-0.6B.
"""
import os
import pytest
import numpy as np

from conftest import HF_CACHE_DIR

pytestmark = pytest.mark.gpu


def _model_path(name):
    return os.path.join(HF_CACHE_DIR, name)


def _skip_if_missing(name):
    path = _model_path(name)
    if not os.path.isdir(path):
        pytest.skip(f"Model not cached: {path}")


# ---------------------------------------------------------------------------
# jina-reranker-m0 (multimodal: text query → image doc)
# ---------------------------------------------------------------------------

class TestJinaRerankerM0:
    MODEL_NAME = "jina-reranker-m0"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoReranker
        return FlagAutoReranker.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices="cuda:0",
        )

    def test_text_to_text(self, model):
        pairs = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the capital of France?", "Berlin is in Germany."),
        ]
        scores = model.compute_score(
            pairs, query_type="text", doc_type="text",
        )
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] > scores[1]

    def test_text_to_image(self, model, test_images):
        pairs = [
            ("a photo of a person", test_images["query"]),
            ("a photo of a cat", test_images["query"]),
        ]
        scores = model.compute_score(
            pairs, query_type="text", doc_type="image",
        )
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_single_pair(self, model):
        score = model.compute_score(
            ("hello", "world"), query_type="text", doc_type="text",
        )
        assert isinstance(score, float)

    def test_score_range_normalized(self, model):
        pairs = [("query", "document")]
        score = model.compute_score(
            pairs, query_type="text", doc_type="text", normalize=True,
        )
        if isinstance(score, list):
            score = score[0]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Qwen3-VL-Reranker-2B (multimodal cross-encoder)
# ---------------------------------------------------------------------------

class TestQwen3VLReranker2B:
    MODEL_NAME = "Qwen3-VL-Reranker-2B"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoReranker
        return FlagAutoReranker.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices="cuda:0",
        )

    def test_text_pairs(self, model):
        pairs = [
            ("What is deep learning?", "Deep learning is a subset of machine learning."),
            ("What is deep learning?", "The weather is sunny today."),
        ]
        scores = model.compute_score(pairs)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_single_pair(self, model):
        score = model.compute_score(
            ("hello", "world"),
        )
        assert isinstance(score, float)

    def test_batch_consistency(self, model):
        pair = ("What is Python?", "Python is a programming language.")
        single_score = model.compute_score(pair)
        batch_scores = model.compute_score([pair, pair])
        assert isinstance(batch_scores, list)
        assert abs(batch_scores[0] - single_score) < 1e-4
        assert abs(batch_scores[1] - single_score) < 1e-4


# ---------------------------------------------------------------------------
# Qwen3-Reranker-0.6B (text-only cross-encoder)
# ---------------------------------------------------------------------------

class TestQwen3Reranker06B:
    MODEL_NAME = "Qwen3-Reranker-0.6B"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoReranker
        return FlagAutoReranker.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices="cuda:0",
        )

    def test_basic_scoring(self, model):
        pairs = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the capital of France?", "Python is a programming language."),
        ]
        scores = model.compute_score(pairs)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_score_range(self, model):
        pairs = [("query", "document")]
        scores = model.compute_score(pairs)
        if isinstance(scores, list):
            score = scores[0]
        else:
            score = scores
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_pair(self, model):
        score = model.compute_score(
            ("What is AI?", "Artificial intelligence is a field of computer science."),
        )
        assert isinstance(score, float)
        assert score > 0.3

    def test_batch_larger(self, model):
        pairs = [
            ("query1", "relevant doc for query1"),
            ("query2", "relevant doc for query2"),
            ("query3", "irrelevant noise"),
            ("query4", "random text here"),
        ]
        scores = model.compute_score(pairs)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)
