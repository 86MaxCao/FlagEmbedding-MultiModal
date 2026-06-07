"""
Tests for multimodal embedder inference.

Covers BGE-VL-MLLM-S1 and Qwen3-VL-Embedding-2B via FlagAutoModel.
"""
import os
import pytest
import numpy as np
import torch

from conftest import HF_CACHE_DIR, PROJECT_ROOT

pytestmark = pytest.mark.gpu


def _model_path(name):
    return os.path.join(HF_CACHE_DIR, name)


def _skip_if_missing(name):
    path = _model_path(name)
    if not os.path.isdir(path):
        pytest.skip(f"Model not cached: {path}")


# ---------------------------------------------------------------------------
# BGE-VL-MLLM-S1
# ---------------------------------------------------------------------------

class TestBGEVLMLLMS1:
    MODEL_NAME = "BGE-VL-MLLM-S1"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoModel
        return FlagAutoModel.from_finetuned(
            _model_path(self.MODEL_NAME),
            model_class="multimodal-mllm",
            use_fp16=True,
            devices="cuda:0",
            trust_remote_code=True,
        )

    def test_encode_text_only(self, model):
        emb = model.encode_queries(
            queries=["What is the capital of France?"],
            task_instruction="Given a question, retrieve relevant passages: ",
        )
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_image_only(self, model, test_images):
        emb = model.encode_corpus(images=[test_images["query"]])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_text_and_image(self, model, test_images):
        emb = model.encode_queries(
            queries=["Make the background dark"],
            images=[test_images["query"]],
            task_instruction="Retrieve the target image: ",
        )
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1

    def test_normalize_embeddings(self, model):
        emb = model.encode_queries(
            queries=["hello world"],
            task_instruction="Retrieve: ",
        )
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_similarity_ranking(self, model, test_images):
        query_emb = model.encode_queries(
            queries=["a dark photo at night"],
            images=[test_images["query"]],
            task_instruction="Retrieve the target image: ",
        )
        corpus_emb = model.encode_corpus(
            images=[test_images["candi_1"], test_images["candi_2"]],
        )
        scores = query_emb @ corpus_emb.T
        assert scores.shape == (1, 2)

    def test_batch_encode(self, model):
        texts = ["hello world", "foo bar", "baz qux"]
        emb = model.encode_corpus(corpus=texts)
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 3


# ---------------------------------------------------------------------------
# Qwen3-VL-Embedding-2B
# ---------------------------------------------------------------------------

class TestQwen3VLEmbedding2B:
    MODEL_NAME = "Qwen3-VL-Embedding-2B"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoModel
        return FlagAutoModel.from_finetuned(
            _model_path(self.MODEL_NAME),
            use_fp16=True,
            devices="cuda:0",
            trust_remote_code=True,
        )

    def test_encode_text_only(self, model):
        emb = model.encode(["What is the capital of France?"])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_with_instruction(self, model):
        emb = model.encode_queries(
            queries=["What is deep learning?"],
            task_instruction="Given a question, retrieve relevant passages that answer it.",
        )
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 1

    def test_encode_multimodal(self, model, test_images):
        emb = model.encode_queries(
            queries=["describe this image"],
            images=[test_images["query"]],
            task_instruction="Retrieve relevant images: ",
        )
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_normalize_embeddings(self, model):
        emb = model.encode(["hello world"])
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_batch_encode(self, model):
        texts = ["hello", "world", "foo", "bar"]
        emb = model.encode(texts)
        assert emb.shape[0] == 4

    def test_corpus_encode(self, model):
        emb = model.encode_corpus(corpus=["Paris is the capital of France."])
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 1


# ---------------------------------------------------------------------------
# jina-embeddings-v4 (multimodal: text + image)
# ---------------------------------------------------------------------------

class TestJinaEmbeddingsV4:
    MODEL_NAME = "jina-embeddings-v4"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoModel
        return FlagAutoModel.from_finetuned(
            _model_path(self.MODEL_NAME),
            model_class="jina-embeddings-v4",
            use_fp16=True,
            devices="cuda:0",
            trust_remote_code=True,
        )

    def test_encode_text_query(self, model):
        emb = model.encode_queries(queries=["What is the capital of France?"])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_text_corpus(self, model):
        emb = model.encode_corpus(corpus=["Paris is the capital of France."])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_image(self, model, test_images):
        emb = model.encode_queries(images=[test_images["query"]])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_normalize_embeddings(self, model):
        emb = model.encode_queries(queries=["hello world"])
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_similarity_text_image(self, model, test_images):
        query_emb = model.encode_queries(queries=["a photo of a person"])
        image_emb = model.encode_corpus(images=[test_images["query"]])
        scores = query_emb @ image_emb.T
        assert scores.shape == (1, 1)

    def test_batch_encode(self, model):
        texts = ["hello world", "foo bar", "baz qux"]
        emb = model.encode_corpus(corpus=texts)
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 3


# ---------------------------------------------------------------------------
# gme-Qwen2-VL-2B-Instruct
# ---------------------------------------------------------------------------

class TestGmeQwen2VL:
    MODEL_NAME = "gme-Qwen2-VL-2B-Instruct"

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_missing(self.MODEL_NAME)
        from FlagEmbedding import FlagAutoModel
        return FlagAutoModel.from_finetuned(
            _model_path(self.MODEL_NAME),
            model_class="gme-qwen2vl",
            use_fp16=True,
            devices="cuda:0",
            trust_remote_code=True,
        )

    def test_encode_text_only(self, model):
        emb = model.encode_queries(queries=["What is the capital of France?"])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_image_only(self, model, test_images):
        emb = model.encode_corpus(images=[test_images["query"]])
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_encode_text_and_image(self, model, test_images):
        emb = model.encode_queries(
            queries=["describe this image"],
            images=[test_images["query"]],
        )
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert not np.isnan(emb).any()

    def test_normalize_embeddings(self, model):
        emb = model.encode_queries(queries=["hello world"])
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_similarity_ranking(self, model, test_images):
        query_emb = model.encode_queries(
            queries=["a dark photo at night"],
            images=[test_images["query"]],
        )
        corpus_emb = model.encode_corpus(
            images=[test_images["candi_1"], test_images["candi_2"]],
        )
        scores = query_emb @ corpus_emb.T
        assert scores.shape == (1, 2)

    def test_batch_encode(self, model):
        texts = ["hello world", "foo bar", "baz qux"]
        emb = model.encode_corpus(corpus=texts)
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 3
