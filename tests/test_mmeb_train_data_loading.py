"""
Tests for MMEB-train Parquet data loading across all multimodal dataset classes.

Covers:
  - Synthetic Parquet loading (always runs, uses tmp_path)
  - Real MMEB-train data loading (skipped when data is unavailable)
  - image_root_dir relative path resolution
  - Backward compatibility with JSON/JSONL loading
  - All 5 dataset classes: embedder/base, embedder/qwen3_vl,
    reranker/base, reranker/jina_reranker_m0, reranker/qwen3_vl
"""

import json
import os
import types
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMEB_ROOT = "/mnt/nas-tbt/tbt/data/hf_cache/MMEB-train"
MMEB_SUBSET = os.path.join(MMEB_ROOT, "A-OKVQA")
MMEB_IMAGE_ROOT = os.path.join(MMEB_ROOT, "images")

REAL_DATA_AVAILABLE = os.path.isdir(MMEB_SUBSET) and any(
    f.endswith(".parquet") for f in os.listdir(MMEB_SUBSET)
)

# Sample record mimicking MMEB-train schema
SAMPLE_RECORD = {
    "qry": "What is the primary color of the car? A) red B) blue C) green D) yellow",
    "qry_image_path": "images/A-OKVQA/Train/A-OKVQA_image_0.jpg",
    "pos_text": "A) red",
    "pos_image_path": None,
    "neg_text": "B) blue",
    "neg_image_path": None,
}


# ---------------------------------------------------------------------------
# Helpers: mock data-argument objects
# ---------------------------------------------------------------------------

def _make_embedder_args(train_data, image_root_dir=None, cache_path=None):
    """Build a minimal mock that satisfies AbsEmbedderDataArguments fields."""
    args = MagicMock()
    args.train_data = train_data
    args.image_root_dir = image_root_dir
    args.cache_path = cache_path
    args.train_group_size = 2
    args.query_max_len = 64
    args.passage_max_len = 64
    args.pad_to_multiple_of = None
    args.max_example_num_per_dataset = 100_000_000
    args.query_instruction_for_retrieval = None
    args.query_instruction_format = "{}{}"
    args.knowledge_distillation = False
    args.shuffle_ratio = 0.0
    return args


def _make_reranker_args(train_data, image_root_dir=None, cache_path=None):
    """Build a minimal mock that satisfies AbsRerankerDataArguments fields."""
    args = MagicMock()
    args.train_data = train_data
    args.image_root_dir = image_root_dir
    args.cache_path = cache_path
    args.train_group_size = 2
    args.query_max_len = 64
    args.passage_max_len = 64
    args.max_len = 512
    args.pad_to_multiple_of = None
    args.max_example_num_per_dataset = 100_000_000
    args.query_instruction_for_rerank = None
    args.query_instruction_format = "{}{}"
    args.knowledge_distillation = False
    args.shuffle_ratio = 0.0
    args.sep_token = "\n"
    return args


def _write_parquet(path, records):
    """Write a list of dicts as a single Parquet file."""
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def _write_jsonl(path, records):
    """Write a list of dicts as a JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_parquet(tmp_path):
    """Create a temporary Parquet file with MMEB-train-style records."""
    records = [
        {
            "qry": f"Question {i}",
            "qry_image_path": f"images/subset/img_{i}.jpg",
            "pos_text": f"Answer {i}",
            "pos_image_path": None,
            "neg_text": f"Wrong {i}",
            "neg_image_path": None,
        }
        for i in range(10)
    ]
    pq_dir = tmp_path / "SyntheticSubset"
    pq_dir.mkdir()
    pq_file = pq_dir / "train-00000-of-00001.parquet"
    _write_parquet(str(pq_file), records)
    return str(pq_dir), records


@pytest.fixture
def synthetic_jsonl(tmp_path):
    """Create a temporary JSONL file with standard-format records."""
    records = [
        {
            "query": f"Question {i}",
            "query_image": f"/abs/path/img_{i}.jpg",
            "pos": [f"Answer {i}"],
            "pos_images": [None],
            "neg": [f"Wrong {i}"],
            "neg_images": [None],
        }
        for i in range(5)
    ]
    jsonl_file = tmp_path / "examples.jsonl"
    _write_jsonl(str(jsonl_file), records)
    return str(jsonl_file), records


@pytest.fixture
def image_root(tmp_path):
    """Create a fake image_root_dir with a subdirectory structure."""
    root = tmp_path / "image_root"
    root.mkdir()
    (root / "images" / "subset").mkdir(parents=True)
    return str(root)


# ===================================================================
# Embedder – base (HF datasets library)
# ===================================================================

class TestEmbedderBaseParquet:
    """Tests for MultimodalEmbedderTrainDataset (base)."""

    def _make_dataset(self, args, tokenizer=None):
        from FlagEmbedding.finetune.embedder.multimodal.base.dataset import (
            MultimodalEmbedderTrainDataset,
        )
        return MultimodalEmbedderTrainDataset(args, tokenizer=tokenizer)

    def test_load_synthetic_parquet(self, synthetic_parquet, tmp_path):
        pq_dir, records = synthetic_parquet
        args = _make_embedder_args(
            train_data=[pq_dir],
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_image_root_dir_resolution(self, synthetic_parquet, image_root, tmp_path):
        pq_dir, records = synthetic_parquet
        args = _make_embedder_args(
            train_data=[pq_dir],
            image_root_dir=image_root,
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        item = ds[0]
        # query_image should now be an absolute path under image_root
        assert item["query_image"].startswith(image_root)
        assert "images/subset/img_0.jpg" in item["query_image"]

    def test_absolute_paths_unchanged(self, tmp_path):
        """Absolute image paths must not be altered by image_root_dir."""
        records = [
            {
                "qry": "Q",
                "qry_image_path": "/already/absolute/path.jpg",
                "pos_text": "A",
                "pos_image_path": None,
                "neg_text": "B",
                "neg_image_path": None,
            }
        ]
        pq_dir = tmp_path / "AbsPathSubset"
        pq_dir.mkdir()
        _write_parquet(str(pq_dir / "train.parquet"), records)

        args = _make_embedder_args(
            train_data=[str(pq_dir)],
            image_root_dir="/some/root",
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        item = ds[0]
        assert item["query_image"] == "/already/absolute/path.jpg"

    def test_jsonl_still_works(self, synthetic_jsonl, tmp_path):
        jsonl_file, records = synthetic_jsonl
        args = _make_embedder_args(
            train_data=[jsonl_file],
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) == len(records)
        item = ds[0]
        assert item["query_text"] == "Question 0"

    def test_mixed_formats_in_directory(self, tmp_path):
        """A directory with both .parquet and .jsonl files should load both."""
        mix_dir = tmp_path / "MixedSubset"
        mix_dir.mkdir()

        parquet_records = [
            {"qry": "PQ", "qry_image_path": None, "pos_text": "PA", "pos_image_path": None, "neg_text": "PB", "neg_image_path": None}
        ]
        jsonl_records = [
            {"query": "JQ", "query_image": None, "pos": ["JA"], "pos_images": [None], "neg": ["JB"], "neg_images": [None]}
        ]
        _write_parquet(str(mix_dir / "data.parquet"), parquet_records)
        _write_jsonl(str(mix_dir / "data.jsonl"), jsonl_records)

        args = _make_embedder_args(
            train_data=[str(mix_dir)],
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) == 2

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MMEB-train data not available")
    def test_load_real_mmeb_train(self, tmp_path):
        args = _make_embedder_args(
            train_data=[MMEB_SUBSET],
            image_root_dir=MMEB_IMAGE_ROOT,
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) > 0
        item = ds[0]
        assert "query_text" in item
        assert "query_image" in item
        assert "passages_text" in item
        # If image path is present, it should be resolved to absolute
        if item["query_image"] is not None:
            assert os.path.isabs(item["query_image"])


# ===================================================================
# Embedder – qwen3_vl (pandas loading)
# ===================================================================

class TestEmbedderQwen3VLParquet:
    """Tests for Qwen3VLEmbedderTrainDataset."""

    def _make_dataset(self, args, tokenizer=None):
        from FlagEmbedding.finetune.embedder.multimodal.qwen3_vl.dataset import (
            Qwen3VLEmbedderTrainDataset,
        )
        return Qwen3VLEmbedderTrainDataset(args, tokenizer=tokenizer)

    def test_load_synthetic_parquet(self, synthetic_parquet):
        pq_dir, records = synthetic_parquet
        args = _make_embedder_args(train_data=[pq_dir])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_image_root_dir_resolution(self, synthetic_parquet, image_root):
        pq_dir, _ = synthetic_parquet
        args = _make_embedder_args(train_data=[pq_dir], image_root_dir=image_root)
        ds = self._make_dataset(args)
        item = ds[0]
        assert item["query_image"].startswith(image_root)

    def test_jsonl_still_works(self, synthetic_jsonl):
        jsonl_file, records = synthetic_jsonl
        args = _make_embedder_args(train_data=[jsonl_file])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_none_image_not_resolved(self, tmp_path):
        """None image paths should remain None even with image_root_dir set."""
        records = [
            {"qry": "Q", "qry_image_path": None, "pos_text": "A", "pos_image_path": None, "neg_text": "B", "neg_image_path": None}
        ]
        pq_dir = tmp_path / "NoneImgSubset"
        pq_dir.mkdir()
        _write_parquet(str(pq_dir / "train.parquet"), records)

        args = _make_embedder_args(train_data=[str(pq_dir)], image_root_dir="/root")
        ds = self._make_dataset(args)
        item = ds[0]
        assert item["query_image"] is None

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MMEB-train data not available")
    def test_load_real_mmeb_train(self):
        args = _make_embedder_args(
            train_data=[MMEB_SUBSET],
            image_root_dir=MMEB_IMAGE_ROOT,
        )
        ds = self._make_dataset(args)
        assert len(ds) > 0
        item = ds[0]
        assert "query_text" in item


# ===================================================================
# Reranker – base (HF datasets library)
# ===================================================================

class TestRerankerBaseParquet:
    """Tests for MultimodalRerankerTrainDataset (base)."""

    def _make_dataset(self, args, tokenizer=None):
        from FlagEmbedding.finetune.reranker.multimodal.base.dataset import (
            MultimodalRerankerTrainDataset,
        )
        return MultimodalRerankerTrainDataset(args, tokenizer=tokenizer)

    def test_load_synthetic_parquet(self, synthetic_parquet, tmp_path):
        pq_dir, records = synthetic_parquet
        args = _make_reranker_args(
            train_data=[pq_dir],
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_image_root_dir_resolution(self, synthetic_parquet, image_root, tmp_path):
        pq_dir, _ = synthetic_parquet
        args = _make_reranker_args(
            train_data=[pq_dir],
            image_root_dir=image_root,
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        item = ds[0]
        if item["query_image"] is not None:
            assert item["query_image"].startswith(image_root)

    def test_jsonl_still_works(self, synthetic_jsonl, tmp_path):
        jsonl_file, records = synthetic_jsonl
        args = _make_reranker_args(
            train_data=[jsonl_file],
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MMEB-train data not available")
    def test_load_real_mmeb_train(self, tmp_path):
        args = _make_reranker_args(
            train_data=[MMEB_SUBSET],
            image_root_dir=MMEB_IMAGE_ROOT,
            cache_path=str(tmp_path / ".cache"),
        )
        ds = self._make_dataset(args)
        assert len(ds) > 0


# ===================================================================
# Reranker – jina_reranker_m0 (pandas loading)
# ===================================================================

class TestRerankerJinaM0Parquet:
    """Tests for JinaRerankerM0TrainDataset."""

    def _make_dataset(self, args, tokenizer=None):
        from FlagEmbedding.finetune.reranker.multimodal.jina_reranker_m0.dataset import (
            JinaRerankerM0TrainDataset,
        )
        return JinaRerankerM0TrainDataset(args, tokenizer=tokenizer)

    def test_load_synthetic_parquet(self, synthetic_parquet):
        pq_dir, records = synthetic_parquet
        args = _make_reranker_args(train_data=[pq_dir])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_image_root_dir_resolution(self, synthetic_parquet, image_root):
        pq_dir, _ = synthetic_parquet
        args = _make_reranker_args(train_data=[pq_dir], image_root_dir=image_root)
        ds = self._make_dataset(args)
        item = ds[0]
        if item["query_image"] is not None:
            assert item["query_image"].startswith(image_root)

    def test_jsonl_still_works(self, synthetic_jsonl):
        jsonl_file, records = synthetic_jsonl
        args = _make_reranker_args(train_data=[jsonl_file])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MMEB-train data not available")
    def test_load_real_mmeb_train(self):
        args = _make_reranker_args(
            train_data=[MMEB_SUBSET],
            image_root_dir=MMEB_IMAGE_ROOT,
        )
        ds = self._make_dataset(args)
        assert len(ds) > 0


# ===================================================================
# Reranker – qwen3_vl (pandas loading)
# ===================================================================

class TestRerankerQwen3VLParquet:
    """Tests for Qwen3VLRerankerTrainDataset."""

    def _make_dataset(self, args, tokenizer=None):
        from FlagEmbedding.finetune.reranker.multimodal.qwen3_vl.dataset import (
            Qwen3VLRerankerTrainDataset,
        )
        return Qwen3VLRerankerTrainDataset(args, tokenizer=tokenizer)

    def test_load_synthetic_parquet(self, synthetic_parquet):
        pq_dir, records = synthetic_parquet
        args = _make_reranker_args(train_data=[pq_dir])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    def test_image_root_dir_resolution(self, synthetic_parquet, image_root):
        pq_dir, _ = synthetic_parquet
        args = _make_reranker_args(train_data=[pq_dir], image_root_dir=image_root)
        ds = self._make_dataset(args)
        item = ds[0]
        if item["query_image"] is not None:
            assert item["query_image"].startswith(image_root)

    def test_jsonl_still_works(self, synthetic_jsonl):
        jsonl_file, records = synthetic_jsonl
        args = _make_reranker_args(train_data=[jsonl_file])
        ds = self._make_dataset(args)
        assert len(ds) == len(records)

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MMEB-train data not available")
    def test_load_real_mmeb_train(self):
        args = _make_reranker_args(
            train_data=[MMEB_SUBSET],
            image_root_dir=MMEB_IMAGE_ROOT,
        )
        ds = self._make_dataset(args)
        assert len(ds) > 0


# ===================================================================
# Cross-class consistency test
# ===================================================================

class TestCrossClassConsistency:
    """Verify that all dataset classes produce the same output schema
    when given the same Parquet input."""

    def test_output_keys_match(self, synthetic_parquet, tmp_path):
        pq_dir, _ = synthetic_parquet
        cache = str(tmp_path / ".cache")

        emb_args = _make_embedder_args(train_data=[pq_dir], cache_path=cache)
        rer_args = _make_reranker_args(train_data=[pq_dir], cache_path=cache)

        from FlagEmbedding.finetune.embedder.multimodal.base.dataset import MultimodalEmbedderTrainDataset
        from FlagEmbedding.finetune.embedder.multimodal.qwen3_vl.dataset import Qwen3VLEmbedderTrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.base.dataset import MultimodalRerankerTrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.jina_reranker_m0.dataset import JinaRerankerM0TrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.qwen3_vl.dataset import Qwen3VLRerankerTrainDataset

        expected_keys = {"query_text", "query_image", "passages_text", "passages_image", "teacher_scores"}

        datasets_to_test = [
            ("embedder/base", MultimodalEmbedderTrainDataset(emb_args, tokenizer=None)),
            ("embedder/qwen3_vl", Qwen3VLEmbedderTrainDataset(emb_args, tokenizer=None)),
            ("reranker/base", MultimodalRerankerTrainDataset(rer_args, tokenizer=None)),
            ("reranker/jina_m0", JinaRerankerM0TrainDataset(rer_args, tokenizer=None)),
            ("reranker/qwen3_vl", Qwen3VLRerankerTrainDataset(rer_args, tokenizer=None)),
        ]

        for name, ds in datasets_to_test:
            item = ds[0]
            assert set(item.keys()) == expected_keys, (
                f"Dataset {name} returned unexpected keys: {set(item.keys())} vs {expected_keys}"
            )


# ===================================================================
# SUPPORTED_EXTENSIONS attribute test
# ===================================================================

class TestSupportedExtensions:
    """All dataset classes must declare .parquet support."""

    def test_all_classes_support_parquet(self):
        from FlagEmbedding.finetune.embedder.multimodal.base.dataset import MultimodalEmbedderTrainDataset
        from FlagEmbedding.finetune.embedder.multimodal.qwen3_vl.dataset import Qwen3VLEmbedderTrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.base.dataset import MultimodalRerankerTrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.jina_reranker_m0.dataset import JinaRerankerM0TrainDataset
        from FlagEmbedding.finetune.reranker.multimodal.qwen3_vl.dataset import Qwen3VLRerankerTrainDataset

        for cls in [
            MultimodalEmbedderTrainDataset,
            Qwen3VLEmbedderTrainDataset,
            MultimodalRerankerTrainDataset,
            JinaRerankerM0TrainDataset,
            Qwen3VLRerankerTrainDataset,
        ]:
            assert ".parquet" in cls.SUPPORTED_EXTENSIONS, (
                f"{cls.__name__} does not list .parquet in SUPPORTED_EXTENSIONS"
            )
            # Backward compat: JSON/JSONL must still be supported
            assert ".json" in cls.SUPPORTED_EXTENSIONS
            assert ".jsonl" in cls.SUPPORTED_EXTENSIONS
