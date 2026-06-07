"""
Training smoke tests for multimodal embedder and reranker fine-tuning.

Each test runs a 3-step ``torchrun`` training job via subprocess and verifies
that the process exits cleanly and produces the expected output directory.

Models covered:
  - gme-Qwen2-VL-2B-Instruct  (embedder)
  - jina-embeddings-v4         (embedder)
  - jina-reranker-m0           (reranker)
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from conftest import HF_CACHE_DIR, PROJECT_ROOT

pytestmark = pytest.mark.gpu

EMBEDDER_TRAIN_DATA = os.path.join(
    PROJECT_ROOT, "examples/finetune/example_data/multimodal/examples.jsonl"
)
RERANKER_TRAIN_DATA = os.path.join(
    PROJECT_ROOT, "examples/finetune/example_data/multimodal/examples_with_scores.jsonl"
)


def _model_path(name: str) -> str:
    return os.path.join(HF_CACHE_DIR, name)


def _skip_if_missing(name: str) -> str:
    path = _model_path(name)
    if not os.path.isdir(path):
        pytest.skip(f"Model not cached: {path}")
    return path


def _run_torchrun(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Launch a ``torchrun`` training process and return the result."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", "1",
    ] + args
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def _embedder_train_args(model_path: str, output_dir: str) -> list[str]:
    """Common training arguments for multimodal embedder fine-tuning."""
    return [
        "-m", "FlagEmbedding.finetune.embedder.multimodal.base",
        "--model_name_or_path", model_path,
        "--use_lora", "True",
        "--lora_rank", "32",
        "--lora_alpha", "64",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj",
        "--save_merged_lora_model", "False",
        "--trust_remote_code", "True",
        "--train_data", EMBEDDER_TRAIN_DATA,
        "--cache_path", os.path.join(output_dir, ".cache"),
        "--train_group_size", "2",
        "--query_max_len", "256",
        "--passage_max_len", "256",
        "--pad_to_multiple_of", "8",
        "--knowledge_distillation", "False",
        "--output_dir", output_dir,
        "--learning_rate", "1e-4",
        "--bf16",
        "--num_train_epochs", "1",
        "--max_steps", "3",
        "--per_device_train_batch_size", "2",
        "--dataloader_drop_last", "True",
        "--warmup_ratio", "0.1",
        "--gradient_checkpointing",
        "--logging_steps", "1",
        "--save_steps", "999",
        "--negatives_cross_device",
        "--temperature", "0.02",
        "--sentence_pooling_method", "last_token",
        "--normalize_embeddings", "True",
        "--kd_loss_type", "m3_kd_loss",
    ]


def _reranker_train_args(model_path: str, output_dir: str) -> list[str]:
    """Common training arguments for multimodal reranker fine-tuning."""
    return [
        "-m", "FlagEmbedding.finetune.reranker.multimodal.base",
        "--model_name_or_path", model_path,
        "--use_lora", "True",
        "--lora_rank", "32",
        "--lora_alpha", "64",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj",
        "--save_merged_lora_model", "False",
        "--trust_remote_code", "True",
        "--train_data", RERANKER_TRAIN_DATA,
        "--cache_path", os.path.join(output_dir, ".cache"),
        "--train_group_size", "2",
        "--query_max_len", "256",
        "--passage_max_len", "256",
        "--pad_to_multiple_of", "8",
        "--output_dir", output_dir,
        "--learning_rate", "1e-4",
        "--bf16",
        "--num_train_epochs", "1",
        "--max_steps", "3",
        "--per_device_train_batch_size", "2",
        "--dataloader_drop_last", "True",
        "--warmup_ratio", "0.1",
        "--gradient_checkpointing",
        "--logging_steps", "1",
        "--save_steps", "999",
    ]


# ---------------------------------------------------------------------------
# gme-Qwen2-VL-2B-Instruct (embedder)
# ---------------------------------------------------------------------------

class TestGmeQwen2VLTraining:
    MODEL_NAME = "gme-Qwen2-VL-2B-Instruct"

    def test_lora_finetune_smoke(self, tmp_path):
        model_path = _skip_if_missing(self.MODEL_NAME)
        output_dir = str(tmp_path / "gme_train_output")

        result = _run_torchrun(_embedder_train_args(model_path, output_dir))

        assert result.returncode == 0, (
            f"Training failed (exit {result.returncode}).\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
        assert os.path.isdir(output_dir), "Output directory was not created"


# ---------------------------------------------------------------------------
# jina-embeddings-v4 (embedder)
# ---------------------------------------------------------------------------

class TestJinaEmbeddingsV4Training:
    MODEL_NAME = "jina-embeddings-v4"

    def test_lora_finetune_smoke(self, tmp_path):
        model_path = _skip_if_missing(self.MODEL_NAME)
        output_dir = str(tmp_path / "jina_emb_train_output")

        result = _run_torchrun(_embedder_train_args(model_path, output_dir))

        assert result.returncode == 0, (
            f"Training failed (exit {result.returncode}).\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
        assert os.path.isdir(output_dir), "Output directory was not created"


# ---------------------------------------------------------------------------
# jina-reranker-m0 (reranker)
# ---------------------------------------------------------------------------

class TestJinaRerankerM0Training:
    MODEL_NAME = "jina-reranker-m0"

    def test_lora_finetune_smoke(self, tmp_path):
        model_path = _skip_if_missing(self.MODEL_NAME)
        output_dir = str(tmp_path / "jina_reranker_train_output")

        result = _run_torchrun(_reranker_train_args(model_path, output_dir))

        assert result.returncode == 0, (
            f"Training failed (exit {result.returncode}).\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
        assert os.path.isdir(output_dir), "Output directory was not created"
