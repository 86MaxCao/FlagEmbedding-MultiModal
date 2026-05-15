"""
Common pytest fixtures and configuration for FlagEmbedding tests.
"""

import os
import pytest
import torch
from packaging import version
import transformers

# Check if we're using transformers v5+
TF_VER = version.parse(getattr(transformers, "__version__", "0.0.0"))
IS_TF_V5_OR_HIGHER = TF_VER >= version.parse("5.0.0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = "/mnt/nas-tbt/tbt/checkpoint/hf_cache"


@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def transformers_version():
    """Return the transformers version."""
    return TF_VER


@pytest.fixture(scope="session")
def hf_cache_dir():
    """Return the path to the HuggingFace cache directory."""
    return HF_CACHE_DIR


@pytest.fixture(scope="session")
def test_images():
    """Return paths to test images."""
    imgs_dir = os.path.join(PROJECT_ROOT, "imgs")
    return {
        "query": os.path.join(imgs_dir, "cir_query.png"),
        "candi_1": os.path.join(imgs_dir, "cir_candi_1.png"),
        "candi_2": os.path.join(imgs_dir, "cir_candi_2.png"),
    }


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires at least 1 GPU")
    config.addinivalue_line("markers", "multi_gpu: requires at least 2 GPUs")


def pytest_collection_modifyitems(config, items):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for item in items:
        if "gpu" in item.keywords and gpu_count < 1:
            item.add_marker(pytest.mark.skip(reason="No GPU available"))
        if "multi_gpu" in item.keywords and gpu_count < 2:
            item.add_marker(pytest.mark.skip(reason="Need at least 2 GPUs"))
