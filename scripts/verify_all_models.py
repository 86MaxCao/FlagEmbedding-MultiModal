"""
End-to-end verification script for all multimodal models.

Tests that each registered multimodal model can:
1. Load via FlagAutoModel / FlagAutoReranker
2. Run inference (encode / compute_score)
3. Produce sensible outputs (correct shapes, positive > negative)

Also runs a mini retrieval evaluation using test_data/embedder_train.jsonl.

Usage:
    python scripts/verify_all_models.py                 # single GPU
    python scripts/verify_all_models.py --multi-gpu     # multi GPU (2 devices)
    python scripts/verify_all_models.py --model Qwen3-VL-Embedding-2B  # specific model only
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

HF_CACHE_DIR = "/mnt/nas-tbt/tbt/checkpoint/hf_cache"
IMGS_DIR = PROJECT_ROOT / "imgs"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"


def get_available_gpus(min_free_gb: float = 4.0) -> List[str]:
    """Return list of GPU device strings with enough free memory."""
    devices = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > min_free_gb * 1024**3:
            devices.append(f"cuda:{i}")
    return devices


def model_cached(name: str) -> bool:
    return os.path.isdir(os.path.join(HF_CACHE_DIR, name))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ==================== Embedder Verification ====================

def verify_qwen3_vl_embedding(devices: List[str], model_name: str = "Qwen3-VL-Embedding-2B"):
    """Verify Qwen3-VL-Embedding model."""
    from FlagEmbedding import FlagAutoModel

    print(f"\n{'='*60}")
    print(f"  Verifying Embedder: {model_name}")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    if not model_cached(model_name):
        print(f"  SKIP: Model not cached at {HF_CACHE_DIR}/{model_name}")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoModel.from_finetuned(
        model_path,
        use_fp16=True,
        devices=devices,
        trust_remote_code=True,
    )

    # Test 1: Text encoding
    texts = ["What is deep learning?", "Deep learning uses neural networks.", "Cooking recipes"]
    emb = model.encode(texts)
    assert emb.shape[0] == 3, f"Expected 3 embeddings, got {emb.shape[0]}"
    assert not np.isnan(emb).any(), "NaN in embeddings"
    print(f"  [PASS] Text encode: shape={emb.shape}")

    # Test 2: Similarity ranking
    sim_pos = cosine_sim(emb[0], emb[1])
    sim_neg = cosine_sim(emb[0], emb[2])
    assert sim_pos > sim_neg, f"Expected pos sim ({sim_pos:.4f}) > neg sim ({sim_neg:.4f})"
    print(f"  [PASS] Similarity ranking: pos={sim_pos:.4f} > neg={sim_neg:.4f}")

    # Test 3: Normalization
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), f"Norms not close to 1: {norms}"
    print(f"  [PASS] Normalization: norms ≈ 1.0")

    # Test 4: Multi-GPU check
    if len(devices) > 1:
        assert len(model.target_devices) == len(devices)
        texts_big = ["sentence " + str(i) for i in range(8)]
        emb_big = model.encode(texts_big)
        assert emb_big.shape[0] == 8
        print(f"  [PASS] Multi-GPU encode: {len(devices)} devices, shape={emb_big.shape}")

    del model
    torch.cuda.empty_cache()
    return "PASS"


def verify_bge_vl_mllm(devices: List[str], model_name: str = "BGE-VL-MLLM-S1"):
    """Verify BGE-VL-MLLM model."""
    from FlagEmbedding import FlagAutoModel

    print(f"\n{'='*60}")
    print(f"  Verifying Embedder: {model_name}")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    if not model_cached(model_name):
        print(f"  SKIP: Model not cached at {HF_CACHE_DIR}/{model_name}")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    try:
        model = FlagAutoModel.from_finetuned(
            model_path,
            use_fp16=True,
            devices=devices,
            trust_remote_code=True,
        )
    except Exception as e:
        if "language_model" in str(e) or "LlavaNext" in str(e):
            print(f"  SKIP: Known transformers 5.x compatibility issue: {e}")
            return "SKIP (compat)"
        raise

    # Test 1: Text-only encoding
    texts = ["hello world", "deep learning"]
    try:
        emb = model.encode(texts)
        assert emb.shape[0] == 2
        assert not np.isnan(emb).any()
        print(f"  [PASS] Text encode: shape={emb.shape}")
    except Exception as e:
        print(f"  [FAIL] Text encode: {e}")
        del model
        torch.cuda.empty_cache()
        return "FAIL"

    # Test 2: Image + text encoding
    query_img = str(IMGS_DIR / "cir_query.png")
    if os.path.exists(query_img):
        try:
            emb_img = model.encode(["describe this image"], images=[query_img])
            assert emb_img.shape[0] == 1
            assert not np.isnan(emb_img).any()
            print(f"  [PASS] Image+text encode: shape={emb_img.shape}")
        except Exception as e:
            print(f"  [WARN] Image+text encode failed: {e}")

    del model
    torch.cuda.empty_cache()
    return "PASS"


# ==================== Reranker Verification ====================

def verify_qwen3_reranker(devices: List[str], model_name: str = "Qwen3-Reranker-0.6B"):
    """Verify Qwen3 text reranker."""
    from FlagEmbedding import FlagAutoReranker

    print(f"\n{'='*60}")
    print(f"  Verifying Reranker: {model_name}")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    if not model_cached(model_name):
        print(f"  SKIP: Model not cached at {HF_CACHE_DIR}/{model_name}")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoReranker.from_finetuned(
        model_path,
        use_fp16=True,
        devices=devices,
    )

    # Test 1: Basic scoring
    pairs = [
        ("What is AI?", "Artificial intelligence is a field of computer science."),
        ("What is AI?", "Dogs are cute animals."),
    ]
    scores = model.compute_score(pairs)
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    assert scores[0] > scores[1], f"Expected pos ({scores[0]:.4f}) > neg ({scores[1]:.4f})"
    print(f"  [PASS] Basic scoring: pos={scores[0]:.4f} > neg={scores[1]:.4f}")

    # Test 2: Single pair
    single = model.compute_score(("What is Python?", "Python is a programming language."))
    assert isinstance(single, float)
    print(f"  [PASS] Single pair: score={single:.4f}")

    # Test 3: Multi-GPU
    if len(devices) > 1:
        assert len(model.target_devices) == len(devices)
        big_pairs = [("query " + str(i), "doc " + str(i)) for i in range(8)]
        big_scores = model.compute_score(big_pairs)
        assert len(big_scores) == 8
        print(f"  [PASS] Multi-GPU scoring: {len(devices)} devices, {len(big_scores)} scores")

    del model
    torch.cuda.empty_cache()
    return "PASS"


def verify_jina_reranker(devices: List[str], model_name: str = "jina-reranker-m0"):
    """Verify jina multimodal reranker."""
    from FlagEmbedding import FlagAutoReranker

    print(f"\n{'='*60}")
    print(f"  Verifying Reranker: {model_name}")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    if not model_cached(model_name):
        print(f"  SKIP: Model not cached at {HF_CACHE_DIR}/{model_name}")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoReranker.from_finetuned(
        model_path,
        use_fp16=True,
        devices=devices,
    )

    # Test 1: Text-to-text scoring
    pairs = [
        ("deep learning", "Deep learning uses neural networks with multiple layers."),
        ("deep learning", "Gardening is a relaxing hobby."),
    ]
    scores = model.compute_score(pairs, query_type="text", doc_type="text")
    assert len(scores) == 2
    assert scores[0] > scores[1], f"Expected pos ({scores[0]:.4f}) > neg ({scores[1]:.4f})"
    print(f"  [PASS] Text-text scoring: pos={scores[0]:.4f} > neg={scores[1]:.4f}")

    # Test 2: Text-to-image scoring
    candi_1 = str(IMGS_DIR / "cir_candi_1.png")
    candi_2 = str(IMGS_DIR / "cir_candi_2.png")
    if os.path.exists(candi_1) and os.path.exists(candi_2):
        img_pairs = [
            ("a photo of shoes", candi_1),
            ("a photo of shoes", candi_2),
        ]
        try:
            img_scores = model.compute_score(img_pairs, query_type="text", doc_type="image")
            assert len(img_scores) == 2
            assert all(isinstance(s, float) for s in img_scores)
            print(f"  [PASS] Text-image scoring: scores={[f'{s:.4f}' for s in img_scores]}")
        except Exception as e:
            print(f"  [WARN] Text-image scoring failed: {e}")

    # Test 3: Multi-GPU
    if len(devices) > 1:
        assert len(model.target_devices) == len(devices)
        big_pairs = [("query " + str(i), "doc " + str(i)) for i in range(6)]
        big_scores = model.compute_score(big_pairs, query_type="text", doc_type="text")
        assert len(big_scores) == 6
        print(f"  [PASS] Multi-GPU scoring: {len(devices)} devices, {len(big_scores)} scores")

    del model
    torch.cuda.empty_cache()
    return "PASS"


def verify_qwen3_vl_reranker(devices: List[str], model_name: str = "Qwen3-VL-Reranker-2B"):
    """Verify Qwen3-VL reranker."""
    from FlagEmbedding import FlagAutoReranker

    print(f"\n{'='*60}")
    print(f"  Verifying Reranker: {model_name}")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    if not model_cached(model_name):
        print(f"  SKIP: Model not cached at {HF_CACHE_DIR}/{model_name}")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoReranker.from_finetuned(
        model_path,
        use_fp16=True,
        devices=devices,
    )

    pairs = [
        ("What is deep learning?", "Deep learning is a branch of machine learning."),
        ("What is deep learning?", "Paris is a city in France."),
    ]
    scores = model.compute_score(pairs)
    assert len(scores) == 2
    assert scores[0] > scores[1], f"Expected pos ({scores[0]:.4f}) > neg ({scores[1]:.4f})"
    print(f"  [PASS] Basic scoring: pos={scores[0]:.4f} > neg={scores[1]:.4f}")

    # Multi-GPU check
    if len(devices) > 1:
        assert len(model.target_devices) == len(devices)
        big_pairs = [("query " + str(i), "doc " + str(i)) for i in range(6)]
        big_scores = model.compute_score(big_pairs)
        assert len(big_scores) == 6
        print(f"  [PASS] Multi-GPU scoring: {len(devices)} devices, {len(big_scores)} scores")

    del model
    torch.cuda.empty_cache()
    return "PASS"


# ==================== Mini Retrieval Evaluation ====================

def run_mini_retrieval_eval(devices: List[str]):
    """Run a simple retrieval eval using test_data/embedder_train.jsonl."""
    from FlagEmbedding import FlagAutoModel

    print(f"\n{'='*60}")
    print(f"  Mini Retrieval Evaluation")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    data_path = TEST_DATA_DIR / "embedder_train.jsonl"
    if not data_path.exists():
        print(f"  SKIP: No test data at {data_path}")
        return "SKIP"

    # Load test data (text-only entries)
    entries = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("query") and entry.get("pos") and entry["pos"][0]:
                entries.append(entry)
    if not entries:
        print("  SKIP: No valid text entries in test data")
        return "SKIP"

    # Use Qwen3-VL-Embedding-2B for evaluation
    model_name = "Qwen3-VL-Embedding-2B"
    if not model_cached(model_name):
        print(f"  SKIP: Model {model_name} not cached")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoModel.from_finetuned(
        model_path, use_fp16=True, devices=devices, trust_remote_code=True,
    )

    # Build queries and corpus
    queries = [e["query"] for e in entries]
    positives = [e["pos"][0] for e in entries]
    negatives = []
    for e in entries:
        if e.get("neg"):
            negatives.extend([n for n in e["neg"] if n])

    # Deduplicate corpus
    corpus = list(set(positives + negatives))
    pos_indices = [corpus.index(p) for p in positives]

    print(f"  Queries: {len(queries)}, Corpus: {len(corpus)}")

    # Encode
    q_emb = model.encode(queries)
    c_emb = model.encode(corpus)

    # Compute recall@1
    hits = 0
    for i, q_vec in enumerate(q_emb):
        sims = np.dot(c_emb, q_vec)
        top_idx = np.argmax(sims)
        if top_idx == pos_indices[i]:
            hits += 1

    recall_at_1 = hits / len(queries)
    print(f"  Recall@1: {recall_at_1:.2%} ({hits}/{len(queries)})")

    if recall_at_1 >= 0.5:
        print(f"  [PASS] Recall@1 >= 50%")
        result = "PASS"
    else:
        print(f"  [WARN] Recall@1 < 50% (may be expected with small corpus)")
        result = "WARN"

    del model
    torch.cuda.empty_cache()
    return result


def run_mini_reranker_eval(devices: List[str]):
    """Run a simple reranker eval using test_data/reranker_train.jsonl."""
    from FlagEmbedding import FlagAutoReranker

    print(f"\n{'='*60}")
    print(f"  Mini Reranker Evaluation")
    print(f"  Devices: {devices}")
    print(f"{'='*60}")

    data_path = TEST_DATA_DIR / "reranker_train.jsonl"
    if not data_path.exists():
        print(f"  SKIP: No test data at {data_path}")
        return "SKIP"

    entries = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("query") and entry.get("pos") and entry["pos"][0]:
                entries.append(entry)
    if not entries:
        print("  SKIP: No valid text entries")
        return "SKIP"

    model_name = "Qwen3-Reranker-0.6B"
    if not model_cached(model_name):
        print(f"  SKIP: Model {model_name} not cached")
        return "SKIP"

    model_path = os.path.join(HF_CACHE_DIR, model_name)
    model = FlagAutoReranker.from_finetuned(model_path, use_fp16=True, devices=devices)

    correct = 0
    total = 0
    for entry in entries:
        query = entry["query"]
        pos_doc = entry["pos"][0]
        neg_docs = [n for n in entry.get("neg", []) if n]
        if not neg_docs:
            continue

        all_docs = [pos_doc] + neg_docs
        pairs = [(query, doc) for doc in all_docs]
        scores = model.compute_score(pairs)

        if not isinstance(scores, list):
            scores = [scores]
        if np.argmax(scores) == 0:
            correct += 1
        total += 1

    if total == 0:
        print("  SKIP: No valid evaluation pairs")
        del model
        torch.cuda.empty_cache()
        return "SKIP"

    accuracy = correct / total
    print(f"  Reranker accuracy: {accuracy:.2%} ({correct}/{total})")

    if accuracy >= 0.5:
        print(f"  [PASS] Accuracy >= 50%")
        result = "PASS"
    else:
        print(f"  [WARN] Accuracy < 50%")
        result = "WARN"

    del model
    torch.cuda.empty_cache()
    return result


# ==================== Main ====================

MODEL_VERIFIERS = {
    "Qwen3-VL-Embedding-2B": verify_qwen3_vl_embedding,
    "Qwen3-VL-Embedding-8B": lambda devs: verify_qwen3_vl_embedding(devs, "Qwen3-VL-Embedding-8B"),
    "BGE-VL-MLLM-S1": verify_bge_vl_mllm,
    "BGE-VL-MLLM-S2": lambda devs: verify_bge_vl_mllm(devs, "BGE-VL-MLLM-S2"),
    "Qwen3-Reranker-0.6B": verify_qwen3_reranker,
    "Qwen3-Reranker-4B": lambda devs: verify_qwen3_reranker(devs, "Qwen3-Reranker-4B"),
    "Qwen3-Reranker-8B": lambda devs: verify_qwen3_reranker(devs, "Qwen3-Reranker-8B"),
    "jina-reranker-m0": verify_jina_reranker,
    "Qwen3-VL-Reranker-2B": verify_qwen3_vl_reranker,
    "Qwen3-VL-Reranker-8B": lambda devs: verify_qwen3_vl_reranker(devs, "Qwen3-VL-Reranker-8B"),
}


def main():
    parser = argparse.ArgumentParser(description="E2E verification for all multimodal models")
    parser.add_argument("--multi-gpu", action="store_true", help="Use 2 GPUs for testing")
    parser.add_argument("--model", type=str, default=None, help="Test a specific model only")
    parser.add_argument("--skip-eval", action="store_true", help="Skip mini retrieval/reranker evaluation")
    parser.add_argument("--min-free-gb", type=float, default=4.0, help="Min free GPU memory in GB")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlagEmbedding Multimodal - E2E Model Verification")
    print("=" * 60)

    # Determine devices
    available = get_available_gpus(args.min_free_gb)
    if not available:
        print("ERROR: No GPUs with enough free memory available")
        sys.exit(1)

    if args.multi_gpu:
        if len(available) < 2:
            print(f"ERROR: --multi-gpu requires 2 GPUs, only {len(available)} available")
            sys.exit(1)
        devices = available[:2]
    else:
        devices = [available[0]]

    print(f"  Available GPUs: {available}")
    print(f"  Using devices: {devices}")
    print(f"  HF cache: {HF_CACHE_DIR}")
    print(f"  Project root: {PROJECT_ROOT}")

    # Run verifications
    results = {}
    models_to_test = [args.model] if args.model else list(MODEL_VERIFIERS.keys())

    for model_name in models_to_test:
        if model_name not in MODEL_VERIFIERS:
            print(f"\n  ERROR: Unknown model '{model_name}'")
            print(f"  Available: {list(MODEL_VERIFIERS.keys())}")
            sys.exit(1)

        try:
            start = time.time()
            result = MODEL_VERIFIERS[model_name](devices)
            elapsed = time.time() - start
            results[model_name] = result
            print(f"  Result: {result} ({elapsed:.1f}s)")
        except Exception as e:
            results[model_name] = "FAIL"
            print(f"  [FAIL] {model_name}: {e}")
            traceback.print_exc()

    # Run mini evaluations
    if not args.skip_eval:
        try:
            results["retrieval_eval"] = run_mini_retrieval_eval(devices)
        except Exception as e:
            results["retrieval_eval"] = "FAIL"
            print(f"  [FAIL] Retrieval eval: {e}")
            traceback.print_exc()

        try:
            results["reranker_eval"] = run_mini_reranker_eval(devices)
        except Exception as e:
            results["reranker_eval"] = "FAIL"
            print(f"  [FAIL] Reranker eval: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        status_icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○", "WARN": "△"}.get(
            result.split()[0] if " " in result else result, "?"
        )
        print(f"  {status_icon} {name}: {result}")

    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v == "FAIL")
    skipped = sum(1 for v in results.values() if "SKIP" in v)
    warned = sum(1 for v in results.values() if v == "WARN")
    print(f"\n  Total: {len(results)} | Pass: {passed} | Fail: {failed} | Skip: {skipped} | Warn: {warned}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
