# Changelog

## 2025-05-10

### Added

#### Multimodal Embedder Training Pipeline

- **Qwen3-VL-Embedding training support** (`FlagEmbedding/finetune/embedder/multimodal/qwen3_vl/`)
  - Full training pipeline: `arguments.py`, `dataset.py`, `modeling.py`, `runner.py`, `trainer.py`, `load_model.py`
  - Uses Qwen3VL processor for text+image input processing
  - Last-token pooling for embedding extraction
  - InfoNCE contrastive loss with in-batch negatives
  - Supports LoRA fine-tuning
  - System prompt: "Represent the user's input."
  - Key fixes:
    - Avoids C++ heap corruption by using plain `json.loads()` instead of `datasets.load_dataset`
    - Forces slow tokenizer (`use_fast=False`) to prevent `tokenizers` + `pyarrow` memory conflict
    - Removes sequence truncation in image-batch padding to preserve image token/feature count invariant
    - Resets `rope_deltas` before each forward to avoid Qwen3VL `compute_3d_position_ids` batch-size mismatch bug

#### Multimodal Reranker Training Pipeline

- **jina-reranker-m0 training support** (`FlagEmbedding/finetune/reranker/multimodal/jina_reranker_m0/`)
  - Full training pipeline with pairwise and listwise loss modes
  - Patches `Qwen2VLConfig` to expose `hidden_size` at top level and disables weight tying (`tie_word_embeddings=False`)
  - Bypasses `AbsRerankerModel.__init__` to avoid `config.pad_token_id` incompatibility with `Qwen2VLConfig`
  - Custom collator processes text-only data with tokenizer and image data with processor
  - Key fixes (same C++ crash prevention as Qwen3-VL):
    - Plain JSON loading instead of `datasets` library
    - Slow tokenizer, `TOKENIZERS_PARALLELISM=false`
    - No `processing_class` passed to `Trainer`

#### Inference Model Support

- **Qwen3-VL-Embedding** (`FlagEmbedding/inference/embedder/qwen3_vl_embedding.py`)
  - 2B and 8B variants
  - `Qwen3VLForConditionalGeneration` based embedder with last-token pooling

- **Qwen3-VL-Reranker** (`FlagEmbedding/inference/reranker/qwen3_vl_reranker.py`)
  - 2B and 8B variants
  - Custom score head for reranking

### Fixed

- `is_torch_fx_available` ImportError in `modeling_minicpm_reranker.py` files (transformers 5.x compatibility)
- `Trainer.__init__` `tokenizer` -> `processing_class` API change in transformers 5.x

### Training Verification

| Model | Steps | Status |
|-------|-------|--------|
| jina-reranker-m0 | 3 | Passed |
| Qwen3-VL-Embedding-2B | 3 | Passed |
