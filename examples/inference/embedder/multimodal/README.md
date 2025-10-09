# Multimodal MLLM Embedder Examples

This directory contains examples for using the multimodal MLLM (Multi-modal Large Language Model) embedder.

## Prerequisites

Before running the examples, make sure you have:

1. Installed the required dependencies
2. Downloaded the BGE-VL-MLLM-S1 model or have access to it via HuggingFace
3. Prepared test images (if testing with images)

## Test Scripts

### 1. `auto_mllm_single_device.py`

This script demonstrates how to use the FlagAutoModel to automatically load and use the multimodal model. It includes three test cases:

- **Multimodal (Image + Text)**: Composed image retrieval with both image and text query
- **Text Only**: Text-based retrieval
- **Image Only**: Image-based retrieval

**Usage:**
```bash
python auto_mllm_single_device.py
```

### 2. `base_mllm_single_device.py`

This script shows how to use the FlagMLLMModel base class directly to load and use the multimodal model.

**Usage:**
```bash
python base_mllm_single_device.py
```

## Example Usage

### Using FlagAutoModel (Recommended)

```python
from FlagEmbedding import FlagAutoModel

# Load the multimodal model
model = FlagAutoModel.from_finetuned(
    'BAAI/BGE-VL-MLLM-S1',
    devices="cuda:0",
)

# Query with image and text
query_emb = model.encode(
    sentences="Make the background dark",
    images="./path/to/image.png",
    q_or_c="q",
    task_instruction="Retrieve the target image: "
)

# Candidate images
candidate_emb = model.encode(
    images=["./path/to/candi1.png", "./path/to/candi2.png"],
    q_or_c="c"
)

# Calculate scores
scores = query_emb @ candidate_emb.T
```

### Using FlagMLLMModel Directly

```python
from FlagEmbedding import FlagMLLMModel

model = FlagMLLMModel(
    model_name_or_path='BAAI/BGE-VL-MLLM-S1',
    normalize_embeddings=True,
    use_fp16=True,
    devices="cuda:0",
    trust_remote_code=True,
)

# Same encode API as above
```

## Parameters

### Key Parameters for Multimodal Encoding

- `sentences` (str or List[str], optional): Text inputs
- `images` (str or List[str], optional): Image file paths
- `q_or_c` (str): Query ("q" or "query") or Candidate ("c" or "candidate")
- `task_instruction` (str, optional): Task-specific instruction for queries
- `batch_size` (int): Batch size for processing
- `convert_to_numpy` (bool): Whether to convert output to numpy array

### Model Parameters

- `model_name_or_path` (str): Model identifier or path
- `normalize_embeddings` (bool): Whether to normalize embeddings (default: True)
- `use_fp16` (bool): Use half precision (default: True)
- `devices` (str): Device to use (e.g., "cuda:0", "cpu")
- `trust_remote_code` (bool): Trust remote code (required for custom models)

## Notes

1. The multimodal model supports three modes:
   - **Multimodal**: Both text and images
   - **Text-only**: Just text input
   - **Image-only**: Just image input

2. For query encoding (q_or_c="q"), you can provide a task instruction to improve retrieval performance.

3. For candidate encoding (q_or_c="c"), task instruction is typically not needed.

4. Make sure image paths are accessible and images are in supported formats (PNG, JPG, etc.).

## Troubleshooting

If you encounter issues:

1. **CUDA out of memory**: Try reducing batch_size or using CPU
2. **Image loading errors**: Check image paths and formats
3. **Model loading errors**: Ensure trust_remote_code=True for custom models

