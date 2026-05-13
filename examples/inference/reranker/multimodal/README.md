# Multimodal Reranker Examples

This directory contains examples for using multimodal reranker models like `jinaai/jina-reranker-m0`.

## Features

The multimodal reranker supports various query-document modality combinations:

- **Text → Image**: Text query with image documents
- **Image → Text**: Image query with text documents  
- **Text → Text**: Text query with text documents
- **Image → Image**: Image query with image documents

## Prerequisites

```bash
pip install transformers torch pillow
```

## Examples

### 1. Using FlagAutoReranker (Recommended)

```python
from FlagEmbedding import FlagAutoReranker

reranker = FlagAutoReranker.from_finetuned(
    'jinaai/jina-reranker-m0',
    devices="cuda:0"
)

# Text query -> Image documents
query = "machine learning paper"
image_docs = ["path/to/image1.png", "path/to/image2.png"]

pairs = [[query, doc] for doc in image_docs]
scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)
```

**Run the example:**
```bash
python auto_reranker_single_device.py
```

### 2. Using MultimodalReranker Directly

```python
from FlagEmbedding import MultimodalReranker

reranker = MultimodalReranker(
    model_name_or_path='jinaai/jina-reranker-m0',
    use_fp16=True,
    devices="cuda:0",
    batch_size=4,
    max_length=2048,
)

scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)
```

**Run the example:**
```bash
python base_reranker_single_device.py
```

## Parameters

### Model Initialization

```python
MultimodalReranker(
    model_name_or_path='jinaai/jina-reranker-m0',
    use_fp16=True,              # Use half precision
    trust_remote_code=True,     # Required for custom models
    devices="cuda:0",            # Device to use
    batch_size=8,               # Batch size
    query_max_length=512,       # Max query length
    max_length=10240,           # Max total length
    normalize=True,             # Normalize scores to [0,1]
    query_type='text',          # Default query type
    doc_type='image',           # Default document type
)
```

### Compute Score

```python
scores = reranker.compute_score(
    sentence_pairs,              # List of [query, doc] pairs
    batch_size=8,               # Optional: override batch size
    query_max_length=512,       # Optional: override query max length
    max_length=10240,           # Optional: override max length
    normalize=True,             # Optional: normalize scores
    query_type='text',          # 'text' or 'image'
    doc_type='image',           # 'text' or 'image'
)
```

## Modality Combinations

### Text Query → Image Documents
```python
query = "scientific paper"
docs = ["image1.png", "image2.jpg"]
scores = reranker.compute_score(
    [[query, doc] for doc in docs],
    query_type='text',
    doc_type='image'
)
```

### Image Query → Text Documents
```python
query = "query_image.png"
docs = ["Document about AI", "Document about cooking"]
scores = reranker.compute_score(
    [[query, doc] for doc in docs],
    query_type='image',
    doc_type='text'
)
```

### Text Query → Text Documents
```python
query = "What is AI?"
docs = ["AI explanation", "Cooking recipe"]
scores = reranker.compute_score(
    [[query, doc] for doc in docs],
    query_type='text',
    doc_type='text'
)
```

### Image Query → Image Documents
```python
query = "query.png"
docs = ["doc1.png", "doc2.png"]
scores = reranker.compute_score(
    [[query, doc] for doc in docs],
    query_type='image',
    doc_type='image'
)
```

## Image Sources

Images can be provided as:
- **Local paths**: `"path/to/image.png"`
- **URLs**: `"https://example.com/image.jpg"`
- **PIL Image objects**: `Image.open("image.png")`

## Score Normalization

By default, scores are normalized to [0, 1] using sigmoid:
```python
score = 1.0 / (1.0 + exp(-(raw_score - 2.65)))
```

To get raw scores, set `normalize=False`.

## Performance Tips

1. **Batch Processing**: Use larger `batch_size` for better throughput
2. **Half Precision**: Enable `use_fp16=True` for faster inference
3. **Image URLs**: Use `lazy_load=True` to avoid loading all images at once
4. **Max Length**: Adjust `max_length` based on your use case

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size`
- Enable `use_fp16=True`
- Reduce `max_length`

### Image Loading Error
- Check image paths/URLs are accessible
- Ensure images are in supported formats (PNG, JPG, WEBP)

### Import Error
- Make sure `trust_remote_code=True` is set
- Update transformers: `pip install -U transformers`

## Model Information

**jina-reranker-m0**:
- Based on Qwen2-VL architecture
- Supports text and image inputs
- Output: Relevance scores in [0, 1] (when normalized)
- Max sequence length: 10240 tokens

## Notes

- The reranker uses a special score token (ID: 100) for computing relevance
- Prompts are formatted as: `**Document**: ...\n**Query**: ...`
- Vision tokens are represented as `<|vision_start|><|image_pad|><|vision_end|>`

