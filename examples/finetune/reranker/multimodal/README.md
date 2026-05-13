# Multimodal Reranker Fine-tuning

Fine-tuning scripts for multimodal reranker models like `jinaai/jina-reranker-m0`.

## Features

- ✅ Support for **Pairwise Loss** (default)
- ✅ Support for **Listwise Loss**
- ✅ Knowledge distillation
- ✅ LoRA efficient training
- ✅ Multi-GPU training with DeepSpeed

## Loss Types

### 1. Pairwise Loss (Default)
Compares positive passages against each negative passage individually using margin ranking loss.

**Formula**: `max(0, margin - (score_pos - score_neg))`

**Advantages**:
- More stable training
- Better for small datasets
- Focuses on relative ordering

**Use when**: You want to ensure positive is ranked higher than each negative.

### 2. Listwise Loss
Uses cross-entropy over all passages simultaneously.

**Formula**: `CrossEntropy(logits, target=0)` where first position is positive.

**Advantages**:
- Considers all candidates together
- Better calibrated scores
- Works well with knowledge distillation

**Use when**: You have reliable teacher scores or want globally calibrated scores.

## Data Format

Your training data should follow this format:

```json
{
  "query": "text query",
  "qry_image_path": "path/to/query/image.jpg",
  "pos": ["positive text"],
  "pos_image_path": ["path/to/positive/image.jpg"],
  "neg": ["negative text 1", "negative text 2"],
  "neg_image_path": ["path/to/neg1.jpg", "path/to/neg2.jpg"],
  "pos_scores": [1.0],
  "neg_scores": [0.5, 0.3]
}
```

**Supported fields**:
- `query` / `qry` - Query text (optional)
- `query_image` / `qry_image_path` - Query image path (optional)
- `pos` / `pos_text` - Positive passages (text)
- `pos_images` / `pos_image_path` - Positive passages (images)
- `neg` / `neg_text` - Negative passages (text)
- `neg_images` / `neg_image_path` - Negative passages (images)
- `pos_scores` / `neg_scores` - Teacher scores for KD (optional)

## Training Scripts

### 1. `base_pairwise.sh` - Pairwise Loss Training

```bash
bash base_pairwise.sh
```

**Configuration**:
- Loss type: `pairwise`
- Default for most use cases
- More stable training

### 2. `base_listwise.sh` - Listwise Loss Training

```bash
bash base_listwise.sh
```

**Configuration**:
- Loss type: `listwise`
- Includes knowledge distillation
- Better for calibrated scores

## Configuration

### Model Arguments

```bash
--model_name_or_path jinaai/jina-reranker-m0  # Base model
--trust_remote_code True                       # Required
--use_lora True                                # Use LoRA
--lora_rank 32                                 # LoRA rank
--lora_alpha 64                                # LoRA alpha
--loss_type pairwise                           # 'pairwise' or 'listwise'
--save_merged_lora_model True                  # Save merged model
```

### Data Arguments

```bash
--train_data <path>              # Training data path
--train_group_size 8            # 1 positive + N-1 negatives
--query_max_len 512             # Max query tokens
--passage_max_len 1536          # Max passage tokens
--knowledge_distillation True   # Use KD (for listwise)
```

### Training Arguments

```bash
--output_dir <path>                  # Output directory
--learning_rate 5e-5                # Learning rate
--num_train_epochs 3                # Number of epochs
--per_device_train_batch_size 4     # Batch size per GPU
--gradient_checkpointing            # Save memory
--weight_decay 0.01                 # Weight decay
--warmup_ratio 0.1                  # Warmup ratio
```

## Usage Example

### Step 1: Prepare Data

Create training data in JSONL format:

```jsonl
{"query": "machine learning", "qry_image_path": "query1.png", "pos": ["ML is..."], "pos_image_path": ["pos1.png"], "neg": ["Cooking..."], "neg_image_path": ["neg1.png"]}
{"query": "deep learning", "pos_text": ["DL is..."], "neg_text": ["Sports..."]}
```

### Step 2: Configure Training Script

Edit `base_pairwise.sh`:

```bash
# Update data path
train_data="path/to/your/data.jsonl"

# Adjust GPUs
num_gpus=4

# Adjust batch size if needed
per_device_train_batch_size=2
```

### Step 3: Start Training

```bash
cd examples/finetune/reranker/multimodal
bash base_pairwise.sh
```

## Loss Comparison

| Aspect | Pairwise Loss | Listwise Loss |
|--------|--------------|---------------|
| **Training** | Pair-by-pair comparison | All candidates together |
| **Stability** | More stable | May be less stable |
| **Use case** | General reranking | With teacher scores |
| **Formula** | Margin ranking | Cross-entropy |
| **KD Support** | ✅ Yes | ✅ Yes |

## Modality Support

The training framework supports:

1. **Text query → Text documents**
2. **Text query → Image documents**
3. **Image query → Text documents**
4. **Image query → Image documents**
5. **Mixed**: Any combination of the above

## Multi-GPU Training

```bash
# Set number of GPUs
num_gpus=4

# The script uses torchrun
torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.multimodal.base \
    ...
```

## DeepSpeed

Uses DeepSpeed Stage 0 (disabled). Make sure you have `../../ds_stage0.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 0
  },
  "fp16": {
    "enabled": "auto"
  }
}
```

## Output Structure

After training:

```
output_multimodal_reranker_jina_m0_pairwise/
├── checkpoint-500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── merged_model/              # If save_merged_lora_model=True
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── trainer_state.json
└── training_args.bin
```

## Using Trained Model

```python
from FlagEmbedding import FlagAutoReranker

# Load trained model
reranker = FlagAutoReranker.from_finetuned(
    './output_multimodal_reranker_jina_m0_pairwise/merged_model',
    devices="cuda:0"
)

# Rerank
query = "text query"
docs = ["doc1.png", "doc2.png"]
pairs = [[query, doc] for doc in docs]
scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)
```

## Troubleshooting

### OOM Error
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `passage_max_len`

### Import Error
- Ensure `trust_remote_code=True`
- Update transformers: `pip install -U transformers`

### Loss NaN
- Try pairwise loss instead of listwise
- Reduce learning rate
- Check data quality

## Advanced: Choosing Loss Type

**Use Pairwise Loss when**:
- Small training data
- No teacher scores
- Want stable training
- Focus on relative ordering

**Use Listwise Loss when**:
- Have teacher scores (KD)
- Large training data
- Want calibrated scores
- Need global ranking

## Notes

1. The processor handles multimodal inputs automatically
2. Images are loaded on-the-fly during training
3. Score token (ID: 100) is appended to each input
4. Scores are normalized with sigmoid during inference

