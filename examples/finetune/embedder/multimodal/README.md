# Multimodal Embedder Fine-tuning

This directory contains scripts for fine-tuning multimodal embedding models like BGE-VL-MLLM-S1.

## Prerequisites

1. Install required packages:
```bash
pip install transformers peft torch deepspeed
```

2. Prepare your training data in the format:
```json
{
  "query": "text query or null",
  "qry_image_path": "path/to/query/image.jpg or null",
  "pos_text": ["positive text passages"],
  "pos_image_path": ["paths/to/positive/images.jpg"],
  "neg_text": ["negative text passages"],
  "neg_image_path": ["paths/to/negative/images.jpg"],
  "pos_scores": [scores],  // Optional: for knowledge distillation
  "neg_scores": [scores]   // Optional: for knowledge distillation
}
```

## Training Scripts

### 1. `base.sh`
Basic training script without knowledge distillation.

**Usage:**
```bash
bash base.sh
```

**Key parameters:**
- `--model_name_or_path`: Base model (default: BAAI/BGE-VL-MLLM-S1)
- `--train_data`: Path to training data (supports HF dataset names)
- `--use_lora`: Use LoRA for efficient training
- `--trust_remote_code`: Required for custom multimodal models

### 2. `base_same_dataset.sh`
Training with knowledge distillation and same-dataset batching.

**Usage:**
```bash
bash base_same_dataset.sh
```

**Additional features:**
- Knowledge distillation from teacher scores
- Same dataset within batch for better contrast

## Configuration

### Model Arguments
```bash
--model_name_or_path BAAI/BGE-VL-MLLM-S1  # Base model
--use_lora True                            # Use LoRA
--lora_rank 32                            # LoRA rank
--lora_alpha 64                           # LoRA alpha
--target_modules q_proj k_proj v_proj ... # LoRA target modules
--save_merged_lora_model True             # Save merged model
--trust_remote_code True                  # Required for multimodal
```

### Data Arguments
```bash
--train_data <path>              # Training data path
--train_group_size 8            # Number of passages per query
--query_max_len 512             # Max query length
--passage_max_len 512           # Max passage length
--knowledge_distillation False  # Use KD or not
--same_dataset_within_batch True # Sample from same dataset
```

### Training Arguments
```bash
--output_dir <path>              # Output directory
--learning_rate 1e-4            # Learning rate
--num_train_epochs 3            # Number of epochs
--per_device_train_batch_size 4 # Batch size per device
--gradient_checkpointing        # Save memory
--negatives_cross_device        # Cross-device negatives
--temperature 0.02              # Temperature for contrastive loss
--normalize_embeddings True     # L2 normalization
```

## Multi-GPU Training

The scripts use `torchrun` for distributed training:

```bash
# Set number of GPUs
num_gpus=4

# The script will automatically use torchrun
torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.multimodal.base \
    ...
```

## DeepSpeed

The scripts use DeepSpeed stage 1 for memory efficiency. Make sure you have `../../ds_stage1.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 1
  },
  "fp16": {
    "enabled": "auto"
  }
}
```

## Data Format Notes

### Multimodal Data
The training data should contain both text and image fields:
- **Query**: Can be text-only, image-only, or both
- **Positive**: Can be text-only, image-only, or both
- **Negative**: Can be text-only, image-only, or both

### Supported Formats
1. Pure text retrieval
2. Pure image retrieval
3. Text-to-image retrieval
4. Image-to-text retrieval
5. Multimodal-to-multimodal retrieval

## Output

After training, you'll find:
- **Checkpoints**: `output_dir/checkpoint-*`
- **Merged model** (if enabled): `output_dir/merged_model/`
- **Training logs**: `output_dir/trainer_state.json`

## Example

```bash
# 1. Update training data path in the script
vim base.sh
# Change: train_data="your/data/path"

# 2. Run training
bash base.sh

# 3. Monitor training
tensorboard --logdir output_multimodal_bge_vl_mllm_s1
```

## Troubleshooting

1. **OOM Error**: Reduce `per_device_train_batch_size` or use gradient accumulation
2. **Import Error**: Make sure `trust_remote_code=True` is set
3. **Data Error**: Ensure your data follows the correct format with image paths
4. **Processor Error**: The model must support `set_processor()` method

## Notes

- The collator uses the model's `data_process()` method to handle multimodal inputs
- Image paths in the data should be accessible from the training environment
- For large datasets, consider using `max_example_num_per_dataset` to limit samples

