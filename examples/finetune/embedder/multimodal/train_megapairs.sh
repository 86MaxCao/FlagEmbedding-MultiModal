#!/bin/bash

# MegaPairs Training Script for Multimodal Embedder
# Train BAAI/BGE-VL-MLLM-S1 on MegaPairs dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled  # Set to 'online' if you want to use wandb

# Model and data paths
model_name_or_path="BAAI/BGE-VL-MLLM-S1"
train_data="path/to/megapairs/train.jsonl"  # Update this path
output_dir="./checkpoints/bge-vl-mllm-megapairs"

# Training arguments
torchrun --nproc_per_node 4 \
    -m FlagEmbedding.finetune.embedder.multimodal \
    --model_name_or_path $model_name_or_path \
    --train_data $train_data \
    --cache_dir ./cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval "Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: " \
    --query_instruction_format "{}{}" \
    --knowledge_distillation False \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage1.json \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div

echo "Training completed! Model saved to $output_dir"

