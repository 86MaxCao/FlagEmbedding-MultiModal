#!/bin/bash

# MegaPairs Training Script for Multimodal Reranker
# Train jinaai/jina-reranker-m0 on MegaPairs dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled  # Set to 'online' if you want to use wandb

# Model and data paths
model_name_or_path="jinaai/jina-reranker-m0"
train_data="path/to/megapairs/train.jsonl"  # Update this path
output_dir="./checkpoints/jina-reranker-m0-megapairs"

# Training arguments
torchrun --nproc_per_node 4 \
    -m FlagEmbedding.finetune.reranker.multimodal \
    --model_name_or_path $model_name_or_path \
    --train_data $train_data \
    --cache_dir ./cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --max_len 2048 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 5e-6 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage1.json \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --use_train_config

echo "Training completed! Model saved to $output_dir"

