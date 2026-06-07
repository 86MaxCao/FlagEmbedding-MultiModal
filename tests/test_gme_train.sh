#!/bin/bash
# Quick training smoke test for gme-Qwen2-VL-2B-Instruct
# Runs a few steps on example data to verify the training pipeline works end-to-end.

set -e

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/.."

TRAIN_DATA="examples/finetune/example_data/multimodal/examples.jsonl"
MODEL_PATH="/mnt/nas-tbt/tbt/checkpoint/hf_cache/gme-Qwen2-VL-2B-Instruct"
OUTPUT_DIR="/tmp/test_gme_train_output_$$"

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

echo "=== GME-Qwen2-VL-2B-Instruct Training Smoke Test ==="
echo "Model: $MODEL_PATH"
echo "Data:  $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.multimodal.base \
    --model_name_or_path "$MODEL_PATH" \
    --cache_dir "$HF_HUB_CACHE" \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model False \
    --trust_remote_code True \
    --train_data "$TRAIN_DATA" \
    --cache_path /tmp/.cache_gme_test \
    --train_group_size 2 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 1 \
    --max_steps 3 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --save_steps 999 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss

echo ""
echo "=== Training completed successfully ==="

# Cleanup
rm -rf "$OUTPUT_DIR" /tmp/.cache_gme_test
echo "Cleaned up output directory."
