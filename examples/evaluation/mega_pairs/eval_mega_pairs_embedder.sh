#!/bin/bash

# MegaPairs Embedder Evaluation Script
# For evaluating BAAI/BGE-VL-MLLM-S1 or other multimodal embedders on MegaPairs dataset

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# Quick test mode (uncomment to test with first 100 samples)
# QUICK_TEST=true

if [ "$QUICK_TEST" = "true" ]; then
    echo "Running in QUICK TEST mode (first 100 samples, no cache)"
    eval_args="\
        --eval_name mega_pairs \
        --dataset_dir ~/.cache/huggingface/datasets/MegaPairs-flat-final/data \
        --use_local_data True \
        --max_samples 100 \
        --skip_corpus_cache True \
        --splits test \
        --corpus_embd_save_dir None \
        --output_dir ./mega_pairs/search_results \
        --search_top_k 1000 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 3 5 10 20 50 100 \
        --eval_output_method markdown \
        --eval_output_path ./mega_pairs/embedder_eval_results.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
        --ignore_identical_ids False \
    "
else
    echo "Running in FULL EVALUATION mode"
    eval_args="\
        --eval_name mega_pairs \
        --dataset_dir ~/.cache/huggingface/datasets/MegaPairs-flat-final/data \
        --use_local_data True \
        --splits test \
        --corpus_embd_save_dir ./mega_pairs/corpus_embd \
        --output_dir ./mega_pairs/search_results \
        --search_top_k 1000 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 3 5 10 20 50 100 \
        --eval_output_method markdown \
        --eval_output_path ./mega_pairs/embedder_eval_results.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
        --ignore_identical_ids False \
    "
fi

# Model arguments
model_args="\
    --embedder_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --embedder_model_class multimodal-mllm \
    --devices cuda:0 \
    --cache_dir $HF_HUB_CACHE \
    --use_fp16 \
    --trust_remote_code \
    --normalize_embeddings \
    --embedder_batch_size 32 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
"

# Run evaluation
cmd="python -m FlagEmbedding.evaluation.mega_pairs \
    $eval_args \
    $model_args \
"

echo "Running MegaPairs Embedder Evaluation..."
echo $cmd
eval $cmd

