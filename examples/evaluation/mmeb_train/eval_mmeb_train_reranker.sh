#!/bin/bash

# MMEB-train Reranker Evaluation Script
# For evaluating jinaai/jina-reranker-m0 or other multimodal rerankers on MMEB-train dataset

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# Quick test mode (uncomment to test with first 100 samples)
# QUICK_TEST=true

if [ "$QUICK_TEST" = "true" ]; then
    echo "Running in QUICK TEST mode (first 100 samples, no cache)"
    eval_args="\
        --eval_name mmeb_train \
        --dataset_dir ~/.cache/huggingface/datasets/MMEB-train-flat \
        --use_local_data True \
        --max_samples 100 \
        --skip_corpus_cache True \
        --sub_datasets DocVQA CIRR MSCOCO \
        --task_types text image mixed \
        --splits test \
        --corpus_embd_save_dir None \
        --output_dir ./mmeb_train/search_results \
        --search_top_k 1000 \
        --rerank_top_k 100 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 3 5 10 20 50 100 \
        --eval_output_method markdown \
        --eval_output_path ./mmeb_train/reranker_eval_results.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
        --ignore_identical_ids False \
    "
else
    echo "Running in FULL EVALUATION mode"
    eval_args="\
        --eval_name mmeb_train \
        --dataset_dir ~/.cache/huggingface/datasets/MMEB-train-flat \
        --use_local_data True \
        --sub_datasets DocVQA CIRR MSCOCO MSCOCO_i2t MSCOCO_t2i VisualNews_i2t VisualNews_t2i Visual7W WebQA \
        --task_types text image mixed \
        --splits test \
        --corpus_embd_save_dir ./mmeb_train/corpus_embd \
        --output_dir ./mmeb_train/search_results \
        --search_top_k 1000 \
        --rerank_top_k 100 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 3 5 10 20 50 100 \
        --eval_output_method markdown \
        --eval_output_path ./mmeb_train/reranker_eval_results.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
        --ignore_identical_ids False \
    "
fi

# Model arguments
model_args="\
    --embedder_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --embedder_model_class multimodal-mllm \
    --reranker_name_or_path jinaai/jina-reranker-m0 \
    --devices cuda:0 \
    --cache_dir $HF_HUB_CACHE \
    --use_fp16 \
    --trust_remote_code \
    --normalize_embeddings \
    --embedder_batch_size 32 \
    --reranker_batch_size 8 \
    --reranker_max_length 2048 \
    --normalize \
"

# Run evaluation
cmd="python -m FlagEmbedding.evaluation.mmeb_train \
    $eval_args \
    $model_args \
"

echo "Running MMEB-train Reranker Evaluation..."
echo $cmd
eval $cmd
