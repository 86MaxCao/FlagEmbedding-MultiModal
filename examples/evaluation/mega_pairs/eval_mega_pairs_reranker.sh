#!/bin/bash

# MegaPairs Reranker Evaluation Script
# For evaluating jinaai/jina-reranker-m0 or other multimodal rerankers on MegaPairs dataset

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# Evaluation arguments
eval_args="\
    --eval_name mega_pairs \
    --dataset_name JUNJIE99/MegaPairs \
    --splits test \
    --corpus_embd_save_dir ./mega_pairs/corpus_embd \
    --output_dir ./mega_pairs/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 1 3 5 10 20 50 100 \
    --eval_output_method markdown \
    --eval_output_path ./mega_pairs/reranker_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
    --ignore_identical_ids False \
"

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
cmd="python -m FlagEmbedding.evaluation.mega_pairs \
    $eval_args \
    $model_args \
"

echo "Running MegaPairs Reranker Evaluation..."
echo $cmd
eval $cmd

