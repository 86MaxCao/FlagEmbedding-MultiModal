export WANDB_MODE=disabled

# Training data - update with your data path
train_data="../example_data/multimodal/examples.jsonl"

# Training parameters
num_train_epochs=3
per_device_train_batch_size=4
gradient_accumulation_steps=1
train_group_size=8

# Set num_gpus
num_gpus=2

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path jinaai/jina-reranker-m0 \
    --cache_dir $HF_HUB_CACHE \
    --trust_remote_code True \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --loss_type listwise \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 512 \
    --passage_max_len 1536 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
"

training_args="\
    --output_dir ./output_multimodal_reranker_jina_m0_listwise \
    --overwrite_output_dir \
    --learning_rate 5e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 10 \
    --save_steps 500 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.multimodal.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd

