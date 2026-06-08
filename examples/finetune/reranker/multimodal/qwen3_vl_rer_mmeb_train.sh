export WANDB_MODE=disabled

# MMEB-train data directories (each contains train/diverse_instruction/original parquet files)
MMEB_ROOT="/mnt/nas-tbt/tbt/data/hf_cache/MMEB-train"
IMAGE_ROOT="${MMEB_ROOT}/images"

train_data="\
    ${MMEB_ROOT}/A-OKVQA \
    ${MMEB_ROOT}/CIRR \
    ${MMEB_ROOT}/ChartQA \
    ${MMEB_ROOT}/DocVQA \
    ${MMEB_ROOT}/HatefulMemes \
    ${MMEB_ROOT}/ImageNet_1K \
    ${MMEB_ROOT}/InfographicsVQA \
    ${MMEB_ROOT}/MSCOCO \
    ${MMEB_ROOT}/MSCOCO_i2t \
    ${MMEB_ROOT}/MSCOCO_t2i \
    ${MMEB_ROOT}/N24News \
    ${MMEB_ROOT}/NIGHTS \
    ${MMEB_ROOT}/OK-VQA \
    ${MMEB_ROOT}/SUN397 \
    ${MMEB_ROOT}/VOC2007 \
    ${MMEB_ROOT}/VisDial \
    ${MMEB_ROOT}/Visual7W \
    ${MMEB_ROOT}/VisualNews_i2t \
    ${MMEB_ROOT}/VisualNews_t2i \
    ${MMEB_ROOT}/WebQA \
"

# Training parameters
num_train_epochs=3
per_device_train_batch_size=2
gradient_accumulation_steps=1
train_group_size=3
num_gpus=4

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path Qwen/Qwen3-VL-Reranker-2B \
    --cache_dir $HF_HUB_CACHE \
    --trust_remote_code True \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --loss_type pairwise \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --image_root_dir $IMAGE_ROOT \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 512 \
    --passage_max_len 1536 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir ./output_qwen3_vl_rer_mmeb_train \
    --learning_rate 5e-5 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 500 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.multimodal.qwen3_vl \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
