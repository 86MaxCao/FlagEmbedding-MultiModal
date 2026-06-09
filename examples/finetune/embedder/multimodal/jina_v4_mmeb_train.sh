export WANDB_MODE=disabled

# MMEB-train data directories (each contains train/diverse_instruction/original parquet files)
MMEB_ROOT="/mnt/nas-tbt/tbt/data/hf_cache/MMEB-train"
IMAGE_ROOT="${MMEB_ROOT}"

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
per_device_train_batch_size=4
num_gpus=4

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path jinaai/jina-embeddings-v4 \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
    --trust_remote_code True \
"

data_args="\
    --train_data $train_data \
    --image_root_dir $IMAGE_ROOT \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir ./output_jina_v4_mmeb_train \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../../ds_stage1.json \
    --logging_steps 10 \
    --save_steps 500 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.multimodal.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
