import os
import logging

from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from FlagEmbedding.compat import apply_qwen3_vl_reranker_patches
from .arguments import Qwen3VLRerankerModelArguments

logger = logging.getLogger(__name__)

# Apply compatibility patches at module import time
apply_qwen3_vl_reranker_patches()


def get_model(model_args: Qwen3VLRerankerModelArguments):
    """Load Qwen3-VL-Reranker model (full model with lm_head) with processor.

    We load Qwen3VLForConditionalGeneration so that:
    1. The lm_head weights are available to initialize the score_linear
       (yes_weight - no_weight from vocabulary projection).
    2. The full model can be used for forward passes during training.

    Args:
        model_args: Model arguments.

    Returns:
        tuple: (model, processor)
    """
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load the full model first to extract lm_head weights, then use base model
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    if model_args.use_lora:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )
        full_model = get_peft_model(full_model, peft_config)
        full_model.print_trainable_parameters()

    # Store processor in model for compatibility
    full_model._processor = processor

    return full_model, processor


def save_merged_model(model_args: Qwen3VLRerankerModelArguments, output_dir: str):
    """Save merged model (merge LoRA weights).

    Args:
        model_args: Model arguments.
        output_dir: Output directory.
    """
    logger.info(f"Saving merged model to {output_dir}/merged_model")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = PeftModel.from_pretrained(model, output_dir)
    model = model.merge_and_unload()

    os.makedirs(os.path.join(output_dir, 'merged_model'), exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, 'merged_model'))

    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.save_pretrained(os.path.join(output_dir, 'merged_model'))

    logger.info("Merged model saved successfully")
