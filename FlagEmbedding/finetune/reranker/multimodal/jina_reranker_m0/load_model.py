import os
import logging
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from FlagEmbedding.compat import apply_jina_reranker_patches
from .arguments import JinaRerankerM0ModelArguments

logger = logging.getLogger(__name__)

# Apply all compatibility patches at module import time
apply_jina_reranker_patches()


def get_model(model_args: JinaRerankerM0ModelArguments):
    """Load jina-reranker-m0 model with processor.

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
    # JinaVLForRanking replaces lm_head with nn.Identity, so disable weight tying
    config.tie_word_embeddings = False

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    # After from_pretrained, lm_head may carry a stale meta-device weight
    # from the Identity patch + ignore_mismatched_sizes; replace with clean Identity
    if isinstance(model.lm_head, nn.Identity):
        model.lm_head = nn.Identity()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        max_pixels=602112,
        min_pixels=3136,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    model._processor = processor

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
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor


def save_merged_model(model_args: JinaRerankerM0ModelArguments, output_dir: str):
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
    config.tie_word_embeddings = False

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    # Clean up lm_head after load
    if isinstance(model.lm_head, nn.Identity):
        model.lm_head = nn.Identity()

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
