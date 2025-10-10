import os
import torch
import logging
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .arguments import MultimodalRerankerModelArguments

logger = logging.getLogger(__name__)


def get_model(model_args: MultimodalRerankerModelArguments):
    """Load multimodal reranker model with processor.

    Args:
        model_args: Model arguments.

    Returns:
        tuple: (model, processor)
    """
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            num_labels=1,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported."
        )

    # Load model
    if model_args.model_name_or_path:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError("model_name_or_path must be provided")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        max_pixels=602112,
        min_pixels=3136,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    # Store processor in model for access
    model._processor = processor

    # Apply LoRA if needed
    if model_args.use_lora:
        # For Qwen2VL-based models, target all linear layers in attention and MLP
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


def save_merged_model(model_args: MultimodalRerankerModelArguments, output_dir: str):
    """Save merged model (merge LoRA weights).

    Args:
        model_args: Model arguments.
        output_dir: Output directory.
    """
    logger.info(f"Saving merged model to {output_dir}/merged_model")
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, output_dir)
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(os.path.join(output_dir, 'merged_model'), exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, 'merged_model'))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir, 
        trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.save_pretrained(os.path.join(output_dir, 'merged_model'))
    
    logger.info("Merged model saved successfully")

