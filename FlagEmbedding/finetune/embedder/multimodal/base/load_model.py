import os
import re
import torch
import logging
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .arguments import MultimodalEmbedderModelArguments
from FlagEmbedding.inference.embedder.multimodal.gme_qwen2vl import (
    _bypass_gme_version_check,
    _restore_require_version,
    _patch_composite_config,
    _patch_visual_output,
    _tie_gme_visual_encoder,
    _patch_rope_init_functions,
    _patch_tied_weights_keys,
    _patch_processor_auto_loading,
)

logger = logging.getLogger(__name__)


def _is_gme_model(model_name_or_path: str) -> bool:
    """Check whether the model path points to a GME-Qwen2-VL model."""
    if model_name_or_path is None:
        return False
    basename = os.path.basename(model_name_or_path.rstrip("/"))
    return basename.lower().startswith("gme-qwen2-vl") or basename.lower().startswith("gme_qwen2_vl")


def _is_jina_model(model_name_or_path: str) -> bool:
    """Check whether the model path points to a jina-embeddings model."""
    if model_name_or_path is None:
        return False
    basename = os.path.basename(model_name_or_path.rstrip("/"))
    return "jina-embeddings" in basename.lower() or "jina_embeddings" in basename.lower()


def _add_jina_data_process(model):
    """Monkey-patch ``data_process`` onto a jina-embeddings model for training collator."""
    import types

    processor = model.processor

    def data_process(self, text=None, images=None, q_or_c="q", task_instruction=None, **kwargs):
        if images is not None and any(img is not None for img in (images if isinstance(images, list) else [images])):
            return self.processor.process_images(
                [img for img in (images if isinstance(images, list) else [images]) if img is not None]
            )

        if text is not None:
            texts = text if isinstance(text, list) else [text]
            prefix = "Query: " if q_or_c == "q" else "Passage: "
            return self.processor.process_texts(texts, prefix=prefix)

        return {}

    model.data_process = types.MethodType(data_process, model)


def _add_gme_data_process(model):
    """Monkey-patch ``data_process`` onto a GME model for training collator compatibility."""
    import types
    from PIL import Image

    processor = model.processor

    def data_process(self, text=None, images=None, q_or_c="q", task_instruction=None, **kwargs):
        instruction = task_instruction or self.default_instruction
        input_texts, input_images = [], []
        batch_size = max(len(text) if text else 0, len(images) if images else 0)
        for i in range(batch_size):
            t = text[i] if text and i < len(text) else None
            img = images[i] if images and i < len(images) else None
            input_str = ""
            if img is not None:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                input_images.append(img)
            if t is not None:
                input_str += t
            if q_or_c == "q":
                msg = (
                    f"<|im_start|>system\n{instruction}<|im_end|>\n"
                    f"<|im_start|>user\n{input_str}<|im_end|>\n"
                    f"<|im_start|>assistant\n<|endoftext|>"
                )
            else:
                default = self.default_instruction
                msg = (
                    f"<|im_start|>system\n{default}<|im_end|>\n"
                    f"<|im_start|>user\n{input_str}<|im_end|>\n"
                    f"<|im_start|>assistant\n<|endoftext|>"
                )
            input_texts.append(msg)
        imgs_arg = input_images if input_images else None
        inputs = processor(
            text=input_texts,
            images=imgs_arg,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v for k, v in inputs.items()}

    model.data_process = types.MethodType(data_process, model)


def _patch_jina_forward_for_training(model):
    """Replace JinaEmbeddingsV4Model.forward() for training compatibility.

    The bundled qwen2_5_vl.py passes ``task_label`` as an extra positional
    arg to every linear projection (e.g. ``self.q_proj(x, task_label)``).
    After unloading the inference LoRA adapters, training wraps nn.Linear
    with peft LoRA Linear whose forward does ``self.base_layer(x, *args)``,
    forwarding the extra arg to nn.Linear which only accepts ``(input,)``.

    Two patches are applied:
    1. All nn.Linear instances inside the language model get a patched
       forward that accepts and silently drops extra positional args.
       This must happen BEFORE ``get_peft_model()`` so the patched
       nn.Linear becomes the ``base_layer`` inside each LoRA wrapper.
    2. The top-level model forward is replaced to bypass task_label
       routing and return the last hidden state directly.
    """
    import types
    import torch.nn as nn
    import torch.nn.functional as F

    lang_model = getattr(getattr(model, 'model', None), 'language_model', None)
    if lang_model is not None:
        def _linear_forward_ignore_extra(self, input, *args, **kwargs):
            return F.linear(input, self.weight, self.bias)

        patched = 0
        for _, module in lang_model.named_modules():
            if type(module) is nn.Linear:
                module.forward = types.MethodType(_linear_forward_ignore_extra, module)
                patched += 1
        logger.info(f"Patched {patched} nn.Linear forwards in language model to ignore task_label.")

    model_cls = type(model)
    _parent_forward = None
    for cls in model_cls.__mro__[1:]:
        if 'forward' in cls.__dict__ and cls.__name__ != model_cls.__name__:
            _parent_forward = cls.forward
            break

    if _parent_forward is None:
        logger.warning("Could not find parent forward for jina model; skipping patch.")
        return

    def _training_forward(self, input_ids=None, attention_mask=None, **kwargs):
        pv = kwargs.get("pixel_values")
        igt = kwargs.get("image_grid_thw")
        if pv is not None and igt is not None:
            try:
                offsets = igt[:, 1] * igt[:, 2]
                kwargs["pixel_values"] = torch.cat(
                    [p[:o] for p, o in zip(pv, offsets)], dim=0
                )
            except (TypeError, IndexError):
                pass

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=igt,
            attention_mask=attention_mask,
        )

        kwargs.pop("task_label", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("rope_deltas", None)
        kwargs["output_hidden_states"] = True

        outputs = _parent_forward(
            self,
            task_label=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
            **kwargs,
        )
        return outputs.hidden_states[-1]

    model.forward = types.MethodType(_training_forward, model)
    logger.info("Patched jina model forward for training (bypasses task_label routing).")


def find_largest_checkpoint(checkpoint_dir):
    """Find the largest checkpoint from directory.

    Args:
        checkpoint_dir (str): Directory to the checkpoint.

    Returns:
        str: Directory to the checkpoint, None no matching found.
    """
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    max_number = -1
    max_checkpoint_file = None
    for file in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.search(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_checkpoint_file = file
    if max_checkpoint_file:
        return os.path.join(checkpoint_dir, max_checkpoint_file)
    else:
        return None


def get_model(model_args: MultimodalEmbedderModelArguments, output_dir: str, resize: bool, resize_tokens: int):
    """Get the multimodal model with processor initialization.

    Args:
        model_args (MultimodalEmbedderModelArguments): Model arguments instance.
        output_dir (str): Directory to save the model.
        resize (bool): Whether to resize the number of tokens.
        resize_tokens (int): The new token size.

    Returns:
        transformers.PreTrainedModel or PeftModel: The loaded model.
    """
    _patch_rope_init_functions()
    _patch_tied_weights_keys()
    _patch_processor_auto_loading()

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    is_gme = _is_gme_model(model_args.model_name_or_path)
    is_jina = _is_jina_model(model_args.model_name_or_path)
    orig_rv = _bypass_gme_version_check() if is_gme else None

    try:
        if is_gme:
            _patch_composite_config(config)

        if model_args.model_name_or_path:
            model_kw = {
                "token": model_args.token,
                "cache_dir": model_args.cache_dir,
                "from_tf": bool(".ckpt" in model_args.model_name_or_path),
                "config": config,
                "trust_remote_code": model_args.trust_remote_code,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            if model_args.use_flash_attn:
                model_kw["attn_implementation"] = "flash_attention_2"
            else:
                model_kw["attn_implementation"] = "sdpa"
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                **model_kw
            )
        else:
            logger.info("Training new model from scratch")
            model = model_args.from_config(config)
    finally:
        if orig_rv is not None:
            _restore_require_version(orig_rv)

    if isinstance(model, PeftModel):
        import gc
        try:
            logger.info("Model loaded with inference LoRA adapters, merging into base weights for training.")
            model = model.merge_and_unload()
        except NotImplementedError:
            logger.info("merge_and_unload() not supported, using unload() to discard inference adapters.")
            model = model.unload()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    if is_gme:
        _patch_visual_output(model)
        _tie_gme_visual_encoder(model)
        _add_gme_data_process(model)
        logger.info("Applied GME compatibility patches and data_process method.")
    elif is_jina:
        if hasattr(model, 'set_processor'):
            model.set_processor(model_args.model_name_or_path)
        _add_jina_data_process(model)
        _patch_jina_forward_for_training(model)
        logger.info("Applied jina-embeddings data_process method and forward patch.")
    elif hasattr(model, 'set_processor'):
        logger.info(f"Initializing processor for multimodal model: {model_args.model_name_or_path}")
        model.set_processor(model_args.model_name_or_path)
    else:
        logger.debug("Model does not have set_processor method.")

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    if resize:
        model.resize_token_embeddings(resize_tokens)
        os.makedirs(os.path.join(output_dir, 'embedding'), exist_ok=True)
        torch.save(model.embed_tokens, os.path.join(output_dir, 'embedding', 'emb.pth'))
        target_modules = model_args.target_modules
    else:
        target_modules = model_args.target_modules
        if 'embed_tokens' in target_modules:
            target_modules.remove('embed_tokens')

    if model_args.from_peft is not None:
        if os.path.exists(os.path.join(model_args.from_peft, 'embedding')):
            model.set_input_embeddings(torch.load(os.path.join(model_args.from_peft, 'embedding', 'emb.pth')))
            torch.save(model.embed_tokens, os.path.join(output_dir, 'embedding', 'emb.pth'))
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=target_modules,
                modules_to_save=model_args.modules_to_save,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model


def save_merged_model(model_args: MultimodalEmbedderModelArguments, output_dir: str):
    """
    Loads a multimodal model with specified configurations, merges it with PEFT layers if available.

    Args:
        model_args (MultimodalEmbedderModelArguments): Model arguments instance.
        output_dir (str): Directory to save the model.
    """
    _patch_rope_init_functions()
    _patch_tied_weights_keys()

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    is_gme = _is_gme_model(model_args.model_name_or_path)
    orig_rv = _bypass_gme_version_check() if is_gme else None

    try:
        if is_gme:
            _patch_composite_config(config)

        if model_args.model_name_or_path:
            model_kw = {
                "token": model_args.token,
                "cache_dir": model_args.cache_dir,
                "from_tf": bool(".ckpt" in model_args.model_name_or_path),
                "config": config,
                "trust_remote_code": model_args.trust_remote_code,
                "torch_dtype": torch.bfloat16,
            }
            if model_args.use_flash_attn:
                model_kw["attn_implementation"] = "flash_attention_2"
            else:
                model_kw["attn_implementation"] = "sdpa"
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                **model_kw
            )
        else:
            model = model_args.from_config(config)
    finally:
        if orig_rv is not None:
            _restore_require_version(orig_rv)

    if is_gme:
        _patch_visual_output(model)
    elif hasattr(model, 'set_processor'):
        model.set_processor(model_args.model_name_or_path)

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=model_args.trust_remote_code)

    if os.path.exists(os.path.join(output_dir, 'embedding', 'emb.pth')):
        model.set_input_embeddings(torch.load(os.path.join(output_dir, 'embedding', 'emb.pth')))
        model.config.vocab_size = len(tokenizer)

    try:
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
    except Exception:
        model = PeftModel.from_pretrained(model, find_largest_checkpoint(output_dir))
        model = model.merge_and_unload()

    tokenizer.save_pretrained(os.path.join(output_dir, 'merged_model'))

    model.save_pretrained(os.path.join(output_dir, 'merged_model'))

