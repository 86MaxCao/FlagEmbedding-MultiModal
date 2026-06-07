"""Jina Embeddings V4 multimodal embedder.

Includes compatibility patches for transformers >= 5.x where the bundled
jina-embeddings-v4 model code was written for transformers 4.x.
"""

import math
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from FlagEmbedding.abc.inference import AbsEmbedder


def _apply_jina_v4_compat_patches():
    """Apply one-time monkey-patches for jina-embeddings-v4 on transformers >= 5.x.

    Fixes two breaking changes:
    1. ``ROPE_INIT_FUNCTIONS`` no longer contains a ``'default'`` key.
    2. ``_tied_weights_keys`` is a list but transformers 5.x expects a dict
       (handled via ``config.tie_word_embeddings = False`` at load time).

    The bundled ``Qwen2_5_VLProcessor`` tokenizer issue is handled separately
    by :func:`_patch_jina_v4_bundled_processor`.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(
            config=None, device=None, seq_len=None, layer_type=None,
        ):
            config.standardize_rope_params()
            rope_params = (
                config.rope_parameters[layer_type]
                if layer_type else config.rope_parameters
            )
            base = rope_params["rope_theta"]
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (
                    torch.arange(0, dim, 2, dtype=torch.int64)
                    .to(device=device, dtype=torch.float) / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_jina_v4_bundled_processor(model_name_or_path, config):
    """Patch the bundled Qwen2_5_VLProcessor for transformers >= 5.x compatibility.

    The jina-embeddings-v4 model ships its own ``qwen2_5_vl.py`` which defines a
    ``Qwen2_5_VLProcessor`` whose ``__init__`` crashes when ``tokenizer=None``.
    Under transformers 5.x, ``ProcessorMixin.from_pretrained`` no longer
    auto-instantiates sub-components, so the tokenizer arrives as ``None``.

    This function force-loads the bundled module (via the ``auto_map`` in config),
    finds the bundled ``Qwen2_5_VLProcessor``, and patches its ``__init__`` to
    auto-create the tokenizer when it is missing.
    """
    import sys
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    auto_map = getattr(config, "auto_map", None) or {}
    model_class_ref = auto_map.get("AutoModel")
    if not model_class_ref:
        return

    try:
        get_class_from_dynamic_module(model_class_ref, model_name_or_path)
    except Exception:
        return

    for mod_name, mod in list(sys.modules.items()):
        if ("qwen2_5_vl" in mod_name
                and "transformers_modules" in mod_name
                and hasattr(mod, "Qwen2_5_VLProcessor")):
            proc_cls = mod.Qwen2_5_VLProcessor
            if getattr(proc_cls, "_jina_patched", False):
                return
            orig_init = proc_cls.__init__

            def _make_patched_init(orig, path):
                def _patched(self, image_processor=None, tokenizer=None,
                             video_processor=None, chat_template=None, **kwargs):
                    if tokenizer is None:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(
                            path, trust_remote_code=True,
                        )
                    if image_processor is None:
                        from transformers import AutoImageProcessor
                        image_processor = AutoImageProcessor.from_pretrained(
                            path, trust_remote_code=True,
                        )
                    if video_processor is None:
                        try:
                            from transformers import AutoVideoProcessor
                            video_processor = AutoVideoProcessor.from_pretrained(
                                path, trust_remote_code=True,
                            )
                        except Exception:
                            pass
                    orig(
                        self, image_processor=image_processor,
                        tokenizer=tokenizer,
                        video_processor=video_processor,
                        chat_template=chat_template, **kwargs,
                    )
                    if not hasattr(self, "tokenizer") or self.tokenizer is None:
                        self.tokenizer = tokenizer
                    if not hasattr(self, "image_processor") or self.image_processor is None:
                        self.image_processor = image_processor
                    if not hasattr(self, "video_processor") or self.video_processor is None:
                        self.video_processor = video_processor
                return _patched

            proc_cls.__init__ = _make_patched_init(orig_init, model_name_or_path)
            proc_cls._jina_patched = True
            return


class JinaEmbeddingsV4Embedder(AbsEmbedder):
    """Embedder for ``jina-embeddings-v4`` multimodal model.

    Wraps the model's native ``encode_text`` / ``encode_image`` API
    (which internally uses multi-adapter LoRA for different tasks).

    Args:
        model_name_or_path: Path or HF repo id.
        normalize_embeddings: L2-normalize output vectors.
        use_fp16: Ignored (model natively uses bf16 via ``torch_dtype='auto'``).
        devices: Target device(s).
        batch_size: Encoding batch size.
        query_max_length: Max token length for queries.
        passage_max_length: Max token length for passages.
        convert_to_numpy: Return ``np.ndarray`` instead of ``torch.Tensor``.
        task: LoRA adapter task name (``'retrieval'``, ``'text-matching'``, ``'code'``).
        trust_remote_code: Trust remote code when loading model.
        cache_dir: HF cache directory.
    """

    DEFAULT_POOLING_METHOD = "mean"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        devices: Optional[Union[str, List[str]]] = None,
        batch_size: int = 8,
        query_max_length: int = 32768,
        passage_max_length: int = 32768,
        convert_to_numpy: bool = True,
        task: str = "retrieval",
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs,
        )

        from transformers import AutoModel, AutoConfig

        _apply_jina_v4_compat_patches()

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        config._attn_implementation = "sdpa"
        config._attn_implementation_internal = "sdpa"
        config.tie_word_embeddings = False

        _patch_jina_v4_bundled_processor(model_name_or_path, config)

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype="auto",
            attn_implementation="sdpa",
        )
        self.model.task = task
        self.model.to(self.target_devices[0])
        self.model.eval()
        self.trust_remote_code = trust_remote_code

    def encode_queries(
        self,
        queries: Union[List[str], str] = None,
        images: Union[List[str], List[Image.Image], str, Image.Image] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        task_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode queries (text and/or images).

        For text queries, uses ``prompt_name='query'`` to prepend the ``Query:`` prefix.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.query_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        if images is not None:
            return self._encode_images(
                images, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            )

        if queries is None:
            raise ValueError("Either queries or images must be provided.")

        return self.encode(
            queries, batch_size=batch_size, max_length=max_length,
            convert_to_numpy=convert_to_numpy, prompt_name="query", **kwargs,
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str] = None,
        images: Union[List[str], List[Image.Image], str, Image.Image] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode corpus documents (text and/or images).

        For text passages, uses ``prompt_name='passage'`` to prepend the ``Passage:`` prefix.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.passage_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        if images is not None:
            return self._encode_images(
                images, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            )

        if corpus is None:
            raise ValueError("Either corpus or images must be provided.")

        return self.encode(
            corpus, batch_size=batch_size, max_length=max_length,
            convert_to_numpy=convert_to_numpy, prompt_name="passage", **kwargs,
        )

    def _encode_images(
        self,
        images: Union[List[str], List[Image.Image], str, Image.Image],
        batch_size: int = 8,
        convert_to_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(images, (str, Image.Image)):
            images = [images]

        if len(self.target_devices) == 1:
            return self._encode_images_single_device(
                images, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
            )

        n = len(images)
        chunk_size = math.ceil(n / len(self.target_devices))
        parts = []
        for i, device in enumerate(self.target_devices):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= n:
                break
            emb = self._encode_images_single_device(
                images[start:end], batch_size=batch_size,
                convert_to_numpy=convert_to_numpy, device=device,
            )
            parts.append(emb)

        if convert_to_numpy:
            return np.concatenate(parts, axis=0)
        return torch.cat(parts, dim=0)

    def _encode_images_single_device(
        self,
        images: list,
        batch_size: int,
        convert_to_numpy: bool,
        device: str,
    ) -> Union[np.ndarray, torch.Tensor]:
        self.model.to(device)
        embeddings = self.model.encode_image(
            images, batch_size=batch_size,
            return_numpy=convert_to_numpy,
        )
        if convert_to_numpy:
            if isinstance(embeddings, np.ndarray):
                return embeddings
            if isinstance(embeddings, list):
                return np.stack([e.cpu().float().numpy() for e in embeddings])
            return embeddings.cpu().float().numpy()
        if isinstance(embeddings, list):
            return torch.stack(embeddings)
        return embeddings

    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        prompt_name = kwargs.pop("prompt_name", "query")
        if device is None:
            device = self.target_devices[0]
        return self._encode_text_single_device(
            sentences if isinstance(sentences, list) else [sentences],
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            device=device,
            prompt_name=prompt_name,
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        prompt_name: str = "query",
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.passage_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        if isinstance(sentences, str):
            sentences = [sentences]

        if len(self.target_devices) == 1:
            return self._encode_text_single_device(
                sentences, batch_size=batch_size, max_length=max_length,
                convert_to_numpy=convert_to_numpy, device=self.target_devices[0],
                prompt_name=prompt_name,
            )

        n = len(sentences)
        chunk_size = math.ceil(n / len(self.target_devices))
        parts = []
        for i, device in enumerate(self.target_devices):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= n:
                break
            emb = self._encode_text_single_device(
                sentences[start:end], batch_size=batch_size, max_length=max_length,
                convert_to_numpy=convert_to_numpy, device=device,
                prompt_name=prompt_name,
            )
            parts.append(emb)

        if convert_to_numpy:
            return np.concatenate(parts, axis=0)
        return torch.cat(parts, dim=0)

    def _encode_text_single_device(
        self,
        sentences: List[str],
        batch_size: int,
        max_length: int,
        convert_to_numpy: bool,
        device: str,
        prompt_name: str = "query",
    ) -> Union[np.ndarray, torch.Tensor]:
        self.model.to(device)
        embeddings = self.model.encode_text(
            sentences, max_length=max_length, batch_size=batch_size,
            return_numpy=convert_to_numpy, prompt_name=prompt_name,
        )
        if convert_to_numpy:
            if isinstance(embeddings, np.ndarray):
                return embeddings
            if isinstance(embeddings, list):
                return np.stack([e.cpu().float().numpy() for e in embeddings])
            return embeddings.cpu().float().numpy()
        if isinstance(embeddings, list):
            return torch.stack(embeddings)
        return embeddings
