"""GME-Qwen2-VL multimodal embedder.

Wraps the GME-Qwen2-VL model (gme-Qwen2-VL-2B-Instruct, etc.) which provides
text, image, and fused (text+image) embedding via a Qwen2-VL backbone with
last-token pooling.

Includes a compatibility patch to bypass the ``require_version("transformers<4.52.0")``
guard in the bundled model code, since the core Qwen2-VL classes remain functional
in transformers >= 5.x.
"""

import math
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from FlagEmbedding.abc.inference import AbsEmbedder


def _bypass_gme_version_check():
    """Temporarily neuter ``require_version`` so the bundled model code loads.

    The bundled ``modeling_gme_qwen2vl.py`` calls
    ``require_version("transformers<4.52.0", ...)`` at import time, but the
    model works fine with transformers >= 5.x.  We monkey-patch
    ``transformers.utils.versions.require_version`` to silently skip any
    check that mentions ``transformers<``, then restore the original after
    the model is loaded.
    """
    import transformers.utils.versions as _ver_mod

    _orig = _ver_mod.require_version

    def _lenient_require_version(requirement, hint=None):
        if isinstance(requirement, str) and requirement.startswith("transformers<"):
            return
        return _orig(requirement, hint)

    _ver_mod.require_version = _lenient_require_version
    return _orig


def _restore_require_version(orig):
    import transformers.utils.versions as _ver_mod
    _ver_mod.require_version = orig


def _patch_rope_init_functions():
    """Add ``"default"`` to ``ROPE_INIT_FUNCTIONS`` if absent.

    Transformers >= 5.x removed the ``"default"`` key from
    ``ROPE_INIT_FUNCTIONS``, but some bundled model code (e.g.
    jina-embeddings-v4's Qwen2.5-VL variant) converts ``rope_scaling``
    type ``"mrope"`` → ``"default"`` and then looks it up.  The
    ``"default"`` initialisation is standard RoPE with no frequency
    scaling applied.
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()
        rope_params = getattr(config, "rope_parameters", None) or {}
        base = (rope_params.get("rope_theta") or getattr(config, "rope_theta", 10000.0))
        partial = rope_params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_processor_auto_loading():
    """Fix ``ProcessorMixin.get_attributes`` for subclasses that use ``*args, **kwargs``.

    Transformers >= 5.x resolves sub-processor attributes (``image_processor``,
    ``tokenizer``, etc.) by inspecting the processor's ``__init__`` signature.
    If the subclass uses ``*args, **kwargs`` (e.g. ``JinaEmbeddingsV4Processor``),
    no modality parameters are discovered.  The legacy fallback then checks
    ``cls.__dict__`` for ``*_class`` attributes, but these are defined on the
    *parent* class (``Qwen2_5_VLProcessor``) and therefore not found either.

    This patch walks the MRO to find modality parameters from parent ``__init__``
    signatures first (preserving correct argument order), then falls back to
    ``*_class`` attributes on parent classes.
    """
    import inspect
    from transformers import ProcessorMixin
    from transformers.processing_utils import MODALITY_TO_BASE_CLASS_MAPPING

    if getattr(ProcessorMixin, "_auto_loading_patched", False):
        return

    _orig_get_attributes = ProcessorMixin.get_attributes

    @classmethod
    def _patched_get_attributes(cls):
        result = _orig_get_attributes.__func__(cls)
        if result:
            return result

        modality_keys = set(MODALITY_TO_BASE_CLASS_MAPPING.keys())

        for parent in cls.__mro__[1:]:
            try:
                params = list(inspect.signature(parent.__init__).parameters.keys())
            except (ValueError, TypeError):
                continue
            for p in params:
                if p in ("self",) or p.startswith("*"):
                    continue
                if p == "audio_tokenizer":
                    continue
                if any(mod in p for mod in modality_keys):
                    if p not in result:
                        result.append(p)
            if result:
                break

        if not result:
            for parent in cls.__mro__[1:]:
                for attr_name, value in parent.__dict__.items():
                    if (value is not None
                            and attr_name.endswith("_class")
                            and attr_name != "audio_tokenizer_class"):
                        inferred = attr_name[: -len("_class")]
                        if inferred == "audio_tokenizer":
                            continue
                        if any(mod in inferred for mod in modality_keys):
                            if inferred not in result:
                                result.append(inferred)
                if result:
                    break

        return result

    ProcessorMixin.get_attributes = _patched_get_attributes
    ProcessorMixin._auto_loading_patched = True


def _patch_tied_weights_keys():
    """Make ``get_expanded_tied_weights_keys`` accept the old list format.

    Transformers >= 5.x changed ``_tied_weights_keys`` from a list of key
    names to a ``dict[target, source]``.  Some bundled model code (e.g.
    jina-embeddings-v4's Qwen2.5-VL variant) still declares it as a list,
    which crashes in ``get_expanded_tied_weights_keys`` when ``.keys()``
    is called on a list.  We wrap the method once to auto-convert.
    """
    from transformers import PreTrainedModel

    if getattr(PreTrainedModel, "_tied_keys_patched", False):
        return

    _orig = PreTrainedModel.get_expanded_tied_weights_keys

    def _compat_get_expanded_tied_weights_keys(self, all_submodels=False):
        twk = self._tied_weights_keys
        if isinstance(twk, list):
            self._tied_weights_keys = {}
        return _orig(self, all_submodels=all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = _compat_get_expanded_tied_weights_keys
    PreTrainedModel._tied_keys_patched = True


_COMPOSITE_PROMOTED_ATTRS = frozenset((
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "hidden_act", "max_position_embeddings",
    "rms_norm_eps", "use_cache", "rope_theta", "sliding_window",
    "max_window_layers", "attention_dropout", "rope_scaling",
))


def _patch_composite_config(config):
    """Promote ``text_config`` attributes to the top-level config.

    Transformers >= 5.x refactored ``Qwen2VLConfig`` into a composite config
    where ``vocab_size``, ``hidden_size``, etc. live under ``text_config``.
    The bundled GME model code expects these at the top level.

    Also patches the config *class* with a ``__getattr__`` fallback so that
    freshly created instances (e.g. from ``from_pretrained`` inside peft)
    can resolve these attributes without an explicit per-instance patch.
    """
    tc = getattr(config, "text_config", None)
    if tc is None:
        return
    for attr in _COMPOSITE_PROMOTED_ATTRS:
        if hasattr(tc, attr) and not hasattr(config, attr):
            setattr(config, attr, getattr(tc, attr))

    cls = type(config)
    if getattr(cls, "_composite_getattr_patched", False):
        return
    _orig_getattr = cls.__dict__.get("__getattr__")

    def _composite_getattr(self, name):
        if name in _COMPOSITE_PROMOTED_ATTRS:
            tc = self.__dict__.get("text_config")
            if tc is not None and hasattr(tc, name):
                return getattr(tc, name)
        if _orig_getattr is not None:
            return _orig_getattr(self, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    cls.__getattr__ = _composite_getattr
    cls._composite_getattr_patched = True


def _patch_visual_output(model):
    """Wrap the vision encoder so it returns a raw tensor.

    Transformers >= 5.x changed ``Qwen2VisionTransformerPretrainedModel`` to
    return ``BaseModelOutputWithPooling`` instead of a plain tensor.  The
    merged (spatially-downsampled + projected) embeddings that the bundled
    GME code expects are now stored in ``pooler_output``, while
    ``last_hidden_state`` holds the pre-merger activations.
    """
    visual = getattr(model, "visual", None)
    if visual is None:
        return
    _orig_forward = visual.forward

    def _unwrap_forward(*args, **kwargs):
        out = _orig_forward(*args, **kwargs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    visual.forward = _unwrap_forward


def _tie_gme_visual_encoder(model):
    """Replace the inner model's duplicate visual encoder with the outer one.

    Transformers >= 5.x creates a separate ``Qwen2VLModel.visual`` inside the
    ``GmeQwen2VL`` wrapper, but the checkpoint only stores weights under the
    top-level ``visual.*`` keys.  The inner copy (``model.model.visual``) ends
    up randomly initialized and wastes memory.  Tying them ensures a single
    set of parameters while preserving ``state_dict`` key compatibility.
    """
    if not (hasattr(model, "visual") and hasattr(model, "model")
            and hasattr(model.model, "visual")):
        return
    if model.visual is model.model.visual:
        return
    model.model.visual = model.visual


class GmeQwen2VLEmbedder(AbsEmbedder):
    """Embedder for ``gme-Qwen2-VL-*-Instruct`` multimodal models.

    Uses the model's native ``embed`` / ``get_text_embeddings`` /
    ``get_image_embeddings`` API with last-token pooling.

    Args:
        model_name_or_path: Path or HF repo id.
        normalize_embeddings: L2-normalize output vectors.
        use_fp16: Use float16 precision.
        devices: Target device(s).
        batch_size: Encoding batch size.
        query_max_length: Max token length for queries.
        passage_max_length: Max token length for passages.
        convert_to_numpy: Return ``np.ndarray`` instead of ``torch.Tensor``.
        trust_remote_code: Trust remote code when loading model.
        cache_dir: HF cache directory.
    """

    DEFAULT_POOLING_METHOD = "last_token"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        devices: Optional[Union[str, List[str]]] = None,
        batch_size: int = 8,
        query_max_length: int = 1800,
        passage_max_length: int = 1800,
        convert_to_numpy: bool = True,
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

        from transformers import AutoConfig, AutoModel

        orig_rv = _bypass_gme_version_check()
        try:
            dtype = torch.float16 if use_fp16 else torch.float32
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
            _patch_composite_config(config)
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        finally:
            _restore_require_version(orig_rv)

        _patch_visual_output(self.model)
        _tie_gme_visual_encoder(self.model)
        self.model.to(self.target_devices[0])
        self.model.eval()

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
        if batch_size is None:
            batch_size = self.batch_size
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        if queries is not None and images is not None:
            return self._encode_fused(
                queries, images, batch_size=batch_size,
                convert_to_numpy=convert_to_numpy,
                is_query=True, instruction=task_instruction,
            )

        if images is not None:
            return self._encode_images(
                images, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            )

        if queries is None:
            raise ValueError("Either queries or images must be provided.")

        if isinstance(queries, str):
            queries = [queries]

        return self._encode_text(
            queries, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            is_query=True, instruction=task_instruction,
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
        if batch_size is None:
            batch_size = self.batch_size
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        if images is not None:
            return self._encode_images(
                images, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            )

        if corpus is None:
            raise ValueError("Either corpus or images must be provided.")

        if isinstance(corpus, str):
            corpus = [corpus]

        return self._encode_text(
            corpus, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            is_query=False,
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy
        if isinstance(sentences, str):
            sentences = [sentences]

        return self._encode_text(
            sentences, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            is_query=True,
        )

    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sentences, str):
            sentences = [sentences]
        if device:
            self.model.to(device)
        return self._encode_text(
            sentences, batch_size=batch_size, convert_to_numpy=convert_to_numpy,
            is_query=True,
        )

    def _encode_text(
        self,
        texts: List[str],
        batch_size: int,
        convert_to_numpy: bool,
        is_query: bool = True,
        instruction: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        emb = self.model.get_text_embeddings(
            texts, batch_size=batch_size,
            is_query=is_query, instruction=instruction,
        )
        return self._post_process(emb, convert_to_numpy)

    def _encode_images(
        self,
        images: Union[List[str], List[Image.Image], str, Image.Image],
        batch_size: int,
        convert_to_numpy: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(images, (str, Image.Image)):
            images = [images]
        emb = self.model.get_image_embeddings(
            images, batch_size=batch_size,
        )
        return self._post_process(emb, convert_to_numpy)

    def _encode_fused(
        self,
        texts: Union[List[str], str],
        images: Union[List[str], List[Image.Image], str, Image.Image],
        batch_size: int,
        convert_to_numpy: bool,
        is_query: bool = True,
        instruction: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, (str, Image.Image)):
            images = [images]
        emb = self.model.get_fused_embeddings(
            texts=texts, images=images, batch_size=batch_size,
            is_query=is_query, instruction=instruction,
        )
        return self._post_process(emb, convert_to_numpy)

    @staticmethod
    def _post_process(
        embeddings: torch.Tensor,
        convert_to_numpy: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        if convert_to_numpy:
            if isinstance(embeddings, np.ndarray):
                return embeddings
            return embeddings.cpu().float().numpy()
        return embeddings
