"""Centralized compatibility patches for transformers 5.8.0+.

All patches are idempotent (safe to call multiple times) and backward-compatible
(applying them in older transformers versions is harmless).
"""
import torch
import torch.nn as nn


def patch_qwen2vl_config():
    """Patch Qwen2VLConfig to expose hidden_size as a top-level property.

    In transformers 5.8.0+, Qwen2VLConfig stores hidden_size inside text_config,
    but JinaVLForRanking's __init__ accesses config.hidden_size directly to create
    its score MLP layers.
    """
    flag = '_flag_qwen2vl_config_patched'
    try:
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
        if not getattr(Qwen2VLConfig, flag, False):
            if not hasattr(Qwen2VLConfig, 'hidden_size'):
                @property
                def _hidden_size(self):
                    return self.text_config.hidden_size
                Qwen2VLConfig.hidden_size = _hidden_size
            setattr(Qwen2VLConfig, flag, True)
    except Exception:
        pass


def patch_identity():
    """Patch nn.Identity to have a weight attribute.

    In transformers 5.8.0+, the tie_weights initialization loop accesses .weight
    on every module. nn.Identity lacks this attribute, causing crashes when
    JinaVLForRanking replaces lm_head with Identity.
    """
    flag = '_flag_identity_patched'
    try:
        if not getattr(nn.Identity, flag, False):
            _orig_init = nn.Identity.__init__

            def _new_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                self.weight = nn.Parameter(torch.empty(0))

            nn.Identity.__init__ = _new_init
            setattr(nn.Identity, flag, True)
    except Exception:
        pass


def patch_tie_weights_for_meta_tensors():
    """Patch PreTrainedModel.tie_weights to handle Meta tensors in torch.equal.

    In transformers 5.8.0+, tie_weights calls torch.equal on parameter pairs.
    If a parameter is still on the meta device (e.g. a mismatched weight that
    was reinitialized), torch.equal raises NotImplementedError. This patch makes
    it return False for meta tensors instead of crashing.
    """
    flag = '_flag_tie_weights_patched'
    try:
        from transformers.modeling_utils import PreTrainedModel
        if not getattr(PreTrainedModel, flag, False):
            _orig_tie_weights = PreTrainedModel.tie_weights

            def _safe_tie_weights(self, missing_keys=None, recompute_mapping=True):
                _orig_torch_equal = torch.equal

                def _safe_equal(a, b):
                    if a.is_meta or b.is_meta:
                        return False
                    return _orig_torch_equal(a, b)

                torch.equal = _safe_equal
                try:
                    return _orig_tie_weights(self, missing_keys=missing_keys, recompute_mapping=recompute_mapping)
                finally:
                    torch.equal = _orig_torch_equal

            PreTrainedModel.tie_weights = _safe_tie_weights
            setattr(PreTrainedModel, flag, True)
    except Exception:
        pass


def register_qwen3_vl_for_sequence_classification():
    """Register Qwen3VLConfig for AutoModelForSequenceClassification.

    Required for CrossEncoder (which uses AutoModelForSequenceClassification)
    to load Qwen3-VL-Reranker models. Without this registration,
    CrossEncoder cannot find the right model class.
    """
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        if Qwen3VLConfig not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content:
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content[Qwen3VLConfig] = Qwen3VLForConditionalGeneration
    except Exception:
        pass


# --- Convenience aggregators ---

def apply_jina_reranker_patches():
    """Apply all patches needed for jina-reranker-m0 (Qwen2VL-based)."""
    patch_qwen2vl_config()
    patch_identity()
    patch_tie_weights_for_meta_tensors()


def apply_qwen3_vl_reranker_patches():
    """Apply all patches needed for Qwen3-VL-Reranker."""
    register_qwen3_vl_for_sequence_classification()
    patch_identity()
    patch_tie_weights_for_meta_tensors()


def apply_qwen3_vl_embedding_patches():
    """Apply all patches needed for Qwen3-VL-Embedding training."""
    patch_identity()
    patch_tie_weights_for_meta_tensors()
