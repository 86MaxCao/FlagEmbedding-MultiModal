from enum import Enum
from typing import Type
from dataclasses import dataclass
from collections import OrderedDict

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.inference.reranker import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker
from FlagEmbedding.inference.reranker.multimodal import MultimodalReranker
from FlagEmbedding.inference.reranker.qwen3_vl_reranker import Qwen3VLReranker
from FlagEmbedding.inference.reranker.qwen3_reranker import Qwen3Reranker


class RerankerModelClass(Enum):
    ENCODER_ONLY_BASE = "encoder-only-base"
    DECODER_ONLY_BASE = "decoder-only-base"
    DECODER_ONLY_LAYERWISE = "decoder-only-layerwise"
    DECODER_ONLY_LIGHTWEIGHT = "decoder-only-lightweight"
    MULTIMODAL_BASE = "multimodal-base"
    QWEN3_VL_RERANKER = "qwen3-vl-reranker"
    QWEN3_RERANKER = "qwen3-reranker"


RERANKER_CLASS_MAPPING = OrderedDict([
    (RerankerModelClass.ENCODER_ONLY_BASE, FlagReranker),
    (RerankerModelClass.DECODER_ONLY_BASE, FlagLLMReranker),
    (RerankerModelClass.DECODER_ONLY_LAYERWISE, LayerWiseFlagLLMReranker),
    (RerankerModelClass.DECODER_ONLY_LIGHTWEIGHT, LightWeightFlagLLMReranker),
    (RerankerModelClass.MULTIMODAL_BASE, MultimodalReranker),
    (RerankerModelClass.QWEN3_VL_RERANKER, Qwen3VLReranker),
    (RerankerModelClass.QWEN3_RERANKER, Qwen3Reranker),
])


@dataclass
class RerankerConfig:
    model_class: Type[AbsReranker]
    trust_remote_code: bool = False


AUTO_RERANKER_MAPPING = OrderedDict([
    # ============================== BGE ==============================
    (
        "bge-reranker-base", 
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-large", 
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-v2-m3",
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-v2-gemma",
        RerankerConfig(FlagLLMReranker)
    ),
    (
        "bge-reranker-v2-minicpm-layerwise",
        RerankerConfig(LayerWiseFlagLLMReranker)
    ),
    (
        "bge-reranker-v2.5-gemma2-lightweight",
        RerankerConfig(LightWeightFlagLLMReranker)
    ),
    # ============================== Others ==============================
    (
        "jinaai/jina-reranker-v2-base-multilingual",
        RerankerConfig(FlagReranker)
    ),
    (
        "Alibaba-NLP/gte-multilingual-reranker-base",
        RerankerConfig(FlagReranker)
    ),
    (
        "maidalun1020/bce-reranker-base_v1",
        RerankerConfig(FlagReranker)
    ),
    (
        "jinaai/jina-reranker-v1-turbo-en",
        RerankerConfig(FlagReranker)
    ),
    # ============================== Multimodal ==============================
    (
        "jina-reranker-m0",
        RerankerConfig(MultimodalReranker, trust_remote_code=True)
    ),
    (
        "Qwen3-VL-Reranker-2B",
        RerankerConfig(Qwen3VLReranker, trust_remote_code=True)
    ),
    (
        "Qwen3-VL-Reranker-8B",
        RerankerConfig(Qwen3VLReranker, trust_remote_code=True)
    ),
    # ============================== Qwen3 Text Rerankers ==============================
    (
        "Qwen3-Reranker-0.6B",
        RerankerConfig(Qwen3Reranker, trust_remote_code=True)
    ),
    (
        "Qwen3-Reranker-4B",
        RerankerConfig(Qwen3Reranker, trust_remote_code=True)
    ),
    (
        "Qwen3-Reranker-8B",
        RerankerConfig(Qwen3Reranker, trust_remote_code=True)
    ),
])
