from dataclasses import dataclass

from FlagEmbedding.finetune.reranker.multimodal.base.arguments import MultimodalRerankerModelArguments


@dataclass
class Qwen3VLRerankerModelArguments(MultimodalRerankerModelArguments):
    """Model argument class for Qwen3-VL-Reranker.

    Inherits all fields from MultimodalRerankerModelArguments.
    """
    pass
