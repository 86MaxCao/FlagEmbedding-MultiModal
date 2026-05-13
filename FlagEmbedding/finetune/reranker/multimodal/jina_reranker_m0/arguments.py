from dataclasses import dataclass

from FlagEmbedding.finetune.reranker.multimodal.base.arguments import MultimodalRerankerModelArguments


@dataclass
class JinaRerankerM0ModelArguments(MultimodalRerankerModelArguments):
    """
    Model argument class for jina-reranker-m0.
    Inherits all fields from MultimodalRerankerModelArguments.
    """
    pass
