import logging
from FlagEmbedding.abc.evaluation import AbsEvaluator

from .data_loader import MegaPairsEvalDataLoader
from .searcher import MultimodalEvalRetriever, MultimodalEvalReranker

logger = logging.getLogger(__name__)


class MegaPairsEvaluator(AbsEvaluator):
    """
    MegaPairs评估器。
    
    继承自AbsEvaluator，使用多模态的Retriever和Reranker。
    """
    
    def __init__(
        self,
        eval_name: str,
        data_loader: MegaPairsEvalDataLoader,
        overwrite: bool = False,
    ):
        super().__init__(
            eval_name=eval_name,
            data_loader=data_loader,
            overwrite=overwrite
        )

