import logging
from FlagEmbedding.abc.evaluation import AbsEvaluator

from .data_loader import MMEBTrainEvalDataLoader
from .searcher import MMEBTrainEvalRetriever, MMEBTrainEvalReranker

logger = logging.getLogger(__name__)


class MMEBTrainEvaluator(AbsEvaluator):
    """
    MMEB-train评估器。
    
    继承自AbsEvaluator，使用多模态的Retriever和Reranker。
    支持混合模态评估：文本问答、图像检索、混合任务。
    """
    
    def __init__(
        self,
        eval_name: str,
        data_loader: MMEBTrainEvalDataLoader,
        overwrite: bool = False,
        skip_corpus_cache: bool = False,
    ):
        super().__init__(
            eval_name=eval_name,
            data_loader=data_loader,
            overwrite=overwrite
        )
        self.skip_corpus_cache = skip_corpus_cache
    
    def get_retriever(self, retriever, **kwargs):
        """获取多模态检索器"""
        return MMEBTrainEvalRetriever(
            embedder=retriever,
            skip_corpus_cache=self.skip_corpus_cache,
            **kwargs
        )
    
    def get_reranker(self, reranker, **kwargs):
        """获取多模态重排序器"""
        return MMEBTrainEvalReranker(
            reranker=reranker,
            **kwargs
        )
