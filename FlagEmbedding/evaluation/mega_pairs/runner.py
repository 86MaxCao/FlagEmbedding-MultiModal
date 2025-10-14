import logging
from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MegaPairsEvalDataLoader
from .evaluator import MegaPairsEvaluator
from .searcher import MultimodalEvalRetriever, MultimodalEvalReranker

logger = logging.getLogger(__name__)


class MegaPairsEvalRunner(AbsEvalRunner):
    """
    MegaPairs评估运行器。
    
    管理整个评估流程。
    """
    
    def load_data_loader(self) -> MegaPairsEvalDataLoader:
        """加载数据加载器"""
        data_loader = MegaPairsEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
            dataset_name=getattr(self.eval_args, 'dataset_name', 'JUNJIE99/MegaPairs'),
            image_root_dir=getattr(self.eval_args, 'image_root_dir', None),
        )
        return data_loader
    
    def load_evaluator(self) -> MegaPairsEvaluator:
        """加载评估器"""
        evaluator = MegaPairsEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator
    
    def load_retriever_and_reranker(self):
        """加载多模态检索器和重排序器"""
        embedder, reranker = self.get_models(self.model_args)
        
        # 使用多模态Retriever
        if embedder is not None:
            retriever = MultimodalEvalRetriever(
                embedder=embedder,
                search_top_k=self.eval_args.search_top_k,
                overwrite=self.eval_args.overwrite
            )
        else:
            retriever = None
        
        # 使用多模态Reranker
        if reranker is not None:
            reranker_wrapper = MultimodalEvalReranker(
                reranker=reranker,
                rerank_top_k=self.eval_args.rerank_top_k
            )
        else:
            reranker_wrapper = None
        
        return retriever, reranker_wrapper

