from FlagEmbedding.abc.evaluation.runner import AbsEvalRunner

from .data_loader import MMEBTrainEvalDataLoader
from .evaluator import MMEBTrainEvaluator


class MMEBTrainEvalRunner(AbsEvalRunner):
    """
    MMEB-train评估运行器。
    
    管理整个评估流程。
    """
    
    def load_data_loader(self) -> MMEBTrainEvalDataLoader:
        """加载数据加载器"""
        data_loader = MMEBTrainEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
            use_local_data=getattr(self.eval_args, 'use_local_data', True),
            max_samples=getattr(self.eval_args, 'max_samples', None),
            sub_datasets=getattr(self.eval_args, 'sub_datasets', None),
            task_types=getattr(self.eval_args, 'task_types', None),
        )
        return data_loader
    
    def load_evaluator(self) -> MMEBTrainEvaluator:
        """加载评估器"""
        evaluator = MMEBTrainEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
            skip_corpus_cache=getattr(self.eval_args, 'skip_corpus_cache', False),
        )
        return evaluator
    
    def load_retriever_and_reranker(self):
        """加载多模态检索器和重排序器"""
        from .searcher import MMEBTrainEvalRetriever, MMEBTrainEvalReranker
        
        embedder, reranker = self.get_models(self.model_args)
        
        # 使用多模态Retriever
        if embedder is not None:
            retriever = MMEBTrainEvalRetriever(
                embedder=embedder,
                search_top_k=self.eval_args.search_top_k,
                overwrite=self.eval_args.overwrite,
                skip_corpus_cache=getattr(self.eval_args, 'skip_corpus_cache', False),
            )
        else:
            retriever = None
        
        # 使用多模态Reranker
        if reranker is not None:
            reranker_wrapper = MMEBTrainEvalReranker(
                reranker=reranker,
                rerank_top_k=self.eval_args.rerank_top_k
            )
        else:
            reranker_wrapper = None
        
        return retriever, reranker_wrapper
