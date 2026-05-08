import os
import gc
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional

from FlagEmbedding.abc.evaluation import EvalDenseRetriever, EvalReranker
from FlagEmbedding.abc.evaluation.utils import index, search

logger = logging.getLogger(__name__)


class MultimodalEvalRetriever(EvalDenseRetriever):
    """
    多模态检索器，支持图像+文本的查询和候选。
    
    扩展基类以支持传递图像给embedder。
    """
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, Any],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        执行多模态检索。
        
        Args:
            corpus: 候选库，格式 {doc_id: {'text': str, 'image': str}}
            queries: 查询，格式 {query_id: {'text': str, 'image': str}}
            corpus_embd_save_dir: 保存corpus embeddings的目录
            ignore_identical_ids: 是否忽略相同ID的结果
            **kwargs: 其他参数
            
        Returns:
            检索结果，格式 {query_id: {doc_id: score}}
        """
        if ignore_identical_ids:
            logger.warning("ignore_identical_ids is set to True.")
        
        # 提取corpus的文本和图像
        corpus_ids = []
        corpus_texts = []
        corpus_images = []
        
        for docid, doc in corpus.items():
            corpus_ids.append(docid)
            # 提取文本（如果有）
            text = doc.get('text', '')
            if 'title' in doc and doc['title']:
                text = f"{doc['title']} {text}".strip()
            corpus_texts.append(text if text else None)
            # 提取图像路径
            corpus_images.append(doc.get('image', None))
        
        # 提取queries的文本和图像
        queries_ids = []
        queries_texts = []
        queries_images = []
        
        for qid, query in queries.items():
            queries_ids.append(qid)
            if isinstance(query, dict):
                queries_texts.append(query.get('text', None))
                queries_images.append(query.get('image', None))
            else:
                # 兼容纯文本query
                queries_texts.append(query)
                queries_images.append(None)
        
        # 编码corpus
        if corpus_embd_save_dir is not None and not getattr(self, 'skip_corpus_cache', False):
            emb_path = os.path.join(corpus_embd_save_dir, "doc.npy")
            if os.path.exists(emb_path) and not self.overwrite:
                logger.info(f"Loading corpus embeddings from {emb_path}")
                corpus_emb = np.load(emb_path)
            else:
                corpus_emb = self._encode_corpus(corpus_texts, corpus_images, **kwargs)
        else:
            if getattr(self, 'skip_corpus_cache', False):
                logger.info("Skipping corpus embedding cache (skip_corpus_cache=True)")
            corpus_emb = self._encode_corpus(corpus_texts, corpus_images, **kwargs)
        
        # 编码queries
        queries_emb = self._encode_queries(queries_texts, queries_images, **kwargs)
        
        # 处理M3Embedder的字典格式
        if isinstance(corpus_emb, dict):
            corpus_emb = corpus_emb["dense_vecs"]
        if isinstance(queries_emb, dict):
            queries_emb = queries_emb["dense_vecs"]
        
        # 保存corpus embeddings
        if corpus_embd_save_dir is not None and not self.skip_corpus_cache and \
            (not os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) or self.overwrite):
            os.makedirs(corpus_embd_save_dir, exist_ok=True)
            np.save(os.path.join(corpus_embd_save_dir, "doc.npy"), corpus_emb)
            logger.info(f"Corpus embeddings saved to {corpus_embd_save_dir}")
        elif self.skip_corpus_cache:
            logger.info("Skipping corpus embedding cache (skip_corpus_cache=True)")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 使用FAISS进行检索
        faiss_index = index(corpus_embeddings=corpus_emb)
        all_scores, all_indices = search(
            query_embeddings=queries_emb, 
            faiss_index=faiss_index, 
            k=self.search_top_k
        )
        
        # 组织结果
        results = {}
        for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            results[queries_ids[idx]] = {}
            for score, indice in zip(scores, indices):
                if indice != -1:
                    if ignore_identical_ids and corpus_ids[indice] == queries_ids[idx]:
                        continue
                    results[queries_ids[idx]][corpus_ids[indice]] = float(score)
        
        return results
    
    def _bytes_to_pil(self, img_bytes):
        """将 bytes 转换为 PIL Image"""
        from PIL import Image
        import io
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    def _encode_pil_images_directly(self, texts, pil_images, q_or_c="c", **kwargs):
        """直接使用 PIL Image 编码，绕过 data_process 方法"""
        if not hasattr(self.embedder.model, 'data_process'):
            raise ValueError("Model does not have data_process method")
        
        # 准备文本输入
        if texts is None:
            texts = [None] * len(pil_images)
        elif not isinstance(texts, list):
            texts = [texts]
        
        # 确保长度一致
        if len(texts) != len(pil_images):
            if len(texts) == 1:
                texts = texts * len(pil_images)
            else:
                raise ValueError(f"Texts and images length mismatch: {len(texts)} vs {len(pil_images)}")
        
        # 直接调用模型的 data_process，但传入 PIL Image
        # 我们需要修改 data_process 的调用方式
        batch_size = kwargs.get('batch_size', 32)
        max_length = kwargs.get('max_length', 512)
        convert_to_numpy = kwargs.get('convert_to_numpy', True)
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_images = pil_images[i:i+batch_size]
            
            # 构造输入
            batch_inputs = []
            for text, image in zip(batch_texts, batch_images):
                # 准备文本输入
                if text is None:
                    text_input = ""
                else:
                    text_input = text
                
                # 直接使用 processor 处理
                if hasattr(self.embedder.model, 'processor'):
                    inputs = self.embedder.model.processor(
                        images=[image],
                        text=[text_input],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    )
                    batch_inputs.append(inputs)
            
            # 批量编码
            if batch_inputs:
                # 合并批次
                import torch
                # from torch.nn.utils.rnn import pad_sequence  # 未使用
                
                # 这里需要根据具体模型调整
                # 暂时使用单个处理的方式
                batch_embeddings = []
                for inputs in batch_inputs:
                    inputs = {k: v.to(self.embedder.model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.embedder.model(**inputs, output_hidden_states=True)
                        # 获取最后一层的隐藏状态
                        embeddings = outputs.hidden_states[-1][:, -1, :]  # 取最后一个token
                        if convert_to_numpy:
                            embeddings = embeddings.cpu().numpy()
                        batch_embeddings.append(embeddings)
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
        
        if all_embeddings:
            import numpy as np
            return np.concatenate(all_embeddings, axis=0)
        else:
            raise ValueError("No embeddings generated")
    
    def _encode_corpus(self, texts, images, **kwargs):
        """编码corpus，支持多模态"""
        # 检查是否有 bytes 数据
        if images and isinstance(images[0], bytes):
            # 将 bytes 转换为 PIL Image，然后直接调用模型
            pil_images = [self._bytes_to_pil(img_bytes) for img_bytes in images]
            return self._encode_pil_images_directly(texts, pil_images, q_or_c="c", **kwargs)
        
        # 检查embedder是否支持图像
        if hasattr(self.embedder, 'encode_corpus'):
            # 检查方法签名
            import inspect
            sig = inspect.signature(self.embedder.encode_corpus)
            if 'images' in sig.parameters:
                # 支持images参数
                return self.embedder.encode_corpus(corpus=texts, images=images, **kwargs)
            else:
                # 不支持images参数，只传文本
                logger.warning("Embedder does not support images in encode_corpus, using text only")
                return self.embedder.encode_corpus(texts, **kwargs)
        else:
            return self.embedder.encode(texts, **kwargs)
    
    def _encode_queries(self, texts, images, **kwargs):
        """编码queries，支持多模态"""
        # 检查是否有 bytes 数据
        if images and isinstance(images[0], bytes):
            # 将 bytes 转换为 PIL Image，然后直接调用模型
            pil_images = [self._bytes_to_pil(img_bytes) for img_bytes in images]
            return self._encode_pil_images_directly(texts, pil_images, q_or_c="q", **kwargs)
        
        # 检查embedder是否支持图像
        if hasattr(self.embedder, 'encode_queries'):
            import inspect
            sig = inspect.signature(self.embedder.encode_queries)
            if 'images' in sig.parameters:
                # 支持images参数
                return self.embedder.encode_queries(queries=texts, images=images, **kwargs)
            else:
                # 不支持images参数，只传文本
                logger.warning("Embedder does not support images in encode_queries, using text only")
                return self.embedder.encode_queries(texts, **kwargs)
        else:
            return self.embedder.encode(texts, **kwargs)


class MultimodalEvalReranker(EvalReranker):
    """
    多模态重排序器，支持图像+文本的查询和文档。
    
    扩展基类以支持传递图像给reranker。
    """
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, Any],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        执行多模态重排序。
        
        Args:
            corpus: 候选库
            queries: 查询
            search_results: 检索结果
            ignore_identical_ids: 是否忽略相同ID
            **kwargs: 其他参数
            
        Returns:
            重排序结果
        """
        # 截取top-k结果
        for qid in search_results:
            search_results[qid] = dict(
                sorted(search_results[qid].items(), key=lambda x: x[1], reverse=True)[
                    :self.rerank_top_k
                ]
            )
        
        # 生成(query, doc)对
        sentence_pairs = []
        query_images = []
        doc_images = []
        
        for qid in search_results:
            query_data = queries[qid]
            
            # 提取query信息
            if isinstance(query_data, dict):
                query_text = query_data.get('text', '')
                query_image = query_data.get('image', None)
            else:
                query_text = query_data
                query_image = None
            
            for docid in search_results[qid]:
                if ignore_identical_ids and qid == docid:
                    continue
                
                doc_data = corpus[docid]
                
                # 提取doc信息
                doc_text = doc_data.get('text', '')
                if 'title' in doc_data and doc_data['title']:
                    doc_text = f"{doc_data['title']} {doc_text}".strip()
                doc_image = doc_data.get('image', None)
                
                sentence_pairs.append({
                    'qid': qid,
                    'docid': docid,
                    'query_text': query_text,
                    'query_image': query_image,
                    'doc_text': doc_text,
                    'doc_image': doc_image
                })
        
        # 调用reranker
        scores = self._compute_scores(sentence_pairs, **kwargs)
        
        # 组织结果
        for i, score in enumerate(scores):
            sentence_pairs[i]['score'] = float(score)
        
        reranked_results = {qid: {} for qid in search_results}
        for pair in sentence_pairs:
            reranked_results[pair['qid']][pair['docid']] = pair['score']
        
        return reranked_results
    
    def _compute_scores(self, sentence_pairs, **kwargs):
        """计算重排序分数，支持多模态"""
        # 检查reranker是否有特殊的多模态接口
        if hasattr(self.reranker, 'compute_score'):
            import inspect
            sig = inspect.signature(self.reranker.compute_score)
            
            # 检查是否支持query_type和doc_type参数（MultimodalReranker的接口）
            if 'query_type' in sig.parameters and 'doc_type' in sig.parameters:
                # 使用MultimodalReranker接口
                pairs = []
                for pair_data in sentence_pairs:
                    # 确定query和doc类型
                    query = pair_data['query_image'] if pair_data['query_image'] else pair_data['query_text']
                    doc = pair_data['doc_image'] if pair_data['doc_image'] else pair_data['doc_text']
                    pairs.append([query, doc])
                
                query_type = 'image' if sentence_pairs[0]['query_image'] else 'text'
                doc_type = 'image' if sentence_pairs[0]['doc_image'] else 'text'
                
                return self.reranker.compute_score(
                    pairs,
                    query_type=query_type,
                    doc_type=doc_type,
                    **kwargs
                )
            else:
                # 标准接口，只传文本对
                pairs = [
                    (pair_data['query_text'], pair_data['doc_text'])
                    for pair_data in sentence_pairs
                ]
                return self.reranker.compute_score(pairs, **kwargs)
        else:
            raise NotImplementedError("Reranker does not have compute_score method")

