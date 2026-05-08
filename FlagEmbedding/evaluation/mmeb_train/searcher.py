import os
import gc
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import faiss

from FlagEmbedding.abc.evaluation.searcher import EvalDenseRetriever, EvalReranker

logger = logging.getLogger(__name__)


class MMEBTrainEvalRetriever(EvalDenseRetriever):
    """
    MMEB-train多模态检索器，支持混合模态检索。
    
    支持：
    - Text query → Text documents
    - Image query → Text documents  
    - Text query → Image documents
    - Image query → Image documents
    """
    
    def __init__(self, embedder, search_top_k: int = 1000, overwrite: bool = False, skip_corpus_cache: bool = False):
        super().__init__(embedder, search_top_k, overwrite)
        self.skip_corpus_cache = skip_corpus_cache
    
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
        
        print(f"[DEBUG] Processing corpus with {len(corpus)} documents")
        for i, (doc_id, doc) in enumerate(corpus.items()):
            if i < 3:  # 只打印前3个文档的详细信息
                print(f"[DEBUG] Doc {i}: id={doc_id}, text={doc.get('text', '')[:50]}..., image={doc.get('image', '')}")
            corpus_ids.append(doc_id)
            corpus_texts.append(doc.get('text', ''))
            # 确保图像路径不是 None
            image_path = doc.get('image', '') or ''
            corpus_images.append(image_path)
        
        print(f"[DEBUG] Corpus summary: {len(corpus_texts)} texts, {len(corpus_images)} images")
        print(f"[DEBUG] Sample corpus_images: {corpus_images[:3]}")
        
        # 提取queries的文本和图像
        queries_ids = []
        queries_texts = []
        queries_images = []
        
        print(f"[DEBUG] Processing queries with {len(queries)} items")
        for i, (query_id, query) in enumerate(queries.items()):
            if i < 3:  # 只打印前3个查询的详细信息
                print(f"[DEBUG] Query {i}: id={query_id}, query={query if isinstance(query, str) else query.get('text', '')[:50]}...")
            queries_ids.append(query_id)
            if isinstance(query, dict):
                queries_texts.append(query.get('text', ''))
                # 确保图像路径不是 None
                image_path = query.get('image', '') or ''
                queries_images.append(image_path)
            else:
                # 兼容纯文本query
                queries_texts.append(query)
                queries_images.append('')
        
        print(f"[DEBUG] Queries summary: {len(queries_texts)} texts, {len(queries_images)} images")
        print(f"[DEBUG] Sample queries_images: {queries_images[:3]}")
        
        # 编码corpus
        print(f"[DEBUG] About to encode corpus...")
        if corpus_embd_save_dir is not None and not getattr(self, 'skip_corpus_cache', False):
            emb_path = os.path.join(corpus_embd_save_dir, "doc.npy")
            if os.path.exists(emb_path) and not self.overwrite:
                logger.info(f"Loading corpus embeddings from {emb_path}")
                corpus_emb = np.load(emb_path)
            else:
                print(f"[DEBUG] Calling _encode_corpus with {len(corpus_texts)} texts and {len(corpus_images)} images")
                corpus_emb = self._encode_corpus(corpus_texts, corpus_images, **kwargs)
        else:
            if getattr(self, 'skip_corpus_cache', False):
                logger.info("Skipping corpus embedding cache (skip_corpus_cache=True)")
            print(f"[DEBUG] Calling _encode_corpus with {len(corpus_texts)} texts and {len(corpus_images)} images")
            corpus_emb = self._encode_corpus(corpus_texts, corpus_images, **kwargs)
        
        # 编码queries
        print(f"[DEBUG] About to encode queries...")
        print(f"[DEBUG] Calling _encode_queries with {len(queries_texts)} texts and {len(queries_images)} images")
        queries_emb = self._encode_queries(queries_texts, queries_images, **kwargs)
        
        # 处理M3Embedder的字典格式
        if isinstance(corpus_emb, dict):
            corpus_emb = corpus_emb["dense_vecs"]
        if isinstance(queries_emb, dict):
            queries_emb = queries_emb["dense_vecs"]
        
        # 保存corpus embeddings
        if corpus_embd_save_dir is not None and not getattr(self, 'skip_corpus_cache', False) and \
            (not os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) or self.overwrite):
            os.makedirs(corpus_embd_save_dir, exist_ok=True)
            np.save(os.path.join(corpus_embd_save_dir, "doc.npy"), corpus_emb)
            logger.info(f"Corpus embeddings saved to {corpus_embd_save_dir}")
        elif getattr(self, 'skip_corpus_cache', False):
            logger.info("Skipping corpus embedding cache (skip_corpus_cache=True)")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 使用FAISS进行检索
        index = faiss.IndexFlatIP(corpus_emb.shape[1])
        index.add(corpus_emb.astype('float32'))
        
        # 计算相似度
        scores, indices = index.search(queries_emb.astype('float32'), len(corpus_ids))
        
        # 构建结果
        results = {}
        for idx, qid in enumerate(queries_ids):
            results[qid] = {}
            for i, (indice, score) in enumerate(zip(indices[idx], scores[idx])):
                if indice != -1:
                    if ignore_identical_ids and corpus_ids[indice] == queries_ids[idx]:
                        continue
                    results[qid][corpus_ids[indice]] = float(score)
        
        return results
    
    def _encode_corpus(self, texts, images, **kwargs):
        """编码corpus，支持多模态"""
        print(f"[DEBUG] _encode_corpus called with:")
        print(f"  texts: {texts[:2] if texts else texts} (length: {len(texts) if texts else 0})")
        print(f"  images: {images[:2] if images else images} (length: {len(images) if images else 0})")
        print(f"  images type: {type(images)}")
        
        # 确保 images 是列表
        if images is None:
            images = []
            print(f"[DEBUG] images was None, set to empty list")
        
        # 检查是否有有效的图像路径
        has_images = any(isinstance(img, str) and img.strip() for img in images if img)
        print(f"[DEBUG] has_images: {has_images}")
        
        if has_images:
            # 有图像，使用多模态编码
            print(f"[DEBUG] Using encode_corpus with images")
            return self.embedder.encode_corpus(corpus=texts, images=images, **kwargs)
        else:
            # 纯文本，不传递 images 参数，让模型使用默认值
            print(f"[DEBUG] Using encode without images parameter")
            return self.embedder.encode(sentences=texts, **kwargs)
    
    def _encode_queries(self, texts, images, **kwargs):
        """编码queries，支持多模态"""
        print(f"[DEBUG] _encode_queries called with:")
        print(f"  texts: {texts[:2] if texts else texts} (length: {len(texts) if texts else 0})")
        print(f"  images: {images[:2] if images else images} (length: {len(images) if images else 0})")
        print(f"  images type: {type(images)}")
        
        # 确保 images 是列表
        if images is None:
            images = []
            print(f"[DEBUG] images was None, set to empty list")
        
        # 检查是否有有效的图像路径
        has_images = any(isinstance(img, str) and img.strip() for img in images if img)
        print(f"[DEBUG] has_images: {has_images}")
        
        if has_images:
            # 有图像，使用多模态编码
            print(f"[DEBUG] Using encode_queries with images")
            return self.embedder.encode_queries(queries=texts, images=images, **kwargs)
        else:
            # 纯文本，不传递 images 参数，让模型使用默认值
            print(f"[DEBUG] Using encode without images parameter")
            return self.embedder.encode(sentences=texts, **kwargs)


class MMEBTrainEvalReranker(EvalReranker):
    """
    MMEB-train多模态重排序器，支持混合模态重排序。
    
    支持：
    - Text query + Text documents
    - Image query + Text documents
    - Text query + Image documents  
    - Image query + Image documents
    """
    
    def __init__(self, reranker, rerank_top_k: int = 100):
        super().__init__(reranker, rerank_top_k)
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, Any],
        top_k: int = 100,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        执行多模态重排序。
        
        Args:
            corpus: 候选库
            queries: 查询
            top_k: 重排序的top-k数量
            **kwargs: 其他参数
            
        Returns:
            重排序结果
        """
        results = {}
        
        for query_id, query in queries.items():
            if isinstance(query, dict):
                query_text = query.get('text', '')
                query_image = query.get('image', '')
            else:
                query_text = query
                query_image = ''
            
            # 获取候选文档
            candidate_docs = list(corpus.items())[:top_k]
            
            if not candidate_docs:
                results[query_id] = {}
                continue
            
            # 准备重排序对
            pairs = []
            doc_ids = []
            
            for doc_id, doc in candidate_docs:
                doc_text = doc.get('text', '')
                doc_image = doc.get('image', '')
                
                # 构建查询-文档对
                if query_image and doc_image:
                    # 图像查询 + 图像文档
                    pairs.append([query_image, doc_image])
                elif query_image and doc_text:
                    # 图像查询 + 文本文档
                    pairs.append([query_image, doc_text])
                elif query_text and doc_image:
                    # 文本查询 + 图像文档
                    pairs.append([query_text, doc_image])
                else:
                    # 文本查询 + 文本文档
                    pairs.append([query_text, doc_text])
                
                doc_ids.append(doc_id)
            
            # 计算重排序分数
            try:
                scores = self.reranker.compute_score(
                    pairs,
                    query_type='image' if query_image else 'text',
                    doc_type='image' if any(doc.get('image', '') for _, doc in candidate_docs) else 'text'
                )
                
                # 构建结果
                results[query_id] = {}
                for doc_id, score in zip(doc_ids, scores):
                    results[query_id][doc_id] = float(score)
                    
            except Exception as e:
                logger.error(f"Error in reranking for query {query_id}: {e}")
                results[query_id] = {}
        
        return results
