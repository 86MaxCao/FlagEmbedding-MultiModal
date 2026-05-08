import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MegaPairsEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader for MegaPairs dataset.
    
    MegaPairs format:
    {
        "q_img": "path/to/query_image.jpg",
        "q_texts": ["text instruction 1", "text instruction 2", ...],
        "t_img": "path/to/target_image.jpg",
        "hns": ["hard_negative_1.jpg", "hard_negative_2.jpg", ...]
    }
    """
    
    def __init__(
        self,
        eval_name: str,
        dataset_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_redownload: bool = False,
        use_local_data: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            eval_name=eval_name,
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            token=token,
            force_redownload=force_redownload
        )
        self.use_local_data = use_local_data
        self.max_samples = max_samples
    
    def available_dataset_names(self) -> List[str]:
        """Get available dataset names."""
        return []
    
    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """Get available splits."""
        return ["train", "test"]
    
    def _load_remote_corpus(
        self,
        dataset_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load corpus from local parquet MegaPairs dataset.
        
        Corpus包含所有候选图像（target_image + hard_negative_images中的所有图像）
        """
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading MegaPairs dataset from local parquet: {self.dataset_dir}")
        
        # Load dataset from local parquet files
        mega_pairs = datasets.load_dataset(
            'parquet',
            data_dir=self.dataset_dir,
            streaming=True
        )
        
        # Collect all unique images as corpus
        corpus_dict = {}
        
        for idx, data in enumerate(tqdm(mega_pairs['train'], desc="Loading corpus")):
            # 限制样本数量
            if self.max_samples is not None and idx >= self.max_samples:
                break
                
            # Add target image
            target_image_id = data['target_image_id']
            target_image_bytes = data['target_image']
            
            if target_image_id not in corpus_dict:
                corpus_dict[target_image_id] = {
                    'image': target_image_bytes,  # 直接存储 bytes
                    'text': ''  # 纯图像检索，文本为空
                }
            
            # Add hard negatives
            hard_negative_ids = data.get('hard_negative_images_id', [])
            hard_negative_images = data.get('hard_negative_images', [])
            
            for hn_id, hn_bytes in zip(hard_negative_ids, hard_negative_images):
                if hn_id not in corpus_dict:
                    corpus_dict[hn_id] = {
                        'image': hn_bytes,  # 直接存储 bytes
                        'text': ''
                    }
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            corpus_path = os.path.join(save_dir, 'corpus.jsonl')
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for doc_id, doc_data in tqdm(corpus_dict.items(), desc="Saving corpus"):
                    # 注意：这里不能直接保存 bytes 到 JSON，需要 base64 编码
                    import base64
                    image_b64 = base64.b64encode(doc_data['image']).decode('utf-8')
                    f.write(json.dumps({
                        'id': doc_id,
                        'image': image_b64,  # base64 编码的 bytes
                        'text': doc_data.get('text', '')
                    }, ensure_ascii=False) + '\n')
            logger.info(f"Corpus saved to {corpus_path}")
        
        return datasets.DatasetDict(corpus_dict)
    
    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load relevance labels from local parquet MegaPairs dataset."""
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading qrels for split: {split}")
        
        # Load dataset from local parquet files
        mega_pairs = datasets.load_dataset(
            'parquet',
            data_dir=self.dataset_dir,
            streaming=True
        )
        
        qrels_dict = {}
        
        # 为每个q_text创建qrel
        for idx, data in enumerate(tqdm(mega_pairs['train'], desc="Loading qrels")):
            # 限制样本数量
            if self.max_samples is not None and idx >= self.max_samples:
                break
                
            target_image_id = data['target_image_id']
            q_texts = data.get('query_texts', [])
            
            # 为每个query text创建一个qrel
            for text_idx, q_text in enumerate(q_texts):
                qid = f"{idx}_{text_idx}"
                qrels_dict[qid] = {target_image_id: 1}  # target_image_id是正确答案
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            qrels_path = os.path.join(save_dir, f'{split}_qrels.jsonl')
            with open(qrels_path, 'w', encoding='utf-8') as f:
                for qid, doc_rels in tqdm(qrels_dict.items(), desc="Saving qrels"):
                    for docid, relevance in doc_rels.items():
                        f.write(json.dumps({
                            'qid': qid,
                            'docid': docid,
                            'relevance': relevance
                        }, ensure_ascii=False) + '\n')
            logger.info(f"Qrels saved to {qrels_path}")
        
        return datasets.DatasetDict(qrels_dict)
    
    def _load_remote_queries(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load queries from local parquet MegaPairs dataset.
        
        每个query包含：query image + query text instruction
        """
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading queries for split: {split}")
        
        # Load dataset from local parquet files
        mega_pairs = datasets.load_dataset(
            'parquet',
            data_dir=self.dataset_dir,
            streaming=True
        )
        
        queries_dict = {}
        
        # 为每个q_text创建一个query
        for idx, data in enumerate(tqdm(mega_pairs['train'], desc="Loading queries")):
            # 限制样本数量
            if self.max_samples is not None and idx >= self.max_samples:
                break
                
            # query_image_id = data['query_image_id']  # 未使用
            query_image_bytes = data['query_image']
            q_texts = data.get('query_texts', [])
            
            # 为每个query text创建一个query
            for text_idx, q_text in enumerate(q_texts):
                qid = f"{idx}_{text_idx}"
                queries_dict[qid] = {
                    'text': q_text,
                    'image': query_image_bytes  # 直接存储 bytes
                }
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            queries_path = os.path.join(save_dir, f'{split}_queries.jsonl')
            with open(queries_path, 'w', encoding='utf-8') as f:
                for qid, query_data in tqdm(queries_dict.items(), desc="Saving queries"):
                    # 注意：这里不能直接保存 bytes 到 JSON，需要 base64 编码
                    import base64
                    image_b64 = base64.b64encode(query_data['image']).decode('utf-8')
                    f.write(json.dumps({
                        'id': qid,
                        'text': query_data['text'],
                        'image': image_b64  # base64 编码的 bytes
                    }, ensure_ascii=False) + '\n')
            logger.info(f"Queries saved to {queries_path}")
        
        return datasets.DatasetDict(queries_dict)
    
    def _load_local_corpus(self, save_dir: str, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load corpus from local files."""
        corpus_path = os.path.join(save_dir, 'corpus.jsonl')
        if self.force_redownload or not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found in {corpus_path}. Loading from parquet data.")
            return self._load_remote_corpus(dataset_name=dataset_name, save_dir=save_dir)
        
        logger.info(f"Loading corpus from {corpus_path}")
        corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']
        
        corpus = {}
        for e in corpus_data:
            # 检查是否是 base64 编码的 bytes
            image_data = e.get('image', '')
            if isinstance(image_data, str) and len(image_data) > 100:  # 可能是 base64
                try:
                    import base64
                    image_data = base64.b64decode(image_data)
                except Exception:
                    pass  # 如果不是 base64，保持原样
            
            corpus[e['id']] = {
                'image': image_data,
                'text': e.get('text', '')
            }
        
        return datasets.DatasetDict(corpus)
    
    def _load_local_qrels(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load qrels from local files."""
        checked_split = self.check_splits(split, dataset_name=dataset_name)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]
        
        qrels_path = os.path.join(save_dir, f'{split}_qrels.jsonl')
        if self.force_redownload or not os.path.exists(qrels_path):
            logger.warning(f"Qrels not found in {qrels_path}. Downloading from remote.")
            return self._load_remote_qrels(dataset_name=dataset_name, split=split, save_dir=save_dir)
        
        logger.info(f"Loading qrels from {qrels_path}")
        qrels_data = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']
        
        qrels = {}
        for data in qrels_data:
            qid = data['qid']
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][data['docid']] = data['relevance']
        
        return datasets.DatasetDict(qrels)
    
    def _load_local_queries(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load queries from local files."""
        checked_split = self.check_splits(split, dataset_name=dataset_name)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]
        
        queries_path = os.path.join(save_dir, f'{split}_queries.jsonl')
        if self.force_redownload or not os.path.exists(queries_path):
            logger.warning(f"Queries not found in {queries_path}. Loading from parquet data.")
            return self._load_remote_queries(dataset_name=dataset_name, split=split, save_dir=save_dir)
        
        logger.info(f"Loading queries from {queries_path}")
        queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']
        
        queries = {}
        for e in queries_data:
            # 检查是否是 base64 编码的 bytes
            image_data = e.get('image', '')
            if isinstance(image_data, str) and len(image_data) > 100:  # 可能是 base64
                try:
                    import base64
                    image_data = base64.b64decode(image_data)
                except Exception:
                    pass  # 如果不是 base64，保持原样
            
            queries[e['id']] = {
                'text': e.get('text', ''),
                'image': image_data
            }
        
        return datasets.DatasetDict(queries)

