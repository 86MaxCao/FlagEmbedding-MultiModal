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
        dataset_name: str = "JUNJIE99/MegaPairs",
        image_root_dir: Optional[str] = None,
    ):
        super().__init__(
            eval_name=eval_name,
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            token=token,
            force_redownload=force_redownload
        )
        self.dataset_name = dataset_name
        self.image_root_dir = image_root_dir
    
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
        """Load corpus from HuggingFace MegaPairs dataset.
        
        Corpus包含所有候选图像（t_img + hns中的所有图像）
        """
        logger.info(f"Loading MegaPairs dataset from {self.dataset_name}")
        
        # Load dataset from HuggingFace
        mega_pairs = datasets.load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir,
            token=self.token,
            download_mode=self.hf_download_mode
        )
        
        # Collect all unique images as corpus
        corpus_dict = {}
        
        for split in mega_pairs.keys():
            for data in tqdm(mega_pairs[split], desc=f"Loading corpus from {split}"):
                # Add target image
                t_img = data['t_img']
                if self.image_root_dir:
                    t_img_path = os.path.join(self.image_root_dir, t_img)
                else:
                    t_img_path = t_img
                
                if t_img not in corpus_dict:
                    corpus_dict[t_img] = {
                        'image': t_img_path,
                        'text': ''  # 纯图像检索，文本为空
                    }
                
                # Add hard negatives
                for hn in data.get('hns', []):
                    if self.image_root_dir:
                        hn_path = os.path.join(self.image_root_dir, hn)
                    else:
                        hn_path = hn
                    
                    if hn not in corpus_dict:
                        corpus_dict[hn] = {
                            'image': hn_path,
                            'text': ''
                        }
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            corpus_path = os.path.join(save_dir, 'corpus.jsonl')
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for doc_id, doc_data in tqdm(corpus_dict.items(), desc="Saving corpus"):
                    f.write(json.dumps({
                        'id': doc_id,
                        'image': doc_data['image'],
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
        """Load relevance labels from HuggingFace MegaPairs dataset."""
        logger.info(f"Loading qrels for split: {split}")
        
        # Load dataset
        mega_pairs = datasets.load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir,
            token=self.token,
            download_mode=self.hf_download_mode
        )
        
        if split not in mega_pairs:
            raise ValueError(f"Split {split} not found in dataset. Available: {list(mega_pairs.keys())}")
        
        qrels_dict = {}
        
        # 为每个q_text创建qrel
        for idx, data in enumerate(tqdm(mega_pairs[split], desc=f"Loading qrels from {split}")):
            t_img = data['t_img']
            q_texts = data.get('q_texts', [])
            
            # 为每个query text创建一个qrel
            for text_idx, q_text in enumerate(q_texts):
                qid = f"{idx}_{text_idx}"
                qrels_dict[qid] = {t_img: 1}  # t_img是正确答案
        
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
        """Load queries from HuggingFace MegaPairs dataset.
        
        每个query包含：query image + query text instruction
        """
        logger.info(f"Loading queries for split: {split}")
        
        # Load dataset
        mega_pairs = datasets.load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir,
            token=self.token,
            download_mode=self.hf_download_mode
        )
        
        if split not in mega_pairs:
            raise ValueError(f"Split {split} not found in dataset. Available: {list(mega_pairs.keys())}")
        
        queries_dict = {}
        
        # 为每个q_text创建一个query
        for idx, data in enumerate(tqdm(mega_pairs[split], desc=f"Loading queries from {split}")):
            q_img = data['q_img']
            q_texts = data.get('q_texts', [])
            
            if self.image_root_dir:
                q_img_path = os.path.join(self.image_root_dir, q_img)
            else:
                q_img_path = q_img
            
            # 为每个query text创建一个query
            for text_idx, q_text in enumerate(q_texts):
                qid = f"{idx}_{text_idx}"
                queries_dict[qid] = {
                    'text': q_text,
                    'image': q_img_path
                }
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            queries_path = os.path.join(save_dir, f'{split}_queries.jsonl')
            with open(queries_path, 'w', encoding='utf-8') as f:
                for qid, query_data in tqdm(queries_dict.items(), desc="Saving queries"):
                    f.write(json.dumps({
                        'id': qid,
                        'text': query_data['text'],
                        'image': query_data['image']
                    }, ensure_ascii=False) + '\n')
            logger.info(f"Queries saved to {queries_path}")
        
        return datasets.DatasetDict(queries_dict)
    
    def _load_local_corpus(self, save_dir: str, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load corpus from local files."""
        corpus_path = os.path.join(save_dir, 'corpus.jsonl')
        if self.force_redownload or not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found in {corpus_path}. Downloading from remote.")
            return self._load_remote_corpus(dataset_name=dataset_name, save_dir=save_dir)
        
        logger.info(f"Loading corpus from {corpus_path}")
        corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']
        
        corpus = {}
        for e in corpus_data:
            corpus[e['id']] = {
                'image': e.get('image', ''),
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
            logger.warning(f"Queries not found in {queries_path}. Downloading from remote.")
            return self._load_remote_queries(dataset_name=dataset_name, split=split, save_dir=save_dir)
        
        logger.info(f"Loading queries from {queries_path}")
        queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']
        
        queries = {}
        for e in queries_data:
            queries[e['id']] = {
                'text': e.get('text', ''),
                'image': e.get('image', '')
            }
        
        return datasets.DatasetDict(queries)

