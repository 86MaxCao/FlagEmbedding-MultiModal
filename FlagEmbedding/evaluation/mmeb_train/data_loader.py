import os
import json
import logging
import glob
from typing import List, Optional
import datasets
from tqdm import tqdm

from FlagEmbedding.abc.evaluation.data_loader import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MMEBTrainEvalDataLoader(AbsEvalDataLoader):
    """
    MMEB-train数据加载器。
    
    支持从本地parquet文件加载MMEB-train数据集。
    数据格式：
    {
        "qry": "question text",
        "qry_image_path": "relative/path/to/image.jpg",
        "pos_text": "positive answer text",
        "pos_image_path": "relative/path/to/positive_image.jpg",
        "neg_text": "negative answer text", 
        "neg_image_path": "relative/path/to/negative_image.jpg"
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
        sub_datasets: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
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
        self.sub_datasets = sub_datasets or [
            "DocVQA", "CIRR", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i", 
            "VisualNews_i2t", "VisualNews_t2i", "Visual7W", "WebQA"
        ]
        self.task_types = task_types or ["text", "image", "mixed"]
    
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
        """Load corpus from local parquet MMEB-train dataset.
        
        Corpus包含所有候选答案（pos_text + neg_text + pos_image_path + neg_image_path）
        """
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading MMEB-train dataset from local parquet: {self.dataset_dir}")
        
        corpus_dict = {}
        sample_count = 0
        
        # 遍历所有子数据集
        for sub_dataset in self.sub_datasets:
            sub_dir = os.path.join(self.dataset_dir, sub_dataset)
            if not os.path.exists(sub_dir):
                logger.warning(f"Sub-dataset directory not found: {sub_dir}")
                continue
                
            # 查找所有parquet文件
            parquet_files = glob.glob(os.path.join(sub_dir, "*.parquet"))
            if not parquet_files:
                logger.warning(f"No parquet files found in: {sub_dir}")
                continue
                
            logger.info(f"Loading {sub_dataset} with {len(parquet_files)} parquet files")
            
            for parquet_file in parquet_files:
                try:
                    dataset = datasets.load_dataset(
                        'parquet',
                        data_files=parquet_file,
                        streaming=True
                    )
                    
                    for data in tqdm(dataset['train'], desc=f"Loading {sub_dataset}"):
                        # 限制样本数量
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                        # 添加文本答案
                        if data.get('pos_text') and 'text' in self.task_types:
                            doc_id = f"{sub_dataset}_{sample_count}_pos_text"
                            corpus_dict[doc_id] = {
                                'text': data['pos_text'],
                                'image': ''  # 文本答案没有图像
                            }
                        
                        if data.get('neg_text') and 'text' in self.task_types:
                            doc_id = f"{sub_dataset}_{sample_count}_neg_text"
                            corpus_dict[doc_id] = {
                                'text': data['neg_text'],
                                'image': ''
                            }
                        
                        # 添加图像答案
                        if data.get('pos_image_path') and 'image' in self.task_types:
                            # 处理相对路径
                            full_path = os.path.join(self.dataset_dir, data['pos_image_path'])
                            doc_id = f"{sub_dataset}_{sample_count}_pos_image"
                            corpus_dict[doc_id] = {
                                'text': '',
                                'image': full_path
                            }
                        
                        if data.get('neg_image_path') and 'image' in self.task_types:
                            full_path = os.path.join(self.dataset_dir, data['neg_image_path'])
                            doc_id = f"{sub_dataset}_{sample_count}_neg_image"
                            corpus_dict[doc_id] = {
                                'text': '',
                                'image': full_path
                            }
                        
                        sample_count += 1
                        
                        # 检查是否达到最大样本数
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                except Exception as e:
                    logger.error(f"Error loading {parquet_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(corpus_dict)} corpus items from {sample_count} samples")
        logger.info(f"Corpus breakdown by sub_dataset:")
        for sub_dataset in self.sub_datasets:
            sub_count = sum(1 for doc_id in corpus_dict.keys() if doc_id.startswith(sub_dataset))
            logger.info(f"  {sub_dataset}: {sub_count} documents")
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            corpus_path = os.path.join(save_dir, 'corpus.jsonl')
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for doc_id, doc_data in tqdm(corpus_dict.items(), desc="Saving corpus"):
                    f.write(json.dumps({
                        'id': doc_id,
                        'text': doc_data.get('text', ''),
                        'image': doc_data.get('image', '')
                    }, ensure_ascii=False) + '\n')
            logger.info(f"Corpus saved to {corpus_path}")
        
        return datasets.DatasetDict(corpus_dict)
    
    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load relevance labels from local parquet MMEB-train dataset."""
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading qrels for split: {split}")
        
        qrels_dict = {}
        sample_count = 0
        
        # 遍历所有子数据集
        for sub_dataset in self.sub_datasets:
            sub_dir = os.path.join(self.dataset_dir, sub_dataset)
            if not os.path.exists(sub_dir):
                continue
                
            parquet_files = glob.glob(os.path.join(sub_dir, "*.parquet"))
            
            for parquet_file in parquet_files:
                try:
                    dataset = datasets.load_dataset(
                        'parquet',
                        data_files=parquet_file,
                        streaming=True
                    )
                    
                    for data in tqdm(dataset['train'], desc=f"Loading qrels from {sub_dataset}"):
                        # 限制样本数量
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                        qid = f"{sub_dataset}_{sample_count}"
                        qrels_dict[qid] = {}
                        
                        # 添加正样本相关性
                        if data.get('pos_text') and 'text' in self.task_types:
                            doc_id = f"{sub_dataset}_{sample_count}_pos_text"
                            qrels_dict[qid][doc_id] = 1
                        
                        if data.get('pos_image_path') and 'image' in self.task_types:
                            doc_id = f"{sub_dataset}_{sample_count}_pos_image"
                            qrels_dict[qid][doc_id] = 1
                        
                        sample_count += 1
                        
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                except Exception as e:
                    logger.error(f"Error loading qrels from {parquet_file}: {e}")
                    continue
        
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
        """Load queries from local parquet MMEB-train dataset.
        
        每个query包含：query image + query text
        """
        if not self.use_local_data:
            raise NotImplementedError("Remote loading not supported, use local parquet data")
        
        logger.info(f"Loading queries for split: {split}")
        
        queries_dict = {}
        sample_count = 0
        
        # 遍历所有子数据集
        for sub_dataset in self.sub_datasets:
            sub_dir = os.path.join(self.dataset_dir, sub_dataset)
            if not os.path.exists(sub_dir):
                continue
                
            parquet_files = glob.glob(os.path.join(sub_dir, "*.parquet"))
            
            for parquet_file in parquet_files:
                try:
                    dataset = datasets.load_dataset(
                        'parquet',
                        data_files=parquet_file,
                        streaming=True
                    )
                    
                    for data in tqdm(dataset['train'], desc=f"Loading queries from {sub_dataset}"):
                        # 限制样本数量
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                        qid = f"{sub_dataset}_{sample_count}"
                        
                        # 处理查询图像路径
                        query_image = ''
                        if data.get('qry_image_path'):
                            query_image = os.path.join(self.dataset_dir, data['qry_image_path'])
                        
                        # 添加任务指令
                        original_text = data.get('qry', '')
                        task_instruction = self._get_task_instruction(sub_dataset, data)
                        enhanced_text = f"{task_instruction}{original_text}"
                        
                        queries_dict[qid] = {
                            'text': enhanced_text,
                            'image': query_image
                        }
                        
                        sample_count += 1
                        
                        if self.max_samples is not None and sample_count >= self.max_samples:
                            break
                            
                except Exception as e:
                    logger.error(f"Error loading queries from {parquet_file}: {e}")
                    continue
        
        # Save to file if save_dir provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            queries_path = os.path.join(save_dir, f'{split}_queries.jsonl')
            with open(queries_path, 'w', encoding='utf-8') as f:
                for qid, query_data in tqdm(queries_dict.items(), desc="Saving queries"):
                    f.write(json.dumps({
                        'id': qid,
                        'text': query_data.get('text', ''),
                        'image': query_data.get('image', '')
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
            corpus[e['id']] = {
                'text': e.get('text', ''),
                'image': e.get('image', '')
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
            logger.warning(f"Qrels not found in {qrels_path}. Loading from parquet data.")
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
            queries[e['id']] = {
                'text': e.get('text', ''),
                'image': e.get('image', '')
            }
        
        return datasets.DatasetDict(queries)
    
    def _get_task_instruction(self, sub_dataset: str, data: Dict[str, Any]) -> str:
        """根据子数据集和数据类型生成任务指令"""
        
        # 检查是否有图像和文本
        has_image = bool(data.get('qry_image_path'))
        has_text = bool(data.get('qry'))
        
        # 根据子数据集和数据类型生成不同的指令
        if sub_dataset in ['DocVQA', 'InfographicsVQA', 'OK-VQA', 'A-OKVQA']:
            if has_image and has_text:
                return "Answer the question based on the provided image: "
            elif has_image:
                return "Describe what you see in the image: "
            else:
                return "Answer the following question: "
        
        elif sub_dataset in ['CIRR', 'MSCOCO', 'MSCOCO_i2t', 'MSCOCO_t2i']:
            if has_image and has_text:
                return "Find the image that best matches the description: "
            elif has_image:
                return "Find similar images to: "
            else:
                return "Find images that match: "
        
        elif sub_dataset in ['Visual7W', 'VisualNews_i2t', 'VisualNews_t2i']:
            if has_image and has_text:
                return "Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
            elif has_image:
                return "Find images similar to: "
            else:
                return "Find images that match: "
        
        elif sub_dataset in ['HatefulMemes', 'N24News', 'NIGHTS']:
            if has_image and has_text:
                return "Analyze the image and text together: "
            elif has_image:
                return "Analyze the image: "
            else:
                return "Analyze the text: "
        
        else:
            # 默认指令
            if has_image and has_text:
                return "Process the image and text together: "
            elif has_image:
                return "Process the image: "
            else:
                return "Process the text: "
