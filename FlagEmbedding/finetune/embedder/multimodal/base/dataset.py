import os
import math
import random
import logging
import datasets
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import torch.distributed as dist

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderDataArguments

logger = logging.getLogger(__name__)


class MultimodalEmbedderTrainDataset(Dataset):
    """Training dataset for multimodal embedder.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsEmbedderDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio

        train_datasets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')):
                    continue
                temp_dataset = self._load_dataset(data_dir)
                if len(temp_dataset) == 0:
                    continue
                train_datasets.append(temp_dataset)
            else:
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')):
                        continue
                    temp_dataset = self._load_dataset(os.path.join(data_dir, file))
                    if len(temp_dataset) == 0:
                        continue
                    train_datasets.append(temp_dataset)
        self.dataset = datasets.concatenate_datasets(train_datasets)

    def _load_dataset(self, file_path: str):
        """Load dataset from path.

        Args:
            file_path (str): Path to load the datasets from.

        Returns:
            datasets.Dataset: Loaded HF dataset.
        """
        safe_rank = dist.get_rank() if dist.is_initialized() else 0
        if safe_rank == 0:
            logger.info(f'loading data from {file_path} ...')

        temp_dataset = datasets.load_dataset(
            'json', data_files=file_path, split='train', cache_dir=self.args.cache_path
        )
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            temp_dataset = temp_dataset.select(
                random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset)
            )
        
        # Handle knowledge distillation
        if not self.args.knowledge_distillation:
            if 'pos_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['pos_scores'])
            if 'neg_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['neg_scores'])
        else:
            if 'pos_scores' not in temp_dataset.column_names or 'neg_scores' not in temp_dataset.column_names:
                raise ValueError(
                    f"`pos_scores` and `neg_scores` not found in {file_path}, "
                    "which is necessary when using knowledge distillation."
                )
        return temp_dataset

    def _shuffle_text(self, text):
        """Shuffle the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Shuffled text.
        """
        if text is None:
            return None
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        # 检测并转换MegaPairs格式
        if 'q_texts' in data and 'q_img' in data:
            # MegaPairs格式: {"q_img": "...", "q_texts": [...], "t_img": "...", "hns": [...]}
            query_text = random.choice(data['q_texts'])  # 随机选择一个query text
            query_image = data['q_img']
            pos_images = [data['t_img']] if 't_img' in data else []
            pos_texts = [None] * len(pos_images)  # MegaPairs是纯图像检索
            neg_images = data.get('hns', [])
            neg_texts = [None] * len(neg_images)
        else:
            # 原有格式
            query_text = data.get('query', data.get('qry', None))
            query_image = data.get('query_image', data.get('qry_image_path', None))
            pos_texts = data.get('pos', data.get('pos_text', []))
            pos_images = data.get('pos_images', data.get('pos_image_path', []))
            neg_texts = data.get('neg', data.get('neg_text', []))
            neg_images = data.get('neg_images', data.get('neg_image_path', []))
        
        if self.args.query_instruction_for_retrieval is not None and query_text:
            query_text = self.args.query_instruction_format.format(
                data['prompt'] if 'prompt' in data else self.args.query_instruction_for_retrieval,
                query_text
            )

        # Passages: positive and negative
        passages_text = []
        passages_image = []
        teacher_scores = []
        
        # Ensure lists
        if not isinstance(pos_texts, list):
            pos_texts = [pos_texts] if pos_texts else []
        if not isinstance(pos_images, list):
            pos_images = [pos_images] if pos_images else []
        
        # Make them same length (pad with None if needed)
        max_pos = max(len(pos_texts), len(pos_images))
        pos_texts = pos_texts + [None] * (max_pos - len(pos_texts))
        pos_images = pos_images + [None] * (max_pos - len(pos_images))

        # Sample one positive
        if max_pos > 0:
            pos_idx = random.choice(list(range(max_pos)))
            passages_text.append(self._shuffle_text(pos_texts[pos_idx]))
            passages_image.append(pos_images[pos_idx])

        # Ensure lists for negatives
        if not isinstance(neg_texts, list):
            neg_texts = [neg_texts] if neg_texts else []
        if not isinstance(neg_images, list):
            neg_images = [neg_images] if neg_images else []
        
        # Make them same length
        max_neg = max(len(neg_texts), len(neg_images))
        neg_texts = neg_texts + [None] * (max_neg - len(neg_texts))
        neg_images = neg_images + [None] * (max_neg - len(neg_images))

        # Sample negatives
        if max_neg > 0:
            neg_all_idx = list(range(max_neg))
            if max_neg < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / max_neg)
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
            
            for neg_idx in neg_idxs:
                passages_text.append(neg_texts[neg_idx])
                passages_image.append(neg_images[neg_idx])

        # Handle knowledge distillation scores
        if self.args.knowledge_distillation:
            pos_scores = data.get('pos_scores', [])
            neg_scores = data.get('neg_scores', [])
            if isinstance(pos_scores, list) and isinstance(neg_scores, list):
                teacher_scores.append(pos_scores[pos_idx] if pos_idx < len(pos_scores) else 1.0)
                for neg_idx in neg_idxs:
                    teacher_scores.append(neg_scores[neg_idx] if neg_idx < len(neg_scores) else 0.0)
            else:
                teacher_scores = None
        else:
            teacher_scores = None

        return {
            'query_text': query_text,
            'query_image': query_image,
            'passages_text': passages_text,
            'passages_image': passages_image,
            'teacher_scores': teacher_scores
        }


@dataclass
class MultimodalEmbedderCollator(DataCollatorWithPadding):
    """
    Collator for multimodal embedder that uses model's data_process method.
    """
    query_max_len: int = 512
    passage_max_len: int = 512
    model: any = None  # The model with data_process method
    
    def __call__(self, features):
        # Extract data
        query_texts = [f['query_text'] for f in features]
        query_images = [f['query_image'] for f in features]
        passages_texts = [f['passages_text'] for f in features]
        passages_images = [f['passages_image'] for f in features]
        teacher_scores = [f['teacher_scores'] for f in features]
        
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])
        
        # Flatten passages
        if isinstance(passages_texts[0], list):
            passages_texts = sum(passages_texts, [])
        if isinstance(passages_images[0], list):
            passages_images = sum(passages_images, [])
        
        # Process queries using model's data_process
        queries_inputs = self.model.data_process(
            text=query_texts,
            images=query_images,
            q_or_c='q',
            task_instruction=None  # Will be set in training args
        )
        
        # Process passages using model's data_process
        passages_inputs = self.model.data_process(
            text=passages_texts,
            images=passages_images,
            q_or_c='c',
            task_instruction=None
        )
        
        return {
            "queries": queries_inputs,
            "passages": passages_inputs,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False
        }

