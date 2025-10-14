import os
import math
import random
import logging
import datasets
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch.distributed as dist
from PIL import Image
from transformers.image_utils import load_image

from FlagEmbedding.abc.finetune.reranker import AbsRerankerDataArguments

logger = logging.getLogger(__name__)


def load_images_batch(images, lazy_load: bool = True):
    """Load a batch of images."""
    pil_max_px = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None

    images_batch = []
    for image in images:
        if image is None:
            images_batch.append(None)
        elif isinstance(image, Image.Image):
            images_batch.append(image)
        else:
            try:
                pil_image = load_image(image)
                if lazy_load:
                    images_batch.append(pil_image)
                else:
                    images_batch.append(pil_image.copy())
                    pil_image.close()
            except Exception as e:
                logger.warning(f"Failed to load image {image}: {e}")
                images_batch.append(None)
    
    Image.MAX_IMAGE_PIXELS = pil_max_px
    return images_batch


def formatting_prompts_func(
    query: str,
    doc: str,
    query_type: str = 'text',
    doc_type: str = 'text',
) -> str:
    """Format prompts for multimodal inputs."""
    if query_type == 'image':
        query_part = "**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        query_part = f"**Query**:\n{query}"

    if doc_type == 'image':
        doc_part = "**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        doc_part = f"**Document**:\n{doc}"

    return doc_part + '\n' + query_part


class MultimodalRerankerTrainDataset(Dataset):
    """Training dataset for multimodal reranker.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsRerankerDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        self.args = args
        self.tokenizer = tokenizer

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
        """Load dataset from path."""
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

        # Ensure lists
        if not isinstance(pos_texts, list):
            pos_texts = [pos_texts] if pos_texts else []
        if not isinstance(pos_images, list):
            pos_images = [pos_images] if pos_images else []
        if not isinstance(neg_texts, list):
            neg_texts = [neg_texts] if neg_texts else []
        if not isinstance(neg_images, list):
            neg_images = [neg_images] if neg_images else []

        # Make same length
        max_pos = max(len(pos_texts), len(pos_images))
        pos_texts = pos_texts + [None] * (max_pos - len(pos_texts))
        pos_images = pos_images + [None] * (max_pos - len(pos_images))

        max_neg = max(len(neg_texts), len(neg_images))
        neg_texts = neg_texts + [None] * (max_neg - len(neg_texts))
        neg_images = neg_images + [None] * (max_neg - len(neg_images))

        # Sample passages
        passages_text = []
        passages_image = []
        teacher_scores = []

        # Sample one positive
        if max_pos > 0:
            pos_idx = random.choice(list(range(max_pos)))
            passages_text.append(pos_texts[pos_idx])
            passages_image.append(pos_images[pos_idx])

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

        # Handle KD scores
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
class MultimodalRerankerCollator:
    """Collator for multimodal reranker using processor.
    
    Args:
        processor: The processor with tokenizer and image processor.
        query_max_len: Maximum query length.
        passage_max_len: Maximum passage length. 
        max_len: Maximum total length.
        score_token_id: ID of the score token to append.
    """
    processor: any = None
    query_max_len: int = 512
    passage_max_len: int = 512
    max_len: int = 10240
    score_token_id: int = 100
    query_type: str = 'text'
    doc_type: str = 'text'

    def __call__(self, features):
        """Process a batch of features.
        
        Args:
            features: List of feature dicts from dataset.
            
        Returns:
            Dict with processed inputs.
        """
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

        # Determine query and doc types
        has_query_image = any(img is not None for img in query_images)
        has_doc_image = any(img is not None for img in passages_images)
        
        query_type = 'image' if has_query_image else 'text'
        doc_type = 'image' if has_doc_image else 'text'

        # Create pairs
        pairs = []
        pair_images = []
        
        for i, (q_text, q_image) in enumerate(zip(query_texts, query_images)):
            # Get corresponding passages for this query
            group_size = len(features[0]['passages_text'])
            start_idx = i * group_size
            end_idx = start_idx + group_size
            
            for j in range(start_idx, end_idx):
                p_text = passages_texts[j]
                p_image = passages_images[j]
                
                # Determine actual query and doc content
                query_content = q_image if query_type == 'image' else (q_text or "")
                doc_content = p_image if doc_type == 'image' else (p_text or "")
                
                # Format prompt
                prompt = formatting_prompts_func(
                    query_content, 
                    doc_content,
                    query_type=query_type,
                    doc_type=doc_type
                )
                pairs.append(prompt)
                
                # Collect images
                images_for_pair = []
                if doc_type == 'image' and p_image:
                    images_for_pair.append(p_image)
                if query_type == 'image' and q_image:
                    images_for_pair.append(q_image)
                
                pair_images.append(images_for_pair if images_for_pair else None)

        # Load images
        loaded_images = []
        for img_list in pair_images:
            if img_list:
                loaded_images.append(load_images_batch(img_list))
            else:
                loaded_images.append(None)

        # Process with processor
        batch = self.processor(
            text=pairs,
            images=[img if img else None for img in loaded_images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len - 1,  # Reserve space for score token
        )

        # Append score token
        batch_size = batch["input_ids"].size(0)
        batch["input_ids"] = torch.cat(
            [
                batch["input_ids"],
                torch.full((batch_size, 1), self.score_token_id, device=batch["input_ids"].device),
            ],
            dim=1,
        )
        batch["attention_mask"] = torch.cat(
            [
                batch["attention_mask"],
                torch.ones((batch_size, 1), device=batch["attention_mask"].device),
            ],
            dim=1,
        )

        return {
            "pair": batch,
            "teacher_scores": teacher_scores
        }

