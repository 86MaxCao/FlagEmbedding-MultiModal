import os
import json
import math
import random
import logging

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderDataArguments

logger = logging.getLogger(__name__)


class Qwen3VLEmbedderTrainDataset(Dataset):
    """Training dataset for Qwen3-VL-Embedding using plain JSON/Parquet loading.

    Avoids the `datasets` library (pyarrow) which causes C++ heap corruption
    with the tokenizers library when both are loaded in the same process.
    """
    SUPPORTED_EXTENSIONS = ('.json', '.jsonl', '.parquet')

    def __init__(
        self,
        args: AbsEmbedderDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio
        self.image_root_dir = getattr(args, 'image_root_dir', None)

        self.data = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not data_dir.endswith(self.SUPPORTED_EXTENSIONS):
                    continue
                self._load_file(data_dir)
            else:
                for file in sorted(os.listdir(data_dir)):
                    if not file.endswith(self.SUPPORTED_EXTENSIONS):
                        continue
                    self._load_file(os.path.join(data_dir, file))

        logger.info(f'Loaded {len(self.data)} training samples')

    def _load_file(self, file_path):
        """Route to the appropriate loader based on file extension."""
        if file_path.endswith('.parquet'):
            self._load_parquet(file_path)
        else:
            self._load_json(file_path)

    def _load_json(self, file_path):
        safe_rank = 0
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                safe_rank = dist.get_rank()
        except Exception:
            pass

        if safe_rank == 0:
            logger.info(f'Loading data from {file_path} ...')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(item)

    def _load_parquet(self, file_path):
        """Load data from a Parquet file using pandas (avoids pyarrow/tokenizers conflict)."""
        safe_rank = 0
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                safe_rank = dist.get_rank()
        except Exception:
            pass

        if safe_rank == 0:
            logger.info(f'Loading data from {file_path} ...')

        import pandas as pd
        df = pd.read_parquet(file_path)
        records = df.to_dict('records')
        self.data.extend(records)

    def _shuffle_text(self, text):
        if text is None:
            return None
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text) // 3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i + chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        train_group_size = self.args.train_group_size

        # MegaPairs format
        if 'q_texts' in data and 'q_img' in data:
            query_text = random.choice(data['q_texts'])
            query_image = data['q_img']
            pos_images = [data['t_img']] if 't_img' in data else []
            pos_texts = [None] * len(pos_images)
            neg_images = data.get('hns', [])
            neg_texts = [None] * len(neg_images)
        else:
            # Standard format
            query_text = data.get('query', data.get('qry', None))
            query_image = data.get('query_image', data.get('qry_image_path', None))
            pos_texts = data.get('pos', data.get('pos_text', []))
            pos_images = data.get('pos_images', data.get('pos_image_path', []))
            neg_texts = data.get('neg', data.get('neg_text', []))
            neg_images = data.get('neg_images', data.get('neg_image_path', []))

        # Resolve relative image paths using image_root_dir
        if self.image_root_dir:
            if query_image and isinstance(query_image, str) and not os.path.isabs(query_image):
                query_image = os.path.join(self.image_root_dir, query_image)
            if isinstance(pos_images, list):
                pos_images = [
                    os.path.join(self.image_root_dir, p) if p and isinstance(p, str) and not os.path.isabs(p) else p
                    for p in pos_images
                ]
            elif pos_images and isinstance(pos_images, str) and not os.path.isabs(pos_images):
                pos_images = os.path.join(self.image_root_dir, pos_images)
            if isinstance(neg_images, list):
                neg_images = [
                    os.path.join(self.image_root_dir, n) if n and isinstance(n, str) and not os.path.isabs(n) else n
                    for n in neg_images
                ]
            elif neg_images and isinstance(neg_images, str) and not os.path.isabs(neg_images):
                neg_images = os.path.join(self.image_root_dir, neg_images)

        if self.args.query_instruction_for_retrieval is not None and query_text:
            query_text = self.args.query_instruction_format.format(
                data['prompt'] if 'prompt' in data else self.args.query_instruction_for_retrieval,
                query_text
            )

        passages_text = []
        passages_image = []
        teacher_scores = []

        # Ensure lists
        if not isinstance(pos_texts, list):
            pos_texts = [pos_texts] if pos_texts else []
        if not isinstance(pos_images, list):
            pos_images = [pos_images] if pos_images else []
        max_pos = max(len(pos_texts), len(pos_images))
        pos_texts = pos_texts + [None] * (max_pos - len(pos_texts))
        pos_images = pos_images + [None] * (max_pos - len(pos_images))

        # Sample one positive
        if max_pos > 0:
            pos_idx = random.choice(list(range(max_pos)))
            passages_text.append(self._shuffle_text(pos_texts[pos_idx]))
            passages_image.append(pos_images[pos_idx])

        # Negatives
        if not isinstance(neg_texts, list):
            neg_texts = [neg_texts] if neg_texts else []
        if not isinstance(neg_images, list):
            neg_images = [neg_images] if neg_images else []
        max_neg = max(len(neg_texts), len(neg_images))
        neg_texts = neg_texts + [None] * (max_neg - len(neg_texts))
        neg_images = neg_images + [None] * (max_neg - len(neg_images))

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
        else:
            neg_idxs = []

        # Knowledge distillation scores
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
class Qwen3VLEmbedderCollator:
    """Collator for Qwen3-VL-Embedding that uses Qwen3VL processor.

    For text-only data, uses the tokenizer directly to avoid C++ heap corruption
    between pyarrow and tokenizers libraries. Falls back to processor for images.
    """
    processor: any = None
    tokenizer: any = None
    query_max_len: int = 512
    passage_max_len: int = 512
    query_instruction: str = "Represent the user's input."

    def _load_image(self, image_path):
        if image_path is None:
            return None
        if isinstance(image_path, str) and os.path.exists(image_path):
            from PIL import Image
            return Image.open(image_path).convert('RGB')
        return None

    def _build_messages(self, text, image, is_query=False):
        content = []
        if image is not None:
            content.append({"type": "image"})
        if text is not None:
            content.append({"type": "text", "text": text})

        if not content:
            content.append({"type": "text", "text": ""})

        messages = []
        if is_query and self.query_instruction:
            messages.append({
                "role": "system",
                "content": self.query_instruction
            })
        messages.append({
            "role": "user",
            "content": content
        })
        return messages

    def _apply_template(self, messages):
        """Apply chat template and append <|endoftext|> as the pooling target.

        Qwen3-VL-Embedding uses last-token pooling. The official approach
        (ms-swift Qwen3VLEmbTemplate) appends <|endoftext|> so the final
        token's hidden state serves as the sentence embedding.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text += "<|endoftext|>"
        return text

    def _process_batch(self, texts, images, max_len, is_query=False):
        """Process a batch of text+image pairs."""
        loaded_images = []
        all_messages = []
        has_any_image = False

        for text, img_path in zip(texts, images):
            img = self._load_image(img_path)
            if img is not None:
                has_any_image = True
            loaded_images.append(img)
            messages = self._build_messages(text, img, is_query=is_query)
            all_messages.append(messages)

        if has_any_image and self.processor is not None:
            # Use processor for multimodal data
            processed = []
            for i, messages in enumerate(all_messages):
                img = loaded_images[i]
                text_input = self._apply_template(messages)
                if img is not None:
                    inputs = self.processor(
                        text=[text_input], images=[img],
                        return_tensors="pt", padding=True,
                    )
                else:
                    inputs = self.processor(
                        text=[text_input],
                        return_tensors="pt", padding=True,
                    )
                processed.append(inputs)

            # Merge into single batch
            batch = {}
            for key in processed[0].keys():
                if key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'):
                    tensors = [p[key] for p in processed if key in p]
                    if tensors:
                        batch[key] = torch.cat(tensors, dim=0)
                else:
                    tensors = [p[key] for p in processed if key in p]
                    if tensors:
                        batch[key] = self._pad_tensors(tensors)
            return batch
        else:
            # Use tokenizer directly for text-only data
            text_inputs = []
            for messages in all_messages:
                text_input = self._apply_template(messages)
                text_inputs.append(text_input)

            batch = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            return batch

    def _pad_tensors(self, tensors):
        max_size = max(t.size(-1) for t in tensors)
        padded = []
        for t in tensors:
            if t.size(-1) < max_size:
                pad_size = max_size - t.size(-1)
                pad = torch.zeros(t.shape[:-1] + (pad_size,), dtype=t.dtype)
                padded.append(torch.cat([t, pad], dim=-1))
            else:
                padded.append(t)
        return torch.cat(padded, dim=0)

    def __call__(self, features):
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

        queries_inputs = self._process_batch(
            query_texts, query_images,
            max_len=self.query_max_len, is_query=True
        )
        passages_inputs = self._process_batch(
            passages_texts, passages_images,
            max_len=self.passage_max_len, is_query=False
        )

        return {
            "queries": queries_inputs,
            "passages": passages_inputs,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False
        }
