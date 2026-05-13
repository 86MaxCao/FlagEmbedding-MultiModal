import os
import json
import math
import random
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from FlagEmbedding.abc.finetune.reranker import AbsRerankerDataArguments

logger = logging.getLogger(__name__)


class Qwen3VLRerankerTrainDataset(Dataset):
    """Training dataset for Qwen3-VL-Reranker using plain JSON loading.

    Avoids the `datasets` library (pyarrow) which causes C++ heap corruption
    with the tokenizers library when both are loaded in the same process.
    """
    def __init__(
        self,
        args: AbsRerankerDataArguments,
        tokenizer=None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio

        self.data = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')):
                    continue
                self._load_json(data_dir)
            else:
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')):
                        continue
                    self._load_json(os.path.join(data_dir, file))

        logger.info(f'Loaded {len(self.data)} training samples')

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

        if self.args.query_instruction_for_rerank is not None and query_text:
            query_text = self.args.query_instruction_format.format(
                data['prompt'] if 'prompt' in data else self.args.query_instruction_for_rerank,
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
class Qwen3VLRerankerCollator:
    """Collator for Qwen3-VL-Reranker using chat-template style formatting.

    Uses the Qwen3VL instruction format:
    - System: "Judge whether the Document meets the requirements based on the
      Query and the Instruct provided. Note that the answer can only be \"yes\"
      or \"no\"."
    - User: "<Instruct>: ...\\n<Query>: ...\\n<Document>: ..."

    For text-only data, uses the tokenizer directly to avoid C++ heap corruption
    between pyarrow and tokenizers libraries. Falls back to processor for images.
    """
    tokenizer: any = None
    processor: any = None
    query_max_len: int = 512
    passage_max_len: int = 512
    max_len: int = 8192
    instruction: str = "Given a search query, retrieve relevant candidates that answer the query."

    def _load_image(self, image_path):
        if image_path is None:
            return None
        if isinstance(image_path, str) and os.path.exists(image_path):
            from PIL import Image
            return Image.open(image_path).convert('RGB')
        return None

    def _build_messages(self, query_text, query_image, doc_text, doc_image, instruction=None):
        """Build chat messages in Qwen3-VL-Reranker format."""
        instruct = instruction or self.instruction

        # System message
        system_msg = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        }

        # User message with Instruct + Query + Document
        content = []
        content.append({"type": "text", "text": f"<Instruct>: {instruct}"})

        # Query part
        content.append({"type": "text", "text": "<Query>:"})
        if query_image is not None:
            content.append({"type": "image"})
        if query_text is not None:
            content.append({"type": "text", "text": query_text})

        # Document part
        content.append({"type": "text", "text": "\n<Document>:"})
        if doc_image is not None:
            content.append({"type": "image"})
        if doc_text is not None:
            content.append({"type": "text", "text": doc_text})

        user_msg = {
            "role": "user",
            "content": content
        }

        return [system_msg, user_msg]

    def __call__(self, features):
        """Process a batch of features."""
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

        # Check if any images present
        has_any_image = any(
            img is not None
            for img in query_images + passages_images
        )

        # Build chat messages for each (query, doc) pair
        all_messages = []
        loaded_images = []
        for i, (q_text, q_image) in enumerate(zip(query_texts, query_images)):
            group_size = len(features[0]['passages_text'])
            start_idx = i * group_size
            for j in range(start_idx, start_idx + group_size):
                p_text = passages_texts[j]
                p_image = passages_images[j]

                q_img = self._load_image(q_image)
                p_img = self._load_image(p_image)

                messages = self._build_messages(q_text, q_img, p_text, p_img)
                all_messages.append(messages)

                if q_img is not None or p_img is not None:
                    imgs = []
                    if q_img is not None:
                        imgs.append(q_img)
                    if p_img is not None:
                        imgs.append(p_img)
                    loaded_images.append(imgs)
                else:
                    loaded_images.append(None)

        if has_any_image and self.processor is not None:
            # Use processor for multimodal data
            text_inputs = []
            for messages in all_messages:
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_inputs.append(text_input)

            # Collect flat image list
            flat_images = []
            for img_list in loaded_images:
                if img_list:
                    flat_images.extend(img_list)

            processor_kwargs = dict(
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            if flat_images:
                processor_kwargs["images"] = flat_images
            batch = self.processor(**processor_kwargs)
        else:
            # Use tokenizer directly for text-only data
            text_inputs = []
            for messages in all_messages:
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_inputs.append(text_input)

            batch = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )

        return {
            "pair": batch,
            "teacher_scores": teacher_scores
        }
