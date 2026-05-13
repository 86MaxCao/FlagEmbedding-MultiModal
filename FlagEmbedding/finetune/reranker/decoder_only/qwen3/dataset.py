import os
import math
import random
import logging
import datasets
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    BatchEncoding,
)
from typing import List

from FlagEmbedding.abc.finetune.reranker.AbsDataset import AbsRerankerTrainDataset
from FlagEmbedding.abc.finetune.reranker.AbsArguments import AbsRerankerDataArguments

logger = logging.getLogger(__name__)


class Qwen3RerankerTrainDataset(AbsRerankerTrainDataset):
    """Training dataset for Qwen3 text reranker.

    Formats inputs with Qwen3 chat template:
      system + user(<Instruct>: ... <Query>: ... <Document>: ...) + assistant suffix
    """

    def __init__(self, args: AbsRerankerDataArguments, tokenizer: PreTrainedTokenizer):
        super().__init__(args, tokenizer)

        # Pre-compute prefix/suffix token IDs
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # Default instruction
        self.default_instruction = "Given the user query, retrieval the relevant passages"

    def create_one_example(self, qry_encoding: str, doc_encoding: str, instruction: str = None):
        """Create a single input example."""
        if instruction is None:
            instruction = self.default_instruction

        text = f"<Instruct>: {instruction}\n<Query>: {qry_encoding}\n<Document>: {doc_encoding}"

        # Reserve space for prefix + suffix
        reserve_len = len(self.prefix_tokens) + len(self.suffix_tokens)
        max_text_len = self.max_length - reserve_len

        text_inputs = self.tokenizer.encode(text, truncation=True, max_length=max_text_len, add_special_tokens=False)

        input_ids = self.prefix_tokens + text_inputs + self.suffix_tokens
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        if self.args.query_instruction_for_rerank is not None:
            query = self.args.query_instruction_format.format(
                data['query_prompt'] if 'query_prompt' in data else self.args.query_instruction_for_rerank,
                query
            )

        passages = []
        teacher_scores = []

        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(self._shuffle_text(data['pos'][pos_idx]))

        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
            teacher_scores.append(data['pos_scores'][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data['neg_scores'][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError("pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_rerank is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    data['passage_prompt'] if 'passage_prompt' in data else self.args.passage_instruction_for_rerank, p
                )
                for p in passages
            ]

        # Get instruction from data if available
        instruction = data.get('instruction', self.default_instruction)

        batch_data = []
        for passage in passages:
            batch_data.append(self.create_one_example(query, passage, instruction))

        return batch_data, teacher_scores


@dataclass
class Qwen3RerankerCollator(DataCollatorWithPadding):
    """Collator for Qwen3 text reranker."""
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features) -> dict:
        teacher_scores = [f[1] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        features = [f[0] for f in features]
        if isinstance(features[0], list):
            features = sum(features, [])

        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.passage_max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return {
            "pair": collated,
            "teacher_scores": teacher_scores,
        }
