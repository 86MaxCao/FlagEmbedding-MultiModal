"""Qwen3-VL-Reranker via native Qwen3VLForConditionalGeneration.

Uses the binary linear scoring approach from the HF model card:
score = sigmoid(linear(last_hidden_state[:, -1]))
where linear.weight = lm_head[yes] - lm_head[no].
"""

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from FlagEmbedding.abc.inference import AbsReranker


class Qwen3VLReranker(AbsReranker):
    """Reranker for Qwen3-VL-Reranker models (2B/8B).

    Supports text-only and multimodal (text+image) reranking.
    Uses the native model with binary linear scoring instead of CrossEncoder.
    """

    DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
    SYSTEM_PROMPT = (
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        query_instruction_for_rerank: Optional[str] = None,
        devices: Optional[Union[str, List[str], List[int]]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 8192,
        normalize: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            query_instruction_for_rerank=query_instruction_for_rerank,
            devices=devices,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            **kwargs,
        )

        self.instruction = query_instruction_for_rerank or self.DEFAULT_INSTRUCTION

        dtype = torch.float16 if use_fp16 else torch.bfloat16
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            padding_side="left",
        )

        token_true_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        token_false_id = self.processor.tokenizer.convert_tokens_to_ids("no")
        self.score_linear = self._build_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()

        del lm

    @staticmethod
    def _build_binary_linear(lm, token_yes: int, token_no: int) -> torch.nn.Linear:
        """Extract yes/no weights from lm_head to build a binary scoring layer."""
        lm_head_weights = lm.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]
        D = weight_yes.size(0)
        linear = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear.weight[0] = weight_yes - weight_no
        return linear

    def _format_pair(self, query: str, doc: str, instruction: Optional[str] = None) -> list:
        """Format a (query, doc) pair into chat messages for the model."""
        instr = instruction or self.instruction
        return [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"<Instruct>: {instr}\n<Query>:\n{query}\n<Document>:\n{doc}"}
            ]},
        ]

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[float, List[float]]:
        if batch_size is None:
            batch_size = self.batch_size
        if normalize is None:
            normalize = self.normalize

        single_input = isinstance(sentence_pairs, tuple) and len(sentence_pairs) == 2 and isinstance(sentence_pairs[0], str)
        if single_input:
            sentence_pairs = [sentence_pairs]

        if len(self.target_devices) <= 1:
            scores = self.compute_score_single_gpu(
                sentence_pairs, batch_size=batch_size, max_length=max_length,
                normalize=normalize, device=self.target_devices[0], **kwargs
            )
        else:
            n = len(sentence_pairs)
            chunk_size = math.ceil(n / len(self.target_devices))
            scores = []
            for i, device in enumerate(self.target_devices):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                if start >= n:
                    break
                chunk_scores = self.compute_score_single_gpu(
                    sentence_pairs[start:end], batch_size=batch_size,
                    max_length=max_length, normalize=normalize, device=device, **kwargs
                )
                scores.extend(chunk_scores)

        if single_input:
            return scores[0] if isinstance(scores, list) else scores
        return scores

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length
        if normalize is None:
            normalize = self.normalize
        if device is None:
            device = self.target_devices[0]

        if isinstance(sentence_pairs, tuple) and len(sentence_pairs) == 2:
            sentence_pairs = [sentence_pairs]

        self.model.to(device)
        self.model.eval()
        self.score_linear.to(device=device, dtype=self.model.dtype)

        all_scores: List[float] = []
        for start in tqdm(
            range(0, len(sentence_pairs), batch_size),
            desc="Computing scores",
            disable=len(sentence_pairs) <= batch_size,
        ):
            batch_pairs = sentence_pairs[start:start + batch_size]

            # Format as chat messages
            messages_batch = [self._format_pair(q, d, instruction) for q, d in batch_pairs]

            # Tokenize via apply_chat_template
            texts = self.processor.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Forward pass through base model
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[:, -1, :]

            # Score via binary linear + sigmoid
            logits = self.score_linear(hidden)
            scores = torch.sigmoid(logits).squeeze(-1).cpu().float().numpy()
            all_scores.extend(scores.tolist())

        return all_scores
