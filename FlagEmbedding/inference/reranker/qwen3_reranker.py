"""Qwen3 text reranker inference.

Based on Qwen3ForCausalLM with a cross-encoder design.
The model is prompted with a system message and computes P("yes")
from the logits at the last token position.
"""

import torch
import numpy as np
from typing import Any, List, Union, Tuple, Optional
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from FlagEmbedding.abc.inference import AbsReranker


class Qwen3Reranker(AbsReranker):
    """Reranker for Qwen3 text reranker models (e.g. Qwen3-Reranker-0.6B/4B/8B).

    Uses a cross-encoder architecture where query and document are concatenated
    into a single prompt. The model outputs the probability of the "yes" token
    at the last position, indicating relevance.

    Args:
        model_name_or_path (str): Model name or path.
        use_fp16 (bool, optional): Use half precision. Defaults to False.
        trust_remote_code (bool, optional): Trust remote code. Defaults to True.
        cache_dir (Optional[str], optional): Cache directory. Defaults to None.
        devices (Optional[Union[str, List[str], List[int]]], optional): Devices. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 128.
        max_length (int, optional): Max total length. Defaults to 2048.
        normalize (bool, optional): Normalize scores to [0,1]. Defaults to True.
        instruction (Optional[str], optional): Task instruction. Defaults to None.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        devices: Optional[Union[str, List[str], List[int]]] = None,
        batch_size: int = 128,
        max_length: int = 2048,
        normalize: bool = True,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            devices=devices,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )

        # Get yes/no token IDs (lowercase, as used by Qwen3 tokenizer)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # System prompt + user prefix
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        # Assistant suffix that forces the model to output yes/no
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        self.instruction = instruction or "Given the user query, retrieval the relevant passages"
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

    def _format_pair(self, query: str, doc: str, instruction: Optional[str] = None) -> str:
        """Format a query-doc pair into the model input string."""
        instr = instruction if instruction is not None else self.instruction
        return f"<Instruct>: {instr}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(
        self,
        pairs: List[Tuple[str, str]],
        max_length: int,
        instruction: Optional[str] = None,
    ) -> dict:
        """Tokenize and prepare inputs with prefix/suffix tokens."""
        texts = [self._format_pair(q, d, instruction) for q, d in pairs]

        # Tokenize without padding first; reserve space for prefix + suffix
        reserve_len = len(self.prefix_tokens) + len(self.suffix_tokens)
        out = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - reserve_len,
        )

        # Prepend prefix and append suffix tokens
        for i in range(len(out["input_ids"])):
            out["input_ids"][i] = self.prefix_tokens + out["input_ids"][i] + self.suffix_tokens

        # Pad to tensor
        out = self.tokenizer.pad(out, padding=True, return_tensors="pt", max_length=max_length)
        return out

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
    ) -> Union[float, List[float]]:
        """Compute reranking scores for sentence pairs."""
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length
        if normalize is None:
            normalize = self.normalize
        if device is None:
            device = self.target_devices[0]

        if device == "cpu":
            self.use_fp16 = False
        if self.use_fp16:
            self.model.half()

        self.model.to(device)
        self.model.eval()

        if isinstance(sentence_pairs, tuple):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start in trange(
            0, len(sentence_pairs), batch_size,
            desc="Computing scores",
            disable=len(sentence_pairs) < batch_size,
        ):
            batch_pairs = sentence_pairs[start:start + batch_size]
            inputs = self._process_inputs(batch_pairs, max_length, instruction)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            logits = self.model(**inputs).logits[:, -1, :]
            true_scores = logits[:, self.token_true_id]
            false_scores = logits[:, self.token_false_id]
            stacked = torch.stack([false_scores, true_scores], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores = log_probs[:, 1].exp().cpu().float().numpy()
            all_scores.extend(scores.tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores
