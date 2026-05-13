"""Qwen3-VL-Reranker via Sentence-Transformers CrossEncoder (see HF model card)."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.compat import apply_qwen3_vl_reranker_patches

apply_qwen3_vl_reranker_patches()


class Qwen3VLReranker(AbsReranker):
    """Reranker for ``Qwen/Qwen3-VL-Reranker-*`` using ``CrossEncoder``.

    Hugging Face usage: ``CrossEncoder.predict(pairs, prompt=..., activation_fn=Sigmoid)``.
    Set ``query_instruction_for_rerank`` on the constructor (or ``FlagAutoReranker.from_finetuned``)
    to override the default instruction string for your task.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        query_instruction_for_rerank: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        passage_instruction_for_rerank: Optional[str] = None,
        passage_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str], List[int]]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
        query_max_length: Optional[int] = None,
        max_length: int = 32768,
        normalize: bool = True,
        **kwargs: Any,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL-Reranker requires `sentence-transformers`. Install with: pip install sentence-transformers"
            ) from e

        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            query_instruction_for_rerank=query_instruction_for_rerank,
            query_instruction_format=query_instruction_format,
            passage_instruction_for_rerank=passage_instruction_for_rerank,
            passage_instruction_format=passage_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            max_length=max_length,
            normalize=normalize,
            **kwargs,
        )

        ce_kw = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            ce_kw["cache_folder"] = cache_dir

        self.cross_encoder = CrossEncoder(
            model_name_or_path,
            device=self.target_devices[0],
            **ce_kw,
        )
        self.model = getattr(self.cross_encoder, "model", None)

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[Any, Any]], Tuple[Any, Any]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[float, List[float]]:
        if batch_size is None:
            batch_size = self.batch_size
        if normalize is None:
            normalize = self.normalize
        if device is None:
            device = self.target_devices[0]

        if isinstance(sentence_pairs, tuple) and len(sentence_pairs) == 2:
            sentence_pairs = [sentence_pairs]

        prompt = self.query_instruction_for_rerank
        activation_fn = torch.nn.Sigmoid() if normalize else None

        all_scores: List[float] = []
        for start in tqdm(
            range(0, len(sentence_pairs), batch_size),
            desc="Computing scores",
            disable=len(sentence_pairs) <= batch_size,
        ):
            batch = sentence_pairs[start : start + batch_size]
            scores = self.cross_encoder.predict(
                batch,
                batch_size=len(batch),
                prompt=prompt,
                activation_fn=activation_fn,
            )
            arr = np.asarray(scores).reshape(-1)
            all_scores.extend(arr.tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores
