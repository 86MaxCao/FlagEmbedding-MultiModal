"""Qwen3-VL-Embedding via Sentence-Transformers (see HF model card)."""

from typing import Any, List, Optional, Sequence, Union, cast

import numpy as np
import torch

from FlagEmbedding.abc.inference import AbsEmbedder


class Qwen3VLEmbeddingModel(AbsEmbedder):
    """Embedder for ``Qwen/Qwen3-VL-Embedding-*`` models.

    Loads with ``sentence_transformers.SentenceTransformer`` as documented on Hugging Face:
    optional ``prompt`` corresponds to task instruction (default inside the model uses
    ``Represent the user's input.``). Pass ``query_instruction_for_retrieval`` / corpus
    instructions via the base ``encode_queries`` / ``encode_corpus`` instruction fields.

    Multimodal inputs follow the model card: each item may be ``str``, image URL/path, or
    ``dict`` with ``text`` / ``image`` keys; ``encode_queries(..., images=...)`` zips text with image paths.
    """

    DEFAULT_POOLING_METHOD = "last_token"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str]]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        query_max_length: int = 32768,
        passage_max_length: int = 32768,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL-Embedding requires `sentence-transformers`. Install with: pip install sentence-transformers"
            ) from e

        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs,
        )

        model_kw: dict = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            model_kw["cache_folder"] = cache_dir

        self.st_model = SentenceTransformer(model_name_or_path, **model_kw)
        self.st_model.to(self.target_devices[0])
        if use_fp16 and self.target_devices[0] != "cpu":
            self.st_model.half()

        self.model = self.st_model
        self.trust_remote_code = trust_remote_code

    @staticmethod
    def _merge_text_and_images(
        texts: Optional[Union[str, List[str]]],
        images: Optional[Union[str, List[str]]],
    ) -> Union[str, List[Any]]:
        if texts is None and images is None:
            raise ValueError("Either texts/sentences or images must be provided.")
        if images is None:
            return cast(Union[str, List[str]], texts)
        if texts is None:
            if isinstance(images, str):
                return images
            return images
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, str):
            images = [images]
        if len(texts) != len(images):
            raise ValueError("texts and images must have the same length when both are provided.")
        return [{"text": t, "image": img} for t, img in zip(texts, images)]

    def encode_queries(
        self,
        queries: Union[List[str], str] = None,
        images: Union[List[str], str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        task_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.query_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        prompt = (
            task_instruction
            if task_instruction is not None
            else self.query_instruction_for_retrieval
        )
        sentences = self._merge_text_and_images(queries, images)
        return self.encode_single_device(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            device=self.target_devices[0],
            prompt=prompt,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str] = None,
        images: Union[List[str], str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.passage_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        passage_instruction = self.kwargs.get("passage_instruction_for_retrieval")
        sentences = self._merge_text_and_images(corpus, images)
        return self.encode_single_device(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            device=self.target_devices[0],
            prompt=passage_instruction,
            **kwargs,
        )

    def encode(
        self,
        sentences: Union[List[str], str, List[Any]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        del instruction_format  # Qwen3-VL uses ``prompt=`` instead of concatenating instruct + text
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.passage_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        return self.encode_single_device(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            device=self.target_devices[0],
            prompt=instruction,
            **kwargs,
        )

    def encode_single_device(
        self,
        sentences: Union[List[str], str, List[Any]],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if device is None:
            device = self.target_devices[0]

        self.st_model.to(device)
        if self.use_fp16 and device != "cpu":
            self.st_model.half()
        else:
            self.st_model.float()

        show_bar = isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)) and len(sentences) >= batch_size
        embeddings = self.st_model.encode(
            sentences,
            batch_size=batch_size,
            prompt=prompt,
            show_progress_bar=show_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
        )

        return cast(Union[np.ndarray, torch.Tensor], embeddings)
