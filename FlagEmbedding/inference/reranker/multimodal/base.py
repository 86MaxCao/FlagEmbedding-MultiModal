import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union, Tuple, Optional
from transformers import AutoModel, AutoProcessor
from PIL import Image
from transformers.image_utils import load_image

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.compat import apply_jina_reranker_patches

LOGIT_BIAS = 2.65  # logit bias for sigmoid normalization

apply_jina_reranker_patches()


def load_images(images, lazy_load: bool = True):
    """Load images from various sources."""
    pil_max_px = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None

    images_batch = []
    for image in images:
        if isinstance(image, Image.Image):
            images_batch.append(image)
        else:
            pil_image = load_image(image)
            if lazy_load:
                images_batch.append(pil_image)
            else:
                images_batch.append(pil_image.copy())
                pil_image.close()
    Image.MAX_IMAGE_PIXELS = pil_max_px

    return images_batch


def formatting_prompts_func(
    query: str,
    doc: str,
    query_type: str = 'text',
    doc_type: str = 'text',
    prefix_str: str = '',
) -> str:
    """Format prompts for different combinations of query and content types."""
    if query_type == 'image':
        query_part = "**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        query_part = f"**Query**:\n{query}"

    if doc_type == 'image':
        doc_part = "**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        doc_part = f"**Document**:\n{doc}"

    prompt = doc_part + '\n' + query_part

    if prefix_str:
        prompt = prefix_str + '\n' + prompt

    return prompt


class MultimodalReranker(AbsReranker):
    """Multimodal reranker class for models like jina-reranker-m0.

    Args:
        model_name_or_path (str): Model name or path.
        use_fp16 (bool, optional): Use half precision. Defaults to False.
        trust_remote_code (bool, optional): Trust remote code. Defaults to True.
        cache_dir (Optional[str], optional): Cache directory. Defaults to None.
        devices (Optional[Union[str, List[str], List[int]]], optional): Devices to use. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 128.
        query_max_length (Optional[int], optional): Max query length. Defaults to None.
        max_length (int, optional): Max total length. Defaults to 10240.
        normalize (bool, optional): Normalize scores. Defaults to True.
        query_type (str, optional): Query type ('text' or 'image'). Defaults to 'text'.
        doc_type (str, optional): Document type ('text' or 'image'). Defaults to 'image'.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        devices: Optional[Union[str, List[str], List[int]]] = None,
        batch_size: int = 8,
        query_max_length: Optional[int] = 512,
        max_length: int = 10240,
        normalize: bool = True,
        query_type: str = 'text',
        doc_type: str = 'image',
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            max_length=max_length,
            normalize=normalize,
            **kwargs
        )
        
        self.query_type = query_type
        self.doc_type = doc_type
        self.trust_remote_code = trust_remote_code
        
        # Load model
        # ignore_mismatched_sizes: JinaVLForRanking replaces lm_head with nn.Identity(),
        # so lm_head.weight in the checkpoint (shape [vocab_size, hidden_size]) mismatches
        # the Identity's empty weight. We tolerate the mismatch and fix lm_head afterwards.
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype="auto" if not use_fp16 else torch.float16,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        # After from_pretrained, lm_head may still carry a meta-device weight from the
        # Identity patch; replace it with a clean Identity so .to(device) works.
        import torch.nn as nn
        if isinstance(self.model.lm_head, nn.Identity):
            self.model.lm_head = nn.Identity()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            max_pixels=602112,
            min_pixels=3136,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        
        # Store processor in model for compatibility
        if hasattr(self.model, '_processor'):
            self.model._processor = self.processor
        
        self.score_token_id = getattr(self.model, 'score_token_id', 100)

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: Optional[str] = None,
        query_type: Optional[str] = None,
        doc_type: Optional[str] = None,
        **kwargs: Any
    ) -> List[float]:
        """Compute reranking scores for sentence pairs.

        Args:
            sentence_pairs: List of (query, doc) pairs.
            batch_size: Batch size for processing.
            query_max_length: Max length for queries.
            max_length: Max total length.
            normalize: Whether to normalize scores.
            device: Device to use.
            query_type: Type of query ('text' or 'image').
            doc_type: Type of document ('text' or 'image').

        Returns:
            List of scores.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if query_max_length is None:
            query_max_length = self.query_max_length
        if max_length is None:
            max_length = self.max_length
        if normalize is None:
            normalize = self.normalize
        if device is None:
            device = self.target_devices[0]
        if query_type is None:
            query_type = self.query_type
        if doc_type is None:
            doc_type = self.doc_type

        # Handle single pair
        if isinstance(sentence_pairs, tuple) and len(sentence_pairs) == 2 and isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        if self.use_fp16 and device != "cpu":
            self.model.half()

        max_doc_length = max(max_length - query_max_length, query_max_length) if query_max_length else max_length
        max_length = max_length - 1  # Reserve space for score token

        all_scores = []

        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Computing scores", disable=len(sentence_pairs) < batch_size):
            mini_batch = sentence_pairs[start_index:start_index + batch_size]

            # Format prompts
            batch_inputs = []
            for q, d in mini_batch:
                # Truncate long documents
                if doc_type == 'text':
                    tokens = self.processor.tokenizer(d, truncation=True, max_length=max_doc_length)
                    if len(tokens['input_ids']) >= max_doc_length:
                        d = self.processor.tokenizer.decode(tokens['input_ids'])

                batch_inputs.append(formatting_prompts_func(q, d, query_type=query_type, doc_type=doc_type))

            # Prepare images
            doc_images = []
            query_images = []
            if doc_type == 'image':
                doc_images = load_images([d for (q, d) in mini_batch])
            if query_type == 'image':
                query_images = load_images([q for (q, d) in mini_batch])

            batch_images = None
            if len(doc_images) == len(query_images) and len(doc_images) > 0:
                batch_images = [[d, q] for q, d in zip(query_images, doc_images)]
            elif len(doc_images) > 0:
                batch_images = doc_images
            elif len(query_images) > 0:
                batch_images = query_images

            # Process inputs
            batch = self.processor(
                text=batch_inputs,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Append score token
            batch_size_current = batch["input_ids"].size(0)
            batch["input_ids"] = torch.cat(
                [
                    batch["input_ids"],
                    torch.full((batch_size_current, 1), self.score_token_id, device=batch["input_ids"].device),
                ],
                dim=1,
            )
            batch["attention_mask"] = torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones((batch_size_current, 1), device=batch["attention_mask"].device),
                ],
                dim=1,
            )

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            scores = self.model(**batch).view(-1).cpu().float().numpy()

            # Normalize scores
            if normalize:
                scores = 1.0 / (1.0 + np.exp(-(scores - LOGIT_BIAS)))

            all_scores.extend(scores.tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        query_type: Optional[str] = None,
        doc_type: Optional[str] = None,
        **kwargs: Any
    ) -> Union[float, List[float]]:
        """Compute reranking scores.

        Args:
            sentence_pairs: (query, document) pairs.
            batch_size: Batch size.
            query_max_length: Max query length.
            max_length: Max total length.
            normalize: Normalize scores.
            query_type: 'text' or 'image'.
            doc_type: 'text' or 'image'.

        Returns:
            Scores for the pairs.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if query_max_length is None:
            query_max_length = self.query_max_length
        if max_length is None:
            max_length = self.max_length
        if normalize is None:
            normalize = self.normalize

        if isinstance(sentence_pairs, tuple) or len(self.target_devices) == 1:
            return self.compute_score_single_gpu(
                sentence_pairs,
                batch_size=batch_size,
                query_max_length=query_max_length,
                max_length=max_length,
                normalize=normalize,
                device=self.target_devices[0],
                query_type=query_type,
                doc_type=doc_type,
                **kwargs
            )

        # Multi-GPU processing
        if self.pool is None:
            self.pool = self.start_multi_process_pool(AbsReranker._compute_score_multi_process_worker)

        scores = self.compute_score_multi_process(
            sentence_pairs,
            self.pool,
            batch_size=batch_size,
            query_max_length=query_max_length,
            max_length=max_length,
            normalize=normalize,
            query_type=query_type,
            doc_type=doc_type,
            **kwargs
        )
        return scores

