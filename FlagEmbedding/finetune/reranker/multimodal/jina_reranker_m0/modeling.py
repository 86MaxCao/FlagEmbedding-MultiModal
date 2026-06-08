import logging

import torch
from torch import nn, Tensor
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel, RerankerOutput

logger = logging.getLogger(__name__)

LOGIT_BIAS = 2.65


class JinaRerankerM0Model(AbsRerankerModel):
    """Multimodal reranker model for jina-reranker-m0.

    Key differences from MultimodalRerankerModel:
    - jina-reranker-m0's forward() already returns scalar scores directly
      (via its custom JinaVLForRanking that extracts last-token hidden state
      and applies a score head), so no score_token_id append is needed.
    - Handles Qwen2VLConfig which lacks top-level pad_token_id and hidden_size.
    - Does not use yes_loc (no score token needed).
    """
    def __init__(
        self,
        base_model,
        tokenizer=None,
        train_group_size: int = 2,
        loss_type: str = 'pairwise',
        pairwise_margin: float = 1.0,
    ):
        nn.Module.__init__(self)
        self.model = base_model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.train_group_size = train_group_size

        # Get config - handle PeftModel wrapping
        model_for_config = self.model
        if hasattr(self.model, 'base_model'):
            # PeftModel: get config from the base model
            model_for_config = self.model.base_model.model
        self.config = getattr(model_for_config, 'config', None)

        # Set pad_token_id on the underlying config if missing
        if self.config is not None and hasattr(self.config, 'text_config'):
            # Qwen2VLConfig: pad_token_id is in text_config
            if not hasattr(self.config, 'pad_token_id') or self.config.pad_token_id is None:
                if hasattr(self.config.text_config, 'pad_token_id') and self.config.text_config.pad_token_id is not None:
                    self.config.pad_token_id = self.config.text_config.pad_token_id
                elif tokenizer is not None and tokenizer.pad_token_id is not None:
                    self.config.pad_token_id = tokenizer.pad_token_id

        self.loss_type = loss_type
        self.pointwise_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.listwise_loss = nn.CrossEntropyLoss(reduction='mean')
        self.pairwise_loss = nn.MarginRankingLoss(margin=pairwise_margin, reduction='mean')

        logger.info(f"Using {loss_type} loss (pairwise_margin={pairwise_margin}) for jina-reranker-m0 training")

    def encode(self, features):
        """Forward through jina-reranker-m0 to get scores.

        Args:
            features (dict): Processed inputs from JinaRerankerM0Collator.

        Returns:
            torch.Tensor: Scalar scores from the model.
        """
        if features is None:
            return None
        scores = self.model(**features)
        return scores

    def forward(
        self,
        pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Optional[Tensor] = None
    ):
        """Forward pass with loss computation.

        Args:
            pair: Processed inputs from collator.
            teacher_scores: Teacher scores for knowledge distillation.

        Returns:
            RerankerOutput: Output containing loss and scores.
        """
        ranker_logits = self.encode(pair)

        if teacher_scores is not None:
            teacher_scores = torch.Tensor(teacher_scores)
            teacher_targets = teacher_scores.view(-1, self.train_group_size)
            teacher_targets = torch.softmax(teacher_targets.detach(), dim=-1)
        else:
            teacher_targets = None

        if self.training:
            if self.loss_type == 'pointwise':
                loss = self.compute_pointwise_loss(ranker_logits)
            elif self.loss_type == 'listwise':
                loss = self.compute_listwise_loss(ranker_logits, teacher_scores, teacher_targets)
            elif self.loss_type == 'pairwise':
                loss = self.compute_pairwise_loss(ranker_logits, teacher_scores, teacher_targets)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'pointwise', 'pairwise', or 'listwise'.")
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_pointwise_loss(self, scores):
        """Compute pointwise BCE loss (matching ms-swift's PointwiseRerankerLoss).

        Each group has 1 positive (label=1) followed by (train_group_size-1) negatives (label=0).
        """
        grouped_logits = scores.view(-1, self.train_group_size)
        labels = torch.zeros_like(grouped_logits)
        labels[:, 0] = 1.0
        return self.pointwise_loss(grouped_logits.view(-1), labels.view(-1))

    def compute_listwise_loss(self, scores, teacher_scores=None, teacher_targets=None):
        """Compute listwise loss."""
        grouped_logits = scores.view(-1, self.train_group_size)
        batch_size = grouped_logits.size(0)
        target = torch.zeros(batch_size, device=grouped_logits.device, dtype=torch.long)
        loss = self.listwise_loss(grouped_logits, target)

        if teacher_scores is not None and teacher_targets is not None:
            teacher_targets = teacher_targets.to(grouped_logits.device)
            kd_loss = -torch.mean(
                torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)
            )
            loss += kd_loss

        return loss

    def compute_pairwise_loss(self, scores, teacher_scores=None, teacher_targets=None):
        """Compute pairwise loss."""
        grouped_logits = scores.view(-1, self.train_group_size)
        pos_scores = grouped_logits[:, 0]
        neg_scores = grouped_logits[:, 1:]

        num_negatives = neg_scores.size(1)
        pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        pos_flat = pos_scores_expanded.reshape(-1)
        neg_flat = neg_scores.reshape(-1)
        target = torch.ones_like(pos_flat)

        loss = self.pairwise_loss(pos_flat, neg_flat, target)

        if teacher_scores is not None and teacher_targets is not None:
            teacher_targets = teacher_targets.to(grouped_logits.device)
            kd_loss = -torch.mean(
                torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)
            )
            loss += kd_loss

        return loss

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        """Save the model."""
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, *args, **kwargs):
        """Save the tokenizer and model."""
        self.tokenizer.save_pretrained(*args, **kwargs)
        return self.model.save_pretrained(*args, **kwargs)
