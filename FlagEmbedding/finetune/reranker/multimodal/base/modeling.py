import torch
from torch import nn, Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel, RerankerOutput

logger = logging.getLogger(__name__)

LOGIT_BIAS = 2.65  # For sigmoid normalization


class MultimodalRerankerModel(AbsRerankerModel):
    """Multimodal reranker model with support for Pairwise and Listwise losses.

    Args:
        base_model (PreTrainedModel): The base model to train on.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use. Defaults to None.
        train_batch_size (int, optional): Batch size used for training. Defaults to 4.
        loss_type (str, optional): Loss type - 'pairwise' or 'listwise'. Defaults to 'pairwise'.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        train_batch_size: int = 4,
        loss_type: str = 'pairwise',
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
        )
        self.loss_type = loss_type
        self.score_token_id = getattr(self.model, 'score_token_id', 100)
        
        # For listwise loss
        self.listwise_loss = nn.CrossEntropyLoss(reduction='mean')
        # For pairwise loss  
        self.pairwise_loss = nn.MarginRankingLoss(margin=0.0, reduction='mean')
        
        logger.info(f"Using {loss_type} loss for training")

    def encode(self, features):
        """Encode features and get scores.

        Args:
            features (dict): Features from processor.

        Returns:
            torch.Tensor: The scores from the model.
        """
        if features is None:
            return None
        
        # For multimodal reranker, features are already processed
        # Just forward through the model
        scores = self.model(**features)  # Shape: (batch_size,)
        
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
        ranker_logits = self.encode(pair)  # (batch_size * num_passages,)
        
        if teacher_scores is not None:
            teacher_scores = torch.Tensor(teacher_scores)
            teacher_targets = teacher_scores.view(self.train_batch_size, -1)
            teacher_targets = torch.softmax(teacher_targets.detach(), dim=-1)

        if self.training:
            if self.loss_type == 'listwise':
                loss = self.compute_listwise_loss(ranker_logits, teacher_scores, teacher_targets if teacher_scores is not None else None)
            elif self.loss_type == 'pairwise':
                loss = self.compute_pairwise_loss(ranker_logits, teacher_scores, teacher_targets if teacher_scores is not None else None)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'pairwise' or 'listwise'.")
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_listwise_loss(
        self, 
        scores: Tensor, 
        teacher_scores: Optional[Tensor] = None,
        teacher_targets: Optional[Tensor] = None
    ):
        """Compute listwise loss (cross-entropy over all passages).

        Args:
            scores: Model scores, shape (batch_size * num_passages,)
            teacher_scores: Raw teacher scores for KD
            teacher_targets: Softmax-normalized teacher targets

        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape to (batch_size, num_passages)
        grouped_logits = scores.view(self.train_batch_size, -1)
        
        # Target is always the first passage (positive)
        target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
        
        # Base cross-entropy loss
        loss = self.listwise_loss(grouped_logits, target)
        
        # Add knowledge distillation loss if available
        if teacher_scores is not None and teacher_targets is not None:
            teacher_targets = teacher_targets.to(grouped_logits.device)
            kd_loss = -torch.mean(
                torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)
            )
            loss += kd_loss
        
        return loss

    def compute_pairwise_loss(
        self,
        scores: Tensor,
        teacher_scores: Optional[Tensor] = None,
        teacher_targets: Optional[Tensor] = None
    ):
        """Compute pairwise loss (margin ranking loss).

        Args:
            scores: Model scores, shape (batch_size * num_passages,)
            teacher_scores: Raw teacher scores for KD
            teacher_targets: Softmax-normalized teacher targets

        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape to (batch_size, num_passages)
        grouped_logits = scores.view(self.train_batch_size, -1)
        
        # Split positive (first) and negatives (rest)
        pos_scores = grouped_logits[:, 0]  # (batch_size,)
        neg_scores = grouped_logits[:, 1:]  # (batch_size, num_negatives)
        
        # Compute pairwise margin ranking loss
        # For each negative, we want pos_score > neg_score
        num_negatives = neg_scores.size(1)
        
        # Expand pos_scores to match negatives
        pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)  # (batch_size, num_negatives)
        
        # Flatten for pairwise comparison
        pos_flat = pos_scores_expanded.reshape(-1)  # (batch_size * num_negatives,)
        neg_flat = neg_scores.reshape(-1)  # (batch_size * num_negatives,)
        
        # Target: pos should be ranked higher than neg (y=1)
        target = torch.ones_like(pos_flat)
        
        # Margin ranking loss
        loss = self.pairwise_loss(pos_flat, neg_flat, target)
        
        # Add knowledge distillation loss if available
        if teacher_scores is not None and teacher_targets is not None:
            teacher_targets = teacher_targets.to(grouped_logits.device)
            kd_loss = -torch.mean(
                torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)
            )
            loss += kd_loss
        
        return loss

    def compute_loss(self, scores, target):
        """Legacy compute_loss for compatibility.

        Args:
            scores: Computed scores.
            target: Target labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        """Save the model.

        Args:
            output_dir (str): Directory for saving the model.
        """
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, *args, **kwargs):
        """Save the tokenizer and model."""
        self.tokenizer.save_pretrained(*args, **kwargs)
        return self.model.save_pretrained(*args, **kwargs)

