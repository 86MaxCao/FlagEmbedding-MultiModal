import logging
import torch
from torch import nn, Tensor
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel, RerankerOutput

logger = logging.getLogger(__name__)


class Qwen3RerankerModel(AbsRerankerModel):
    """Reranker model for Qwen3 text reranker training.

    Uses AutoModelForCausalLM and computes scores from the logits at the last
    token position over the "yes" and "no" token IDs, matching the official
    Qwen3-Reranker inference mechanism.

    Supports both pairwise and listwise loss.
    """

    def __init__(
        self,
        base_model,
        tokenizer=None,
        train_batch_size: int = 4,
        loss_type: str = 'pairwise',
    ):
        nn.Module.__init__(self)
        self.model = base_model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.train_batch_size = train_batch_size

        model_for_config = self.model
        if hasattr(self.model, 'base_model'):
            model_for_config = self.model.base_model.model
        self.config = getattr(model_for_config, 'config', None)

        self.loss_type = loss_type
        self.listwise_loss = nn.CrossEntropyLoss(reduction='mean')
        self.pairwise_loss = nn.MarginRankingLoss(margin=0.0, reduction='mean')

        # Get yes/no token IDs (lowercase, as in Qwen3 tokenizer)
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        logger.info(
            f"Qwen3RerankerModel: true_id={self.true_id}, false_id={self.false_id}, "
            f"loss_type={loss_type}"
        )

    def encode(self, features):
        """Forward through the causal LM to get relevance scores.

        Returns logit differences (yes - no) at the last token position.
        """
        if features is None:
            return None
        outputs = self.model(**features)
        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        true_logits = logits[:, self.true_id]
        false_logits = logits[:, self.false_id]
        # Use logit difference as scalar score
        scores = true_logits - false_logits
        return scores.contiguous()

    def forward(
        self,
        pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Optional[Tensor] = None
    ):
        """Forward pass with loss computation."""
        ranker_logits = self.encode(pair)

        if teacher_scores is not None:
            teacher_scores = torch.Tensor(teacher_scores)
            teacher_targets = teacher_scores.view(self.train_batch_size, -1)
            teacher_targets = torch.softmax(teacher_targets.detach(), dim=-1)
        else:
            teacher_targets = None

        if self.training:
            if self.loss_type == 'listwise':
                loss = self.compute_listwise_loss(ranker_logits, teacher_targets)
            elif self.loss_type == 'pairwise':
                loss = self.compute_pairwise_loss(ranker_logits, teacher_targets)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_listwise_loss(self, scores, teacher_targets=None):
        """Compute listwise cross-entropy loss."""
        grouped_logits = scores.view(self.train_batch_size, -1)
        target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
        loss = self.listwise_loss(grouped_logits, target)

        if teacher_targets is not None:
            teacher_targets = teacher_targets.to(grouped_logits.device)
            kd_loss = -torch.mean(
                torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)
            )
            loss += kd_loss

        return loss

    def compute_pairwise_loss(self, scores, teacher_targets=None):
        """Compute pairwise margin ranking loss."""
        grouped_logits = scores.view(self.train_batch_size, -1)
        pos_scores = grouped_logits[:, 0]
        neg_scores = grouped_logits[:, 1:]

        num_negatives = neg_scores.size(1)
        pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        pos_flat = pos_scores_expanded.reshape(-1)
        neg_flat = neg_scores.reshape(-1)
        target = torch.ones_like(pos_flat)

        loss = self.pairwise_loss(pos_flat, neg_flat, target)

        if teacher_targets is not None:
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
