import os
import logging

import torch
from torch import nn, Tensor
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel, RerankerOutput

logger = logging.getLogger(__name__)


class Qwen3VLRerankerModel(AbsRerankerModel):
    """Reranker model for Qwen3-VL-Reranker training.

    Uses the Qwen3VL base model's last hidden state + a binary score linear
    (yes - no token weights from lm_head) to produce scalar relevance scores.
    This matches the official Qwen3-VL-Reranker scoring mechanism.

    Key differences from JinaRerankerM0Model:
    - Uses base model (no lm_head) + binary linear for scoring
    - Adds rope_deltas reset to avoid batch-size mismatch bug
    - Does NOT bypass AbsRerankerModel.__init__ (Qwen3VLConfig has pad_token_id)
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
            model_for_config = self.model.base_model.model
        self.config = getattr(model_for_config, 'config', None)

        self.loss_type = loss_type
        self.pointwise_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.listwise_loss = nn.CrossEntropyLoss(reduction='mean')
        self.pairwise_loss = nn.MarginRankingLoss(margin=pairwise_margin, reduction='mean')

        # Build binary score linear from lm_head weights (yes - no)
        self._init_score_linear(model_for_config)

        logger.info(f"Using {loss_type} loss (pairwise_margin={pairwise_margin}) for Qwen3-VL-Reranker training")

    def _init_score_linear(self, model_for_config):
        """Initialize score linear from lm_head weights: weight = W[yes] - W[no].

        This mirrors the official Qwen3-VL-Reranker scoring:
        score = sigmoid(linear(last_hidden_state[:, -1]))
        where linear.weight = lm_head.weight[yes_id] - lm_head.weight[no_id]
        """
        try:
            # Navigate to the underlying Qwen3VL model to get lm_head
            lm_head = None
            if hasattr(model_for_config, 'lm_head'):
                lm_head = model_for_config.lm_head
            elif hasattr(model_for_config, 'model') and hasattr(model_for_config.model, 'lm_head'):
                lm_head = model_for_config.model.lm_head

            if lm_head is not None and hasattr(lm_head, 'weight'):
                # Get yes/no token IDs from tokenizer
                yes_id = self.tokenizer.convert_tokens_to_ids('yes')
                no_id = self.tokenizer.convert_tokens_to_ids('no')

                lm_head_weights = lm_head.weight.data
                weight_yes = lm_head_weights[yes_id]
                weight_no = lm_head_weights[no_id]

                D = weight_yes.size(0)
                self.score_linear = nn.Linear(D, 1, bias=False)
                with torch.no_grad():
                    self.score_linear.weight[0] = weight_yes - weight_no
                self.score_linear = self.score_linear.to(weight_yes.dtype)
                logger.info(f"Initialized score linear from lm_head (yes_id={yes_id}, no_id={no_id}, dim={D}, dtype={weight_yes.dtype})")
            else:
                logger.warning("Could not find lm_head; score_linear will not be initialized")
                self.score_linear = None
        except Exception as e:
            logger.warning(f"Failed to initialize score_linear: {e}")
            self.score_linear = None

    def _reset_rope_deltas(self):
        """Reset rope_deltas before each forward to avoid batch-size mismatch.

        When an image batch sets rope_deltas for batch_size=N and a subsequent
        text-only batch has batch_size=M where M < N, compute_3d_position_ids
        produces position_ids with a zero batch dimension, corrupting attention.
        """
        base = getattr(self.model, 'base_model', None)
        if base is not None:
            qwen_model = getattr(base, 'model', None)
            if qwen_model is not None and hasattr(qwen_model, 'rope_deltas'):
                qwen_model.rope_deltas = None

    def encode(self, features):
        """Forward through Qwen3-VL to get relevance scores.

        Uses base model + score_linear (matching official Qwen3-VL-Reranker).
        """
        if features is None:
            return None
        self._reset_rope_deltas()
        outputs = self.model(**features, output_hidden_states=True)
        # Get last hidden state from the model output
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state[:, -1]
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1][:, -1]
        else:
            raise ValueError("Model output has neither last_hidden_state nor hidden_states")

        # Apply score linear
        if self.score_linear is not None:
            scores = self.score_linear(hidden).squeeze(-1)
        else:
            # Fallback: use raw hidden state norm as proxy score
            scores = hidden.norm(dim=-1)
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
            teacher_scores = torch.Tensor(teacher_scores).to(ranker_logits.device)
            teacher_scores = teacher_scores.view(-1, self.train_group_size)
            teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1)
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
