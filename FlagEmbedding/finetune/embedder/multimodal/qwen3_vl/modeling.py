import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderOutput

logger = logging.getLogger(__name__)


class Qwen3VLEmbedderModel(AbsEmbedderModel):
    """Embedder model for Qwen3-VL-Embedding training.

    Uses Qwen3VL processor for multimodal input processing,
    and last-token pooling for embedding extraction.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        normalize_embeddings: bool = True,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        # Dummy buffer so DataParallel moves it to each replica's device
        self.register_buffer('_dp_device', torch.zeros(1))

    def encode(self, features):
        """Encode multimodal inputs into embeddings using last-token pooling.

        Args:
            features (dict): Processed inputs from the collator, containing
                input_ids, attention_mask, and optionally pixel_values, image_grid_thw.

        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, hidden_size).
        """
        if features is None:
            return None

        device = self._dp_device.device
        model_kwargs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in features.items()}

        # Reset rope_deltas to avoid Qwen3VL batch-size mismatch bug:
        # When an image batch sets rope_deltas for batch_size=N and a subsequent
        # text-only batch has batch_size=M where M < N, compute_3d_position_ids
        # produces position_ids with a zero batch dimension, corrupting attention.
        base = getattr(self.model, 'base_model', None)
        if base is not None:
            qwen_model = getattr(base, 'model', None)
            if qwen_model is not None and hasattr(qwen_model, 'rope_deltas'):
                qwen_model.rope_deltas = None

        outputs = self.model(**model_kwargs, output_hidden_states=True)

        # Last-token pooling: take the hidden state of the last token
        # Qwen3VL uses left-padding, so last token is at position -1
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = last_hidden_state[:, -1]

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings.contiguous()

    def compute_loss(self, scores, target):
        """Compute InfoNCE / contrastive loss.

        Args:
            scores (torch.Tensor): Similarity scores matrix.
            target (torch.Tensor): Target labels (indices of positive pairs).

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.cross_entropy(scores, target)

    def compute_score(self, q_reps, p_reps):
        """Compute similarity scores between query and passage embeddings.

        Scores are scaled by 1/temperature before loss computation so that
        the softmax in InfoNCE is sharper (temperature < 1 amplifies signal).

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: Similarity score matrix scaled by temperature.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
        return torch.matmul(q_reps, p_reps.transpose(-2, -1)) / self.temperature

    def forward(self, queries=None, passages=None, teacher_scores=None, no_in_batch_neg_flag=False):
        """Forward pass for training with in-batch negatives.

        Args:
            queries (dict): Processed query inputs.
            passages (dict): Processed passage inputs.
            teacher_scores: Teacher model scores for knowledge distillation.
            no_in_batch_neg_flag: Whether to skip in-batch negatives.

        Returns:
            EmbedderOutput: Output containing loss and embeddings.
        """
        q_reps = self.encode(queries)
        p_reps = self.encode(passages)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()
                teacher_targets = F.softmax(teacher_scores, dim=-1)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
        else:
            scores = self.compute_score(q_reps, p_reps)
            loss = None

        return EmbedderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def save(self, output_dir: str):
        """Save the model and tokenizer.

        Args:
            output_dir (str): Output directory.
        """
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
