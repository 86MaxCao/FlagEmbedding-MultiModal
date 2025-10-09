import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel

logger = logging.getLogger(__name__)


class BiMultimodalEmbedderModel(AbsEmbedderModel):
    """Embedder model class for multimodal model.

    Args:
        base_model (PreTrainedModel): The base model to train on.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``'kl_div'``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'last_token'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'last_token',
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')

    def encode(self, features):
        """
        Encode and get the embedding for multimodal inputs.

        Args:
            features (dict): Features from model.data_process(), containing processed inputs.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        if features is None:
            return None
        
        # For multimodal models, features are already processed by data_process
        # and we just need to get the outputs with output_hidden_states=True
        outputs = self.model(**features, output_hidden_states=True)
        
        # Extract embeddings from the last token
        # outputs shape: (batch_size, seq_len, hidden_size)
        embeddings = outputs[:, -1, :]
        
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings.contiguous()

    def compute_loss(self, scores, target):
        """Compute the contrastive loss with temperature scaling.

        Args:
            scores (torch.Tensor): Computed similarity scores.
            target (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.cross_entropy(scores, target)

    def forward(self, queries=None, passages=None, teacher_scores=None, no_in_batch_neg_flag=False):
        """
        Forward pass of the multimodal embedder model.

        Args:
            queries (dict): Processed query inputs from data_process.
            passages (dict): Processed passage inputs from data_process.
            teacher_scores (torch.Tensor, optional): Teacher model scores for knowledge distillation.
            no_in_batch_neg_flag (bool): Whether to use in-batch negatives.

        Returns:
            EmbedderOutput: Model output containing loss, scores, and embeddings.
        """
        q_reps = self.encode(queries)
        p_reps = self.encode(passages)

        if self.training:
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            
            if teacher_scores is not None:
                scores = scores.view(q_reps.size(0), -1)
                teacher_targets = torch.tensor(
                    teacher_scores, dtype=scores.dtype, device=scores.device
                ).view(q_reps.size(0), -1)
                loss = self.compute_kd_loss(scores, teacher_targets)
            else:
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                target = target * (p_reps.size(0) // q_reps.size(0))
                loss = self.compute_loss(scores, target)
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        from FlagEmbedding.abc.finetune.embedder import EmbedderOutput
        return EmbedderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_similarity(self, q_reps, p_reps):
        """Compute similarity scores between queries and passages.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: Similarity scores.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_kd_loss(self, scores, teacher_scores):
        """Compute knowledge distillation loss.

        Args:
            scores (torch.Tensor): Student model scores.
            teacher_scores (torch.Tensor): Teacher model scores.

        Returns:
            torch.Tensor: KD loss.
        """
        if self.kd_loss_type == 'm3_kd_loss':
            # M3 KD loss
            teacher_targets = torch.softmax(teacher_scores, dim=-1)
            teacher_targets = teacher_targets * len(teacher_targets)
            return self.compute_loss(scores, teacher_targets.argmax(dim=-1))
        elif self.kd_loss_type == 'kl_div':
            # KL divergence loss
            log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
            teacher_probs = torch.nn.functional.softmax(teacher_scores, dim=-1)
            return torch.nn.functional.kl_div(
                log_scores, teacher_probs, reduction='batchmean'
            )
        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")

    def save(self, output_dir: str):
        """Save the model.

        Args:
            output_dir (str): Output directory.
        """
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        """
        For multimodal models, we use the last token directly.
        This method is kept for compatibility but not used.
        """
        return last_hidden_state[:, -1, :]

