import os
import logging

import torch
from transformers import PreTrainedModel

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class Qwen3VLEmbedderTrainer(AbsEmbedderTrainer):
    """Trainer class for Qwen3-VL-Embedding models."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)

        tokenizer = getattr(self.model, 'tokenizer', None) or getattr(self, 'processing_class', None)
        if tokenizer is not None and self.is_world_process_zero():
            tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
