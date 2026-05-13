import os
import torch
import logging
from typing import Optional

from FlagEmbedding.abc.finetune.reranker import AbsRerankerTrainer

logger = logging.getLogger(__name__)


class Qwen3VLRerankerTrainer(AbsRerankerTrainer):
    """Trainer class for Qwen3-VL-Reranker models.

    Overrides _save to handle the case where processing_class (tokenizer)
    is not passed to the Trainer (to avoid C++ crash with fast tokenizer).
    The tokenizer is still accessible via model.tokenizer.
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)

        # Use model.tokenizer since we don't pass processing_class to Trainer
        tokenizer = getattr(self.model, 'tokenizer', None) or getattr(self, 'processing_class', None)
        if tokenizer is not None and self.is_world_process_zero():
            tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
