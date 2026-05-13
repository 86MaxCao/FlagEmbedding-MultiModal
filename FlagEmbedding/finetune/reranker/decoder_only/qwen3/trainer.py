import os
import torch
import logging
from typing import Optional

from FlagEmbedding.abc.finetune.reranker import AbsRerankerTrainer

logger = logging.getLogger(__name__)


class Qwen3RerankerTrainer(AbsRerankerTrainer):
    """Trainer for Qwen3 text reranker."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} does not support save interface'
            )
        else:
            self.model.save(output_dir)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
