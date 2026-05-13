import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments,
    AbsRerankerRunner,
    AbsRerankerModel,
)

from .arguments import Qwen3VLRerankerModelArguments
from .trainer import Qwen3VLRerankerTrainer
from .modeling import Qwen3VLRerankerModel
from .dataset import Qwen3VLRerankerTrainDataset, Qwen3VLRerankerCollator
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class Qwen3VLRerankerRunner(AbsRerankerRunner):
    """Runner class for Qwen3-VL-Reranker training.

    Args:
        model_args: Model arguments instance.
        data_args: Data arguments instance.
        training_args: Trainer arguments.
    """
    def __init__(
        self,
        model_args: Qwen3VLRerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: Qwen3VLRerankerModelArguments
        self.data_args: AbsRerankerDataArguments
        self.training_args: AbsRerankerTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load tokenizer and model.

        Returns:
            Tuple of tokenizer and model.
        """
        # Force slow tokenizer (use_fast=False) to avoid C++ heap corruption
        # between pyarrow (datasets) and tokenizers libraries.
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = 'left'

        base_model, processor = get_model(self.model_args)
        self.processor = processor

        model = Qwen3VLRerankerModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            loss_type=self.model_args.loss_type,
        )

        if self.training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return tokenizer, model

    def load_train_dataset(self):
        """Load the training dataset."""
        self.train_dataset = Qwen3VLRerankerTrainDataset(
            args=self.data_args,
            tokenizer=self.tokenizer
        )
        return self.train_dataset

    def load_data_collator(self):
        """Load the data collator."""
        self.data_collator = Qwen3VLRerankerCollator(
            tokenizer=self.tokenizer,
            processor=self.processor,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            max_len=self.data_args.query_max_len + self.data_args.passage_max_len,
        )
        return self.data_collator

    def load_trainer(self) -> Qwen3VLRerankerTrainer:
        """Load the trainer.

        Returns:
            Loaded trainer instance.
        """
        trainer = Qwen3VLRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            # Don't pass processing_class: the Qwen3VL fast tokenizer triggers
            # a C++ crash (data.cc:185) when Trainer uses it internally for logging.
            # The tokenizer is still available via model.tokenizer for saving.
        )
        return trainer

    def run(self):
        """Run the finetune."""
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
