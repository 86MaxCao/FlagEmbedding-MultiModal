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

from .arguments import MultimodalRerankerModelArguments
from .trainer import MultimodalRerankerTrainer
from .modeling import MultimodalRerankerModel
from .dataset import MultimodalRerankerTrainDataset, MultimodalRerankerCollator
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class MultimodalRerankerRunner(AbsRerankerRunner):
    """Runner class for multimodal reranker.

    Args:
        model_args: Model arguments instance.
        data_args: Data arguments instance.
        training_args: Trainer arguments.
    """
    def __init__(
        self,
        model_args: MultimodalRerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: MultimodalRerankerModelArguments
        self.data_args: AbsRerankerDataArguments
        self.training_args: AbsRerankerTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load tokenizer and model.

        Returns:
            Tuple of tokenizer and model.
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set padding side to left (required for reranker)
        tokenizer.padding_side = 'left'

        # Load model and processor
        base_model, processor = get_model(self.model_args)
        
        # Store processor for later use
        self.processor = processor

        # Wrap in training model
        model = MultimodalRerankerModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            loss_type=self.model_args.loss_type,
        )

        if self.training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return tokenizer, model

    def load_dataset(self):
        """Load the training dataset."""
        self.train_dataset = MultimodalRerankerTrainDataset(
            args=self.data_args,
            tokenizer=self.tokenizer
        )

    def load_data_collator(self):
        """Load the data collator."""
        self.data_collator = MultimodalRerankerCollator(
            processor=self.processor,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            max_len=self.data_args.query_max_len + self.data_args.passage_max_len,
            score_token_id=self.model.score_token_id,
        )

    def load_trainer(self) -> MultimodalRerankerTrainer:
        """Load the trainer.

        Returns:
            Loaded trainer instance.
        """
        trainer = MultimodalRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer

    def run(self):
        """Run the finetune."""
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        # Save merged model if needed
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)

