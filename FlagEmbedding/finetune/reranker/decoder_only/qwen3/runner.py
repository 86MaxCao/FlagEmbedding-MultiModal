import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.reranker.AbsArguments import AbsRerankerDataArguments, AbsRerankerTrainingArguments
from FlagEmbedding.abc.finetune.reranker import AbsRerankerRunner, AbsRerankerModel

from .modeling import Qwen3RerankerModel
from .arguments import Qwen3RerankerModelArguments
from .dataset import Qwen3RerankerTrainDataset, Qwen3RerankerCollator
from .trainer import Qwen3RerankerTrainer
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class Qwen3RerankerRunner(AbsRerankerRunner):
    """Runner for Qwen3 text reranker fine-tuning."""

    def __init__(
        self,
        model_args: Qwen3RerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments,
    ):
        super().__init__(model_args, data_args, training_args)

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load tokenizer and model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,  # Qwen3 tokenizer works better with slow tokenizer
            trust_remote_code=self.model_args.trust_remote_code,
            padding_side='left',
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token = tokenizer.eod
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token = tokenizer.im_start
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token = tokenizer.im_end
                tokenizer.eos_token_id = tokenizer.im_end_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = get_model(self.model_args)

        model = Qwen3RerankerModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            loss_type=self.model_args.loss_type,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def load_dataset(self):
        """Load training dataset."""
        self.train_dataset = Qwen3RerankerTrainDataset(
            args=self.data_args,
            tokenizer=self.tokenizer,
        )
        self.data_collator = Qwen3RerankerCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
        )

    def load_trainer(self) -> Qwen3RerankerTrainer:
        """Load the trainer."""
        trainer = Qwen3RerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        return trainer

    def run(self):
        """Run the finetuning."""
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        # Save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
