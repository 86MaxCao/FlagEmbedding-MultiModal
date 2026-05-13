import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments,
    AbsEmbedderRunner,
    AbsEmbedderModel,
    EmbedderTrainerCallbackForDataRefresh,
)

from .arguments import Qwen3VLEmbedderModelArguments
from .trainer import Qwen3VLEmbedderTrainer
from .modeling import Qwen3VLEmbedderModel
from .dataset import Qwen3VLEmbedderTrainDataset, Qwen3VLEmbedderCollator
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class Qwen3VLEmbedderRunner(AbsEmbedderRunner):
    """Runner class for Qwen3-VL-Embedding training.

    Args:
        model_args: Model arguments instance.
        data_args: Data arguments instance.
        training_args: Trainer arguments.
    """
    def __init__(
        self,
        model_args: Qwen3VLEmbedderModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: Qwen3VLEmbedderModelArguments
        self.data_args: AbsEmbedderDataArguments
        self.training_args: AbsEmbedderTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load tokenizer, processor, and model.

        Returns:
            Tuple of tokenizer and model.
        """
        # Force slow tokenizer to avoid C++ heap corruption between pyarrow and tokenizers
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

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer.")
            else:
                logger.warning(f"Special tokens already exist in the tokenizer.")

        base_model, processor = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))
        self.processor = processor

        model = Qwen3VLEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size if self.training_args.sub_batch_size else -1,
            kd_loss_type=self.training_args.kd_loss_type,
            normalize_embeddings=self.training_args.normalize_embeddings,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False

        return tokenizer, model

    def load_train_dataset(self):
        """Load the training dataset."""
        return Qwen3VLEmbedderTrainDataset(
            args=self.data_args,
            tokenizer=self.tokenizer
        )

    def load_data_collator(self):
        """Load the data collator using Qwen3VL processor."""
        return Qwen3VLEmbedderCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            query_instruction=self.data_args.query_instruction_for_retrieval or "Represent the user's input.",
        )

    def load_trainer(self) -> Qwen3VLEmbedderTrainer:
        """Load the trainer.

        Returns:
            Qwen3VLEmbedderTrainer instance.
        """
        trainer = Qwen3VLEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            # Don't pass processing_class: avoids C++ crash (data.cc:185)
            # when Trainer uses the fast tokenizer internally.
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def run(self):
        """Run the finetune."""
        if not self.model_args.only_merge_lora_model:
            Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            self.trainer.save_model()

        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
