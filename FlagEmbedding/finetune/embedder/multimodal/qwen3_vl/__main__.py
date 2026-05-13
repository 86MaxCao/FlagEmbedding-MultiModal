import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.multimodal.qwen3_vl import (
    Qwen3VLEmbedderModelArguments,
    Qwen3VLEmbedderDataArguments,
    Qwen3VLEmbedderTrainingArguments,
    Qwen3VLEmbedderRunner,
)


def main():
    parser = HfArgumentParser((
        Qwen3VLEmbedderModelArguments,
        Qwen3VLEmbedderDataArguments,
        Qwen3VLEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: Qwen3VLEmbedderModelArguments
    data_args: Qwen3VLEmbedderDataArguments
    training_args: Qwen3VLEmbedderTrainingArguments

    runner = Qwen3VLEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
