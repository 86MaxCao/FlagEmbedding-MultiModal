import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from transformers import HfArgumentParser

from FlagEmbedding.finetune.reranker.multimodal.qwen3_vl import (
    Qwen3VLRerankerDataArguments,
    Qwen3VLRerankerTrainingArguments,
    Qwen3VLRerankerModelArguments,
    Qwen3VLRerankerRunner,
)


def main():
    parser = HfArgumentParser((
        Qwen3VLRerankerModelArguments,
        Qwen3VLRerankerDataArguments,
        Qwen3VLRerankerTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: Qwen3VLRerankerModelArguments
    data_args: Qwen3VLRerankerDataArguments
    training_args: Qwen3VLRerankerTrainingArguments

    runner = Qwen3VLRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
