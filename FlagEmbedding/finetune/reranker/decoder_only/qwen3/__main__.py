from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)

from FlagEmbedding.finetune.reranker.decoder_only.qwen3 import (
    Qwen3RerankerRunner,
    Qwen3RerankerModelArguments
)


def main():
    parser = HfArgumentParser((Qwen3RerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: Qwen3RerankerModelArguments
    data_args: AbsRerankerDataArguments
    training_args: AbsRerankerTrainingArguments

    runner = Qwen3RerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
