from transformers import HfArgumentParser

from FlagEmbedding.finetune.reranker.multimodal.base import (
    MultimodalRerankerDataArguments,
    MultimodalRerankerTrainingArguments,
    MultimodalRerankerModelArguments,
    MultimodalRerankerRunner,
)


def main():
    parser = HfArgumentParser((
        MultimodalRerankerModelArguments,
        MultimodalRerankerDataArguments,
        MultimodalRerankerTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: MultimodalRerankerModelArguments
    data_args: MultimodalRerankerDataArguments
    training_args: MultimodalRerankerTrainingArguments

    runner = MultimodalRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()

