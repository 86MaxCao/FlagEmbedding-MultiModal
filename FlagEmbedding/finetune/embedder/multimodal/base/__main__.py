from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.multimodal.base import (
    MultimodalEmbedderDataArguments,
    MultimodalEmbedderTrainingArguments,
    MultimodalEmbedderModelArguments,
    MultimodalEmbedderRunner,
)


def main():
    parser = HfArgumentParser((
        MultimodalEmbedderModelArguments,
        MultimodalEmbedderDataArguments,
        MultimodalEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: MultimodalEmbedderModelArguments
    data_args: MultimodalEmbedderDataArguments
    training_args: MultimodalEmbedderTrainingArguments

    runner = MultimodalEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()

