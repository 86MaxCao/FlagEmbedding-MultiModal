import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from transformers import HfArgumentParser

from FlagEmbedding.finetune.reranker.multimodal.jina_reranker_m0 import (
    JinaRerankerM0DataArguments,
    JinaRerankerM0TrainingArguments,
    JinaRerankerM0ModelArguments,
    JinaRerankerM0Runner,
)


def main():
    parser = HfArgumentParser((
        JinaRerankerM0ModelArguments,
        JinaRerankerM0DataArguments,
        JinaRerankerM0TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: JinaRerankerM0ModelArguments
    data_args: JinaRerankerM0DataArguments
    training_args: JinaRerankerM0TrainingArguments

    runner = JinaRerankerM0Runner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
