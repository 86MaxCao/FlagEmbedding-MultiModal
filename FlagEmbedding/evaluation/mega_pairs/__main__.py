from transformers import HfArgumentParser

from FlagEmbedding.abc.evaluation import AbsEvalModelArgs
from FlagEmbedding.evaluation.mega_pairs import (
    MegaPairsEvalArgs,
    MegaPairsEvalRunner
)


def main():
    parser = HfArgumentParser((
        MegaPairsEvalArgs,
        AbsEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: MegaPairsEvalArgs
    model_args: AbsEvalModelArgs

    runner = MegaPairsEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()

