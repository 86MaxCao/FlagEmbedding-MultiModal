from transformers import HfArgumentParser

from FlagEmbedding.abc.evaluation import AbsEvalModelArgs
from .arguments import MMEBTrainEvalArgs
from .runner import MMEBTrainEvalRunner


def main():
    """Main function for MMEB-train evaluation."""
    parser = HfArgumentParser((
        MMEBTrainEvalArgs,
        AbsEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: MMEBTrainEvalArgs
    model_args: AbsEvalModelArgs

    runner = MMEBTrainEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
