from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class MegaPairsEvalArgs(AbsEvalArgs):
    """
    Arguments for MegaPairs evaluation.
    """
    dataset_name: str = field(
        default="JUNJIE99/MegaPairs",
        metadata={"help": "HuggingFace dataset name for MegaPairs. Default: JUNJIE99/MegaPairs"}
    )
    image_root_dir: str = field(
        default=None,
        metadata={"help": "Root directory for images. If None, will use paths from dataset directly."}
    )

