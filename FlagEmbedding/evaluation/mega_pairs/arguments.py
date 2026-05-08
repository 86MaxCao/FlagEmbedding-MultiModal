from dataclasses import dataclass, field
from typing import Optional

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class MegaPairsEvalArgs(AbsEvalArgs):
    """
    Arguments for MegaPairs evaluation.
    """
    dataset_dir: str = field(
        default="~/.cache/huggingface/datasets/MegaPairs-flat-final/data",
        metadata={"help": "Local parquet data directory for MegaPairs"}
    )
    use_local_data: bool = field(
        default=True,
        metadata={"help": "Whether to use local parquet data instead of HF dataset"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to evaluate (for quick testing)"}
    )
    skip_corpus_cache: bool = field(
        default=False,
        metadata={"help": "Skip saving corpus embeddings to disk"}
    )

