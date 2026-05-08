from dataclasses import dataclass, field
from typing import Optional, List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class MMEBTrainEvalArgs(AbsEvalArgs):
    """
    Arguments for MMEB-train evaluation.
    """
    dataset_dir: str = field(
        default="~/.cache/huggingface/datasets/MMEB-train-flat",
        metadata={"help": "Local MMEB-train data directory"}
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
    sub_datasets: List[str] = field(
        default_factory=lambda: [
            "DocVQA", "CIRR", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i", 
            "VisualNews_i2t", "VisualNews_t2i", "Visual7W", "WebQA"
        ],
        metadata={"help": "List of sub-datasets to evaluate"}
    )
    task_types: List[str] = field(
        default_factory=lambda: ["text", "image", "mixed"],
        metadata={"help": "Task types to evaluate: text, image, mixed"}
    )
