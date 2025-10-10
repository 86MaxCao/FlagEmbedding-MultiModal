from typing import Optional
from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModelArguments


@dataclass
class MultimodalRerankerModelArguments(AbsRerankerModelArguments):
    """
    Model argument class for multimodal reranker model.
    """
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    loss_type: str = field(
        default="pairwise",
        metadata={"help": "Loss type: 'pairwise' or 'listwise'. Defaults to 'pairwise'."}
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )

