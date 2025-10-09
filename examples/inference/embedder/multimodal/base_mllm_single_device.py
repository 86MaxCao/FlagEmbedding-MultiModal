import os
import torch
from FlagEmbedding import FlagMLLMModel


def test_multimodal_base():
    """
    Test multimodal MLLM model using the base class directly.
    """
    # Load the multimodal model using base class
    model = FlagMLLMModel(
        model_name_or_path='BAAI/BGE-VL-MLLM-S1',
        normalize_embeddings=True,
        use_fp16=True,
        devices="cuda:0",  # if you don't have a GPU, you can use "cpu"
        trust_remote_code=True,
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    # Example: Composed image retrieval task
    # Query with both image and text instruction
    query_embeddings = model.encode(
        sentences="Make the background dark, as if the camera has taken the photo at night",
        images="./assets/cir_query.png",
        q_or_c="q",
        task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
    )
    
    # Candidate images
    candidate_embeddings = model.encode(
        images=["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"],
        q_or_c="c"
    )
    
    # Calculate similarity scores
    scores = torch.matmul(
        torch.tensor(query_embeddings).unsqueeze(0),
        torch.tensor(candidate_embeddings).T
    )
    
    print("Similarity scores:")
    print(scores)


if __name__ == '__main__':
    test_multimodal_base()

