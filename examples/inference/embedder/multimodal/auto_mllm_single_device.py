import os
import torch
from FlagEmbedding import FlagAutoModel


def test_multimodal_single_device():
    """
    Test multimodal MLLM model with images and text on a single device.
    """
    # Load the multimodal model
    model = FlagAutoModel.from_finetuned(
        'BAAI/BGE-VL-MLLM-S1',
        devices="cuda:0",  # if you don't have a GPU, you can use "cpu"
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
    

def test_multimodal_text_only():
    """
    Test multimodal MLLM model with text only (no images).
    """
    model = FlagAutoModel.from_finetuned(
        'BAAI/BGE-VL-MLLM-S1',
        devices="cuda:0",
    )
    
    # Query with text only
    query_embeddings = model.encode(
        sentences="What is the capital of France?",
        q_or_c="q",
        task_instruction="Given a question, retrieve relevant passages that answer the question: "
    )
    
    # Candidate texts
    candidate_embeddings = model.encode(
        sentences=["Paris is the capital of France.", "London is the capital of England."],
        q_or_c="c"
    )
    
    # Calculate similarity scores
    scores = torch.matmul(
        torch.tensor(query_embeddings).unsqueeze(0),
        torch.tensor(candidate_embeddings).T
    )

    print("Text-only similarity scores:")
    print(scores)


def test_multimodal_image_only():
    """
    Test multimodal MLLM model with images only (no text).
    """
    model = FlagAutoModel.from_finetuned(
        'BAAI/BGE-VL-MLLM-S1',
        devices="cuda:0",
    )

    # Query with image only
    query_embeddings = model.encode(
        images="./assets/cir_query.png",
        q_or_c="q"
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

    print("Image-only similarity scores:")
    print(scores)


if __name__ == '__main__':
    print("=" * 50)
    print("Test 1: Multimodal (Image + Text)")
    print("=" * 50)
    test_multimodal_single_device()

    print("\n" + "=" * 50)
    print("Test 2: Text Only")
    print("=" * 50)
    test_multimodal_text_only()

    print("\n" + "=" * 50)
    print("Test 3: Image Only")
    print("=" * 50)
    test_multimodal_image_only()
