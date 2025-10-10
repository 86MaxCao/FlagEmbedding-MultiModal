"""
Example script for using MultimodalReranker directly.
"""

from FlagEmbedding import MultimodalReranker


def test_multimodal_reranker():
    """Test multimodal reranker with various modality combinations."""
    
    # Initialize reranker
    reranker = MultimodalReranker(
        model_name_or_path='jinaai/jina-reranker-m0',
        use_fp16=True,
        devices="cuda:0",
        batch_size=4,
        max_length=2048,
        normalize=True,
    )
    
    print("=" * 70)
    print("Multimodal Reranker Test")
    print("=" * 70)
    
    # Example 1: Text query -> Image documents
    print("\n1. Text Query -> Image Documents")
    print("-" * 70)
    
    query = "research paper about AI"
    image_docs = [
        "assets/paper-11.png",
        "assets/wired-preview.png",
    ]
    
    pairs = [[query, doc] for doc in image_docs]
    scores = reranker.compute_score(
        pairs,
        query_type='text',
        doc_type='image'
    )
    
    print(f"Query: {query}")
    for doc, score in zip(image_docs, scores):
        print(f"  Score: {score:.4f} - {doc[-50:]}")
    
    # Example 2: Image query -> Text documents
    print("\n2. Image Query -> Text Documents")
    print("-" * 70)
    
    image_query = "assets/paper-11.png"
    text_docs = [
        "This paper discusses deep learning and neural networks.",
        "A guide to cooking Italian pasta.",
        "Research on natural language processing.",
    ]
    
    pairs = [[image_query, doc] for doc in text_docs]
    scores = reranker.compute_score(
        pairs,
        query_type='image',
        doc_type='text'
    )
    
    print(f"Query: Image ({image_query[-40:]})")
    for doc, score in zip(text_docs, scores):
        print(f"  Score: {score:.4f} - {doc[:50]}...")
    
    # Example 3: Text query -> Text documents (baseline)
    print("\n3. Text Query -> Text Documents")
    print("-" * 70)
    
    query = "What is deep learning?"
    text_docs = [
        "Deep learning is a subset of machine learning using neural networks.",
        "Python is a programming language.",
        "Neural networks have multiple layers for processing data.",
    ]
    
    pairs = [[query, doc] for doc in text_docs]
    scores = reranker.compute_score(
        pairs,
        query_type='text',
        doc_type='text'
    )
    
    print(f"Query: {query}")
    for doc, score in zip(text_docs, scores):
        print(f"  Score: {score:.4f} - {doc[:50]}...")


if __name__ == '__main__':
    test_multimodal_reranker()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

