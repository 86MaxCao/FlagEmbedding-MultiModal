"""
Example script for using FlagAutoReranker with multimodal reranker (jina-reranker-m0).
"""

from FlagEmbedding import FlagAutoReranker


def test_text_to_image_reranking():
    """Test text query with image documents."""
    print("=" * 70)
    print("Test 1: Text Query -> Image Documents Reranking")
    print("=" * 70)
    
    # Load multimodal reranker
    reranker = FlagAutoReranker.from_finetuned(
        'jinaai/jina-reranker-m0',
        devices="cuda:0",  # or "cpu" if no GPU
    )
    
    # Text query
    query = "slm markdown"
    
    # Image document URLs (can be local paths too)
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png",
        "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp"
    ]
    
    # Construct sentence pairs
    image_pairs = [[query, doc] for doc in documents]
    
    # Compute scores with doc_type='image'
    scores = reranker.compute_score(
        image_pairs,
        max_length=2048,
        query_type='text',
        doc_type='image'
    )
    
    print(f"\nQuery: {query}")
    print(f"\nDocument scores:")
    for doc, score in zip(documents, scores):
        print(f"  {doc[-50:]}: {score:.4f}")
    print(f"\nTop document: {documents[scores.index(max(scores))]}")


def test_text_to_text_reranking():
    """Test text query with text documents."""
    print("\n" + "=" * 70)
    print("Test 2: Text Query -> Text Documents Reranking")
    print("=" * 70)
    
    reranker = FlagAutoReranker.from_finetuned(
        'jinaai/jina-reranker-m0',
        devices="cuda:0",
    )
    
    query = "What is machine learning?"
    
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Python is a popular programming language used for web development.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "The weather is nice today.",
    ]
    
    pairs = [[query, doc] for doc in documents]
    
    scores = reranker.compute_score(
        pairs,
        query_type='text',
        doc_type='text'
    )
    
    print(f"\nQuery: {query}")
    print(f"\nDocument scores:")
    for doc, score in zip(documents, scores):
        print(f"  {doc[:60]}...: {score:.4f}")
    
    # Sort by score
    sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    print(f"\nTop ranked document: {sorted_docs[0][0]}")


def test_image_to_text_reranking():
    """Test image query with text documents."""
    print("\n" + "=" * 70)
    print("Test 3: Image Query -> Text Documents Reranking")
    print("=" * 70)
    
    reranker = FlagAutoReranker.from_finetuned(
        'jinaai/jina-reranker-m0',
        devices="cuda:0",
    )
    
    # Image query
    query_image = "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
    
    # Text documents
    documents = [
        "This is a research paper about neural networks and deep learning.",
        "A cooking recipe for chocolate cake.",
        "Academic article discussing machine learning models.",
        "Travel guide to Paris, France.",
    ]
    
    pairs = [[query_image, doc] for doc in documents]
    
    scores = reranker.compute_score(
        pairs,
        query_type='image',
        doc_type='text'
    )
    
    print(f"\nQuery: Image ({query_image[-40:]})")
    print(f"\nDocument scores:")
    for doc, score in zip(documents, scores):
        print(f"  {doc[:60]}...: {score:.4f}")


if __name__ == '__main__':
    test_text_to_image_reranking()
    test_text_to_text_reranking()
    test_image_to_text_reranking()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

