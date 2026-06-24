from typing import List
from langchain_chroma import Chroma
from loguru import logger

from src.retrieval.base import BaseRetriever, RetrievalResult

class DenseRetrieverWrapper(BaseRetriever):
    """Wrapper for ChromaDB dense vector retrieval."""
    
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        logger.debug(f"Dense retrieval for query: '{query}' (top_k={top_k})")
        
        # Chroma's similarity_search_with_score returns (Document, distance)
        # We need to convert distance to a similarity score.
        # Cosine distance to similarity: similarity = 1 - distance
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        retrieval_results = []
        for rank, (doc, distance) in enumerate(results_with_scores, 1):
            # Convert distance to a pseudo-similarity score (higher is better)
            # Depending on Chroma's metric (usually L2 or cosine), we invert it.
            # We'll use 1 / (1 + distance) to ensure it's bounded [0, 1] and higher is better.
            similarity_score = 1.0 / (1.0 + distance)
            
            result = RetrievalResult(
                document=doc,
                score=similarity_score,
                rank=rank
            )
            retrieval_results.append(result)
            
        return retrieval_results

if __name__ == "__main__":
    from src.embedding.embedder import create_embeddings, build_vector_store
    
    # Quick test assuming Task 4 vector store exists
    embed_model = create_embeddings("all-MiniLM-L6-v2")
    vstore = build_vector_store([], embed_model, "test_fixed_256_all-MiniLM-L6-v2")
    
    retriever = DenseRetrieverWrapper(vstore)
    results = retriever.retrieve("testing query", top_k=2)
    
    for r in results:
        print(f"Rank {r.rank}: Score {r.score:.4f} | {r.document.page_content[:50]}...")
