from typing import List, Dict
from loguru import logger
from langchain_core.documents import Document

from src.retrieval.base import BaseRetriever, RetrievalResult

class HybridRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float], rrf_k: int = 60):
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")
            
        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k
        
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        logger.debug(f"Hybrid retrieval for query: '{query}'")
        
        # Dictionary to store accumulated scores by document ID
        # Format: doc_id -> {"doc": Document, "score": float}
        fused_scores: Dict[str, Dict] = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            # Fetch more documents initially to ensure good fusion overlap
            results = retriever.retrieve(query, top_k=top_k * 2)
            
            for result in results:
                doc = result.document
                # Use source_id and chunk_index from metadata or page_content as fallback ID
                source_id = doc.metadata.get("source_id", "unknown")
                chunk_index = doc.metadata.get("chunk_index", 0)
                doc_id = f"{source_id}_{chunk_index}" if source_id != "unknown" else doc.page_content[:100]
                
                # RRF Formula: 1 / (rank + k)
                rrf_score = 1.0 / (result.rank + self.rrf_k)
                weighted_score = rrf_score * weight
                
                if doc_id in fused_scores:
                    fused_scores[doc_id]["score"] += weighted_score
                else:
                    fused_scores[doc_id] = {"doc": doc, "score": weighted_score}
                    
        # Sort by accumulated fused score
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Take top_k
        final_results = []
        for rank, item in enumerate(sorted_docs[:top_k], 1):
            result = RetrievalResult(
                document=item["doc"],
                score=item["score"],
                rank=rank
            )
            final_results.append(result)
            
        return final_results

if __name__ == "__main__":
    # Mock setup
    class MockRetriever(BaseRetriever):
        def __init__(self, doc_ids):
            self.doc_ids = doc_ids
            
        def retrieve(self, query: str, top_k: int = 5):
            results = []
            for rank, doc_id in enumerate(self.doc_ids[:top_k], 1):
                doc = Document(page_content=f"Content for {doc_id}", metadata={"source_id": doc_id, "chunk_index": 0})
                results.append(RetrievalResult(doc, 1.0/rank, rank))
            return results
            
    # r1 finds docs A, B, C. r2 finds docs B, C, D
    r1 = MockRetriever(["doc_A", "doc_B", "doc_C"])
    r2 = MockRetriever(["doc_B", "doc_D", "doc_A"])
    
    hybrid = HybridRetriever([r1, r2], weights=[0.5, 0.5])
    results = hybrid.retrieve("test", top_k=3)
    
    for r in results:
        print(f"Rank {r.rank}: Score {r.score:.4f} | {r.document.metadata['source_id']}")
