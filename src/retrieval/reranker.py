import os
import torch
from typing import List
from loguru import logger
from sentence_transformers import CrossEncoder
from src.retrieval.base import RetrievalResult

class CrossEncoderReranker:
    """
    Reranks a list of RetrievalResult objects using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading CrossEncoder model '{self.model_name}' on {self.device}...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        logger.info("CrossEncoder model loaded successfully.")

    def rerank(self, query: str, results: List[RetrievalResult], top_n: int = 5) -> List[RetrievalResult]:
        """
        Rerank a list of RetrievalResult objects based on cross-encoder similarity.
        
        Args:
            query: The user query
            results: List of RetrievalResult from an initial retriever
            top_n: Number of documents to return after reranking
            
        Returns:
            List of top_n reranked RetrievalResult objects.
        """
        if not results:
            return []
            
        # Create (query, text) pairs for the cross-encoder
        pairs = [[query, r.document.page_content] for r in results]
        
        # Compute scores
        scores = self.model.predict(pairs)
        
        # Attach new scores to results
        reranked_results = []
        for i, r in enumerate(results):
            # We create a new RetrievalResult to avoid mutating the original
            reranked_results.append(RetrievalResult(
                document=r.document,
                score=float(scores[i]),
                rank=0 # Will be updated after sorting
            ))
            
        # Sort by score in descending order (higher score = more relevant)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, r in enumerate(reranked_results):
            r.rank = i + 1
            
        return reranked_results[:top_n]
