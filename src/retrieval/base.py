from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class RetrievalResult:
    """Standardized output format for all retrievers."""
    document: Document
    score: float
    rank: int

class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieves top_k documents for a given query.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """
        pass
