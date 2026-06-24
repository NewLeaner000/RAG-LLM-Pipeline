from enum import Enum
from typing import Dict, Any

class QueryType(Enum):
    FACTOID = "FACTOID"
    MULTI_HOP = "MULTI_HOP"
    UNANSWERABLE_RISK = "UNANSWERABLE_RISK"

class QueryRouter:
    @staticmethod
    def classify(query: str) -> QueryType:
        """
        Classifies the incoming query into one of three categories based on heuristic rules.
        """
        q = query.lower().strip()
        
        # 1. Check for Multi-Hop triggers
        multi_hop_keywords = ['compare', 'difference', 'vs', 'versus', 'why', 'how', 'relationship', 'connect', 'both']
        if any(kw in q for kw in multi_hop_keywords) or len(q.split()) > 15:
            return QueryType.MULTI_HOP
            
        # 2. Check for Factoid triggers
        factoid_starts = ['who', 'what', 'when', 'where', 'which', 'name']
        if any(q.startswith(start) for start in factoid_starts):
            # Short queries starting with factoid words are safe bets
            if len(q.split()) < 12:
                return QueryType.FACTOID
                
        # 3. Default fallback
        return QueryType.UNANSWERABLE_RISK
        
    @staticmethod
    def get_retrieval_config(query_type: QueryType) -> Dict[str, Any]:
        """
        Returns the optimal retrieval configuration for a given query type.
        """
        if query_type == QueryType.FACTOID:
            # Factoids benefit from exact keyword matching
            return {
                "strategy": "bm25",
                "use_reranker": False,
                "top_k": 3
            }
        elif query_type == QueryType.MULTI_HOP:
            # Multi-hop needs broad semantic search + cross-encoding
            return {
                "strategy": "hybrid",
                "use_reranker": True,
                "top_k": 5,
                "reranker_top_n": 3
            }
        else: # UNANSWERABLE_RISK
            # Dense vectors are generally safer for vague queries
            return {
                "strategy": "dense",
                "use_reranker": True,
                "top_k": 5,
                "reranker_top_n": 2
            }
