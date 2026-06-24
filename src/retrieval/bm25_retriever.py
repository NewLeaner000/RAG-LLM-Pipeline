import pickle
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from loguru import logger

from src.retrieval.base import BaseRetriever, RetrievalResult

class BM25RetrieverWrapper(BaseRetriever):
    """Wrapper for BM25 keyword-based retrieval."""
    
    def __init__(self, documents: Optional[List[Document]] = None, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        
        if persist_path and Path(persist_path).exists():
            logger.info(f"Loading BM25 index from {persist_path}")
            with open(persist_path, "rb") as f:
                self.retriever = pickle.load(f)
        elif documents:
            logger.info(f"Building new BM25 index with {len(documents)} documents")
            self.retriever = BM25Retriever.from_documents(documents)
            if persist_path:
                logger.info(f"Saving BM25 index to {persist_path}")
                Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
                with open(persist_path, "wb") as f:
                    pickle.dump(self.retriever, f)
        else:
            raise ValueError("Must provide either documents to build index or a valid persist_path to load.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        logger.debug(f"BM25 retrieval for query: '{query}' (top_k={top_k})")
        
        try:
            # Tokenize query using the same preprocess method
            tokenized_query = self.retriever.preprocess_func(query)
            
            # Get scores for all documents
            doc_scores = self.retriever.vectorizer.get_scores(tokenized_query)
            
            # Pair docs with scores
            docs_with_scores = list(zip(self.retriever.docs, doc_scores))
            
            # Sort by score descending
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            top_docs = docs_with_scores[:top_k]
            
            retrieval_results = []
            for rank, (doc, score) in enumerate(top_docs, 1):
                result = RetrievalResult(
                    document=doc,
                    score=float(score),
                    rank=rank
                )
                retrieval_results.append(result)
                
            return retrieval_results
            
        except Exception as e:
            logger.warning(f"Failed to get exact BM25 scores: {e}. Falling back to default retrieval.")
            docs = self.retriever.invoke(query)
            retrieval_results = []
            for rank, doc in enumerate(docs[:top_k], 1):
                # Pseudo score
                result = RetrievalResult(
                    document=doc,
                    score=1.0 / rank,
                    rank=rank
                )
                retrieval_results.append(result)
            return retrieval_results

if __name__ == "__main__":
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog", metadata={"id": 1}),
        Document(page_content="A quick brown dog outpaces a fast fox", metadata={"id": 2}),
        Document(page_content="Something completely different about space", metadata={"id": 3})
    ]
    
    # Build
    bm25 = BM25RetrieverWrapper(documents=docs, persist_path="chroma_db/test_bm25.pkl")
    
    # Query
    results = bm25.retrieve("quick brown fox", top_k=2)
    for r in results:
        print(f"Rank {r.rank}: Score {r.score:.4f} | {r.document.page_content}")
