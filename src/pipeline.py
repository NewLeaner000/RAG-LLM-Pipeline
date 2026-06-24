import time
from typing import List, Dict, Any, Optional
import yaml
from dataclasses import dataclass
from langchain_core.documents import Document
from loguru import logger

from src.retrieval.dense_retriever import DenseRetrieverWrapper
from src.retrieval.bm25_retriever import BM25RetrieverWrapper
from src.retrieval.hybrid_retriever import HybridRetriever
from src.embedding.embedder import build_vector_store
from src.generation.generator import LLMGenerator
from langchain_ollama import OllamaEmbeddings
from src.retrieval.reranker import CrossEncoderReranker

@dataclass
class RAGResult:
    answer: str
    retrieved_docs: List[Document]
    retrieval_scores: List[float]
    generation_latency_ms: float
    total_latency_ms: float

class RAGPipeline:
    def __init__(self, config_path: str = "config/config.yaml", strategy: str = "hybrid"):
        """
        Initializes the entire RAG pipeline (Embedding, Retrieval, Generation).
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.strategy = strategy
        self._init_components()

    def _init_components(self):
        # 1. Embedding
        # Use HuggingFaceEmbeddings for stability with large batches
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # We need a dummy corpus to initialize BM25 if not provided
        # In a real app, the orchestrator loads the vector store from disk.
        from pathlib import Path
        import json
        import os
        
        corpus_path = "data/processed/corpus.json"
        if not os.path.exists(corpus_path):
            corpus_path = "../data/processed/corpus.json"
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
            
        from src.ingestion.chunker import chunk_documents
        
        chunk_config = self.config.get("chunking", {}).get("strategies", [{"name": "recursive_512", "chunk_size": 512, "chunk_overlap": 100}])[0]
        docs = chunk_documents(corpus_data, chunk_config)

        # 2. Vector Store
        self.vectorstore = build_vector_store(docs, self.embedder, "local_all_minilm_chunked")
        
        # 3. Retriever setup
        self.dense_retriever = DenseRetrieverWrapper(self.vectorstore)
        self.bm25_retriever = BM25RetrieverWrapper(documents=docs)
        self.hybrid_retriever = HybridRetriever([self.dense_retriever, self.bm25_retriever], weights=[0.5, 0.5])
        
        if self.strategy == "dense":
            self.default_retriever = self.dense_retriever
        elif self.strategy == "bm25":
            self.default_retriever = self.bm25_retriever
        else: # hybrid
            self.default_retriever = self.hybrid_retriever
            
        # 4. Reranker setup
        reranker_config = self.config.get("reranker", {})
        # ENABLE FOR ALL: Test every strategy WITH the reranker
        self.use_reranker = True
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name=reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
            self.reranker_top_n = reranker_config.get("top_n", 3)
            
        # 5. Generator setup
        self.generator = LLMGenerator(model_name=self.config["generation"]["primary_model"])
        
    def _correct_query_typos(self, raw_query: str) -> str:
        prompt = (
            "You are a professional spelling and grammar corrector for search queries. "
            "Your task is to correct any typos or misspellings in the user's query while preserving the exact original meaning and language. "
            "Do not answer the question. Do not add any conversational text. Output ONLY the corrected query.\n\n"
            f"Original Query: {raw_query}\nCorrected Query:"
        )
        try:
            from langchain_core.messages import HumanMessage
            response = self.generator.llm.invoke(prompt)
            # Ollama LLM returns a string directly
            corrected = str(response).strip().strip('"').strip("'")
            # Fallback if LLM generates too much text
            if len(corrected) > len(raw_query) * 3 or '\n' in corrected:
                return raw_query
            if corrected.lower() != raw_query.lower():
                logger.info(f"Typo corrected: '{raw_query}' -> '{corrected}'")
            return corrected
        except Exception as e:
            logger.warning(f"Typo correction failed: {e}")
            return raw_query

    def query(self, question: str, top_k: int = 5, use_router: bool = False, correct_typos: bool = False) -> RAGResult:
        start_time = time.time()
        
        if correct_typos:
            question = self._correct_query_typos(question)
        
        # Determine strategy
        active_retriever = self.default_retriever
        active_use_reranker = self.use_reranker
        active_top_k = top_k
        active_reranker_top_n = self.reranker_top_n
        
        if use_router:
            from src.routing.query_router import QueryRouter
            q_type = QueryRouter.classify(question)
            config = QueryRouter.get_retrieval_config(q_type)
            
            strat = config["strategy"]
            if strat == "bm25":
                active_retriever = self.bm25_retriever
            elif strat == "dense":
                active_retriever = self.dense_retriever
            else:
                active_retriever = self.hybrid_retriever
                
            active_use_reranker = config["use_reranker"]
            active_top_k = config["top_k"]
            active_reranker_top_n = config["reranker_top_n"]
            from loguru import logger
            logger.info(f"Router classified query as {q_type.name} -> Using {strat.upper()} strategy")
        
        # Retrieval
        # Retrieve more if reranking is enabled to give it a good pool
        initial_k = active_top_k * 2 if active_use_reranker else active_top_k
        retrieved_results = active_retriever.retrieve(question, top_k=initial_k)
        
        # Reranking
        if active_use_reranker and retrieved_results:
            retrieved_results = self.reranker.rerank(question, retrieved_results, top_n=min(active_top_k, active_reranker_top_n))
            
        retrieved_docs = [r.document for r in retrieved_results]
        retrieval_scores = [r.score for r in retrieved_results]
        
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Layer 2 Protection: Retrieval Threshold
        # Different strategies have different score scales. 
        # For production, tune this threshold specifically for Dense/BM25/Hybrid.
        min_score_threshold = 0.0  # Set to > 0 to activate strict filtering
        max_score = max(retrieval_scores) if retrieval_scores else 0.0
        
        if not retrieved_docs or (min_score_threshold > 0 and max_score < min_score_threshold):
            # Bypass LLM generation if we are not confident in the retrieved documents
            answer = "I don't have information."
            gen_time = 0.0
        else:
            # Generation
            gen_start = time.time()
            answer = self.generator.generate(question, context_str)
            gen_time = (time.time() - gen_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return RAGResult(
            answer=answer,
            retrieved_docs=retrieved_docs,
            retrieval_scores=retrieval_scores,
            generation_latency_ms=gen_time,
            total_latency_ms=total_time
        )
