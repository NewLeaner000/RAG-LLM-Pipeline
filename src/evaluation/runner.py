import json
import yaml
from pathlib import Path
from loguru import logger
import argparse

from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import create_embeddings, build_vector_store
from src.retrieval.dense_retriever import DenseRetrieverWrapper
from src.retrieval.bm25_retriever import BM25RetrieverWrapper
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.evaluate_pipeline import PipelineEvaluator

def run_experiments(num_samples: int = 10):
    logger.info(f"Starting Experiment Runner on {num_samples} samples...")
    
    # 1. Load Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Load Evaluation Dataset and Corpus
    with open("data/processed/eval_dataset.json", "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)[:num_samples] # Run on subset for speed
        
    with open("data/processed/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    results_summary = []
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    
    # Test 1 chunking strategy + 1 embedding model + 3 retrieval methods for a quick log
    chunk_strategy = config["chunking"]["strategies"][0] # fixed_256
    embed_config = config["embedding"]["models"][0] # all-MiniLM-L6-v2
    
    logger.info(f"Chunking corpus with {chunk_strategy['name']}...")
    chunks = chunk_documents(corpus, chunk_strategy)
    
    logger.info(f"Creating embeddings with {embed_config['name']}...")
    embed_model = create_embeddings(embed_config["name"])
    
    collection_name = f"exp_{chunk_strategy['name']}_{embed_config['name'].split('/')[-1]}"
    vector_store = build_vector_store(chunks, embed_model, collection_name)
    
    dense_retriever = DenseRetrieverWrapper(vector_store)
    bm25_retriever = BM25RetrieverWrapper(documents=chunks)
    hybrid_retriever = HybridRetriever(
        retrievers=[dense_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    
    retrievers = {
        "Dense": dense_retriever,
        "BM25": bm25_retriever,
        "Hybrid": hybrid_retriever
    }
    
    from src.generation.generator import LLMGenerator
    llm_model_name = config["generation"]["primary_model"]
    logger.info(f"Initializing LLM Generator with model: {llm_model_name}")
    generator = LLMGenerator(model_name=llm_model_name)
    
    for r_name, retriever in retrievers.items():
        logger.info(f"--- Evaluating Strategy: {r_name} ---")
        # Enable LLM generator to evaluate F1 and ROUGE-L
        evaluator = PipelineEvaluator(retriever=retriever, generator=generator) 
        
        results = evaluator.evaluate_dataset(eval_dataset, k_values=[1, 3, 5])
        agg_metrics = results["aggregate_metrics"]
        
        # Save detailed logs
        out_file = f"results/metrics/{chunk_strategy['name']}_{r_name.lower()}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        summary = {
            "Strategy": r_name,
            "HitRate@1": agg_metrics["retrieval"]["hit_rate@1"],
            "HitRate@5": agg_metrics["retrieval"]["hit_rate@5"],
            "MRR@5": agg_metrics["retrieval"]["mrr@5"],
            "Retrieval_Latency_ms": agg_metrics["latency"]["retrieval_ms"]
        }
        results_summary.append(summary)
        
    logger.info("=== EXPERIMENT RESULTS ===")
    for s in results_summary:
        logger.info(f"{s['Strategy']:>10} | HR@1: {s['HitRate@1']:.3f} | HR@5: {s['HitRate@5']:.3f} | MRR@5: {s['MRR@5']:.3f} | Latency: {s['Retrieval_Latency_ms']:.1f}ms")
        
if __name__ == "__main__":
    run_experiments(num_samples=10)
