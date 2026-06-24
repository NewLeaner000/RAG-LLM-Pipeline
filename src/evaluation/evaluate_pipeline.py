import json
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from loguru import logger

from src.retrieval.base import BaseRetriever
from src.generation.generator import LLMGenerator
from src.evaluation.evaluator import ClassicalEvaluator
from src.evaluation.retrieval_metrics import evaluate_retrieval

class PipelineEvaluator:
    def __init__(self, retriever: BaseRetriever, generator: LLMGenerator = None):
        self.retriever = retriever
        self.generator = generator
        self.classical_evaluator = ClassicalEvaluator()
        
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """Evaluates the pipeline on a dataset."""
        logger.info(f"Starting evaluation on {len(dataset)} examples...")
        
        all_metrics = {
            "retrieval": {f"hit_rate@{k}": [] for k in k_values},
            "generation": {"exact_match": [], "f1": [], "rouge_l": []},
            "latency": {"retrieval_ms": [], "generation_ms": []}
        }
        
        # Add mrr@k to metrics
        for k in k_values:
            all_metrics["retrieval"][f"mrr@{k}"] = []
            
        results_log = []
        
        for item in tqdm(dataset, desc="Evaluating RAG pipeline"):
            query = item["question"]
            ground_truth_answer = item["answer"]
            ground_truth_context = item["context"]
            is_answerable = item.get("is_answerable", True)
            
            # --- Retrieval ---
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(query, top_k=max(k_values))
            retrieval_docs_obj = [res.document for res in retrieved_docs]
            retrieval_time = (time.time() - start_time) * 1000
            
            retrieval_scores = evaluate_retrieval(retrieval_docs_obj, ground_truth_context, k_values)
            
            for k, v in retrieval_scores.items():
                all_metrics["retrieval"][k].append(v)
            all_metrics["latency"]["retrieval_ms"].append(retrieval_time)
            
            # --- Generation ---
            generation_scores = {"exact_match": 0.0, "f1": 0.0, "rouge_l": 0.0}
            generated_answer = ""
            generation_time = 0.0
            
            if self.generator:
                start_time = time.time()
                # Combine retrieved contexts
                context_str = "\n\n".join([doc.page_content for doc in retrieval_docs_obj])
                generated_answer = self.generator.generate(query, context_str)
                generation_time = (time.time() - start_time) * 1000
                
                # We only penalize generation if there is an answer to match against.
                if is_answerable and ground_truth_answer:
                    generation_scores = self.classical_evaluator.evaluate_generation(generated_answer, ground_truth_answer)
                    for k, v in generation_scores.items():
                        all_metrics["generation"][k].append(v)
                all_metrics["latency"]["generation_ms"].append(generation_time)
                
            # Log individual result
            results_log.append({
                "id": item.get("id"),
                "question": query,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": generated_answer,
                "retrieval_metrics": retrieval_scores,
                "generation_metrics": generation_scores,
                "latency_ms": {
                    "retrieval": retrieval_time,
                    "generation": generation_time
                }
            })
            
        # Aggregate metrics
        agg_metrics = {"retrieval": {}, "generation": {}, "latency": {}}
        for category in all_metrics:
            for metric, values in all_metrics[category].items():
                if values:
                    agg_metrics[category][metric] = sum(values) / len(values)
                else:
                    agg_metrics[category][metric] = 0.0
                    
        logger.info(f"Evaluation complete. Hit Rate@5: {agg_metrics['retrieval'].get('hit_rate@5', 0):.4f}")
        if self.generator:
            logger.info(f"Average F1: {agg_metrics['generation'].get('f1', 0):.4f}")
            
        return {
            "aggregate_metrics": agg_metrics,
            "details": results_log
        }
