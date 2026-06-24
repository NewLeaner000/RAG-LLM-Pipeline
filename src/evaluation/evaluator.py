import os
import json
import time
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger

from src.pipeline import RAGPipeline
from src.evaluation.metrics import exact_match, f1_score, rouge_l_score, hit_rate_at_k, mrr_at_k

class Evaluator:
    def __init__(self, pipeline: RAGPipeline, eval_dataset: List[Dict]):
        self.pipeline = pipeline
        self.eval_dataset = eval_dataset

    def evaluate(self, num_questions: int = 50, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the evaluation over the subset of questions.
        Returns aggregate metrics and raw per-question results.
        """
        dataset = self.eval_dataset[:num_questions]
        
        all_results = []
        metrics_history = {
            "answerable": {
                "exact_match": [],
                "f1_score": [],
                "rouge_l": [],
                f"hit_rate@{top_k}": [],
                f"mrr@{top_k}": []
            },
            "unanswerable": {
                "rejection_accuracy": []
            },
            "latency_ms": []
        }
        
        start_time = time.time()
        
        for sample in tqdm(dataset, desc=f"Evaluating {self.pipeline.strategy}"):
            question = sample["question"]
            ground_truth = sample.get("answer", "")
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ""
            
            # Run pipeline
            try:
                res = self.pipeline.query(question, top_k=top_k)
                
                metrics_history["latency_ms"].append(res.total_latency_ms)
                
                is_answerable = bool(ground_truth.strip())
                metrics_dict = {"latency_ms": res.total_latency_ms}
                
                if is_answerable:
                    em = exact_match(res.answer, ground_truth)
                    f1 = f1_score(res.answer, ground_truth)
                    try:
                        rl = rouge_l_score(res.answer, ground_truth)
                    except ImportError:
                        rl = 0.0 # fallback if rouge-score not installed
                    
                    ground_truth_context = sample.get("context", ground_truth)
                    hr = hit_rate_at_k(res.retrieved_docs, ground_truth_context, top_k)
                    mrr = mrr_at_k(res.retrieved_docs, ground_truth_context, top_k)
                    
                    metrics_history["answerable"]["exact_match"].append(em)
                    metrics_history["answerable"]["f1_score"].append(f1)
                    metrics_history["answerable"]["rouge_l"].append(rl)
                    metrics_history["answerable"][f"hit_rate@{top_k}"].append(hr)
                    metrics_history["answerable"][f"mrr@{top_k}"].append(mrr)
                    
                    metrics_dict.update({"em": em, "f1": f1, "rouge_l": rl, "hit_rate": hr, "mrr": mrr})
                else:
                    # For unanswerable queries, check if the model correctly rejected
                    rejected = 1.0 if "i don't have information" in res.answer.lower() else 0.0
                    metrics_history["unanswerable"]["rejection_accuracy"].append(rejected)
                    metrics_dict.update({"rejection_accuracy": rejected})
                
                all_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": res.answer,
                    "type": "answerable" if is_answerable else "unanswerable",
                    "metrics": metrics_dict
                })
            except Exception as e:
                logger.error(f"Error evaluating question: {question} -> {e}")
                
        total_time = time.time() - start_time
        
        # Aggregate metrics
        aggregated = {
            "answerable": {},
            "unanswerable": {},
            "latency_ms": {
                "mean": float(np.mean(metrics_history["latency_ms"])) if metrics_history["latency_ms"] else 0.0,
                "std": float(np.std(metrics_history["latency_ms"])) if metrics_history["latency_ms"] else 0.0
            }
        }
        
        for k, v in metrics_history["answerable"].items():
            aggregated["answerable"][k] = {
                "mean": float(np.mean(v)) if v else 0.0,
                "std": float(np.std(v)) if v else 0.0
            }
            
        for k, v in metrics_history["unanswerable"].items():
            aggregated["unanswerable"][k] = {
                "mean": float(np.mean(v)) if v else 0.0,
                "std": float(np.std(v)) if v else 0.0
            }
            
        return {
            "config": self.pipeline.strategy,
            "total_time_seconds": total_time,
            "metrics": aggregated,
            "per_question_results": all_results
        }

def run_experiment_matrix(config_path: str, eval_dataset_path: str, output_dir: str = "results/metrics", num_questions: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)
        
    strategies = ["dense", "bm25", "hybrid"]
    
    for strategy in strategies:
        logger.info(f"Initializing pipeline with strategy: {strategy}")
        pipeline = RAGPipeline(config_path=config_path, strategy=strategy)
        evaluator = Evaluator(pipeline, eval_dataset)
        
        results = evaluator.evaluate(num_questions=num_questions)
        
        out_file = os.path.join(output_dir, f"exp_{strategy}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved results for {strategy} -> {out_file}")
        ans_metrics = results['metrics']['answerable']
        f1_mean = ans_metrics['f1_score']['mean'] if 'f1_score' in ans_metrics else 0.0
        hr_mean = ans_metrics.get('hit_rate@5', {}).get('mean', 0.0)
        unans_metrics = results['metrics']['unanswerable']
        rej_mean = unans_metrics.get('rejection_accuracy', {}).get('mean', 0.0)
        
        logger.info(f"Answerable F1: {f1_mean:.3f} | Hit Rate: {hr_mean:.3f} || Unanswerable Rejection Acc: {rej_mean:.3f}")
