import json
import numpy as np
from tqdm import tqdm
from src.pipeline import RAGPipeline
from src.evaluation.metrics import f1_score

def grid_search(strategy="hybrid", num_questions=20):
    print(f"Running grid search for strategy: {strategy} with {num_questions} questions...")
    pipeline = RAGPipeline(strategy=strategy)
    
    with open("data/processed/eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)[:num_questions]
        
    results = []
    
    print("Gathering raw predictions and retrieval scores...")
    for sample in tqdm(dataset):
        query = sample["question"]
        gt = sample.get("answer", "")
        if isinstance(gt, list): gt = gt[0] if gt else ""
        
        rag_res = pipeline.query(query, top_k=5)
        # For BM25 and Hybrid, higher score is better.
        max_score = max(rag_res.retrieval_scores) if rag_res.retrieval_scores else 0.0
        
        results.append({
            "query": query,
            "gt": gt,
            "raw_answer": rag_res.answer,
            "max_score": max_score,
            "is_unanswerable": (gt == "")
        })
        
    print("\n" + "="*80)
    print(" GRID SEARCH RESULTS ".center(80, "="))
    print("="*80)
    
    # Define threshold range based on strategy
    if strategy == "hybrid":
        # RRF scores are usually very small (e.g., 1/60 = 0.016)
        thresholds = np.linspace(0.0, 0.03, 15)
    elif strategy == "bm25":
        # BM25 scores can be > 1.0
        thresholds = np.linspace(0.0, 10.0, 15)
    else:
        thresholds = np.linspace(0.0, 1.0, 15)
        
    print(f"{'Threshold':<12} | {'Hallucination Rate (Empty GT)':<32} | {'F1 Score (Valid GT)':<20}")
    print("-" * 80)
    
    for t in thresholds:
        hallucinations = 0
        total_empty = 0
        f1_scores = []
        
        for r in results:
            final_ans = r["raw_answer"]
            
            # Simulate threshold cutoff (higher is better)
            if r["max_score"] < t:
                final_ans = "I don't have information."
                
            if r["is_unanswerable"]:
                total_empty += 1
                # If model didn't say "don't have information" or "don't know", it hallucinated
                ans_lower = final_ans.lower()
                if "don't have information" not in ans_lower and "don't know" not in ans_lower:
                    hallucinations += 1
            else:
                f1_scores.append(f1_score(final_ans, r["gt"]))
                
        hal_rate = hallucinations / total_empty if total_empty > 0 else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        print(f"{t:<12.4f} | {hal_rate:<32.2%} | {avg_f1:<20.4f}")

if __name__ == "__main__":
    grid_search("hybrid", 20)
