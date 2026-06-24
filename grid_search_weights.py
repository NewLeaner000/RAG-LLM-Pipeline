import json
import numpy as np
from tqdm import tqdm
from src.pipeline import RAGPipeline
from src.evaluation.metrics import normalize_answer

def grid_search_hybrid_weights(num_questions=50):
    print(f"Running grid search for Hybrid weights with {num_questions} questions...")
    pipeline = RAGPipeline(strategy="hybrid")
    
    with open("data/processed/eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)[:num_questions]
        
    alphas = np.linspace(0.0, 1.0, 11)
    
    print(f"{'Dense Weight (Alpha)':<20} | {'BM25 Weight (1-Alpha)':<20} | {'Context Hit Rate (%)':<20}")
    print("-" * 65)
    
    for alpha in alphas:
        bm25_weight = 1.0 - alpha
        # Update pipeline retriever weights
        pipeline.retriever.weights = [alpha, bm25_weight]
        
        hits = 0
        total_answerable = 0
        
        for sample in dataset:
            query = sample["question"]
            gt = sample.get("answer", "")
            if isinstance(gt, list): gt = gt[0] if gt else ""
            
            if not gt:
                continue # Only check answerable questions for hit rate
                
            total_answerable += 1
            
            # Retrieve documents
            retrieved_results = pipeline.retriever.retrieve(query, top_k=5)
            context_str = "\n\n".join([r.document.page_content for r in retrieved_results])
            
            # Simple hit check: is the normalized ground truth answer in the normalized context?
            # Or just lowercase
            if gt.lower() in context_str.lower():
                hits += 1
                
        hit_rate = hits / total_answerable if total_answerable > 0 else 0
        
        print(f"{alpha:<20.1f} | {bm25_weight:<20.1f} | {hit_rate:<20.2%}")

if __name__ == "__main__":
    grid_search_hybrid_weights(50)
