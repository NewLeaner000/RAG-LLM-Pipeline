import json
import time
import random
from tqdm import tqdm
from src.pipeline import RAGPipeline
from src.evaluation.metrics import f1_score

def eval_router():
    print("Loading pipeline and dataset...")
    pipeline = RAGPipeline("config/config.yaml", strategy="hybrid")
    
    with open("data/processed/eval_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    random.seed(42)
    sample = random.sample(data, 50)
    
    # 1. No Router
    print("\nEvaluating WITHOUT Router (Always Hybrid + Reranker)...")
    no_router_f1 = []
    no_router_time = []
    
    for item in tqdm(sample):
        q = item["question"]
        ans = item.get("answer", "")
        
        start = time.time()
        res = pipeline.query(q, top_k=5, use_router=False)
        latency = (time.time() - start) * 1000
        no_router_time.append(latency)
        
        if item.get("is_answerable", True):
            no_router_f1.append(f1_score(res.answer, ans))
            
    # 2. With Router
    print("\nEvaluating WITH Router (Dynamic Routing)...")
    router_f1 = []
    router_time = []
    
    for item in tqdm(sample):
        q = item["question"]
        ans = item.get("answer", "")
        
        start = time.time()
        res = pipeline.query(q, top_k=5, use_router=True)
        latency = (time.time() - start) * 1000
        router_time.append(latency)
        
        if item.get("is_answerable", True):
            router_f1.append(f1_score(res.answer, ans))
            
    print(f"\n" + "="*40)
    print(f"          ROUTER BENCHMARK RESULTS")
    print(f"="*40)
    print(f"WITHOUT ROUTER (Static Hybrid):")
    print(f"  - F1 Score : {sum(no_router_f1)/len(no_router_f1):.3f}")
    print(f"  - Latency  : {sum(no_router_time)/len(no_router_time):.1f} ms / query")
    print(f"-"*40)
    print(f"WITH ROUTER (Dynamic Strategies):")
    print(f"  - F1 Score : {sum(router_f1)/len(router_f1):.3f}")
    print(f"  - Latency  : {sum(router_time)/len(router_time):.1f} ms / query")
    print(f"="*40)

if __name__ == "__main__":
    eval_router()
