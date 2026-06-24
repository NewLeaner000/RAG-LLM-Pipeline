import json
import random
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger

from src.pipeline import RAGPipeline
from src.generation.generator import LLMGenerator
from src.evaluation.metrics import f1_score, exact_match

def run_benchmark(num_questions=50):
    logger.info("Initializing RAG Pipeline (Hybrid Strategy with Reranker)...")
    # Initialize pipeline once to build/load vector store and reranker
    pipeline = RAGPipeline("config/config.yaml", strategy="hybrid")
    
    # We will benchmark these three models
    models_to_test = [
        "qwen2.5:latest",   # 7B Baseline
        "llama3.1:8b",      # 8B Equivalent Parameter Model
        "qwen2.5:1.5b"      # 1.5B Small Parameter Model
    ]
    
    # Load dataset
    dataset_path = "data/processed/eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Stratified sampling
    answerable = [d for d in data if d.get('is_answerable', True)]
    unanswerable = [d for d in data if not d.get('is_answerable', True)]
    
    random.seed(42)
    sample_size = min(num_questions // 2, len(unanswerable))
    eval_set = random.sample(answerable, num_questions - sample_size) + random.sample(unanswerable, sample_size)
    random.shuffle(eval_set)
    
    logger.info(f"Sampled {len(eval_set)} queries for benchmarking.")
    
    results_data = []
    
    for model_name in models_to_test:
        logger.info(f"--- Benchmarking Model: {model_name} ---")
        # Swap the generator model
        pipeline.generator = LLMGenerator(model_name=model_name)
        
        f1_scores = []
        em_scores = []
        latencies = []
        
        for item in tqdm(eval_set, desc=f"Evaluating {model_name}"):
            query = item["question"]
            ground_truth = item.get("answer", "")
            is_answerable = item.get("is_answerable", True)
            
            try:
                result = pipeline.query(query, top_k=5)
                pred = result.answer
                
                # Metrics computation
                if not is_answerable:
                    if "i don't have information" in pred.lower() or "i don't know" in pred.lower() or "not mention" in pred.lower() or "không có thông tin" in pred.lower():
                        f1_scores.append(1.0)
                        em_scores.append(1.0)
                    else:
                        f1_scores.append(0.0)
                        em_scores.append(0.0)
                else:
                    f1 = f1_score(pred, ground_truth)
                    em = exact_match(pred, ground_truth)
                    f1_scores.append(f1)
                    em_scores.append(em)
                    
                latencies.append(result.generation_latency_ms)
                
            except Exception as e:
                logger.error(f"Error querying model {model_name}: {e}")
                
        # Calculate means
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        mean_em = sum(em_scores) / len(em_scores) if em_scores else 0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        
        logger.info(f"Model {model_name} | F1: {mean_f1:.3f} | Latency: {mean_latency:.1f} ms")
        
        results_data.append({
            "Model": model_name,
            "F1 Score": mean_f1,
            "Exact Match": mean_em,
            "Latency (ms)": mean_latency
        })
        
    # Plotting
    df = pd.DataFrame(results_data)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Latency vs F1 Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Colors for the 3 categories
    colors = {"qwen2.5:latest": "blue", "llama3.1:8b": "purple", "qwen2.5:1.5b": "green"}
    
    ax = sns.scatterplot(data=df, x="Latency (ms)", y="F1 Score", hue="Model", s=300, palette=colors)
    
    plt.title("LLM Benchmark: F1 Score vs Latency", fontsize=16, fontweight='bold')
    plt.xlabel("Generation Latency (ms) [< Lower is Better]")
    plt.ylabel("F1 Score [> Higher is Better]")
    
    for i in range(df.shape[0]):
        plt.text(x=df["Latency (ms)"].iloc[i] + (df["Latency (ms)"].max() * 0.02), 
                 y=df["F1 Score"].iloc[i], 
                 s=df["Model"].iloc[i], 
                 fontdict=dict(color='black', size=12, weight='bold'))
                 
    plt.tight_layout()
    plt.savefig("results/figures/llm_benchmark.png", dpi=300)
    logger.info("Saved benchmark plot to results/figures/llm_benchmark.png")
    
    print("\n--- Benchmark Final Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark(num_questions=50)
