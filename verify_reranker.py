import json
import logging
from src.pipeline import RAGPipeline

logging.getLogger("src").setLevel(logging.WARNING)

def verify():
    # Initialize pipeline
    print("Initializing pipeline (with Hybrid strategy)...")
    pipeline = RAGPipeline("config/config.yaml", strategy="hybrid")
    
    # We choose a question that we know was improved by reranking
    # From SQuAD, let's use: "What alumni wrote \"The Closing of the American Mind\"?"
    # The answer is "Allan Bloom".
    query = 'What alumni wrote "The Closing of the American Mind"?'
    
    print(f"\n--- QUERY: '{query}' ---")
    
    # 1. Get Top 10 from Hybrid Retriever BEFORE reranking
    initial_k = 10
    retrieved_results = pipeline.retriever.retrieve(query, top_k=initial_k)
    
    print("\n[BEFORE RERANKING] Top 5 from Hybrid (RRF Scores):")
    for r in retrieved_results[:5]:
        content = r.document.page_content.replace('\n', ' ')[:80]
        has_answer = "Allan Bloom" in r.document.page_content
        marker = "[YES]" if has_answer else "[NO ]"
        print(f"Rank {r.rank:2d} (Score: {r.score:.4f}) | {marker} | {content}...")

    # Find where the answer is in the Top 10
    answer_rank = -1
    for r in retrieved_results:
        if "Allan Bloom" in r.document.page_content:
            answer_rank = r.rank
            break
            
    if answer_rank != -1:
        print(f"\n* The correct document is currently at Rank: {answer_rank} (Out of Top 5!)")
    else:
        print("\n* The correct document is NOT in the Top 10 at all.")
    
    # 2. Apply Reranker
    if hasattr(pipeline, 'reranker'):
        print("\n[APPLYING CROSS-ENCODER RERANKER...]")
        reranked_results = pipeline.reranker.rerank(query, retrieved_results, top_n=5)
        
        print("\n[AFTER RERANKING] New Top 5 (Cross-Encoder Scores):")
        for r in reranked_results:
            content = r.document.page_content.replace('\n', ' ')[:80]
            has_answer = "Allan Bloom" in r.document.page_content
            marker = "[YES]" if has_answer else "[NO ]"
            print(f"Rank {r.rank:2d} (Score: {r.score:.4f}) | {marker} | {content}...")
            
    else:
        print("Reranker is not enabled in config.yaml!")

if __name__ == "__main__":
    verify()
