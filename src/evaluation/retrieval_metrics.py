from typing import List, Dict, Any
from langchain_core.documents import Document

def is_match(retrieved_doc: Document, ground_truth_context: str) -> bool:
    """
    Checks if a retrieved document matches the ground truth context.
    For this lab, we use substring matching since the ingested chunks 
    are derived directly from the ground truth contexts.
    """
    r_text = retrieved_doc.page_content.strip()
    gt_text = ground_truth_context.strip()
    
    if not r_text or not gt_text:
        return False
        
    return r_text in gt_text or gt_text in r_text

def evaluate_retrieval(retrieved_docs: List[Document], ground_truth_context: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Calculates Hit Rate@K and MRR@K for various K values.
    
    Args:
        retrieved_docs: Sorted list of retrieved documents (best first)
        ground_truth_context: The context that actually contains the answer
        k_values: List of K thresholds to evaluate at
        
    Returns:
        Dictionary mapping metric names (e.g. 'hit_rate@5', 'mrr@5') to scores
    """
    metrics = {}
    
    # Find rank of first relevant document (1-indexed)
    first_relevant_rank = -1
    for i, doc in enumerate(retrieved_docs):
        if is_match(doc, ground_truth_context):
            first_relevant_rank = i + 1
            break
            
    for k in k_values:
        if first_relevant_rank != -1 and first_relevant_rank <= k:
            metrics[f"hit_rate@{k}"] = 1.0
            metrics[f"mrr@{k}"] = 1.0 / first_relevant_rank
        else:
            metrics[f"hit_rate@{k}"] = 0.0
            metrics[f"mrr@{k}"] = 0.0
            
    return metrics

if __name__ == "__main__":
    gt = "Albert Einstein was a German-born theoretical physicist."
    docs = [
        Document(page_content="Isaac Newton was an English mathematician."),
        Document(page_content="Albert Einstein was a German-born theoretical physicist. He is widely held to be one of the greatest and most influential scientists of all time."),
        Document(page_content="Marie Curie was a Polish and naturalized-French physicist.")
    ]
    
    scores = evaluate_retrieval(docs, gt, k_values=[1, 2, 3])
    print(f"Retrieval scores: {scores}")
