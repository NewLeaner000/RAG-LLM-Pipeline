import string
import re
from typing import List, Dict
from langchain_core.documents import Document

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    if not isinstance(s, str):
        return ""
        
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction: str, truth: str) -> float:
    """
    Computes Exact Match (EM) score.
    Returns 1.0 if normalized strings match exactly, else 0.0.
    """
    if not prediction and not truth:
        return 1.0 # Both empty -> Match
    if not prediction or not truth:
        return 0.0 # One empty -> No match
        
    return 1.0 if normalize_answer(prediction) == normalize_answer(truth) else 0.0

def f1_score(prediction: str, truth: str) -> float:
    """
    Computes token-level F1 score between prediction and truth.
    """
    if not prediction and not truth:
        return 1.0
    if not prediction or not truth:
        return 0.0
        
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    num_same = len(common_tokens)
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_l_score(prediction: str, truth: str) -> float:
    """
    Computes ROUGE-L score using rouge-score library.
    """
    if not prediction and not truth:
        return 1.0
    if not prediction or not truth:
        return 0.0
        
    if rouge_scorer is None:
        raise ImportError("Please install rouge-score: pip install rouge-score")
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(truth, prediction)
    return scores['rougeL'].fmeasure

def is_match(retrieved_doc: Document, ground_truth_context: str) -> bool:
    def normalize(text):
        return re.sub(r'\W+', '', text.lower())
        
    r_text = normalize(retrieved_doc.page_content)
    gt_text = normalize(ground_truth_context)
    
    if not r_text or not gt_text:
        return False
        
    return r_text in gt_text or gt_text in r_text

def hit_rate_at_k(retrieved_docs: List[Document], ground_truth_context: str, k: int) -> float:
    if not retrieved_docs or not ground_truth_context:
        return 0.0
        
    for i, doc in enumerate(retrieved_docs[:k]):
        if is_match(doc, ground_truth_context):
            return 1.0
    return 0.0

def mrr_at_k(retrieved_docs: List[Document], ground_truth_context: str, k: int) -> float:
    if not retrieved_docs or not ground_truth_context:
        return 0.0
        
    for i, doc in enumerate(retrieved_docs[:k]):
        if is_match(doc, ground_truth_context):
            return 1.0 / (i + 1)
    return 0.0

if __name__ == "__main__":
    print("--- Edge Case Testing for Task 11 ---")
    
    # Edge case 1: Both empty
    print(f"Both Empty -> EM: {exact_match('', '')}, F1: {f1_score('', '')}, ROUGE: {rouge_l_score('', '')}")
    
    # Edge case 2: One empty
    print(f"One Empty -> EM: {exact_match('hello', '')}, F1: {f1_score('hello', '')}, ROUGE: {rouge_l_score('hello', '')}")
    
    # Edge case 3: Exact identical
    print(f"Identical -> EM: {exact_match('hello world', 'hello world')}, F1: {f1_score('hello world', 'hello world')}")
    
    # Edge case 4: No overlap
    print(f"No Overlap -> EM: {exact_match('apple orange', 'banana grape')}, F1: {f1_score('apple orange', 'banana grape')}")
    
    # Edge case 5: Partial overlap with punctuation and articles
    pred = "The Albert Einstein was a great physicist!"
    truth = "Albert Einstein was a physicist."
    print(f"Partial + Punctuation -> EM: {exact_match(pred, truth)}, F1: {f1_score(pred, truth):.3f}")
    
    print("\n--- Retrieval Metrics Edge Cases ---")
    docs = [Document(page_content="wrong"), Document(page_content="correct context")]
    
    # Edge case: Empty context
    print(f"Empty Truth Context -> Hit@2: {hit_rate_at_k(docs, '', 2)}")
    
    # Edge case: Empty docs
    print(f"Empty Docs -> Hit@2: {hit_rate_at_k([], 'correct', 2)}")
    
    # Edge case: Match at index 2 (k=1 vs k=2)
    print(f"Match at idx 2 (K=1) -> Hit@1: {hit_rate_at_k(docs, 'correct context', 1)}, MRR@1: {mrr_at_k(docs, 'correct context', 1)}")
    print(f"Match at idx 2 (K=2) -> Hit@2: {hit_rate_at_k(docs, 'correct context', 2)}, MRR@2: {mrr_at_k(docs, 'correct context', 2)}")
    
    print("Task 11 completed successfully.")
