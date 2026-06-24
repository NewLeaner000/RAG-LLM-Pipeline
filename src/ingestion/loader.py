import json
import random
from pathlib import Path
from datasets import load_dataset
from loguru import logger
from typing import List, Dict, Any

def load_squad_data(sample_size: int, random_seed: int = 42) -> List[Dict[str, Any]]:
    """Loads SQuAD 2.0 and samples answerable and unanswerable questions."""
    logger.info(f"Loading SQuAD 2.0 dataset (sample size: {sample_size})")
    try:
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    except Exception as e:
        logger.error(f"Failed to load SQuAD 2.0: {e}")
        raise
    
    answerable = []
    unanswerable = []
    
    for item in dataset:
        is_ans = len(item["answers"]["text"]) > 0
        formatted_item = {
            "id": f"squad_{item['id']}",
            "question": item["question"],
            "answer": item["answers"]["text"][0] if is_ans else "",
            "context": item["context"],
            "is_answerable": is_ans,
            "question_type": "factoid"
        }
        
        if is_ans:
            answerable.append(formatted_item)
        else:
            unanswerable.append(formatted_item)
            
    # Stratified sampling
    random.seed(random_seed)
    half_size = sample_size // 2
    
    # Ensure we don't sample more than available
    n_ans = min(half_size, len(answerable))
    n_unans = min(sample_size - n_ans, len(unanswerable))
    
    sampled = random.sample(answerable, n_ans) + random.sample(unanswerable, n_unans)
    random.shuffle(sampled)
    
    logger.info(f"Sampled {len(sampled)} SQuAD questions ({n_ans} answerable, {n_unans} unanswerable)")
    return sampled

def load_hotpotqa_data(sample_size: int, random_seed: int = 42) -> List[Dict[str, Any]]:
    """Loads HotpotQA and samples questions."""
    logger.info(f"Loading HotpotQA dataset (sample size: {sample_size})")
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        raise
        
    formatted_data = []
    for item in dataset:
        # HotpotQA context is a list of [title, list of sentences]
        # We need to construct the supporting context
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context_parts.append(f"{title}: {' '.join(sentences)}")
        
        full_context = "\n\n".join(context_parts)
        
        formatted_item = {
            "id": f"hotpot_{item['id']}",
            "question": item["question"],
            "answer": item["answer"],
            "context": full_context,
            "is_answerable": True,  # HotpotQA distractor questions are answerable
            "question_type": "multi-hop"
        }
        formatted_data.append(formatted_item)
        
    random.seed(random_seed)
    sampled = random.sample(formatted_data, min(sample_size, len(formatted_data)))
    logger.info(f"Sampled {len(sampled)} HotpotQA questions")
    return sampled

def create_eval_dataset(squad_sample_size: int = 500, hotpotqa_sample_size: int = 100, random_seed: int = 42, output_dir: str = "data/processed"):
    """Loads datasets, extracts unique contexts, and saves to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    squad_data = load_squad_data(squad_sample_size, random_seed) if squad_sample_size > 0 else []
    hotpot_data = load_hotpotqa_data(hotpotqa_sample_size, random_seed) if hotpotqa_sample_size > 0 else []
    
    eval_dataset = squad_data + hotpot_data
    
    # Extract unique contexts
    unique_contexts = set()
    for item in eval_dataset:
        unique_contexts.add(item["context"])
        
    corpus = [{"id": f"doc_{i}", "content": ctx} for i, ctx in enumerate(unique_contexts)]
    
    # Save to files
    eval_file = Path(output_dir) / "eval_dataset.json"
    corpus_file = Path(output_dir) / "corpus.json"
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
        
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Saved {len(eval_dataset)} Q&A pairs to {eval_file}")
    logger.info(f"Saved {len(corpus)} unique contexts to {corpus_file}")

if __name__ == "__main__":
    import yaml
    
    # Load config to get sample sizes
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            squad_size = config.get("data", {}).get("squad_sample_size", 500)
            hotpot_size = config.get("data", {}).get("hotpotqa_sample_size", 100)
            seed = config.get("data", {}).get("random_seed", 42)
            
        create_eval_dataset(squad_size, hotpot_size, seed)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
