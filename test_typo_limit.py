import os
import time
from loguru import logger
from src.pipeline import RAGPipeline

def run_typo_stress_test():
    # Initialize the pipeline
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline("config/config.yaml", strategy="hybrid")
    
    # Extreme typo test cases
    test_cases = [
        # 1. Missing vowels (fast phone typing)
        "Wht yer did th Normns cnqur Englnd?",
        
        # 2. Phonetic spelling + bad grammar
        "whare waz nicola tesla borm at?",
        
        # 3. Transpositions + bad capitalization
        "teh cpital of frnace is wht city",
        
        # 4. Severe slurring/fat fingering (Apollo 11)
        "whem did a polla levn lnadd on th mon?",
        
        # 5. The original problem query but worse
        "wht ctiy srvd az Polnads cpatial n 1313?"
    ]
    
    print("\n" + "="*80)
    print("EXTREME TYPO STRESS TEST (LLM AUTO-CORRECT)")
    print("="*80)
    
    for i, raw_query in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}]")
        print(f"RAW USER INPUT : {raw_query}")
        
        # The query method internally calls _correct_query_typos and logs it
        corrected = pipeline._correct_query_typos(raw_query)
        print(f"LLM CORRECTED  : {corrected}")
        
        result = pipeline.query(raw_query, top_k=5, use_router=False, correct_typos=True)
        
        # Print a short snippet of the answer
        ans_snippet = result.answer.replace('\n', ' ')
        if len(ans_snippet) > 100:
            ans_snippet = ans_snippet[:100] + "..."
            
        print(f"FINAL ANSWER   : {ans_snippet}")
        print("-" * 80)

if __name__ == "__main__":
    run_typo_stress_test()
