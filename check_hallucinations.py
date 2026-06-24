import json
import sys

strategies = ["hybrid", "dense", "bm25"]

for strategy in strategies:
    print(f"\n{'='*20} STRATEGY: {strategy.upper()} {'='*20}")
    try:
        with open(f"results/metrics/exp_{strategy}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        results = data["per_question_results"]
        empty_gts = [x for x in results if not x["ground_truth"]]
        
        print(f"Total empty GTs: {len(empty_gts)}")
        for x in empty_gts:
            print(f"Q: {x['question']}")
            print(f"GT: {x['ground_truth']}")
            print(f"PRED: {x['prediction']}")
            print("-" * 50)
    except FileNotFoundError:
        print(f"No result file found for {strategy}")
