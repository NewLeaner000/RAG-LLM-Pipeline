import argparse
from loguru import logger
from src.evaluation.evaluator import run_experiment_matrix

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, choices=["quick", "full"], default="quick", 
                        help="Quick (10 questions) or full (500 questions)")
    
    args = parser.parse_args()
    
    num_questions = 10 if args.stage == "quick" else 500
    
    eval_dataset_path = "data/processed/eval_dataset.json"
    output_dir = "results/metrics"
    
    logger.info(f"Starting evaluation matrix (Stage: {args.stage}, Questions: {num_questions})")
    run_experiment_matrix(args.config, eval_dataset_path, output_dir, num_questions=num_questions)
    
    logger.info("Evaluation complete! Check results/metrics directory for JSON files.")

if __name__ == "__main__":
    main()
