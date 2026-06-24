import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_results(results_dir: str = "results/metrics"):
    """Loads all experiment JSON files into a Pandas DataFrame."""
    results_dir_path = Path(results_dir)
    json_files = list(results_dir_path.glob("exp_*.json"))
    
    data = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            # Extract strategy
            strategy = content.get("config", file.stem.split('_')[-1]).capitalize()
            ans_metrics = content.get("metrics", {}).get("answerable", {})
            unans_metrics = content.get("metrics", {}).get("unanswerable", {})
            lat_metrics = content.get("metrics", {}).get("latency_ms", {})
            
            data.append({
                "Strategy": strategy,
                "Hit Rate@5": ans_metrics.get("hit_rate@5", {}).get("mean", 0.0),
                "MRR@5": ans_metrics.get("mrr@5", {}).get("mean", 0.0),
                "F1 Score": ans_metrics.get("f1_score", {}).get("mean", 0.0),
                "ROUGE-L": ans_metrics.get("rouge_l", {}).get("mean", 0.0),
                "Rejection Acc": unans_metrics.get("rejection_accuracy", {}).get("mean", 0.0),
                "Latency (ms)": lat_metrics.get("mean", 0.0)
            })
            
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).sort_values(by="Hit Rate@5", ascending=False)
    return df

def plot_retrieval_comparison(results_dir: str = "results/metrics", output_dir: str = "results/figures"):
    """Plots Hit Rate and MRR."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        return
        
    df_melted = df.melt(id_vars=["Strategy"], 
                        value_vars=["Hit Rate@5", "MRR@5"], 
                        var_name="Metric", 
                        value_name="Score")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Strategy", palette="viridis")

    plt.title("Retrieval Performance Comparison (Answerable Queries)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')

    for p in ax.patches:
        h = p.get_height()
        if h > 0.001:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retrieval_comparison.png"))
    plt.close()

def plot_generation_comparison(results_dir: str = "results/metrics", output_dir: str = "results/figures"):
    """Plots F1, ROUGE-L and Rejection Accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        return
        
    df_melted = df.melt(id_vars=["Strategy"], 
                        value_vars=["F1 Score", "ROUGE-L", "Rejection Acc"], 
                        var_name="Metric", 
                        value_name="Score")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Strategy", palette="magma")

    plt.title("Generation & Hallucination Defense Performance", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')

    for p in ax.patches:
        h = p.get_height()
        if h > 0.001:  # Skip zero-height patches to prevent 0.000 artifacts
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_comparison.png"))
    plt.close()

def plot_latency_comparison(results_dir: str = "results/metrics", output_dir: str = "results/figures"):
    """Plots Latency vs F1 Score."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        return
        
    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(data=df, x="Latency (ms)", y="F1 Score", hue="Strategy", s=200, palette="deep")

    plt.title("Latency vs F1 Score Trade-off", fontsize=14, fontweight='bold')
    plt.xlabel("Latency (ms) - Lower is better")
    plt.ylabel("F1 Score - Higher is better")
    
    for i in range(df.shape[0]):
        plt.text(x=df["Latency (ms)"].iloc[i] + 50, 
                 y=df["F1 Score"].iloc[i], 
                 s=df["Strategy"].iloc[i], 
                 fontdict=dict(color='black', size=10))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_tradeoff.png"))
    plt.close()

def create_results_table(results_dir: str = "results/metrics"):
    """Prints a markdown table of the results."""
    df = load_results(results_dir)
    if df.empty:
        print("No results found.")
        return
        
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    print("Generating visualizations...")
    plot_retrieval_comparison()
    plot_generation_comparison()
    plot_latency_comparison()
    create_results_table()
    print("Visualizations saved to results/figures/")
