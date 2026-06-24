import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import numpy as np

# Set style for academic look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def analyze_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract features
    sources = []
    types = []
    answerability = []
    lengths = []
    
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 'do', 'does', 'did', 'can', 'could']
    
    for item in data:
        q = item['question'].lower()
        
        # 1. Source
        qtype = item.get('question_type', '')
        if qtype == 'multi-hop':
            sources.append('HotpotQA')
        else:
            sources.append('SQuAD 2.0')
            
        # 2. Answerability
        answerability.append(item.get('is_answerable', True))
        
        # 3. Question Length (words)
        lengths.append(len(q.split()))
        
        # 4. Question Type (first matching WH-word)
        q_type = 'other'
        for w in question_words:
            if q.startswith(w) or f" {w} " in q:
                q_type = w.capitalize()
                break
        types.append(q_type)
        
    df = pd.DataFrame({
        'Source': sources,
        'Type': types,
        'Answerable': answerability,
        'Length': lengths
    })
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Academic Analysis of Evaluation Dataset (N=1000)', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Dataset Source (Pie Chart)
    ax1 = plt.subplot(2, 2, 1)
    source_counts = df['Source'].value_counts()
    colors = ['#4C72B0', '#DD8452']
    ax1.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax1.set_title('Dataset Source Distribution', fontweight='bold')
    
    # 2. Answerability (Bar Chart)
    ax2 = plt.subplot(2, 2, 2)
    ans_counts = df['Answerable'].value_counts()
    sns.barplot(x=ans_counts.index.map({True: 'Answerable', False: 'Unanswerable'}), 
                y=ans_counts.values, palette=['#55A868', '#C44E52'], ax=ax2)
    ax2.set_title('Query Feasibility', fontweight='bold')
    ax2.set_ylabel('Number of Queries')
    for i, v in enumerate(ans_counts.values):
        ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
    # 3. Question Types (Horizontal Bar)
    ax3 = plt.subplot(2, 2, 3)
    type_counts = df['Type'].value_counts().head(8) # Top 8
    sns.barplot(y=type_counts.index, x=type_counts.values, palette='viridis', ax=ax3)
    ax3.set_title('Top Interrogative Typologies (Question Words)', fontweight='bold')
    ax3.set_xlabel('Frequency')
    
    # 4. Question Length Distribution (Histogram/KDE)
    ax4 = plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='Length', hue='Source', kde=True, bins=20, palette=colors, ax=ax4, alpha=0.6)
    ax4.set_title('Lexical Complexity (Query Length Distribution)', fontweight='bold')
    ax4.set_xlabel('Number of Words in Query')
    ax4.set_ylabel('Density')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = 'results/metrics/dataset_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    analyze_dataset('data/processed/eval_dataset.json')
