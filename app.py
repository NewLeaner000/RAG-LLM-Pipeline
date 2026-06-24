import streamlit as st
import time
import os
from PIL import Image

# Initialize the RAG Pipeline in session state to avoid reloading it on every interaction
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
    .doc-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .doc-score {
        float: right;
        font-size: 12px;
        background-color: #ff4b4b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("Engine Config")

st.sidebar.header("1. Language Model")
selected_model = st.sidebar.selectbox(
    "Generator Model",
    options=["qwen2.5:latest", "llama3.1:8b", "qwen2.5:1.5b"],
    index=0
)

st.sidebar.header("2. Retrieval Strategy")
use_router = st.sidebar.toggle("Use Query Router", value=False, help="Dynamically routes factoid queries to BM25 and complex queries to Hybrid.")

if not use_router:
    selected_strategy = st.sidebar.radio(
        "Static Strategy",
        options=["hybrid", "dense", "bm25"],
        index=0,
        format_func=lambda x: x.upper()
    )
else:
    st.sidebar.info("Static strategy is disabled when Query Router is ON.")
    selected_strategy = "hybrid" # default fallback

st.sidebar.header("3. Advanced Settings")
use_reranker = st.sidebar.toggle("Use Cross-Encoder Reranker", value=True)
use_autocorrect = st.sidebar.toggle("Auto-Correct Typos (LLM)", value=True, help="Uses the LLM to silently fix spelling/grammar errors before retrieving documents.")
top_k = st.sidebar.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)

if st.sidebar.button("Apply Configuration & Reload Engine"):
    with st.spinner("Loading RAG Engine (Models, VectorStore, BM25 Index)..."):
        from src.pipeline import RAGPipeline
        st.session_state.pipeline = RAGPipeline("config/config.yaml", strategy=selected_strategy)
        st.session_state.pipeline.generator.model_name = selected_model
        st.session_state.pipeline.use_reranker = use_reranker
        st.success("Engine successfully loaded!")

# --- Main Dashboard ---
st.title("RAG Evaluation Lab")
st.markdown("A robust RAG architecture featuring **Dense Vectors**, **Sparse BM25**, **Cross-Encoder Reranking**, and a **Query Router**.")

# Tabs
tab1, tab2 = st.tabs(["Interactive Chat", "Evaluation Dashboard"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Ask a question to test the RAG engine:", placeholder="e.g. Who is Einstein? or What is the difference between Dense and BM25?")
        
        if query:
            if st.session_state.pipeline is None:
                st.warning("Please click 'Apply Configuration & Reload Engine' in the sidebar first.")
            else:
                with st.spinner("Processing query..."):
                    # Apply runtime settings that don't need re-init
                    if not use_router:
                        if selected_strategy == "bm25":
                            st.session_state.pipeline.default_retriever = st.session_state.pipeline.bm25_retriever
                        elif selected_strategy == "dense":
                            st.session_state.pipeline.default_retriever = st.session_state.pipeline.dense_retriever
                        else:
                            st.session_state.pipeline.default_retriever = st.session_state.pipeline.hybrid_retriever
                            
                    st.session_state.pipeline.generator.model_name = selected_model
                    st.session_state.pipeline.use_reranker = use_reranker
                    
                    try:
                        result = st.session_state.pipeline.query(query, top_k=top_k, use_router=use_router, correct_typos=use_autocorrect)
                        
                        st.markdown("### Generated Answer")
                        st.info(result.answer)
                        
                        st.markdown(f"### Retrieved Context ({len(result.retrieved_docs)} chunks)")
                        for idx, (doc, score) in enumerate(zip(result.retrieved_docs, result.retrieval_scores)):
                            with st.expander(f"Chunk {idx+1} [Score: {score:.3f}]"):
                                st.markdown(f"<div class='doc-card'><span class='doc-score'>Score: {score:.3f}</span>{doc.page_content}</div>", unsafe_allow_html=True)
                                
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    with col2:
        st.markdown("### Diagnostics")
        if query and st.session_state.pipeline is not None and 'result' in locals():
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total End-to-End Latency</div>
                <div class="metric-value">{result.total_latency_ms:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">LLM Generation Latency</div>
                <div class="metric-value">{result.generation_latency_ms:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
            
            retrieval_time = result.total_latency_ms - result.generation_latency_ms
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Retrieval + Rerank Latency</div>
                <div class="metric-value">{max(0, retrieval_time):.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Submit a query to see performance metrics.")

with tab2:
    st.header("Results Benchmark (1000 Samples)")
    st.markdown("These charts are pre-computed results from evaluating 1000 queries across SQuAD and HotpotQA datasets.")
    
    figures_dir = "results/figures"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(f"{figures_dir}/retrieval_comparison.png"):
            st.image(Image.open(f"{figures_dir}/retrieval_comparison.png"), caption="Retrieval Strategy Comparison", use_column_width=True)
        else:
            st.warning("Retrieval comparison chart not found.")
            
        if os.path.exists(f"{figures_dir}/llm_benchmark.png"):
            st.image(Image.open(f"{figures_dir}/llm_benchmark.png"), caption="LLM Parameter Scale Benchmark", use_column_width=True)
            
    with col2:
        if os.path.exists(f"{figures_dir}/generation_comparison.png"):
            st.image(Image.open(f"{figures_dir}/generation_comparison.png"), caption="Generation Quality (F1 & ROUGE-L)", use_column_width=True)
        else:
            st.warning("Generation comparison chart not found.")
            
        if os.path.exists(f"{figures_dir}/latency_tradeoff.png"):
            st.image(Image.open(f"{figures_dir}/latency_tradeoff.png"), caption="Latency vs Quality Trade-off", use_column_width=True)
