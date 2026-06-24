from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from pathlib import Path
import os

# We store our DB persistently here
PERSIST_DIRECTORY = "chroma_db"

def create_embeddings(model_name: str) -> Embeddings:
    """
    Creates a HuggingFace embedding model.
    """
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        # For HuggingFaceEmbeddings, it will auto-detect CUDA if torch is installed with CUDA support.
        encode_kwargs = {"normalize_embeddings": True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}")
        raise

def build_vector_store(documents: List[Document], embedding_model: Embeddings, collection_name: str) -> Chroma:
    """
    Creates or loads a ChromaDB collection with the given documents and embeddings.
    """
    logger.info(f"Accessing vector store for collection: '{collection_name}'")
    persist_dir = Path(PERSIST_DIRECTORY) / collection_name
    
    # Check if the directory exists and has files (indicating it was already built)
    is_existing = persist_dir.exists() and os.listdir(persist_dir)
    
    if is_existing:
        logger.info(f"Loading existing collection '{collection_name}' from {persist_dir}")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(persist_dir)
        )
    else:
        logger.info(f"Building new collection '{collection_name}' with {len(documents)} documents")
        # Ensure the directory exists
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Chroma will batch inserts automatically
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=str(persist_dir)
        )
        logger.info(f"Finished building collection '{collection_name}'")
        
    return vector_store

if __name__ == "__main__":
    import json
    import yaml
    from src.ingestion.chunker import chunk_documents
    
    # Simple test script
    try:
        # 1. Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # 2. Load 50 paragraphs for testing
        with open("data/processed/corpus.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)[:50] 
            
        # 3. Chunk
        chunk_config = config["chunking"]["strategies"][0] # fixed_256
        chunks = chunk_documents(corpus, chunk_config)
        
        # 4. Embed and store
        embed_config = config["embedding"]["models"][0] # all-MiniLM-L6-v2
        embed_model = create_embeddings(embed_config["name"])
        
        # Clean model name for collection (remove slashes)
        safe_model_name = embed_config["name"].split("/")[-1]
        collection_name = f"test_{chunk_config['name']}_{safe_model_name}"
        
        vector_store = build_vector_store(chunks, embed_model, collection_name)
        
        # Query test
        results = vector_store.similarity_search("testing", k=1)
        print(f"Vector store created. Query returned {len(results)} result(s).")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
