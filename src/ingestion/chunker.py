from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

def chunk_documents(documents: List[Dict[str, Any]], strategy_config: Dict[str, Any]) -> List[Document]:
    """
    Chunks a list of documents according to the provided strategy configuration.
    
    Expected format of documents: [{"id": "doc_0", "content": "text..."}, ...]
    """
    strategy_name = strategy_config.get("name", "unknown")
    chunk_size = strategy_config.get("chunk_size", 512)
    chunk_overlap = strategy_config.get("chunk_overlap", 100)
    
    logger.info(f"Chunking {len(documents)} documents using strategy '{strategy_name}' (size: {chunk_size}, overlap: {chunk_overlap})")
    
    if strategy_name.startswith("fixed"):
        # Use simple character text splitter for "fixed" with blank separator
        # to strictly enforce chunk size boundary
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    elif strategy_name.startswith("recursive"):
        separators = strategy_config.get("separators", ["\n\n", "\n", ". ", " "])
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")

    all_chunks = []
    
    for doc in documents:
        doc_id = doc["id"]
        content = doc["content"]
        original_length = len(content)
        
        # Split text
        texts = splitter.split_text(content)
        
        for i, text in enumerate(texts):
            chunk = Document(
                page_content=text,
                metadata={
                    "source_id": doc_id,
                    "chunk_index": i,
                    "strategy": strategy_name,
                    "original_length": original_length
                }
            )
            all_chunks.append(chunk)
            
    logger.info(f"Produced {len(all_chunks)} chunks using strategy '{strategy_name}'")
    return all_chunks

if __name__ == "__main__":
    import json
    try:
        with open("data/processed/corpus.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)
            
        strategies = [
            {"name": "fixed_256", "chunk_size": 256, "chunk_overlap": 50},
            {"name": "fixed_512", "chunk_size": 512, "chunk_overlap": 100},
            {"name": "recursive_512", "chunk_size": 512, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " "]}
        ]
        
        for s in strategies:
            chunks = chunk_documents(corpus, s)
            print(f"Strategy: {s['name']} -> {len(chunks)} chunks, example size: {len(chunks[0].page_content) if chunks else 0}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
