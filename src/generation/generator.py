from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
import requests

RAG_PROMPT_TEMPLATE = """You are a strict QA assistant.
Use the following pieces of retrieved context to answer the question.
If the exact answer cannot be logically deduced from the context, you MUST output exactly 'I don't have information.'. Do not guess. Do not extrapolate or use outside knowledge.
Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:"""

class LLMGenerator:
    """Wrapper for Ollama-based local LLM generation."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", temperature: float = 0.1):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        
        logger.info(f"Initializing LLM generator with model '{model_name}' at {base_url}")
        
        # Check if Ollama is reachable
        self._check_ollama_status()
        
        self.llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )
        
        self.prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def _check_ollama_status(self):
        """Checks if Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                # Try exact match or base name match (e.g. "phi3:mini" vs "phi3")
                has_model = False
                for m in models:
                    if self.model_name == m or self.model_name == m.split(":")[0]:
                        has_model = True
                        break
                        
                if not has_model:
                    logger.warning(f"Model '{self.model_name}' not found in Ollama. It may need to be pulled: `ollama pull {self.model_name}`")
                else:
                    logger.info("Ollama connection successful.")
            else:
                logger.warning(f"Ollama responded with status code {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}")
            
    def generate(self, query: str, context: str) -> str:
        """Generates an answer based on query and context."""
        logger.debug(f"Generating answer for query: '{query[:50]}...'")
        try:
            answer = self.chain.invoke({
                "context": context,
                "question": query
            })
            return answer.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {e}"

if __name__ == "__main__":
    generator = LLMGenerator(model_name="phi3")
    test_context = "The quick brown fox jumps over the lazy dog. The dog's name is Max."
    test_query = "What is the dog's name?"
    print(f"Query: {test_query}")
    print(f"Answer: {generator.generate(test_query, test_context)}")
