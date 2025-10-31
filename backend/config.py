"""Configuration settings for Deep Thinking RAG pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration dictionary
config = {
    "data_dir": "./data",
    "vector_store_dir": "./vector_store",
    "persistent_db_dir": "./chroma_db",
    "llm_provider": "deepseek",
    "reasoning_llm": "deepseek-reasoner",
    "fast_llm": "deepseek-chat",
    "embedding_model": "models/gemini-embedding-001",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_reasoning_iterations": 7,
    "top_k_retrieval": 10,
    "top_n_rerank": 3,
}

# Environment setup
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_PROJECT"] = "Advanced-Deep-Thinking-RAG"

print("Config ready")