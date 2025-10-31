"""Main execution script for Deep Thinking RAG."""
from rich.markdown import Markdown
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import config
from data_processing import (
    download_and_parse_10k, load_and_chunk_document,
    create_metadata_chunks, url_10k, doc_path_raw, doc_path_clean
)
from vector_store import init_baseline_vector_store, init_advanced_vector_store
from agents import fast_llm
from utils import console, format_docs
from graph_builder import build_graph


def run_baseline_rag(query: str, retriever):
    """Run baseline RAG pipeline."""
    template = """You are an AI financial analyst. Answer the question based upon only on the following context

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    baseline_rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | fast_llm
        | StrOutputParser()
    )
    
    print("Executing complex query on the baseline RAG chain...")
    result = baseline_rag_chain.invoke(query)
    console.print("\n--- BASELINE RAG OUTPUT ---")
    console.print(Markdown(result))
    return result


def run_deep_thinking_rag(query: str, graph):
    """Run Deep Thinking RAG pipeline."""
    final_state = None
    graph_input = {"original_question": query}
    
    print("--- Invoking Deep Thinking RAG Graph ---")
    for chunk in graph.stream(graph_input, stream_mode="values"):
        final_state = chunk
    print("\n--- Graph Stream Finished ---")
    
    console.print("\n--- DEEP THINKING RAG FINAL ANSWER ---")
    console.print(Markdown(final_state['final_answer']))
    return final_state


if __name__ == "__main__":
    # Step 1: Download and process data
    print("=== STEP 1: DATA ACQUISITION ===")
    download_and_parse_10k(url_10k, doc_path_raw, doc_path_clean)
    
    # Step 2: Load and chunk documents
    print("\n=== STEP 2: DOCUMENT PROCESSING ===")
    documents, doc_chunks, text_splitter = load_and_chunk_document(doc_path_clean)
    
    # Step 3: Create metadata-aware chunks
    print("\n=== STEP 3: METADATA ENRICHMENT ===")
    doc_chunks_with_metadata = create_metadata_chunks(documents, text_splitter, doc_path_clean)
    
    # Step 4: Initialize vector stores
    print("\n=== STEP 4: VECTOR STORE INITIALIZATION ===")
    baseline_retriever = init_baseline_vector_store(doc_chunks)  # Capture the return value!
    init_advanced_vector_store(doc_chunks_with_metadata)
    
    # Step 5: Define complex query
    complex_query = (
        "Based on NVIDIA's 2025 10-K filing, identify their key risks related to competition. "
        "Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain "
        "how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."
    )
    
    # Step 6: Run baseline RAG
    print("\n=== STEP 5: BASELINE RAG ===")
    baseline_result = run_baseline_rag(complex_query, baseline_retriever)  # Pass it as parameter!
    
    # Step 7: Build and run Deep Thinking RAG
    print("\n=== STEP 6: DEEP THINKING RAG ===")
    deep_thinking_graph = build_graph()
    final_state = run_deep_thinking_rag(complex_query, deep_thinking_graph)
    
    print("\n=== EXECUTION COMPLETE ===")