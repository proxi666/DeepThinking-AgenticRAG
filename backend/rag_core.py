"""Core logic for the Deep Thinking RAG system, refactored for a service architecture."""
from rich.markdown import Markdown
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from vector_store import load_persistent_stores 
from agents import fast_llm
from utils import console, format_docs
from graph_builder import build_graph

# --- GLOBAL VARIABLES ---
baseline_rag_chain = None
deep_thinking_graph = None

def initialize_rag_components():
    """
    Performs the one-time setup for the RAG system.
    LOADS pre-built vector stores from disk and builds the RAG components.
    This function is called once when the application starts and should be VERY FAST.
    """
    global baseline_rag_chain, deep_thinking_graph

    # Step 1: Load pre-built vector stores and indexes from disk
    console.print("=== STEP 1: LOADING PERSISTENT VECTOR STORES ===")
    load_persistent_stores()
    
    # After load_persistent_stores() runs, the 'baseline_retriever' variable
    # in vector_store.py is populated. We need to import it here to use it.
    from vector_store import baseline_retriever 

    # Step 2: Build Baseline RAG Chain using the loaded retriever
    console.print("\n=== STEP 2: BUILDING BASELINE RAG CHAIN ===")
    template = """You are an AI financial analyst. Answer the question based upon only on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    baseline_rag_chain = (
        {"context": baseline_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | fast_llm
        | StrOutputParser()
    )
    console.print("Baseline RAG chain created successfully.")

    # Step 3: Build Deep Thinking RAG Graph
    console.print("\n=== STEP 3: BUILDING DEEP THINKING GRAPH ===")
    deep_thinking_graph = build_graph()
    console.print("Deep Thinking RAG graph built and compiled successfully.")
    
    console.print("\n\n=== ALL COMPONENTS INITIALIZED SUCCESSFULLY ===")

# BASELINE RAG SYSTEM
# def run_baseline_rag(query: str) -> dict:
#     """
#     Executes ONLY the baseline RAG pipeline for a given query.
#     """
#     if not baseline_rag_chain:
#         raise RuntimeError("Baseline RAG chain is not initialized. Call initialize_rag_components() first.")
    
#     print("\n--- Executing Baseline RAG ---")
#     baseline_result = baseline_rag_chain.invoke(query)
#     console.print(Markdown(baseline_result))
    
#     return {"baseline_output": baseline_result}

def run_baseline_rag(query: str) -> dict:
    """
    Executes ONLY the baseline RAG pipeline and returns the output AND contexts.
    """
    if not baseline_rag_chain:
        raise RuntimeError("Baseline RAG chain is not initialized...")
    
    # We need access to the retriever itself to get the contexts
    from vector_store import baseline_retriever

    print("\n--- Executing Baseline RAG ---")
    
    # First, retrieve the documents separately to capture them
    retrieved_docs = baseline_retriever.invoke(query)
    
    # Then, run the full chain to get the answer
    baseline_result = baseline_rag_chain.invoke(query)
    console.print(Markdown(baseline_result))
    
    return {
        "baseline_output": baseline_result,
        # Convert Document objects to a serializable format (their page content)
        "contexts": [doc.page_content for doc in retrieved_docs]
    }


# def get_deep_thinking_stream(query: str):
#     """
#     Executes the Deep Thinking RAG pipeline and YIELDS each chunk from the stream.
#     This is a generator function.
#     """
#     if not deep_thinking_graph:
#         raise RuntimeError("Deep Thinking graph is not initialized. Call initialize_rag_components() first.")

#     graph_input = {"original_question": query}
#     graph_config = {"recursion_limit": 50}

#     print("\n--- Invoking Deep Thinking RAG Graph Stream ---")
#     # This will yield each intermediate step as it happens
#     yield from deep_thinking_graph.stream(
#         graph_input, 
#         config=graph_config,
#         stream_mode="values"
#     )

def get_deep_thinking_stream(query: str):
    """
    Executes the Deep Thinking RAG pipeline and YIELDS each chunk, plus a final
    event with all contexts.
    """
    if not deep_thinking_graph:
        raise RuntimeError("Deep Thinking graph is not initialized...")

    graph_input = {"original_question": query}
    graph_config = {"recursion_limit": 50}

    print("\n--- Invoking Deep Thinking RAG Graph Stream ---")
    
    final_state = None
    # Iterate through the stream to get the final state
    for chunk in deep_thinking_graph.stream(graph_input, config=graph_config, stream_mode="values"):
        final_state = chunk
        yield chunk # Yield the original chunks for the UI

    print("\n--- Graph Stream Finished ---")
    
    # After the stream is done, aggregate all contexts from the final state
    if final_state and "past_steps" in final_state:
        all_contexts = []
        for step in final_state["past_steps"]:
            # We save the reranked docs as they are the most precise evidence
            for doc in step["retrieved_docs"]:
                 all_contexts.append(doc.page_content)
        
        # Yield one final, special event containing all the contexts
        yield {"contexts_complete": all_contexts}
    print("\n--- Graph Stream Finished ---")