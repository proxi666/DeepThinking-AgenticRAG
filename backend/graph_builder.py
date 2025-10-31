"""LangGraph construction and compilation."""
from langgraph.graph import StateGraph, END

from models import RAGState
from graph_nodes import (
    plan_node, retrieval_node, web_search_node, rerank_node,
    compression_node, reflection_node, final_answer_node,
    route_by_tool, should_continue_node
)


def build_graph():
    """Build and compile the Deep Thinking RAG graph."""
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_10k", retrieval_node)
    graph.add_node("retrieve_web", web_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compression_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("generate_final_answer", final_answer_node)
    
    # Set entry point
    graph.set_entry_point("plan")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "plan",
        route_by_tool,
        {
            "search_10k": "retrieve_10k",
            "search_web": "retrieve_web",
        },
    )
    
    # Add linear edges
    graph.add_edge("retrieve_10k", "rerank")
    graph.add_edge("retrieve_web", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "reflect")
    
    # Add main control loop
    graph.add_conditional_edges(
        "reflect",
        should_continue_node,
        {
            "continue": "plan",
            "finish": "generate_final_answer",
        },
    )
    
    graph.add_edge("generate_final_answer", END)
    
    print("StateGraph constructed successfully.")
    return graph.compile()


print("Graph builder ready.")