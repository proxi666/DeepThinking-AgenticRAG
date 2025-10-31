"""LangGraph node functions for Deep Thinking RAG."""
import json
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.pretty import pprint as rprint

from config import config
from models import RAGState
from agents import (
    planner_agent, query_rewriter_agent, retrieval_supervisor_agent,
    distiller_agent, reflection_agent, policy_agent, reasoning_llm
)
from retrieval import (
    vector_search_only, bm25_search_only, hybrid_search,
    rerank_documents_function, web_search_function
)
from utils import console, format_docs, get_past_context_str


def plan_node(state: RAGState) -> Dict:
    """Generate initial plan or pass through."""
    if not state.get("plan"):
        console.print("--- 🧠: Generating Plan ---")
        plan = planner_agent.invoke({"question": state["original_question"]})
        rprint(plan)
        return {"plan": plan, "current_step_index": 0, "past_steps": []}
    else:
        return {}


def retrieval_node(state: RAGState) -> Dict:
    """Retrieve documents from 10-K."""
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 📚: Retrieving from 10-K (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")
    
    retrieval_decision = retrieval_supervisor_agent.invoke({"sub_question": rewritten_query})
    console.print(f"  Supervisor Decision: Use `{retrieval_decision.strategy}`. Justification: {retrieval_decision.justification}")

    if retrieval_decision.strategy == 'vector_search':
        retrieved_docs = vector_search_only(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])
    elif retrieval_decision.strategy == 'keyword_search':
        retrieved_docs = bm25_search_only(rewritten_query, k=config['top_k_retrieval'])
    else:
        retrieved_docs = hybrid_search(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])
    
    return {"retrieved_docs": retrieved_docs}


def web_search_node(state: RAGState) -> Dict:
    """Search the web for information."""
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 🌐: Searching Web (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")
    retrieved_docs = web_search_function(rewritten_query)
    return {"retrieved_docs": retrieved_docs}


def rerank_node(state: RAGState) -> Dict:
    """Rerank retrieved documents."""
    console.print("--- 🎯: Reranking Documents ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    reranked_docs = rerank_documents_function(current_step.sub_question, state["retrieved_docs"])
    console.print(f"  Reranked to top {len(reranked_docs)} documents.")
    return {"reranked_docs": reranked_docs}


def compression_node(state: RAGState) -> Dict:
    """Distill context from reranked documents."""
    console.print("--- ✂️: Distilling Context ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    context = format_docs(state["reranked_docs"])
    synthesized_context = distiller_agent.invoke({"question": current_step.sub_question, "context": context})
    console.print(f"  Distilled Context Snippet: {synthesized_context[:200]}...")
    return {"synthesized_context": synthesized_context}


def reflection_node(state: RAGState) -> Dict:
    """Reflect on findings and update history."""
    console.print("--- 🔄: Reflecting on Findings ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    summary = reflection_agent.invoke({"sub_question": current_step.sub_question, "context": state['synthesized_context']})
    console.print(f"  Summary: {summary}")
    
    new_past_step = {
        "step_index": current_step_index + 1,
        "sub_question": current_step.sub_question,
        "retrieved_docs": state['reranked_docs'],
        "summary": summary
    }
    return {"past_steps": state["past_steps"] + [new_past_step], "current_step_index": current_step_index + 1}


def final_answer_node(state: RAGState) -> Dict:
    """Generate final answer with citations."""
    console.print("--- ✅: Generating Final Answer with Citations ---")
    final_context = ""
    for i, step in enumerate(state['past_steps']):
        final_context += f"\\n--- Findings from Research Step {i+1} ---\\n"
        for doc in step['retrieved_docs']:
            source = doc.metadata.get('section') or doc.metadata.get('source')
            final_context += f"Source: {source}\\nContent: {doc.page_content}\\n\\n"
    
    final_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Synthesize the research findings from internal documents and web searches into a comprehensive, multi-paragraph answer for the user's original question.
Your answer must be grounded in the provided context. At the end of any sentence that relies on specific information, you MUST add a citation. For 10-K documents, use [Source: <section title>]. For web results, use [Source: <URL>]."""),
        ("human", "Original Question: {question}\n\nResearch History and Context:\n{context}")
    ])
    
    final_answer_agent = final_answer_prompt | reasoning_llm | StrOutputParser()
    final_answer = final_answer_agent.invoke({"question": state['original_question'], "context": final_context})
    return {"final_answer": final_answer}


def route_by_tool(state: RAGState) -> str:
    """Conditional edge to route by tool."""
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    return current_step.tool


def should_continue_node(state: RAGState) -> str:
    """Conditional edge to control the main loop."""
    console.print("--- 🚦: Evaluating Policy ---")
    current_step_index = state["current_step_index"]
    
    if current_step_index >= len(state["plan"].steps):
        console.print("  -> Plan complete. Finishing.")
        return "finish"
    
    if current_step_index >= config["max_reasoning_iterations"]:
        console.print("  -> Max iterations reached. Finishing.")
        return "finish"

    if state.get("reranked_docs") is not None and not state["reranked_docs"]:
        console.print("  -> Retrieval failed for the last step. Continuing with next step in plan.")
        return "continue"

    history = get_past_context_str(state['past_steps'])
    plan_str = json.dumps([s.model_dump() for s in state['plan'].steps])

    decision = policy_agent.invoke({"question": state["original_question"], "plan": plan_str, "history": history})
    console.print(f"  -> Decision: {decision.next_action} | Justification: {decision.justification}")
    
    if decision.next_action == "FINISH":
        return "finish"
    else:
        return "continue"


print("All graph nodes ready.")