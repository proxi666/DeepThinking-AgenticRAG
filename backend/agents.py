"""LLM agent definitions for the Deep Thinking RAG system."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser

from config import config
from models import Plan, RetrievalDecision, Decision


# Initialize LLMs
reasoning_llm = ChatDeepSeek(
    model=config["reasoning_llm"],
    temperature=0,
    timeout=120
)

fast_llm = ChatDeepSeek(model=config["fast_llm"], temperature=0)


# Planner Agent
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Your task is to create a clear, multi-step plan to answer a complex user query by retrieving information from multiple sources.
Decompose the user's query into a series of simple, sequential sub-questions. For each step, decide which tool is more appropriate.
For `search_10k` steps, also identify the most likely section of the 10-K (e.g., 'Item 1A. Risk Factors', 'Item 7. Management's Discussion and Analysis...').
It is critical to use the exact section titles found in a 10-K filing where possible."""),
    ("human", "User Query: {question}")
])

planner_agent = planner_prompt | reasoning_llm.with_structured_output(Plan)


# Query Rewriter Agent
query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimization expert. Your task is to rewrite a given sub-question into a highly effective search query for a vector database or web search engine, using keywords and context from the research plan.
The rewritten query should be specific, use terminology likely to be found in the target source (a financial 10-K or news articles), and be structured to retrieve the most relevant text snippets. Do not mention anything apart from the optimized query, no need to narrate."""),
    ("human", "Current sub-question: {sub_question}\n\nRelevant keywords from plan: {keywords}\n\nContext from past steps:\n{past_context}")
])

query_rewriter_agent = query_rewriter_prompt | reasoning_llm | StrOutputParser()


# Retrieval Supervisor Agent
retrieval_supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval strategy expert. Based on the user's query, you must decide the best retrieval strategy.
You have three options:
1. `vector_search`: Best for conceptual, semantic, or similarity-based queries.
2. `keyword_search`: Best for queries with specific, exact terms, names, or codes (e.g., 'Item 1A', 'Hopper architecture').
3. `hybrid_search`: A good default that combines both, but may be less precise than a targeted strategy."""),
    ("human", "User Query: {sub_question}")
])

retrieval_supervisor_agent = retrieval_supervisor_prompt | reasoning_llm.with_structured_output(RetrievalDecision)


# Distiller Agent
distiller_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Your task is to synthesize the following retrieved document snippets into a single, concise paragraph.
The goal is to provide a clear and coherent context that directly answers the question: '{question}'.
Focus on removing redundant information and organizing the content logically. Answer only with the synthesized context."""),
    ("human", "Retrieved Documents:\n{context}")
])

distiller_agent = distiller_prompt | reasoning_llm | StrOutputParser()


# Reflection Agent
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Based on the retrieved context for the current sub-question, write a concise, one-sentence summary maintaining to use main Key Words of the key findings.
This summary will be added to our research history. Be factual and to the point."""),
    ("human", "Current sub-question: {sub_question}\n\nDistilled context:\n{context}")
])

reflection_agent = reflection_prompt | reasoning_llm | StrOutputParser()


# Policy Agent
policy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a easy strategist. Your role is to analyze the research progress and decide the next action.
You have the original question, the initial plan, and a log of completed steps with their summaries.
- If the collected information in the Research History is sufficient to comprehensively answer the Original Question, decide to FINISH.
- Otherwise, if the plan is not yet complete, decide to CONTINUE_PLAN."""),
    ("human", "Original Question: {question}\n\nInitial Plan:\n{plan}\n\nResearch History (Completed Steps):\n{history}")
])

policy_agent = policy_prompt | reasoning_llm.with_structured_output(Decision)


print("All agents created successfully.")