"""Pydantic models for the Deep Thinking RAG system."""
from typing import List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class Step(BaseModel):
    """A single step in the agent's reasoning plan."""
    sub_question: str = Field(description="A specific, answerable question for this step.")
    justification: str = Field(description="A brief explanation of why this step is necessary to answer the main query.")
    tool: Literal["search_10k", "search_web"] = Field(description="The tool to use for this step.")
    keywords: List[str] = Field(description="A list of critical keywords for searching relevant document sections.")
    document_section: Optional[str] = Field(description="A likely document section title (e.g., 'Item 1A. Risk Factors') to search within. Only for 'search_10k' tool.")


class Plan(BaseModel):
    """The overall plan, which is a list of individual steps."""
    steps: List[Step] = Field(description="A detailed, multi-step plan to answer the user's query.")


class PastStep(TypedDict):
    """Results of a completed step in our research history."""
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str


class RAGState(TypedDict):
    """Main state dictionary passed between all nodes in the LangGraph agent."""
    original_question: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    synthesized_context: str
    final_answer: str


class RetrievalDecision(BaseModel):
    """Retrieval strategy decision."""
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"]
    justification: str


class Decision(BaseModel):
    """Policy decision for continuing or finishing."""
    next_action: Literal["CONTINUE_PLAN", "FINISH"]
    justification: str