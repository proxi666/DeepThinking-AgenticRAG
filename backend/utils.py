"""Helper utility functions."""
from typing import List
from langchain_core.documents import Document
from rich.console import Console

# Initialize console for pretty printing
console = Console()


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_past_context_str(past_steps: List[dict]) -> str:
    """Format research history for prompts."""
    return "\\n\\n".join([
        f"Step {s['step_index']}: {s['sub_question']}\\nSummary: {s['summary']}"
        for s in past_steps
    ])