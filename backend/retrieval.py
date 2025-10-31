"""Retrieval strategies and reranking functions."""
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from sentence_transformers import CrossEncoder

from config import config
import vector_store as vs


# Initialize web search tool
web_search_tool = TavilySearch(topic="general", k=3)

# Initialize reranker
print("Initializing CrossEncoder reranker...")
reranker = CrossEncoder(config["reranker_model"])


def vector_search_only(query: str, section_filter: str = None, k: int = 10) -> List[Document]:
    """Pure vector search with optional metadata filtering."""
    filter_dict = {"section": section_filter} if section_filter and "Unknown" not in section_filter else None
    return vs.advanced_vector_store.similarity_search(query, k=k, filter=filter_dict)


def bm25_search_only(query: str, k: int = 10) -> List[Document]:
    """Pure keyword search using BM25."""
    if vs.bm25 is None or vs.doc_map is None:
        raise RuntimeError(
            "BM25 index is not initialised. Call `init_advanced_vector_store` before using keyword search."
        )

    tokenized_query = query.split(' ')
    bm25_scores = vs.bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    return [vs.doc_map[vs.doc_ids[i]] for i in top_k_indices]


def hybrid_search(query: str, section_filter: str = None, k: int = 10) -> List[Document]:
    """Hybrid search with Reciprocal Rank Fusion (RRF)."""
    bm25_docs = bm25_search_only(query, k=k)
    semantic_docs = vector_search_only(query, section_filter=section_filter, k=k)
    
    all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}.values()
    ranked_lists = [
        [doc.metadata["id"] for doc in bm25_docs],
        [doc.metadata["id"] for doc in semantic_docs]
    ]
    
    rrf_scores = {}
    for doc_list in ranked_lists:
        for i, doc_id in enumerate(doc_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (i + 61)
    
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    final_docs = [vs.doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]
    return final_docs


def rerank_documents_function(query: str, documents: List[Document]) -> List[Document]:
    """Rerank documents using cross-encoder."""
    if not documents:
        return []
    
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in doc_scores[:config["top_n_rerank"]]]
    return reranked_docs


def web_search_function(query: str) -> List[Document]:
    """Search the web using Tavily."""
    response = web_search_tool.invoke({"query": query})
    results = response.get('results', [])
    return [
        Document(
            page_content=res["content"],
            metadata={"source": res["url"]}
        ) for res in results
    ]


print("All retrieval functions ready.")
