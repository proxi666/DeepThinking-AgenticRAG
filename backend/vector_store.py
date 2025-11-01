"""Vector store and BM25 index initialization."""
import os
import pickle
from typing import List
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config import config

# --- SHARED COMPONENTS ---
# Initialize embedding function once
embedding_function = GoogleGenerativeAIEmbeddings(model=config['embedding_model'])

# Define the persistent directory from the config file
PERSIST_DIRECTORY = config['persistent_db_dir']

# Define the path for our saved BM25 index file
BM25_PICKLE_PATH = os.path.join(PERSIST_DIRECTORY, "bm25_index.pkl")


# --- GLOBAL VARIABLES (to be used by other parts of the app) ---
# These will be populated by the loading functions.
baseline_retriever = None
advanced_vector_store = None
bm25 = None
doc_map = None
doc_ids = None

# ==============================================================================
# SECTION 1: CREATION AND SAVING LOGIC
# This part of the file is ONLY used by the offline 'build_vector_store.py' script.
# ==============================================================================

def create_and_save_stores(doc_chunks: List[Document], doc_chunks_with_metadata: List[Document]):
    """
    Creates and persists the baseline and advanced vector stores and the BM25 index.
    This function is meant to be run offline by the build_vector_store.py script.
    """
    print(f"Creating and saving all data to: {PERSIST_DIRECTORY}")
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # --- Create and Persist Baseline Store ---
    print("\n--- Processing Baseline Vector Store ---")
    Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_function,
        # This tells Chroma where to save the database files
        persist_directory=os.path.join(PERSIST_DIRECTORY, "baseline")
    )
    print("Baseline vector store created and saved successfully.")

    # --- Create and Persist Advanced Store ---
    print("\n--- Processing Advanced Vector Store ---")
    Chroma.from_documents(
        documents=doc_chunks_with_metadata,
        embedding=embedding_function,
        # This tells Chroma where to save the database files
        persist_directory=os.path.join(PERSIST_DIRECTORY, "advanced")
    )
    print("Advanced vector store created and saved successfully.")

    # --- Build and Save BM25 Index ---
    print("\n--- Processing BM25 Index ---")
    tokenized_corpus = [doc.page_content.split(' ') for doc in doc_chunks_with_metadata]
    doc_ids_local = [doc.metadata["id"] for doc in doc_chunks_with_metadata]
    doc_map_local = {doc.metadata["id"]: doc for doc in doc_chunks_with_metadata}
    bm25_local = BM25Okapi(tokenized_corpus)

    # Save the necessary BM25 components to a single file using pickle
    with open(BM25_PICKLE_PATH, "wb") as f:
        pickle.dump((bm25_local, doc_map_local, doc_ids_local), f)
    print("BM25 index built and saved successfully.")


# ==============================================================================
# SECTION 2: LOADING LOGIC
# This part of the file is used by 'rag_core.py' when the server starts.
# ==============================================================================

def load_persistent_stores():
    """
    Loads the persisted vector stores and BM25 index from disk into memory.
    This function is called by rag_core.py at startup.
    """
    # Use 'global' to modify the variables defined at the top of the file
    global baseline_retriever, advanced_vector_store, bm25, doc_map, doc_ids

    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Persistence directory not found at '{PERSIST_DIRECTORY}'. "
            "Please run the `build_vector_store.py` script first to create the necessary files."
        )

    # --- Load Baseline Store from Disk ---
    print("Loading baseline vector store from disk...")
    baseline_vs = Chroma(
        persist_directory=os.path.join(PERSIST_DIRECTORY, "baseline"),
        embedding_function=embedding_function
    )
    # This creates the retriever that the rest of our app will use
    baseline_retriever = baseline_vs.as_retriever(search_kwargs={"k": 10})
    print(f"DEBUG: Created baseline_retriever with k=10")

    # ADD TEST
    print("\n🧪 Testing baseline retriever...")
    test_query = "What are NVIDIA's main risks?"
    test_docs = baseline_retriever.invoke(test_query)
    print(f"   Test retrieval returned {len(test_docs)} documents")
    if test_docs:
        print(f"   First doc preview: {test_docs[0].page_content[:100]}...")
    else:
        print("   ⚠️ WARNING: Test retrieval returned NO documents!")

    # --- Load Advanced Store from Disk ---
    print("\nLoading advanced vector store from disk...")
    advanced_vector_store = Chroma(
        persist_directory=os.path.join(PERSIST_DIRECTORY, "advanced"),
        embedding_function=embedding_function
    )
    print("Advanced vector store loaded.")
    
    # --- Load BM25 Index from Disk ---
    print("\nLoading BM25 index from disk...")
    with open(BM25_PICKLE_PATH, "rb") as f:
        bm25, doc_map, doc_ids = pickle.load(f)
    print("BM25 index loaded.")