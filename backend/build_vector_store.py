"""
Offline script to download, process, and embed the 10-K document.
This script creates and saves all necessary vector stores and indexes
so that the main application can start up quickly.
"""
from data_processing import (
    download_and_parse_10k, load_and_chunk_document,
    create_metadata_chunks, url_10k, doc_path_raw, doc_path_clean
)
from vector_store import create_and_save_stores
from utils import console

def main():
    """Main function to run the offline processing pipeline."""
    console.print("[bold yellow]=== STARTING OFFLINE VECTOR STORE BUILD ===", style="yellow")

    # Step 1: Data Acquisition
    console.print("\n[bold cyan]STEP 1: DATA ACQUISITION[/bold cyan]")
    download_and_parse_10k(url_10k, doc_path_raw, doc_path_clean)
    
    # Step 2: Document Processing
    console.print("\n[bold cyan]STEP 2: DOCUMENT PROCESSING[/bold cyan]")
    documents, doc_chunks, text_splitter = load_and_chunk_document(doc_path_clean)
    
    # Step 3: Metadata Enrichment
    console.print("\n[bold cyan]STEP 3: METADATA ENRICHMENT[/bold cyan]")
    doc_chunks_with_metadata = create_metadata_chunks(documents, text_splitter, doc_path_clean)
    
    # Step 4: Create and Save Vector Stores
    console.print("\n[bold cyan]STEP 4: VECTOR STORE CREATION[/bold cyan]")
    create_and_save_stores(doc_chunks, doc_chunks_with_metadata)

    console.print("\n[bold green]=== OFFLINE BUILD COMPLETE ===", style="green")
    console.print(f"Vector stores and indexes are saved in the 'chroma_db' directory.")
    console.print("You can now start the main application with `uvicorn server:app --reload`.")

if __name__ == "__main__":
    main()