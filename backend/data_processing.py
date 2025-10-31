"""Document downloading, cleaning, and chunking utilities."""
import os
import re
import time
import uuid
import requests
from bs4 import BeautifulSoup
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config import config


def download_and_parse_10k(url: str, doc_path_raw: str, doc_path_clean: str):
    """Download and parse SEC 10-K filing."""
    match = re.search(r'doc=(.+?)(?:&|$)', url)
    if match:
        url = f"https://www.sec.gov{match.group(1)}"
    
    headers = {
        'User-Agent': 'sid sv6659@srmist.edu.in',  # CHANGE THIS
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    
    response = requests.get(url, headers=headers)
    time.sleep(1)
    
    with open(doc_path_raw, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    soup = BeautifulSoup(response.text, 'xml')
    
    for hidden in soup.find_all(['ix:hidden', 'ix:header', 'ix:resources']):
        hidden.decompose()
    for tag in soup(['script', 'style', 'meta', 'link']):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.startswith('ix:')):
        tag.unwrap()
    
    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    with open(doc_path_clean, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Success: {len(text)} chars")


def load_and_chunk_document(doc_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Load and chunk a text document."""
    print("Loading and chunking the doc")
    loader = TextLoader(doc_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Document loaded and split into {len(doc_chunks)} chunks.")
    return documents, doc_chunks, text_splitter


def create_metadata_chunks(documents: List[Document], text_splitter, doc_path_clean: str) -> List[Document]:
    """Create metadata-aware chunks from documents."""
    raw_text = documents[0].page_content
    section_pattern = r"(^ITEM\s+\d[A-Z]?\..*)"
    split_text = re.split(section_pattern, raw_text, flags=re.MULTILINE | re.IGNORECASE)
    sections = split_text[1:]
    
    doc_chunks_with_metadata = []
    print(f"Regex split the document into {len(sections)} parts (titles and content).")
    
    for i in range(0, len(sections), 2):
        section_title = sections[i].strip()
        section_content = sections[i+1].strip()
        section_chunks = text_splitter.split_text(section_content)
        
        for chunk in section_chunks:
            if len(chunk) < 150:
                continue
                
            chunk_id = str(uuid.uuid4())
            doc_chunks_with_metadata.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "section": section_title,
                        "source_doc": doc_path_clean,
                        "id": chunk_id
                    }
                )
            )
    
    print(f"Created {len(doc_chunks_with_metadata)} chunks with section metadata.")
    return doc_chunks_with_metadata


# Initialize data
os.makedirs(config["data_dir"], exist_ok=True)
url_10k = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581025000209/nvda-20250727.htm"
doc_path_raw = f"{config['data_dir']}/nvda_q2_2025_raw.html"
doc_path_clean = f"{config['data_dir']}/nvda_q2_2025_clean.txt"