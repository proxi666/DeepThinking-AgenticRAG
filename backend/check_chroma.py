
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

baseline_path = './chroma_db/baseline'
if not os.path.exists(baseline_path):
    print(f'❌ Baseline chroma_db not found at {baseline_path}')
    exit(1)

chroma = Chroma(
    persist_directory=baseline_path,
    embedding_function=embeddings
)

count = chroma._collection.count()
print(f'✅ Baseline store has {count} documents')

if count == 0:
    print('❌ ERROR: Vector store is EMPTY!')
    print('Run: python build_vector_store.py')
else:
    # Test retrieval
    results = chroma.similarity_search('NVIDIA risks', k=5)
    print(f'✅ Test query returned {len(results)} documents')
    if results:
        print(f'   First result: {results[0].page_content[:200]}...')
