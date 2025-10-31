import os
import requests
import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision, answer_correctness
from datasets import Dataset
from langchain_openai import ChatOpenAI  # Use a more reliable provider
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURE THE EVALUATION ---

# ✅ OPTION 1: Use OpenAI (most reliable for evaluation)
# eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)

# ✅ OPTION 2: Use Google's Gemini (if you have access)
# from langchain_google_genai import ChatGoogleGenerativeAI
# eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, timeout=120)

# ✅ OPTION 3: Use DeepSeek Chat (faster than reasoner)
from langchain_deepseek import ChatDeepSeek
eval_llm = ChatDeepSeek(
    model="deepseek-chat",  # Changed from deepseek-reasoner
    temperature=0, 
    timeout=120  # Reduced timeout
)

eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# The metrics we want to calculate
metrics = [
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
]

# The complex_query_adv we will be testing on
complex_query_adv = "Based on NVIDIA's 2025 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."

# The "ground truth" 
ground_truth_answer_adv = "NVIDIA's 2025 10-K lists intense competition and rapid technological change as key risks. This risk is exacerbated by AMD's 2026 strategy, specifically the launch of the MI300X AI accelerator, which directly competes with NVIDIA's H100 and has been adopted by major cloud providers, threatening NVIDIA's market share in the data center segment."



# --- 2. GATHER RESULTS FROM YOUR API ENDPOINTS ---

def get_baseline_results(query: str):
    """Get results from the baseline RAG system."""
    print("Querying Baseline RAG endpoint...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/query/baseline", 
            json={"query": query},
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        data = response.json()
        return {
            "answer": data.get("baseline_output", ""), 
            "contexts": data.get("contexts", [])
        }
    except Exception as e:
        print(f"Error querying baseline: {e}")
        return {"answer": "", "contexts": []}

def get_deep_thinking_results(query: str):
    """Get results from the deep thinking RAG system via streaming."""
    print("Querying Deep Thinking RAG streaming endpoint...")
    final_answer = ""
    contexts = []
    
    try:
        with requests.post(
            "http://127.0.0.1:8000/stream_query/deep_thinking", 
            json={"query": query}, 
            stream=True,
            timeout=600  # 10 minute timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        try:
                            data_str = decoded_line[len('data:'):].strip()
                            event = json.loads(data_str)
                            if event.get("type") == "final_answer":
                                final_answer = event.get("data", "")
                            elif event.get("type") == "contexts":
                                contexts = event.get("data", [])
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Error querying deep thinking: {e}")
        
    return {"answer": final_answer, "contexts": contexts}

# --- 3. RUN THE EVALUATION ---

if __name__ == "__main__":
    # Ensure server is running
    try:
        requests.get("http://127.0.0.1:8000/", timeout=5)
    except:
        print("ERROR: Server is not running! Start it with: uvicorn server:app --reload")
        exit(1)
    
    # Get results for both RAG systems
    baseline_result = get_baseline_results(complex_query_adv)
    deep_thinking_result = get_deep_thinking_results(complex_query_adv)
    
    # Check if we got valid results
    if not baseline_result["answer"] or not deep_thinking_result["answer"]:
        print("ERROR: Failed to get results from one or both systems")
        exit(1)
    
    print(f"\nBaseline answer length: {len(baseline_result['answer'])}")
    print(f"Deep thinking answer length: {len(deep_thinking_result['answer'])}")
    print(f"Baseline contexts: {len(baseline_result['contexts'])}")
    print(f"Deep thinking contexts: {len(deep_thinking_result['contexts'])}")
    
    # Format the results into a Dataset object RAGAs can use
    data = {
        "user_input": [complex_query_adv, complex_query_adv],
        "response": [baseline_result["answer"], deep_thinking_result["answer"]],
        "retrieved_contexts": [baseline_result["contexts"], deep_thinking_result["contexts"]],
        "reference": [ground_truth_answer_adv, ground_truth_answer_adv],
    }
    dataset = Dataset.from_dict(data)
    
    # Run the evaluation with error handling
    print("\nRunning RAGAs evaluation... (This may take a few minutes)")
    try:
        result = evaluate(
            dataset, 
            metrics=metrics, 
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False  # Don't fail on individual metric errors
        )
        
        # Display the results
        df = result.to_pandas()
        df['system'] = ["Baseline RAG", "Deep Thinking RAG"]
        
        print("\n\n=== RAGAs Evaluation Scorecard ===")
        print(df[['system', 'faithfulness', 'context_recall', 'context_precision', 'answer_correctness']].to_string(index=False))
        
        # Save results
        df.to_csv("evaluation_results.csv", index=False)
        print("\nResults saved to evaluation_results.csv")
        
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()