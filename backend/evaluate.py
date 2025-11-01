"""
RAGAs Evaluation using Google Gemini 2.5 Flash
Simpler methodology with multi-query evaluation
"""
import os
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
import pandas as pd
from dotenv import load_dotenv

# Import Gemini LangChain wrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Verify API key is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Add it to your .env file.")

print("=" * 80)
print("🔬 RAGAs Evaluation with Google Gemini 2.5 Flash")
print("=" * 80)

# --- 1. CONFIGURE GEMINI MODELS ---
print("\n✅ Initializing Gemini 2.5 Flash for evaluation...")

# Use Gemini 2.5 Flash - Best balance of speed, cost, and quality
eval_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Stable model, production-ready
    temperature=0,  # Deterministic for evaluation
)

# Use Google's embedding model (same as in your vector store)
eval_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("✅ Gemini 2.5 Flash initialized successfully")

# --- CHECK SERVER ---
print("\n🔍 Checking if server is running...")
try:
    response = requests.get("http://127.0.0.1:8000/", timeout=5)
    if response.status_code == 200:
        print("✅ Server is running")
    else:
        print(f"⚠️ Server returned status {response.status_code}")
except Exception as e:
    print("❌ Server is NOT running!")
    print(f"   Error: {e}")
    print("\n💡 Start the server with:")
    print("   uvicorn server:app --reload")
    exit(1)

# --- 2. DEFINE TEST CASES ---
TEST_CASES = [
    {
        "id": "Q1_Competition_Risk",
        "query": """From NVIDIA's FY2025 Form 10-K, extract the explicitly stated risks tied to competition. 
Then pull post-filing 2024 news about AMD's AI GPU strategy and explain precisely how AMD's 2024 moves 
either mitigate or amplify one of NVIDIA's stated competition risks. Answer in 100 words.""",
        "ground_truth": """From NVIDIA's FY2025 10-K: "Competition could adversely impact our market share and 
financial results," as rivals may offer cheaper/better products, have greater fab capacity, and customers may 
build in-house alternatives—pressuring prices and demand. In 2024 AMD accelerated its AI GPU strategy: MI325X 
shipping in Q4'24 with broad OEM support; MI350 (CDNA 4) slated for 2025 promising up to 35× inference gains 
vs. MI300; plus ROCm updates doubling MI300X training/inference and expanding model support. These moves 
strengthen a credible alternative for hyperscalers, increasing price/performance and ecosystem competition—directly 
amplifying NVIDIA's stated competition risk."""
    },
    {
        "id": "Q2_Supply_Chain",
        "query": """List NVIDIA's FY2025 10-K disclosures on supply/manufacturing concentration and long lead times. 
Cross-check 2024 reports about AMD AI GPU availability/shipments to hyperscalers. Score (0–5) how much AMD's 2024 
supply posture could worsen NVIDIA's lead-time risk in 2025–26, citing evidence, and answer in 100 words.""",
        "ground_truth": """NVIDIA FY2025 10-K flags: (1) "Long manufacturing lead times," sometimes ">12 months," 
creating supply-demand mismatches; (2) reliance on third-party suppliers/limited capacity for manufacturing, assembly, 
test, packaging; (3) geopolitical exposure where key suppliers/assemblers are in Taiwan/China, risking continuity. 
2024 AMD supply posture: MI300X made generally available on Microsoft Azure (May 2024); AMD said MI325X would ship 
in Q4'24 with broad OEM availability in Q1'25; industry estimates suggest ~307k MI300X shipped to Meta, Microsoft, 
Oracle in 2024. Score: 4/5. Evidence of real 2024 shipments plus expanding cloud/OEM channels likely shortens 
non-NVIDIA procurement paths, worsening NVIDIA's lead-time risk into 2025–26."""
    },
    {
        "id": "Q3_Ecosystem_Lock_in",
        "query": """Identify language in NVIDIA's FY2025 10-K about needing to expand/maintain its ecosystem. 
Compare with 2024 news on ROCm updates, cloud availability of AMD Instinct, and major model workloads running 
on AMD. Explain whether ROCm in 2024 measurably reduces CUDA lock-in, using at least 2 concrete developer-facing 
examples and answer in 100 words.""",
        "ground_truth": """10-K ecosystem language: NVIDIA says CUDA is the "foundational" model for its platforms 
and that a "large and growing number of developers… strengthens our ecosystem," with software support a key 
competitive factor. ROCm/AMD 2024: ROCm 6.2 shipped (Aug 2024). Azure made MI300X VMs GA (May 2024). OCI showed 
Llama-3.1-405B running on MI300X. vLLM published MI300X best-practices and AMD released a prebuilt ROCm+vLLM Docker. 
Does this reduce CUDA lock-in? Partly: • vLLM on ROCm serves Llama 3.x with strong throughput—drop-in 
OpenAI-compatible API. • Azure ND MI300X + ROCm gives managed, ready images/drivers for PyTorch/ROCm. Conclusion: 
ROCm meaningfully narrows dependence, but CUDA's ecosystem remains larger."""
    },
    {
        "id": "Q4_Customer_Concentration",
        "query": """Summarize NVIDIA's FY2025 10-K on customer concentration and risks from partners/cloud providers. 
Then gather 2024 items where hyperscalers adopted or trialed AMD Instinct. Build two scenarios (Base, Adverse) 
estimating how a 10–20% workload shift to AMD in inference could impact NVIDIA's pricing power and gross margin 
qualitatively. Answer in 100 words.""",
        "ground_truth": """NVIDIA FY2025 10-K flags revenue concentration in a limited set of partners and customers, 
with top direct customers at 12%, 11%, 11%, and warns partners may change purchasing or build in-house, harming 
sales. It also depends on CSPs to host DGX Cloud, where timing and availability shifts can affect results. 2024 AMD 
adoption: Azure made MI300X VMs GA, Microsoft offered AMD as an alternative, and Oracle Cloud launched MI300X 
instances. Reports cited strong hyperscaler demand. Scenarios Base: 10% inference shift to AMD trims NVIDIA pricing 
leverage at renewals, modest gross margin pressure. Adverse: 20% shift plus multi cloud AMD capacity triggers 
discounting and slower mix to premium SKUs, meaningfully compressing margins."""
    },
    {
        "id": "Q5_Product_Cadence",
        "query": """Extract where NVIDIA's FY2025 10-K highlights risks from rapid tech cycles and competitor releases. 
Overlay AMD's 2024–2025 public accelerator roadmap milestones (e.g., MI300X/MI325X) with NVIDIA's announced platforms. 
Argue whether AMD's 2024 cadence directly targets any 10-K-stated vulnerability windows for NVIDIA. Answer in 100 words.""",
        "ground_truth": """NVIDIA's FY2025 10-K warns product cycles are "rapid," with success hinging on "timely" 
new releases; competitor products or shifts in standards could render offerings less competitive. AMD's 2024–25 
cadence: MI300X cloud availability (OCI), MI325X "available Q4 2024" with broad OEMs, and MI350 (CDNA 4) expected 
2025; NVIDIA's counter cadence is Blackwell (GB200/NVL72) announced Mar 2024, broad cloud in 2025. Assessment: 
Yes—AMD's late-2024 MI325X shipments and 2025 MI350 target NVIDIA's transition window from Hopper to Blackwell; 
any Blackwell slippage magnifies this vulnerability."""
    }
]

# --- 3. GATHER RESULTS FROM API ---
import json  # Add this import

def get_results_from_api(query: str, endpoint: str):
    """Helper to get results from FastAPI endpoints."""
    try:
        if "baseline" in endpoint:
            print(f"    Calling: POST http://127.0.0.1:8000/query/baseline")
            response = requests.post(
                "http://127.0.0.1:8000/query/baseline",
                json={"query": query},
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("baseline_output", "")
            contexts = data.get("contexts", [])
            
            print(f"    Got: {len(answer)} chars, {len(contexts)} contexts")
            return {"answer": answer, "contexts": contexts}
            
        else:  # deep_thinking
            print(f"    Calling: POST http://127.0.0.1:8000/stream_query/deep_thinking")
            final_answer = ""
            contexts = []
            
            response = requests.post(
                "http://127.0.0.1:8000/stream_query/deep_thinking",
                json={"query": query},
                stream=True,
                timeout=600
            )
            response.raise_for_status()
            
            # Collect ALL streaming events
            for line in response.iter_lines():
                if not line:
                    continue
                    
                decoded_line = line.decode('utf-8')
                if not decoded_line.startswith('data:'):
                    continue
                
                try:
                    data_str = decoded_line[5:].strip()  # Remove 'data:' prefix
                    event = json.loads(data_str)
                    
                    event_type = event.get("type")
                    if event_type == "final_answer":
                        final_answer = event.get("data", "")
                        print(f"      → Got final answer: {len(final_answer)} chars")
                    elif event_type == "contexts":
                        contexts = event.get("data", [])
                        print(f"      → Got contexts: {len(contexts)} items")
                except json.JSONDecodeError as e:
                    print(f"      → JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"      → Event processing error: {e}")
                    continue
            
            print(f"    Final: {len(final_answer)} chars, {len(contexts)} contexts")
            
            if not final_answer:
                print(f"    ⚠️ WARNING: No final_answer received!")
            if not contexts:
                print(f"    ⚠️ WARNING: No contexts received!")
            
            return {"answer": final_answer, "contexts": contexts}
            
    except requests.exceptions.ConnectionError:
        print(f"    ❌ Connection Error: Is the server running?")
        print(f"       Start server with: uvicorn server:app --reload")
        return {"answer": "", "contexts": []}
    except requests.exceptions.Timeout:
        print(f"    ❌ Timeout Error: Request took too long")
        return {"answer": "", "contexts": []}
    except Exception as e:
        print(f"    ❌ Error: {type(e).__name__}: {e}")
        return {"answer": "", "contexts": []}

# --- 4. COLLECT ALL RESULTS ---
print("\n" + "=" * 80)
print("📥 Collecting Results from Both RAG Systems")
print("=" * 80)

questions = []
baseline_answers = []
deep_thinking_answers = []
baseline_contexts = []
deep_thinking_contexts = []
ground_truths = []

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\n[{i}/5] Processing: {test_case['id']}")
    
    # Get baseline result
    print("  → Querying Baseline RAG...")
    baseline_result = get_results_from_api(test_case['query'], 'baseline')
    
    # Get deep thinking result
    print("  → Querying Deep Thinking RAG...")
    deep_result = get_results_from_api(test_case['query'], 'deep_thinking')
    
    # Check if BOTH results are valid (have answers)
    has_baseline = bool(baseline_result['answer'])
    has_deep = bool(deep_result['answer'])
    
    if has_baseline and has_deep:
        questions.append(test_case['query'])
        baseline_answers.append(baseline_result['answer'])
        deep_thinking_answers.append(deep_result['answer'])
        baseline_contexts.append(baseline_result['contexts'])
        deep_thinking_contexts.append(deep_result['contexts'])
        ground_truths.append(test_case['ground_truth'])
        print(f"  ✅ Success: Baseline ({len(baseline_result['answer'])} chars, {len(baseline_result['contexts'])} ctx), "
              f"Deep Thinking ({len(deep_result['answer'])} chars, {len(deep_result['contexts'])} ctx)")
    else:
        missing = []
        if not has_baseline:
            missing.append("Baseline")
        if not has_deep:
            missing.append("Deep Thinking")
        print(f"  ⚠️ Skipped - missing {', '.join(missing)} answer(s)")

print(f"\n✅ Collected {len(questions)} complete query results")

# --- VALIDATE DATA ---
if len(questions) == 0:
    print("\n" + "=" * 80)
    print("❌ ERROR: No results collected!")
    print("=" * 80)
    print("\nPossible causes:")
    print("1. Server is not running")
    print("   → Start with: uvicorn server:app --reload")
    print("2. API endpoints are returning errors")
    print("   → Check server logs for errors")
    print("3. Network/connection issues")
    print("   → Try: curl http://127.0.0.1:8000/")
    exit(1)

# --- 5. PREPARE DATASET ---
print("\n" + "=" * 80)
print("📊 Preparing RAGAs Evaluation Dataset")
print("=" * 80)

# Combine data for both systems
eval_data = {
    'user_input': questions + questions,  # Repeat for both systems
    'response': baseline_answers + deep_thinking_answers,
    'retrieved_contexts': baseline_contexts + deep_thinking_contexts,
    'reference': ground_truths + ground_truths,
}

# Create Dataset
eval_dataset = Dataset.from_dict(eval_data)
print(f"✅ Dataset created with {len(eval_dataset)} samples")

# --- 6. RUN EVALUATION ---
print("\n" + "=" * 80)
print("🔬 Running RAGAs Evaluation with Gemini 2.5 Flash")
print("=" * 80)
print("⏳ This may take 5-10 minutes...\n")

metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
]

try:
    result = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    
    print("✅ Evaluation complete!")
    
    # --- 7. FORMAT AND DISPLAY RESULTS ---
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS")
    print("=" * 80)
    
    # Convert to DataFrame
    results_df = result.to_pandas()
    
    # Split into baseline and deep thinking
    n_queries = len(questions)
    baseline_df = results_df.iloc[:n_queries].copy()
    deep_thinking_df = results_df.iloc[n_queries:].copy()
    
    # Calculate averages
    baseline_avg = baseline_df[['faithfulness', 'context_recall', 'context_precision', 'answer_correctness']].mean()
    deep_thinking_avg = deep_thinking_df[['faithfulness', 'context_recall', 'context_precision', 'answer_correctness']].mean()
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': ['Faithfulness', 'Context Recall', 'Context Precision', 'Answer Correctness'],
        'Baseline RAG': [
            f"{baseline_avg['faithfulness']:.3f}",
            f"{baseline_avg['context_recall']:.3f}",
            f"{baseline_avg['context_precision']:.3f}",
            f"{baseline_avg['answer_correctness']:.3f}",
        ],
        'Deep Thinking RAG': [
            f"{deep_thinking_avg['faithfulness']:.3f}",
            f"{deep_thinking_avg['context_recall']:.3f}",
            f"{deep_thinking_avg['context_precision']:.3f}",
            f"{deep_thinking_avg['answer_correctness']:.3f}",
        ],
        'Improvement': [
            f"+{((deep_thinking_avg['faithfulness'] - baseline_avg['faithfulness']) / baseline_avg['faithfulness'] * 100):.1f}%",
            f"+{((deep_thinking_avg['context_recall'] - baseline_avg['context_recall']) / (baseline_avg['context_recall'] + 0.001) * 100):.1f}%",
            f"+{((deep_thinking_avg['context_precision'] - baseline_avg['context_precision']) / (baseline_avg['context_precision'] + 0.001) * 100):.1f}%",
            f"+{((deep_thinking_avg['answer_correctness'] - baseline_avg['answer_correctness']) / baseline_avg['answer_correctness'] * 100):.1f}%",
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save detailed results
    baseline_df.index = [f"Q{i+1}_Baseline" for i in range(n_queries)]
    deep_thinking_df.index = [f"Q{i+1}_DeepThinking" for i in range(n_queries)]
    
    combined_df = pd.concat([baseline_df, deep_thinking_df])
    combined_df.to_csv("evaluation_results.csv")
    
    # Save summary
    with open("evaluation_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GEMINI 2.5 FLASH - RAGAs EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluation Model: Gemini 2.5 Flash\n")
        f.write(f"Embedding Model: Gemini Embedding 001\n")
        f.write(f"Test Queries: {n_queries}\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
    
    print("\n✅ Results saved:")
    print("   • evaluation_results.csv (detailed per-query results)")
    print("   • evaluation_summary.txt (aggregate summary)")
    
    print("\n" + "=" * 80)
    print("✨ EVALUATION COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()