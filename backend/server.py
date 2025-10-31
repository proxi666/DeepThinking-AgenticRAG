import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict

from fastapi.middleware.cors import CORSMiddleware


# Import the new, separated functions from rag_core
from rag_core import initialize_rag_components, run_baseline_rag, get_deep_thinking_stream
# Import StreamingResponse for our new endpoint
from fastapi.responses import StreamingResponse

# --- Pydantic Model (stays the same) ---
class QueryRequest(BaseModel):
    query: str

# --- FastAPI Lifespan (stays the same) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    initialize_rag_components()
    print("RAG components initialized.")
    yield
    print("Server shutting down...")

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_rag_events(query: str):
    """
    An async generator that yields formatted Server-Sent Events (SSE).
    This version is designed to send each unique event only once.
    """
    try:
        loop = asyncio.get_event_loop()
        stream_iterator = await loop.run_in_executor(None, get_deep_thinking_stream, query)

        # Keep track of which step summaries we've already sent
        sent_step_summaries = set()

        for chunk in stream_iterator:
            event_data = None  # Reset for each chunk

            # --- Event 1: The Plan (Sent Once) ---
            # The 'plan' key is added only by the plan_node. We can check for its presence.
            # We also check if the step index is 0 to ensure it's the first time.
            if "plan" in chunk and 0 not in sent_step_summaries:
                event_data = {"type": "plan", "data": [step.dict() for step in chunk["plan"].steps]}
                # Mark step 0 as "sent" so we don't send the plan again
                sent_step_summaries.add(0)

            # --- Event 2: A Step Result (Sent Once Per Step) ---
            # The 'past_steps' list grows after the reflection_node runs.
            # We can check its length to see if a new step has been completed.
            if "past_steps" in chunk:
                for step_result in chunk["past_steps"]:
                    step_index = step_result["step_index"]
                    if step_index not in sent_step_summaries:
                        event_data = {
                            "type": "step_result", 
                            "data": {
                                "step": step_result["step_index"],
                                "sub_question": step_result["sub_question"],
                                "summary": step_result["summary"],
                            }
                        }
                        sent_step_summaries.add(step_index)
            
            # --- Event 3: The Final Answer (Sent Once) ---
            # The 'final_answer' key is only added by the final_answer_node at the very end.
            if "final_answer" in chunk:
                event_data = {"type": "final_answer", "data": chunk["final_answer"]}

            # --- Event 4: All Contexts (for evaluation) ---
            elif "contexts_complete" in chunk:
                event_data = {"type": "contexts", "data": chunk["contexts_complete"]}

            if event_data:
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0.01) # A tiny sleep helps with streaming delivery

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        error_data = {"type": "error", "data": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Deep Thinking RAG API is running."}

### NEW ENDPOINT 1: BASELINE RAG ###
@app.post("/query/baseline")
async def process_baseline_query(request: QueryRequest) -> Dict:
    """
    Processes a query using ONLY the baseline RAG pipeline.
    """
    try:
        print(f"Received baseline query: {request.query}")
        result = await asyncio.to_thread(run_baseline_rag, request.query)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

### NEW ENDPOINT 2: DEEP THINKING RAG (STREAMING) ###
@app.post("/stream_query/deep_thinking")
async def stream_deep_thinking_query(request: QueryRequest):
    """
    Processes a query using the Deep Thinking RAG pipeline and streams the
    intermediate steps and final answer as Server-Sent Events (SSE).
    """
    print(f"Received deep thinking stream query: {request.query}")
    return StreamingResponse(stream_rag_events(request.query), media_type="text/event-stream")