import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


from src.graph import app as compiled_graph


app = FastAPI(
    title="Secure Compliance orchestrator (SCO) API",
    description="production grade AI Agent API for legal telemetry sanitization and compliance evaluation.",
    version="1.0.0"
)

class TelemetryRequest(BaseModel):
    raw_telemetry: str
    thread_id: str = "default_session"


@app.post("/run-compliance-review")
async def run_compliance_review(request: TelemetryRequest):
    """
    Triggers the SCO LangGraph pipeline:
    1. Sanitizes telemetry data (Presidio)
    2. Queries compliance policies via hybrid search (Qdrant + RRF)
    3. Drafts structured risk assessment (LLM)
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_input = {"raw_telemetry": request.raw_telemetry}

        final_state = compiled_graph.invoke(initial_input, config=config)

        return {
            "status": "success",
            "thread_id": request.thread_id,
            "sanitized_telemetry": final_state.get("sanitized_telemetry"),
            "retrieved_policies_count": len(final_state.get("retrieved_policies",[])),
            "retrieved_policies": final_state.get("retrieved_policies"),
            "risk_summary": final_state.get("risk_summary")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thread/{thread_id}")
async def get_thread_state(thread_id: str):
    """
    Retrieves the current saved checkpoint state for a given thread ID,
    demonstrating production state management and recovery.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = compiled_graph.get_state(config)
        if not state_snapshot.values:
            return {"thread_id": thread_id, "status": "No active state found."}
        return{
            "thread_id": thread_id,
            "values": state_snapshot.values,
            "next_steps": state_snapshot.next
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
