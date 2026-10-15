from typing import Annotated, List, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):

    # Ensures new messages are appended and not overwritten
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Raw input from the FastAPI Endpoint
    raw_telemetry: str = Field(default="")

    # The scrubbed text 
    sanitized_telemetry: str = Field(default="")

    # Documents pulled from Hybrid Search
    retrieved_policies: List[Dict[str, Any]] = Field(default_factory=list)

    # Tracks Tool failures to prevent infinite loops
    error_count: int = Field(default=0)

    # The final LLM output
    risk_summary: str = Field(default="")