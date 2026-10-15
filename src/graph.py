import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import AgentState
from src.nodes.sanitize import sanitize_data
from src.nodes.retrieve import query_compliance_db
from src.nodes.draft import draft_assessment


def error_counting(state: AgentState) -> str:
    """
    Conditional edge : Checks if tool failures have occurred.
    If the errors exceeds threshold , routes to END. Otherwise proceeds.
    """

    if state.error_count >= 3:
        return "end"
    return "continue"

# Initialize Graph
workflow  = StateGraph(AgentState)


# Add Nodes
workflow.add_node("sanitize",sanitize_data)
workflow.add_node("retrieve", query_compliance_db)
workflow.add_node("draft",draft_assessment)


# Add Edges (Pipeline Flow)
workflow.set_entry_point("sanitize")
workflow.add_edge("sanitize","retrieve")
workflow.add_edge("retrieve","draft")
workflow.add_edge("draft",END)


# SetUp SQLite for persistent state storage and recovery
db_conn = sqlite3.connect("sco_checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(db_conn)


app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":

    config = {"configurable": {"thread_id": "test-session-001"}}

    initial_input = {
        "raw_telemetry": (
            "Event ID: EVT-88392-NL. Customer Bram Van der Berg called from +31-6-12345678 "
            "at location 52.3676 N, 4.9041 E (Amsterdam, Netherlands). "
            "Vehicle VIN WBA00000000000000 initiated a diagnostic loop. "
            "Continuous location polling active during infotainment hard reset. "
            "User voice commands recorded during reset cycle to debug microphone dropout."
        )
    }


    final_state = app.invoke(initial_input, config=config)


    print("\n[RESULT] Sanitized Telemetry:")
    print(final_state["sanitized_telemetry"])
    
    print(f"\n[RESULT] Policies Retrieved: {len(final_state['retrieved_policies'])}")
    
    print("\n[RESULT] Risk Assessment Summary:")
    print(final_state["risk_summary"])






