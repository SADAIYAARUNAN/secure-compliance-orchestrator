import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


from src.state import AgentState


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


def draft_assessment(state: AgentState) -> Dict[str, Any]:
    """
    Combines sanitized telemetry and retrieved Qdrant compliance policies 
    to generate a structured risk assessment summary.
    """

    sanitized_text = state.sanitized_telemetry
    policies = state.retrieved_policies

    if not sanitized_text:
        return{
            "risk_summary": "Error: No sanitized telemetry available for assessment."
        }

    formatted_policies = "\n".join(
        f"-[{p['policy_id']}] (Severity: {p['severity']}): {p['content']}"
        for p in policies
    )if policies else "No matching compliance policies found."


    prompt=f"""
    You are an expert AI Compliance and Legal Research Agent for European automotive systems.
    Review the following sanitized telemetry log against the retrieved regulatory policies.

    === SANITIZED TELEMETRY ===
    {sanitized_text}

    === RETRIEVED POLICIES ===
    {formatted_policies}

    === INSTRUCTIONS ===
    Provide a professional compliance risk assessment containing:
    1. Violation Status (Compliant / Non-Compliant / Review Required)
    2. Applicable Policy Risks identified from the retrieved database
    3. Recommended remediation steps for the engineering team.
    """


    try:
        response = llm.invoke([
            SystemMessage(
                content="You are a precise legal and regulatory compliance engineering assistant."),
                HumanMessage(content=prompt)
        ])
        return{
            "risk_summary": response.content,
            "message": [response]
        }
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return{
            "risk_summary": f"Failed to generate assessment due to API error: {e}",
            "error_count": state.error_count + 1
        }


