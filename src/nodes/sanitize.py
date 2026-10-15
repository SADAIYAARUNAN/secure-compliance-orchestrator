import operator
from pydoc import text
from typing import Dict, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, analyzer_engine
from presidio_anonymizer import AnonymizerEngine, anonymizer_engine, entities
from presidio_anonymizer.entities import OperatorConfig


# Import state scheme
from src.state import AgentState

# Initiate presidio engine
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# Create custom recognizer for Vehicle Identification Number (VIN)
vin_pattern = Pattern(
    name="vin_pattern",
    regex=r"[A-HJ-NPR-Z0-9]{17}",
    score=0.9
)
vin_recognizer = PatternRecognizer(
    supported_entity="VIN",
    patterns=[vin_pattern]
)
analyzer.registry.add_recognizer(vin_recognizer)


def sanitize_data(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: sanitize_date
    Reads raw_telemetry, scrubs the standard PII and custom VIN'sm,
    and outputs the snitized telemetry.
    """

    raw_text = state.raw_telemetry

    if not raw_text:
        return {"sanitized_telemetry": "No raw telemetry provided."}


    # Analyse the text for PII and VIN
    results = analyzer.analyze(
        text = raw_text,
        entities=["PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS", "VIN"],
        language="en"
    )


    # detect the specifc entities and anonymize with masks
    anaonymized_result  = anonymizer.anonymize(
        text=raw_text,
        analyzer_results=results,
        operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON_REDACTED>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION_REDACTED>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_REDACTED>"}),
            "VIN": OperatorConfig("replace", {"new_value": "<VIN_REDACTED>"}),
        }
    )


    # return the updated state
    return {"sanitized_telemetry": anaonymized_result.text}



if __name__ == "__main__":

    # Mock state:
    mock_state = AgentState(
        raw_telemetry="""
        Event ID: EVT-88392-NL. Customer Bram Van der Berg called from +31-6-12345678 
        at location 52.3676 N, 4.9041 E (Amsterdam, Netherlands). 
        Vehicle VIN WBA00000000000000 initiated a diagnostic loop. 
        Continuous location polling active during infotainment hard reset. 
        User voice commands recorded during reset cycle to debug microphone dropout.
        """
    )


    result = sanitize_data(mock_state)

    print("RAW TELEMETERY")
    print(mock_state.raw_telemetry.strip())
    print("\n-SANITIZED TELEMETRY--")
    print(result["sanitized_telemetry"].strip())






