import os
import atexit
from typing import Dict, Any, List
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

from src.state import AgentState


# Initializing Enbeddings Models
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
COLLECTION_NAME = "compliance_policies"

dense_embedder = TextEmbedding(model_name=DENSE_MODEL_NAME)
sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

# Initialize Qdrant Client
_client = None
def get_client() -> QdrantClient:
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        _client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path="./qdrant_data")
        atexit.register(_client.close)
    return _client


# MOCK Data Policies
MOCK_POLICIES: List[Dict[str, Any]] = [
{
        "id": 1,
        "policy_id": "GDPR-TELEMETRY-NL",
        "category": "Location Data",
        "content": (
            "Continuous GPS location polling is strictly prohibited during infotainment "
            "or system resets unless explicit user consent is re-verified via the mobile application."
        ),
        "severity": "High",
        "access_level": "Tier-1",
    },
    {
        "id": 2,
        "policy_id": "EU-VOICE-ACT",
        "category": "Audio Recording",
        "content": (
            "In-cabin voice commands and audio recordings must be processed strictly in volatile "
            "memory and cannot be written to persistent storage during diagnostic loops."
        ),
        "severity": "Critical",
        "access_level": "Tier-1",
    },
    {
        "id": 3,
        "policy_id": "ISO-21434-INTERNAL",
        "category": "Firmware Key Management",
        "content": (
            "Root cryptographic keys for ECUs must be rotated internally every 90 days."
        ),
        "severity": "Low",
        "access_level": "Restricted-Admin",  # Will be filtered out for Tier-1 queries
    },
]


def _embed_dense(text: str) -> List[float]:
    """
    Generate a 384 dimensional dense vector
    """
    return next(dense_embedder.embed([text])).tolist()


def _embed_sparse(text: str) -> models.SparseVector:
    """
    Generate a BM25 Sparse vector.
    """
    sparse_vec = next(sparse_embedder.embed([text]))
    return models.SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist(),
    )


def seed_knowledge_base() -> None:
    """
    Creates hybrid collection and indexes.
    """
    db = get_client()
    if db.collection_exists(COLLECTION_NAME):
        return

    db.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )

    if db.count(collection_name=COLLECTION_NAME).count > 0:
        return

    points = []
    for policy in MOCK_POLICIES:
        content = policy["content"]
        points.append(
            models.PointStruct(
                id=policy["id"],
                vector={
                    "dense":_embed_dense(content),
                    "sparse": _embed_sparse(content)
                },
                payload={
                    "policy_id": policy["policy_id"],
                    "category": policy["category"],
                    "content": content,
                    "severity": policy["severity"],
                    "access_level": policy["access_level"]
                },
            )
        )

    db.upsert(collection_name=COLLECTION_NAME, points=points)


def query_compliance_db(state: AgentState) -> Dict[str, Any]:
    """
    Executes a hybrid search (Dense + Sparse BM25 with RRF fusion) against Qdrant
    using the sanitized telemetry text and enforces- access level payload filtering. 
    """

    query_text = state.sanitized_telemetry

    if not query_text:
        return{
            "retrieved_policies": [],
            "error_count": state.error_count + 1,
        }

    try:
        seed_knowledge_base()
        db = get_client()

        dense_query = _embed_dense(query_text)
        sparse_query = _embed_sparse(query_text)

        # Access control via Qdrant Payload Filter
        access_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="access_level",
                    match=models.MatchValue(value="Tier-1"),
                )
            ]
        )


        # Execute Hybrid Search using Reciprocal rank Fusion
        response = db.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=access_filter,
                    limit=5,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    filter=access_filter,
                    limit=5
                ),
            ],
            query = models.FusionQuery(fusion=models.Fusion.RRF),
            limit=2,
            with_payload=True

        )


        policies  = [
            {
                "policy_id": point.payload.get("policy_id"),
                "category": point.payload.get("category"),
                "severity": point.payload.get("severity"),
                "content": point.payload.get("content"),
                "score": round(point.score, 4),
            }
            for point in response.points
            if point.payload
        ]

        return {"retrieved_policies": policies}

    except Exception as e:
        print(f"[ERROR] Qdrant retrieval failed: {e}")
        return{
            "retrieved_policies": [],
            "error_count": state.error_count + 1,
        }


if __name__ == "__main__":
    mock_state = AgentState(
        sanitized_telemetry=(
            "Event ID: EVT-88392-NL. Customer <PERSON_REDACTED> called from <PHONE_REDACTED> "
            "at location <LOCATION_REDACTED>. Vehicle VIN <VIN_REDACTED> initiated a diagnostic loop. "
            "Continuous location polling active during infotainment hard reset. "
            "User voice commands recorded during reset cycle to debug microphone dropout."
        )
    )

    output = query_compliance_db(mock_state)

    print(f"\n--- Retrieved policies (Count: {len(output['retrieved_policies'])}) ---")
    for idx, doc in enumerate(output['retrieved_policies'], start=1):
        print(f"\n{idx}. [{doc['policy_id']}] (Severity: {doc['severity']} | Score: {doc['score']})")
        print(f"   Category: {doc['category']}")
        print(f"   Rule: {doc['content']}")
