"""
Vector memory store using ChromaDB.

ChromaDB is used for retrieval and persistence only.
Semantic create/update/ignore decisions are made by the LLM in nodes.py.
"""

from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, MEMORY_COLLECTION, MEMORY_MAX_SNAPSHOT_CHARS
from memory_utils import cap_snapshot, sanitize_topic_id


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _format_memory_result(doc_id: str, document: str, meta: dict, distance: float) -> dict:
    similarity = 1 - distance
    return {
        "id": doc_id,
        "topic": meta.get("topic", ""),
        "snapshot": document,
        "similarity": similarity,
        "hit_count": int(meta.get("hit_count", 0)),
        "source_questions": meta.get("source_questions", "").split("|||"),
        "last_updated": meta.get("last_updated", ""),
    }


def fetch_relevant_memory(question: str, top_k: int = 3) -> list[dict]:
    """Query ChromaDB for memories relevant to the question."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    memories = []
    for i, doc_id in enumerate(results["ids"][0]):
        memories.append(
            _format_memory_result(
                doc_id,
                results["documents"][0][i],
                results["metadatas"][0][i],
                results["distances"][0][i],
            )
        )
    return memories


def fetch_candidate_memories(query: str, top_k: int = 3) -> list[dict]:
    """Alias for candidate retrieval during LLM write decisions."""
    return fetch_relevant_memory(query, top_k=top_k)


def get_memory_by_id(memory_id: str) -> dict | None:
    collection = _get_collection()
    result = collection.get(ids=[memory_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    meta = result["metadatas"][0]
    return {
        "id": memory_id,
        "topic": meta.get("topic", ""),
        "snapshot": result["documents"][0],
        "hit_count": int(meta.get("hit_count", 0)),
        "source_questions": meta.get("source_questions", "").split("|||"),
        "last_updated": meta.get("last_updated", ""),
    }


def create_memory(topic: str, snapshot: str, source_question: str) -> str:
    """Create a new memory topic. Returns 'created'."""
    collection = _get_collection()
    snapshot = cap_snapshot(snapshot, MEMORY_MAX_SNAPSHOT_CHARS)

    topic_id = sanitize_topic_id(topic)
    existing_ids = collection.get(ids=[topic_id])
    if existing_ids["ids"]:
        topic_id = f"{topic_id}_{datetime.now().strftime('%H%M%S')}"

    collection.add(
        ids=[topic_id],
        documents=[snapshot],
        metadatas=[{
            "topic": topic,
            "source_questions": source_question,
            "hit_count": 0,
            "last_updated": datetime.now().isoformat(),
        }],
    )
    print(f"[vector_memory] Created new topic: '{topic}'")
    return "created"


def update_memory_by_id(
    memory_id: str,
    topic: str,
    snapshot: str,
    source_question: str,
) -> str:
    """Update an existing memory by ID. Returns 'updated' or 'missing'."""
    collection = _get_collection()
    result = collection.get(ids=[memory_id], include=["metadatas"])
    if not result["ids"]:
        print(f"[vector_memory] Memory id not found: {memory_id}")
        return "missing"

    meta = result["metadatas"][0]
    snapshot = cap_snapshot(snapshot, MEMORY_MAX_SNAPSHOT_CHARS)

    existing_questions = meta.get("source_questions", "")
    questions_list = [q for q in existing_questions.split("|||") if q]
    if source_question and source_question not in questions_list:
        questions_list.append(source_question)

    collection.update(
        ids=[memory_id],
        documents=[snapshot],
        metadatas=[{
            "topic": topic or meta.get("topic", "unknown"),
            "source_questions": "|||".join(questions_list),
            "hit_count": int(meta.get("hit_count", 0)),
            "last_updated": datetime.now().isoformat(),
        }],
    )
    print(f"[vector_memory] Updated topic: '{topic}' (id={memory_id})")
    return "updated"


def increment_hit_count(memory_id: str):
    """Called when a memory answers a question directly — tracks usage."""
    collection = _get_collection()
    result = collection.get(ids=[memory_id], include=["metadatas"])
    if result["ids"]:
        meta = result["metadatas"][0]
        meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
        collection.update(ids=[memory_id], metadatas=[meta])
        print(f"[vector_memory] Hit count for '{meta.get('topic')}': {meta['hit_count']}")


def fetch_all_memories() -> list[dict]:
    """For display/debugging."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["documents", "metadatas"])
    memories = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        memories.append({
            "id": doc_id,
            "topic": meta.get("topic", ""),
            "snapshot": results["documents"][i][:300],
            "hit_count": meta.get("hit_count", 0),
            "source_questions": meta.get("source_questions", "").split("|||"),
            "last_updated": meta.get("last_updated", ""),
        })
    return memories
