"""Shared helpers for long-term memory read/write flows."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

SUMMARY_SOURCE = "chat_context_summary"


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def get_current_user_question(state: dict) -> str:
    """Return the latest real user question, skipping conversation summaries."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        if msg.additional_kwargs.get("lc_source") == SUMMARY_SOURCE:
            continue
        content = _content_to_str(msg.content).strip()
        if content:
            return content
    return ""


def parse_llm_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def build_snapshot_text(snapshot: str, key_facts: list[str]) -> str:
    facts = [f for f in key_facts if f]
    if not facts:
        return snapshot.strip()
    return (
        f"{snapshot.strip()}\n\nKey facts:\n"
        + "\n".join(f"- {fact}" for fact in facts)
    )


def cap_snapshot(snapshot: str, max_chars: int) -> str:
    snapshot = snapshot.strip()
    if len(snapshot) <= max_chars:
        return snapshot
    return snapshot[: max_chars - 20].rstrip() + "\n...[truncated]"


def sanitize_topic_id(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return (slug or "unknown_topic")[:50]
