from typing_extensions import Literal
from config import MAX_VERIFY_ATTEMPTS, DIRECT_ANSWER_THRESHOLD
from state import State


def route_start(state: State) -> Literal["summarize_history", "fetch_long_term_memory"]:
    from nodes import should_summarize
    if should_summarize(state):
        return "summarize_history"
    return "fetch_long_term_memory"


def route_after_memory(state: State) -> Literal["direct_answer_node", "planner"]:
    raw = state.get("matched_similarity")
    try:
        similarity = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        similarity = 0.0

    can_answer = bool(state.get("memory_can_answer", False))
    print(
        f"[routing] similarity={similarity:.3f} "
        f"threshold={DIRECT_ANSWER_THRESHOLD} can_answer={can_answer}"
    )

    if similarity >= DIRECT_ANSWER_THRESHOLD and can_answer:
        print("[routing] → DIRECT ANSWER from memory")
        return "direct_answer_node"

    print("[routing] → FULL RESEARCH pipeline")
    return "planner"


def route_after_verification(state: State) -> Literal["planner", "human_approval"]:
    if state["plan_verified"]:
        return "human_approval"
    if state["verify_attempts"] >= MAX_VERIFY_ATTEMPTS:
        print("[routing] max verify attempts reached, escalating to human")
        return "human_approval"
    return "planner"


def route_after_human(state: State) -> Literal["planner", "start_research"]:
    if state["plan_approved"]:
        return "start_research"
    return "planner"
