import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import interrupt

from llm import llm
from prompts import SYSTEM_PROMPTS
from state import State
from tool_executor import invoke_with_tools
from pdf_agent import pdf_research_agent
from vector_memory import (
    fetch_relevant_memory,
    fetch_candidate_memories,
    get_memory_by_id,
    create_memory,
    update_memory_by_id,
    increment_hit_count,
)
from memory_utils import (
    get_current_user_question,
    parse_llm_json,
    build_snapshot_text,
    cap_snapshot,
)
from chat_summarizer import ChatContextSummaryNode
from config import (
    DIRECT_ANSWER_THRESHOLD,
    MEMORY_CONTEXT_THRESHOLD,
    MEMORY_CANDIDATE_TOP_K,
    MEMORY_MAX_SNAPSHOT_CHARS,
)


# ─────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────

class OllamaSummaryNode(ChatContextSummaryNode):
    def _create_model_instance(self):
        return llm


_summarizer = OllamaSummaryNode(
    model_name="minimax-m2.5:cloud",
    trigger=("messages", 20),
    keep=("messages", 6),
    trim_tokens_to_summarize=4000,
)


def summarize_history(state: State) -> dict:
    print("[summarizer] Checking if summarization needed...")
    result = _summarizer.summarize(state)
    if result:
        print("[summarizer] Summarized. Kept last 6 messages.")
    else:
        print("[summarizer] No summarization needed yet.")
    return result


def should_summarize(state: State) -> bool:
    return len(state.get("messages", [])) > 20


# ─────────────────────────────────────────────
# VECTOR MEMORY FETCH
# ─────────────────────────────────────────────

def fetch_long_term_memory(state: State) -> dict:
    user_question = get_current_user_question(state)

    if not user_question:
        return {
            "long_term_context": "",
            "matched_memory_id": "",
            "matched_similarity": 0.0,
            "memory_can_answer": False,
            "memory_coverage_reason": "No user question found.",
        }

    memories = fetch_relevant_memory(user_question, top_k=MEMORY_CANDIDATE_TOP_K)

    if not memories:
        print("[memory] No relevant memories found in vector DB.")
        return {
            "long_term_context": "",
            "matched_memory_id": "",
            "matched_similarity": 0.0,
            "memory_can_answer": False,
            "memory_coverage_reason": "No relevant memories found.",
        }

    best = memories[0]
    similarity = float(best["similarity"])
    print(f"[memory] Best match: topic='{best['topic']}' similarity={similarity:.3f} hits={best['hit_count']}")

    parts = []
    for m in memories:
        if float(m["similarity"]) > MEMORY_CONTEXT_THRESHOLD:
            parts.append(
                f"Topic: {m['topic']}\n"
                f"Similarity: {m['similarity']:.2f}\n\n"
                f"{m['snapshot']}"
            )

    context = "PRIOR RESEARCH MEMORY:\n\n" + "\n\n---\n\n".join(parts) if parts else ""

    result = {
        "long_term_context": context,
        "matched_memory_id": str(best["id"]) if context else "",
        "matched_similarity": similarity if context else 0.0,
        "memory_can_answer": False,
        "memory_coverage_reason": "",
    }
    print(f"[memory] Returning to state: matched_similarity={result['matched_similarity']}")
    return result


def check_memory_coverage(state: State) -> dict:
    """LLM gate: only allow direct answers when memory truly covers the question."""
    similarity = float(state.get("matched_similarity", 0.0) or 0.0)
    context = state.get("long_term_context", "")

    if similarity < DIRECT_ANSWER_THRESHOLD or not context:
        reason = (
            "Similarity below direct-answer threshold."
            if similarity < DIRECT_ANSWER_THRESHOLD
            else "No usable memory context."
        )
        print(f"[memory_coverage] Skipping coverage check: {reason}")
        return {"memory_can_answer": False, "memory_coverage_reason": reason}

    user_question = get_current_user_question(state)
    print("[memory_coverage] Running LLM coverage check...")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["memory_coverage_checker"]),
        HumanMessage(content=(
            f"User question:\n{user_question}\n\n"
            f"Prior research memory:\n{context}"
        )),
    ])

    try:
        data = parse_llm_json(response.content)
        can_answer = bool(data.get("can_answer", False))
        reason = data.get("reason", "No reason provided.")
        missing = data.get("missing_information", [])
        if missing and not can_answer:
            reason = f"{reason} Missing: {', '.join(missing[:3])}"
    except Exception as e:
        can_answer = False
        reason = f"Coverage check parse failed: {e}"

    print(f"[memory_coverage] can_answer={can_answer} reason={reason}")
    return {"memory_can_answer": can_answer, "memory_coverage_reason": reason}


def _format_candidates_for_decider(candidates: list[dict]) -> str:
    if not candidates:
        return "No existing memory candidates."
    blocks = []
    for c in candidates:
        blocks.append(
            f"ID: {c['id']}\n"
            f"Topic: {c.get('topic', '')}\n"
            f"Similarity: {c.get('similarity', 'n/a')}\n"
            f"Snapshot:\n{c.get('snapshot', '')[:1500]}"
        )
    return "\n\n---\n\n".join(blocks)


def _merge_snapshots_with_llm(existing_snapshot: str, new_snapshot: str) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["snapshot_merger"]),
        HumanMessage(content=(
            f"Snapshot 1 (existing memory):\n{existing_snapshot[:3000]}\n\n"
            f"Snapshot 2 (new session):\n{new_snapshot[:3000]}"
        )),
    ])
    return parse_llm_json(response.content)


def _collect_write_candidates(state: State, new_snapshot: str) -> list[dict]:
    candidates = fetch_candidate_memories(new_snapshot, top_k=MEMORY_CANDIDATE_TOP_K)
    by_id = {c["id"]: c for c in candidates}

    matched_id = state.get("matched_memory_id", "")
    if matched_id and matched_id not in by_id:
        existing = get_memory_by_id(matched_id)
        if existing:
            existing["similarity"] = state.get("matched_similarity", 0.0)
            by_id[matched_id] = existing

    return list(by_id.values())

# ─────────────────────────────────────────────
# DIRECT ANSWER NODE (short circuit)
# ─────────────────────────────────────────────

def direct_answer_node(state: State) -> dict:
    """
    Called when similarity and LLM coverage both pass.
    Answers directly from memory and increments hit counter.
    """
    user_question = get_current_user_question(state)

    print(
        f"[direct_answer] Answering from memory "
        f"(similarity={state.get('matched_similarity', 0):.3f}, "
        f"coverage={state.get('memory_coverage_reason', '')})"
    )

    memory_id = state.get("matched_memory_id", "")
    if memory_id:
        increment_hit_count(memory_id)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["direct_answer_agent"]),
        HumanMessage(content=(
            f"Memory snapshot:\n{state.get('long_term_context', '')}\n\n"
            f"User question:\n{user_question}"
        ))
    ])

    answer = (
        f"[Answered from research memory — similarity: "
        f"{state.get('matched_similarity', 0):.0%}]\n\n{response.content}"
    )
    return {"final_output": answer, "direct_answer": answer}


# ─────────────────────────────────────────────
# VECTOR MEMORY SAVE (end of pipeline)
# ─────────────────────────────────────────────

def save_long_term_memory(state: State) -> dict:
    """
    After final answer: extract snapshot, let the LLM decide create/update/ignore,
    then persist only the compact LLM-produced snapshot in ChromaDB.
    """
    if state.get("direct_answer"):
        print("[memory_save] Direct answer session — skipping save.")
        return {}

    user_question = get_current_user_question(state)
    final_answer = state.get("final_output", "")
    if not user_question or not final_answer:
        print("[memory_save] No question or answer — skipping.")
        return {}

    print("[memory_save] Extracting snapshot...")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["snapshot_extractor"]),
        HumanMessage(content=(
            f"User question:\n{user_question}\n\n"
            f"Final answer:\n{final_answer[:4000]}"
        )),
    ])

    try:
        data = parse_llm_json(response.content)

        if not data.get("worth_saving", True):
            print("[memory_save] LLM marked as not worth saving.")
            return {}

        topic = data.get("topic", "unknown")
        snapshot = data.get("snapshot", "")
        key_facts = data.get("key_facts", [])

        if not snapshot:
            print("[memory_save] Empty snapshot — skipping.")
            return {}

        new_snapshot = cap_snapshot(
            build_snapshot_text(snapshot, key_facts),
            MEMORY_MAX_SNAPSHOT_CHARS,
        )

        candidates = _collect_write_candidates(state, new_snapshot)
        print("[memory_save] Asking LLM for memory write decision...")
        decision_response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPTS["memory_write_decider"]),
            HumanMessage(content=(
                f"User question:\n{user_question}\n\n"
                f"New memory candidate:\n"
                f"Topic: {topic}\n"
                f"Snapshot:\n{new_snapshot}\n\n"
                f"Existing memory candidates:\n"
                f"{_format_candidates_for_decider(candidates)}"
            )),
        ])
        decision = parse_llm_json(decision_response.content)
        action = decision.get("action", "create")
        target_id = decision.get("target_memory_id")
        topic = decision.get("topic", topic) or topic
        print(
            f"[memory_save] LLM decision: action={action} "
            f"target={target_id} reason={decision.get('reason', '')}"
        )

        if action == "ignore":
            print("[memory_save] LLM chose to ignore memory write.")
            return {}

        if action == "update":
            if not target_id:
                print("[memory_save] Update requested without target_memory_id — creating instead.")
                action = "create"
            else:
                existing = get_memory_by_id(str(target_id))
                if not existing:
                    print(f"[memory_save] Target memory missing ({target_id}) — creating instead.")
                    action = "create"
                else:
                    merge_data = _merge_snapshots_with_llm(existing["snapshot"], new_snapshot)
                    merged_topic = merge_data.get("topic", topic)
                    merged_snapshot = cap_snapshot(
                        build_snapshot_text(
                            merge_data.get("snapshot", new_snapshot),
                            merge_data.get("key_facts", key_facts),
                        ),
                        MEMORY_MAX_SNAPSHOT_CHARS,
                    )
                    result = update_memory_by_id(
                        memory_id=str(target_id),
                        topic=merged_topic,
                        snapshot=merged_snapshot,
                        source_question=user_question,
                    )
                    print(f"[memory_save] {result}: topic='{merged_topic}'")
                    return {}

        create_memory(topic=topic, snapshot=new_snapshot, source_question=user_question)
        print(f"[memory_save] created: topic='{topic}'")

    except Exception as e:
        print(f"[memory_save] Failed: {e}")

    return {}


# ─────────────────────────────────────────────
# PIPELINE NODES
# ─────────────────────────────────────────────

def planner(state: State):
    print("\n[planner]")

    feedback = state.get("verify_feedback", "")

    # Only pass HumanMessages — strip any AI/tool messages from prior rounds
    # to prevent context overflow from accumulated tool results
    from langchain_core.messages import HumanMessage as HM, AIMessage, ToolMessage
    from memory_utils import SUMMARY_SOURCE
    prompt_parts = [
        msg for msg in state["messages"]
        if isinstance(msg, HM)
        and msg.additional_kwargs.get("lc_source") != SUMMARY_SOURCE
    ]

    summary = state.get("conversation_summary", "")
    if summary:
        prompt_parts = [HumanMessage(content=f"[Conversation Summary]\n{summary}")] + prompt_parts

    lt_context = state.get("long_term_context", "")
    if lt_context:
        prompt_parts = [HumanMessage(content=lt_context)] + prompt_parts

    if feedback:
        prompt_parts.append(
            HumanMessage(content=f"Verifier Feedback:\n\n{feedback}\n\nRevise the plan accordingly.")
        )
    else:
        prompt_parts.append(
            HumanMessage(content="Generate a research plan for the above query.")
        )

    response_content = invoke_with_tools(SYSTEM_PROMPTS["planner"], prompt_parts)
    return {"plan": response_content, "plan_verified": False}


def verifier(state: State):
    attempts = state.get("verify_attempts", 0) + 1
    print(f"\n[verifier] attempt={attempts}")

    user_question = get_current_user_question(state)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["verifier"]),
        HumanMessage(content=(
            f"Original user query:\n{user_question}\n\n"
            f"Planner output:\n{state['plan']}"
        ))
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        approved = bool(data.get("final_decision", {}).get("approved_for_execution", False))
        feedback = data.get("final_decision", {}).get("reason", "No reason provided.")
    except Exception as e:
        approved = False
        feedback = f"Verifier returned invalid JSON: {e}"

    print(f"[verifier] approved={approved}")
    return {
        "verify_attempts": attempts,
        "verify_feedback": feedback,
        "plan_verified": approved
    }


def human_approval(state: State):
    approved = interrupt({
        "message": "Approve this research plan?",
        "plan": state["plan"]
    })
    return {"plan_approved": approved}


def start_research(state: State):
    """No-op fan-out node: triggers parallel web and PDF research branches."""
    print("\n[start_research] Launching parallel web + PDF research branches")
    return {}


def _research_memory_message(state: State) -> list[HumanMessage]:
    """Provide prior memory as active research context for worker branches."""
    lt_context = state.get("long_term_context", "")
    if not lt_context:
        return []

    return [HumanMessage(content=(
        "Relevant long-term research memory was found for this question. "
        "Use it as continuation context for the current research: reuse useful "
        "facts, avoid repeating already-covered background, verify or extend "
        "important claims with fresh sources, and clearly note when current "
        "findings confirm, update, or contradict the memory.\n\n"
        f"{lt_context}"
    ))]


def run_worker_1(state):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {threading.current_thread().name} | Worker 1 START")
    result = invoke_with_tools(
        SYSTEM_PROMPTS["worker_1"],
        [
            *_research_memory_message(state),
            *state["messages"],
            HumanMessage(content=f"Approved Research Plan:\n\n{state['plan']}\n\nExecute the FIRST HALF of the subtasks.")
        ]
    )
    print(f"{datetime.now().strftime('%H:%M:%S')} | {threading.current_thread().name} | Worker 1 END")
    return result


def run_worker_2(state):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {threading.current_thread().name} | Worker 2 START")
    result = invoke_with_tools(
        SYSTEM_PROMPTS["worker_2"],
        [
            *_research_memory_message(state),
            *state["messages"],
            HumanMessage(content=f"Approved Research Plan:\n\n{state['plan']}\n\nExecute the SECOND HALF of the subtasks.")
        ]
    )
    print(f"{datetime.now().strftime('%H:%M:%S')} | {threading.current_thread().name} | Worker 2 END")
    return result


def parallel_workers(state: State):
    print("\n[parallel_workers] START")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(run_worker_1, state)
        future_2 = executor.submit(run_worker_2, state)
        worker_1_output = future_1.result()
        worker_2_output = future_2.result()
    print(f"\n[parallel_workers] FINISHED in {time.perf_counter() - start:.2f} sec")
    return {"worker_1_output": worker_1_output, "worker_2_output": worker_2_output}


def pdf_agent_node(state: State):
    return pdf_research_agent(state)


def final_output_agent(state: State):
    user_question = get_current_user_question(state)

    lt_context = state.get("long_term_context", "")
    summary_context = state.get("conversation_summary", "")

    context_block = ""
    if lt_context:
        context_block += f"{lt_context}\n\n"
    if summary_context:
        context_block += f"[Conversation Summary]\n{summary_context}\n\n"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPTS["final"]),
        HumanMessage(content=(
            f"{context_block}"
            f"User's original question:\n{user_question}\n\n"
            f"Research Plan:\n\n{state['plan']}\n\n"
            f"Worker 1 Findings:\n\n{state['worker_1_output']}\n\n"
            f"Worker 2 Findings:\n\n{state['worker_2_output']}\n\n"
            f"Academic PDF Summaries:\n\n{state.get('pdf_output', 'No PDF data available.')}"
        ))
    ])
    return {"final_output": response.content}