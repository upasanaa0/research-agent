import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from config import DB_URI
from graph import build_graph
from vector_memory import fetch_all_memories
from history_store import (
    setup_history_table,
    save_session,
    fetch_history_for_thread,
    fetch_all_history,
    fetch_all_threads
)


def get_initial_state(user_question: str) -> dict:
    return {
        "plan": "",
        "verify_feedback": "",
        "verify_attempts": 0,
        "plan_verified": False,
        "plan_approved": False,
        "worker_1_output": "",
        "worker_2_output": "",
        "pdf_output": "",
        "final_output": "",
        "direct_answer": "",
        "conversation_summary": "",
        "long_term_context": "",
        "matched_memory_id": "",
        "matched_similarity": 0.0,
        "memory_can_answer": False,
        "memory_coverage_reason": "",
        "messages": [HumanMessage(content=user_question)]
    }


def print_memory_store():
    memories = fetch_all_memories()
    if not memories:
        print("\n(No memories stored yet.)")
        return
    print(f"\n{'='*60}")
    print(f"VECTOR MEMORY STORE ({len(memories)} topics)")
    print(f"{'='*60}")
    for m in memories:
        print(f"\nTopic:    {m['topic']}")
        print(f"Hits:     {m['hit_count']}")
        print(f"Updated:  {m['last_updated'][:10]}")
        qs = [q for q in m['source_questions'] if q]
        for q in qs[:3]:
            print(f"  Q: {q[:80]}")
        print(f"Snapshot: {m['snapshot'][:200]}...")


def print_thread_history(thread_id: str):
    sessions = fetch_history_for_thread(thread_id)
    if not sessions:
        print(f"\n(No history found for thread {thread_id})")
        return
    print(f"\n{'='*60}")
    print(f"HISTORY FOR THREAD: {thread_id} ({len(sessions)} session(s))")
    print(f"{'='*60}")
    for s in sessions:
        print(f"\n[{s['asked_at'][:16]}] [{s['answered_from'].upper()}]")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:500]}{'...' if len(s['answer']) > 500 else ''}")
        print("-" * 40)


def print_all_threads():
    threads = fetch_all_threads()
    if not threads:
        print("\n(No sessions recorded yet.)")
        return
    print(f"\n{'='*60}")
    print(f"ALL THREADS ({len(threads)} total)")
    print(f"{'='*60}")
    for t in threads:
        print(f"\nThread:    {t['thread_id']}")
        print(f"Questions: {t['question_count']}")
        print(f"First:     {t['first_asked'][:16]}")
        print(f"Last:      {t['last_asked'][:16]}")


def print_all_history():
    sessions = fetch_all_history(limit=50)
    if not sessions:
        print("\n(No history found.)")
        return
    print(f"\n{'='*60}")
    print(f"RECENT HISTORY (last {len(sessions)} sessions)")
    print(f"{'='*60}")
    for s in sessions:
        print(f"\n[{s['asked_at'][:16]}] Thread: {s['thread_id']} [{s['answered_from'].upper()}]")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:300]}{'...' if len(s['answer']) > 300 else ''}")
        print("-" * 40)


def run_research_session():
    """Run a single research session and return."""
    print("\nEnter a thread ID to continue a session, or press Enter for a new one:")
    thread_input = input("Thread ID: ").strip()
    thread_id = thread_input if thread_input else f"task-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    print(f"[thread_id] {thread_id}")

    existing = fetch_history_for_thread(thread_id)
    if existing:
        show = input(f"\nFound {len(existing)} prior Q&A(s) for this thread. Show? (y/n): ").strip().lower()
        if show == "y":
            print_thread_history(thread_id)

    user_question = input("\nEnter your research question: ").strip()
    if not user_question:
        print("No question provided.")
        return

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        initial_state = get_initial_state(user_question)
        result = graph.invoke(initial_state, config=cfg)

        while "__interrupt__" in result:
            data = result["__interrupt__"][0].value
            print("\n" + "=" * 60)
            print("RESEARCH PLAN FOR REVIEW")
            print("=" * 60)
            print(data["plan"])
            decision = input("\nApprove this plan? (y/n): ").strip().lower()
            result = graph.invoke(Command(resume=decision == "y"), config=cfg)

        final_answer = result.get("final_output", "")
        is_direct    = bool(result.get("direct_answer", ""))
        answered_from = "memory" if is_direct else "research"

        print("\n" + "=" * 60)
        print(f"ANSWER (from {answered_from})")
        print("=" * 60 + "\n")
        print(final_answer)

        if final_answer:
            save_session(
                thread_id=thread_id,
                question=user_question,
                answer=final_answer,
                answered_from=answered_from
            )


def main():
    setup_history_table()

    while True:
        print("\n" + "="*60)
        print("RESEARCH PIPELINE")
        print("="*60)
        print("\nOptions:")
        print("  1. Start / continue research session")
        print("  2. View memory store (topics learned)")
        print("  3. View history for a thread")
        print("  4. View all threads")
        print("  5. View recent history (all threads)")
        print("  6. Exit")

        choice = input("\nChoice (1-6): ").strip()

        if choice == "1":
            run_research_session()

        elif choice == "2":
            print_memory_store()

        elif choice == "3":
            thread_id = input("Enter thread ID: ").strip()
            if thread_id:
                print_thread_history(thread_id)

        elif choice == "4":
            print_all_threads()

        elif choice == "5":
            print_all_history()

        elif choice == "6":
            print("Bye.")
            break

        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()