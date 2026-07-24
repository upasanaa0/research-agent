from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    planner, verifier, human_approval, start_research,
    parallel_workers, pdf_agent_node, final_output_agent,
    summarize_history, fetch_long_term_memory, check_memory_coverage,
    save_long_term_memory, direct_answer_node
)
from routing import (
    route_start, route_after_memory,
    route_after_verification, route_after_human
)


def build_graph(checkpointer):
    builder = StateGraph(State)

    # All nodes
    builder.add_node("summarize_history", summarize_history)
    builder.add_node("fetch_long_term_memory", fetch_long_term_memory)
    builder.add_node("check_memory_coverage", check_memory_coverage)
    builder.add_node("direct_answer_node", direct_answer_node)
    builder.add_node("planner", planner)
    builder.add_node("verifier", verifier)
    builder.add_node("human_approval", human_approval)
    builder.add_node("start_research", start_research)
    builder.add_node("web_search_research", parallel_workers)
    builder.add_node("paper_pdf_research", pdf_agent_node)
    builder.add_node("final_output_agent", final_output_agent)
    builder.add_node("save_long_term_memory", save_long_term_memory)

    # START → summarize check
    builder.add_conditional_edges(
        START,
        route_start,
        {
            "summarize_history": "summarize_history",
            "fetch_long_term_memory": "fetch_long_term_memory"
        }
    )

    # After summarization → fetch memory
    builder.add_edge("summarize_history", "fetch_long_term_memory")

    # After memory fetch → coverage check → route: direct answer OR full research
    builder.add_edge("fetch_long_term_memory", "check_memory_coverage")
    builder.add_conditional_edges(
        "check_memory_coverage",
        route_after_memory,
        {
            "direct_answer_node": "direct_answer_node",
            "planner": "planner"
        }
    )

    # Direct answer → save (no-op for direct) → END
    builder.add_edge("direct_answer_node", "save_long_term_memory")

    # Full research pipeline
    builder.add_edge("planner", "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"planner": "planner", "human_approval": "human_approval"}
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"planner": "planner", "start_research": "start_research"}
    )

    # Fan-out: web search and PDF research run in parallel
    builder.add_edge("start_research", "web_search_research")
    builder.add_edge("start_research", "paper_pdf_research")

    # Fan-in: final output waits for both branches
    builder.add_edge("web_search_research", "final_output_agent")
    builder.add_edge("paper_pdf_research", "final_output_agent")

    builder.add_edge("final_output_agent", "save_long_term_memory")
    builder.add_edge("save_long_term_memory", END)

    return builder.compile(checkpointer=checkpointer)
