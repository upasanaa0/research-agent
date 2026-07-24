from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    conversation_summary: str

    long_term_context: str
    matched_memory_id: str
    matched_similarity: float
    memory_can_answer: bool
    memory_coverage_reason: str
    direct_answer: str

    plan: str
    verify_feedback: str
    verify_attempts: int
    plan_verified: bool
    plan_approved: bool

    worker_1_output: str
    worker_2_output: str
    pdf_output: str
    final_output: str
