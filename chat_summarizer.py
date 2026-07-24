"""Utilities for compressing chat history into a structured context summary.

This module defines prompt templates and helper functions used by a LangGraph
node that trims long conversations. When token or message limits are reached,
the node summarizes prior messages into a single context-preserving message so
the workflow can continue within model context limits.
"""
import functools
import os
import uuid
from typing import Any, Callable, Literal, Union

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT
What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY
Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS
What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS
What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

_DEFAULT_TRIM_TOKEN_LIMIT = 4000
REMOVE_ALL_MESSAGES = "__remove_all__"

TriggerKey = Literal["tokens", "messages"]
KeepKey = Literal["messages"]

TriggerSpec = Union[
    tuple[TriggerKey, int],
    list[tuple[TriggerKey, int]],
    dict[TriggerKey, int],
]
KeepSpec = Union[
    tuple[KeepKey, int],
    list[tuple[KeepKey, int]],
    dict[KeepKey, int],
]
PolicySpec = Union[TriggerSpec, KeepSpec]


@functools.lru_cache(maxsize=16)
def _get_encoding(model_name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for ``model_name``.

    Falls back to the ``cl100k_base`` encoding for models that tiktoken does
    not know about (e.g. Gemini), so this helper is safe across providers.
    Cached so repeated calls are essentially free.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _content_to_text(content: Any) -> str:
    """Coerce a LangChain ``BaseMessage.content`` value to a plain string.

    LangChain message content can be ``str`` or a list of content parts
    (``str`` or ``dict`` with a ``"text"`` key) for multimodal messages.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def count_tokens_approximately(model_name: str, messages: list[BaseMessage]) -> int:
    """Approximate the total number of tokens across ``messages`` for ``model_name``.

    Uses ``tiktoken.Encoding.encode_batch`` so that encoding runs in parallel
    Rust threads with the GIL released, which is several times faster than a
    Python-level per-message loop on long histories.
    """
    if not messages:
        return 0
    encoding = _get_encoding(model_name)
    texts = [_content_to_text(m.content) for m in messages]
    return sum(len(t) for t in encoding.encode_batch(texts))


def render_messages_compact(messages: list[BaseMessage]) -> str:
    """Render ``messages`` as a compact ``[role]\\ncontent`` block.

    The default ``str(messages)`` calls ``repr()`` on every ``BaseMessage``,
    which includes ``additional_kwargs``, ``response_metadata``, IDs, and
    other metadata. For typical histories that is 5–10x larger than the
    actual content. This renderer keeps only what the summarizer model
    needs (role + text + tool calls if present), which dramatically shrinks
    the prompt token count and the in-process string buffer.
    """
    parts: list[str] = []
    for message in messages:
        body = _content_to_text(message.content)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            body = f"{body}\n[tool_calls: {tool_calls}]"
        parts.append(f"[{message.type}]\n{body}")
    return "\n\n".join(parts)


def _normalize_policy(spec: PolicySpec | None) -> list[tuple[str, int]]:
    """Normalize a ``trigger`` or ``keep`` spec to a uniform list of ``(key, value)`` pairs."""
    if spec is None:
        return []
    if isinstance(spec, tuple):
        return [spec]
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        return list(spec.items())
    raise TypeError(
        f"Unsupported policy type: {type(spec).__name__}; "
        "expected tuple, list[tuple], or dict."
    )


def _validate_policy(
    spec: PolicySpec | None,
    name: str,
    allowed_keys: tuple[str, ...],
) -> None:
    """Raise ``ValueError``/``TypeError`` if ``spec`` is not a valid policy."""
    if spec is None:
        return
    items = _normalize_policy(spec)
    if not items:
        raise ValueError(
            f"{name} must contain at least one entry, got empty {type(spec).__name__}."
        )
    for entry in items:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise ValueError(
                f"Invalid {name} entry {entry!r}; expected a (key, value) pair."
            )
        key, value = entry
        if key not in allowed_keys:
            raise ValueError(
                f"Invalid {name} key {key!r}; expected one of {allowed_keys}."
            )
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Invalid {name} value for {key!r}: must be a non-negative int, got {value!r}."
            )


class ChatContextSummaryNode:
    """LangGraph node that summarizes a long chat history into a single message.

    When invoked it inspects the current message list and, if it exceeds the
    configured ``trigger`` threshold, replaces the older portion with a
    single summary ``HumanMessage`` while preserving the most recent messages
    according to ``keep``.

    Args:
        model_name: The chat model used to produce the summary. Names
            starting with ``"gemini"`` use Google GenAI; names starting with
            ``"gpt"`` use OpenAI.
        messages_key: Key in the LangGraph state that holds the message list.
        trigger: When to summarize. Accepts:

            - ``("tokens", n)`` — summarize when total tokens > ``n``
            - ``("messages", n)`` — summarize when message count > ``n``
            - a ``list`` of such tuples — summarize as soon as the **first**
              condition is met (short-circuit any-of)
            - a ``dict`` like ``{"tokens": n, "messages": m}`` — same
              short-circuit any-of semantics as a list
            - ``None`` — never summarize.
        keep: How many of the most recent messages to retain verbatim. The
            ``"messages"`` key is the only valid policy key here; the type
            still accepts ``tuple``, ``list[tuple]``, or ``dict`` shapes for
            consistency. With multiple entries the largest count wins so
            recent context is never accidentally summarized away. Examples:
            ``("messages", 10)``, ``[("messages", 10)]``, ``{"messages": 10}``.
        summary_prompt: Prompt template containing a ``{messages}`` placeholder.
        token_counter: Callable ``(model_name, messages) -> int`` used for
            token-based triggers and trimming. Defaults to a tiktoken-based
            counter with a safe fallback encoding.
        trim_tokens_to_summarize: Maximum number of tokens allowed in the
            generated summary. The summary is truncated if it exceeds this
            limit. Set to ``None`` to disable trimming.
        message_renderer: Callable ``(messages) -> str`` used to format the
            history into the ``{messages}`` slot of the prompt. Defaults to
            :func:`render_messages_compact`, which is much smaller than
            ``str(messages)``. Pass ``str`` to fall back to the legacy
            verbose rendering.
    """

    _TRIGGER_ORDER = {"messages": 0, "tokens": 1}

    def __init__(
        self,
        model_name: str,
        messages_key: str = "messages",
        trigger: TriggerSpec | None = ("tokens", 40000),
        keep: KeepSpec | None = ("messages", 10),
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        token_counter: Callable[[str, list[BaseMessage]], int] = count_tokens_approximately,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        message_renderer: Callable[[list[BaseMessage]], str] = render_messages_compact,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(messages_key, str) or not messages_key:
            raise ValueError("messages_key must be a non-empty string.")
        _validate_policy(trigger, "trigger", allowed_keys=("tokens", "messages"))
        _validate_policy(keep, "keep", allowed_keys=("messages",))
        if "{messages}" not in summary_prompt:
            raise ValueError("summary_prompt must contain a '{messages}' placeholder.")
        if trim_tokens_to_summarize is not None and (
            not isinstance(trim_tokens_to_summarize, int)
            or isinstance(trim_tokens_to_summarize, bool)
            or trim_tokens_to_summarize <= 0
        ):
            raise ValueError(
                "trim_tokens_to_summarize must be a positive int or None, "
                f"got {trim_tokens_to_summarize!r}."
            )

        self._model_name = model_name
        self._messages_key = messages_key
        self._summary_prompt = summary_prompt
        self._token_counter = token_counter
        self._trim_tokens_to_summarize = trim_tokens_to_summarize
        self._message_renderer = message_renderer

        trigger_items = _normalize_policy(trigger)
        self._trigger_items: list[tuple[str, int]] = sorted(
            trigger_items, key=lambda kv: self._TRIGGER_ORDER.get(kv[0], 99)
        )

        keep_items = _normalize_policy(keep)
        self._max_keep: int = max((v for _, v in keep_items), default=0)

        self._cached_model: Any | None = None

    def _get_model(self) -> Any:
        """Return the cached chat model, creating it on first use."""
        if self._cached_model is None:
            self._cached_model = self._create_model_instance()
        return self._cached_model

    def _create_model_instance(self) -> Any:
        """Instantiate the chat model used for summarization."""
        if self._model_name.startswith("gemini"):
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key is None:
                raise ValueError("GOOGLE_API_KEY is not set")
            
            
        if self._model_name.startswith("gpt"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError("OPENAI_API_KEY is not set")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self._model_name,
                temperature=0.2,
                max_tokens=128000,
                api_key=openai_api_key,
            )
        raise ValueError(f"Invalid model name: {self._model_name}")

    def _ensure_message_ids(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Assign a UUID to any message missing an ``id`` (in place)."""
        for message in messages:
            if message.id is None:
                message.id = str(uuid.uuid4())
        return messages

    def _should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Return True as soon as any configured trigger threshold is exceeded.

        Triggers are pre-sorted cheap-first in ``__init__`` (``"messages"``
        before ``"tokens"``) so the O(1) message-count check short-circuits
        before the O(N) token count is ever computed.
        """
        if not self._trigger_items:
            return False
        for key, threshold in self._trigger_items:
            if key == "messages":
                if len(messages) > threshold:
                    return True
            elif key == "tokens":
                if self._token_counter(self._model_name, messages) > threshold:
                    return True
        return False

    def _partition_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split ``messages`` into ``(to_summarize, preserved)``.

        The most recent messages are preserved according to ``self._max_keep``;
        everything older is returned as the segment to be summarized.
        """
        keep_n = self._max_keep
        total = len(messages)
        if keep_n <= 0:
            return messages, []
        if keep_n >= total:
            return [], messages
        return messages[:-keep_n], messages[-keep_n:]

    def _trim_summary(self, summary: str) -> str:
        """Truncate ``summary`` to ``trim_tokens_to_summarize`` tokens, if needed.

        Uses two cheap short-circuits before calling tiktoken:

        1. Conservative safe bound — ``num_tokens ≤ num_bytes ≤ 4·num_chars``,
           so if ``4·len(summary) ≤ limit`` we are guaranteed under budget.
        2. Practical English bound — BPE averages ~3–4 chars/token, so if
           ``len(summary) ≤ limit`` the token count is almost certainly ≤
           ``limit``. We still verify with tiktoken before returning, but this
           avoids the encode entirely for the majority of short summaries.
        """
        limit = self._trim_tokens_to_summarize
        if limit is None:
            return summary
        # Fast guaranteed safe path: 4·chars ≤ limit → tokens ≤ limit for certain.
        if len(summary) * 4 <= limit:
            return summary
        encoding = _get_encoding(self._model_name)
        tokens = encoding.encode(summary)
        if len(tokens) <= limit:
            return summary
        return encoding.decode(tokens[:limit])

    def _create_summary(self, messages: list[BaseMessage]) -> str:
        """Invoke the configured chat model and return the summary text."""
        model = self._get_model()
        rendered = self._message_renderer(messages)
        response = model.invoke(self._summary_prompt.format(messages=rendered))
        return _content_to_text(response.content)

    def _create_summary_message(self, summary: str) -> BaseMessage:
        """Wrap ``summary`` in a HumanMessage tagged with this node's source."""
        return HumanMessage(
            content=summary,
            additional_kwargs={"lc_source": "chat_context_summary"},
        )

    def summarize(self, state: dict[str, Any]) -> dict[str, list[BaseMessage]]:
        """LangGraph node entry point.

        Returns a partial state update. When summarization is not required,
        returns an empty dict (no-op). When triggered, returns a state update
        that removes all existing messages and replaces them with the summary
        followed by the preserved tail of the history.
        """
        messages = state.get(self._messages_key, [])
        if not isinstance(messages, list):
            raise TypeError(
                f"State key {self._messages_key!r} must be a list of BaseMessage, "
                f"got {type(messages).__name__}."
            )
        if not self._should_summarize(messages):
            return {}

        self._ensure_message_ids(messages)
        messages_to_summarize, preserved_messages = self._partition_messages(messages)
        if not messages_to_summarize:
            return {}

        summary = self._create_summary(messages_to_summarize)
        trimmed_summary = self._trim_summary(summary)
        summary_message = self._create_summary_message(trimmed_summary)

        return {
            self._messages_key: [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_message,
                *preserved_messages,
            ]
        }